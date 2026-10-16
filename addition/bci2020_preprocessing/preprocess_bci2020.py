"""Create model-ready BCI2020 Track-3 epochs with a standard MNE-style flow.

Pipeline: restore the original epochs from BIDS events -> trial-wise 0.5--40 Hz
zero-phase Butterworth filtering -> common-average reference -> conservative
automatic amplitude/flatness rejection -> save MNE Epochs plus an audit table.

The BIDS EDF is a concatenation of originally independent epochs, so filtering
is deliberately applied after epoch restoration and never crosses a boundary.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mne
import numpy as np
import pandas as pd
from scipy import signal


CLASS_NAMES = ("Hello", "Help me", "Stop", "Thank you", "Yes")
EVENT_ID = {name: index + 1 for index, name in enumerate(CLASS_NAMES)}
RECORD_PATTERN = re.compile(
    r"^(sub-(?P<subject>\d+))_task-imaginedSpeech_run-(?P<run>\d+)_eeg\.edf$"
)


@dataclass(frozen=True)
class Record:
    subject: str
    run: str
    edf: Path
    events: Path
    channels: Path


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the configuration") from exc
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    for section in ("dataset", "preprocessing", "rejection", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping section: {section}")
    return config


def _subject(value: str | int) -> str:
    text = str(value).strip().casefold().removeprefix("sub-")
    if not text.isdigit():
        raise ValueError(f"Invalid subject: {value!r}")
    return f"sub-{int(text):02d}"


def _run(value: str | int) -> str:
    text = str(value).strip().casefold().removeprefix("run-")
    if not text.isdigit():
        raise ValueError(f"Invalid run: {value!r}")
    return f"{int(text):02d}"


def discover_records(
    root: str | Path,
    subjects: Iterable[str | int] | str = "auto",
    runs: Iterable[str | int] | str = "auto",
) -> tuple[Record, ...]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BCI2020 root does not exist: {root}")
    selected_subjects = None if subjects == "auto" else {_subject(x) for x in subjects}
    selected_runs = None if runs == "auto" else {_run(x) for x in runs}
    records: list[Record] = []
    for edf in sorted(root.glob("sub-*/eeg/*_task-imaginedSpeech_run-*_eeg.edf")):
        match = RECORD_PATTERN.fullmatch(edf.name)
        if match is None:
            continue
        subject, run = match.group(1), match.group("run")
        if selected_subjects is not None and subject not in selected_subjects:
            continue
        if selected_runs is not None and run not in selected_runs:
            continue
        prefix = edf.name.removesuffix("_eeg.edf")
        events = edf.with_name(prefix + "_events.tsv")
        channels = edf.with_name(prefix + "_channels.tsv")
        if not events.is_file() or not channels.is_file():
            raise FileNotFoundError(f"Missing BIDS sidecar for {edf}")
        records.append(Record(subject, run, edf, events, channels))
    if not records:
        raise FileNotFoundError(f"No selected BCI2020 records under {root}")
    return tuple(records)


def _read_events(path: Path, sfreq: float) -> pd.DataFrame:
    events = pd.read_csv(path, sep="\t")
    required = {"onset", "duration", "trial_type", "value"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    events = events.copy()
    samples = np.rint(events["onset"].astype(float) * sfreq).astype(np.int64)
    if "sample" in events and not np.array_equal(
        events["sample"].astype(np.int64).to_numpy(), samples.to_numpy()
    ):
        raise ValueError(f"onset/sample mismatch in {path}")
    events["sample"] = samples
    events["n_samples"] = np.rint(
        events["duration"].astype(float) * sfreq
    ).astype(np.int64)
    if events["n_samples"].nunique() != 1:
        raise ValueError(f"Variable trial length in {path}")
    codes = events["value"].astype(np.int64)
    if not set(codes).issubset(set(EVENT_ID.values())):
        raise ValueError(f"Unexpected labels in {path}")
    expected_names = codes.map({code: name for name, code in EVENT_ID.items()})
    if not np.array_equal(
        expected_names.to_numpy(), events["trial_type"].astype(str).to_numpy()
    ):
        raise ValueError(f"trial_type/value mismatch in {path}")
    events["label"] = codes - 1
    events["trial_index"] = np.arange(len(events), dtype=np.int64)
    return events


def load_trials(
    record: Record,
    *,
    expected_sfreq: float = 256.0,
    expected_channels: int = 64,
) -> tuple[np.ndarray, pd.DataFrame, tuple[str, ...], float]:
    raw = mne.io.read_raw_edf(record.edf, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, expected_sfreq):
        raise ValueError(f"Expected {expected_sfreq} Hz, found {sfreq} in {record.edf}")
    channels = pd.read_csv(record.channels, sep="\t")
    names = tuple(
        channels.loc[
            channels["type"].astype(str).str.casefold() == "eeg", "name"
        ].astype(str)
    )
    if len(names) != expected_channels or len(set(names)) != len(names):
        raise ValueError(f"Expected {expected_channels} unique EEG channels in {record.edf}")
    if set(raw.ch_names) != set(names):
        raise ValueError(f"EDF/channels.tsv mismatch in {record.edf}")
    raw.reorder_channels(list(names))
    events = _read_events(record.events, sfreq)
    n_times = int(events["n_samples"].iloc[0])
    trials = []
    for row in events.itertuples(index=False):
        start, stop = int(row.sample), int(row.sample) + n_times
        if start < 0 or stop > raw.n_times:
            raise ValueError(f"Trial outside EDF bounds in {record.edf}")
        trials.append(raw.get_data(start=start, stop=stop))
    return np.stack(trials), events, names, sfreq


def preprocess_trials(
    trials: np.ndarray,
    sfreq: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return preprocessed trials and a raw non-finite-trial mask."""
    nonfinite_trial = ~np.isfinite(trials).all(axis=(1, 2))
    safe = np.asarray(trials, dtype=np.float64).copy()
    for trial, channel in np.argwhere(~np.isfinite(safe).all(axis=2)):
        row = safe[trial, channel]
        finite = np.isfinite(row)
        row[~finite] = np.median(row[finite]) if finite.any() else 0.0
    low = float(config["l_freq_hz"])
    high = float(config["h_freq_hz"])
    order = int(config.get("filter_order", 4))
    if not 0 < low < high < sfreq / 2:
        raise ValueError(f"Invalid bandpass {low}-{high} Hz at {sfreq} Hz")
    sos = signal.butter(order, (low, high), btype="bandpass", fs=sfreq, output="sos")
    filtered = signal.sosfiltfilt(sos, safe, axis=-1, padtype="odd")
    if bool(config.get("average_reference", True)):
        filtered = filtered - filtered.mean(axis=1, keepdims=True)
    return filtered, nonfinite_trial


def automatic_trial_rejection(
    trials: np.ndarray,
    nonfinite_trial: np.ndarray,
    events: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Conservative MNE-style peak-to-peak and flatness rejection."""
    peak_to_peak = np.ptp(trials, axis=2)
    max_ptp_uv = float(config["max_peak_to_peak_uv"])
    min_ptp_uv = float(config["min_peak_to_peak_uv"])
    if not 0 <= min_ptp_uv < max_ptp_uv:
        raise ValueError("Require 0 <= min_peak_to_peak_uv < max_peak_to_peak_uv")
    high = peak_to_peak > max_ptp_uv * 1e-6
    flat = peak_to_peak < min_ptp_uv * 1e-6
    keep = ~(nonfinite_trial | high.any(axis=1) | flat.any(axis=1))
    rows: list[dict[str, Any]] = []
    for index in range(len(trials)):
        reasons: list[str] = []
        if nonfinite_trial[index]:
            reasons.append("nonfinite_raw")
        if high[index].any():
            reasons.append("peak_to_peak_above_threshold")
        if flat[index].any():
            reasons.append("flat_channel_below_threshold")
        rows.append(
            {
                "trial_index": index,
                "original_sample": int(events.iloc[index]["sample"]),
                "original_onset_s": float(events.iloc[index]["onset"]),
                "trial_type": str(events.iloc[index]["trial_type"]),
                "label": int(events.iloc[index]["label"]),
                "max_channel_peak_to_peak_uv": float(peak_to_peak[index].max() * 1e6),
                "min_channel_peak_to_peak_uv": float(peak_to_peak[index].min() * 1e6),
                "channels_above_threshold": int(high[index].sum()),
                "channels_below_flat_threshold": int(flat[index].sum()),
                "keep": bool(keep[index]),
                "reject_reason": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def _save_epochs(
    record: Record,
    trials: np.ndarray,
    events: pd.DataFrame,
    decisions: pd.DataFrame,
    channel_names: tuple[str, ...],
    sfreq: float,
    output_root: Path,
    *,
    overwrite: bool,
) -> Path:
    keep = decisions["keep"].to_numpy(dtype=bool)
    if not keep.any():
        raise RuntimeError(f"All trials were rejected in {record.subject} run-{record.run}")
    info = mne.create_info(channel_names, sfreq, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        match_case=False,
        on_missing="raise",
    )
    kept_events = events.loc[keep].reset_index(drop=True)
    kept_decisions = decisions.loc[keep].reset_index(drop=True)
    codes = kept_events["value"].astype(np.int64).to_numpy()
    present_codes = set(codes.tolist())
    event_id = {name: code for name, code in EVENT_ID.items() if code in present_codes}
    mne_events = np.column_stack(
        (
            kept_events["sample"].astype(np.int64),
            np.zeros(len(kept_events), dtype=np.int64),
            codes,
        )
    )
    metadata = pd.DataFrame(
        {
            "subject": record.subject,
            "run": record.run,
            "original_trial_index": kept_decisions["trial_index"].astype(int),
            "original_sample": kept_decisions["original_sample"].astype(int),
            "trial_type": kept_decisions["trial_type"].astype(str),
            "label": kept_decisions["label"].astype(int),
        }
    )
    epochs = mne.EpochsArray(
        trials[keep].astype(np.float32, copy=False),
        info,
        events=mne_events,
        event_id=event_id,
        tmin=0.0,
        baseline=None,
        metadata=metadata,
        verbose="ERROR",
    )
    out_dir = output_root / record.subject / "eeg"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (
        f"{record.subject}_task-imaginedSpeech_run-{record.run}_desc-preproc_epo.fif"
    )
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}; use --overwrite")
    epochs.save(path, overwrite=overwrite, verbose="ERROR")
    return path


def process_record(
    record: Record,
    config: dict[str, Any],
    output_root: Path,
    *,
    overwrite: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    data_cfg = config["dataset"]
    raw, events, names, sfreq = load_trials(
        record,
        expected_sfreq=float(data_cfg.get("expected_sfreq_hz", 256.0)),
        expected_channels=int(data_cfg.get("expected_eeg_channels", 64)),
    )
    processed, nonfinite = preprocess_trials(raw, sfreq, config["preprocessing"])
    decisions = automatic_trial_rejection(
        processed, nonfinite, events, config["rejection"]
    )
    decisions.insert(0, "run", record.run)
    decisions.insert(0, "subject", record.subject)
    qc_dir = output_root / "qc" / record.subject
    qc_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(
        qc_dir / f"{record.subject}_run-{record.run}_trial_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    output_path = _save_epochs(
        record,
        processed,
        events,
        decisions,
        names,
        sfreq,
        output_root,
        overwrite=overwrite,
    )
    kept = int(decisions["keep"].sum())
    summary: dict[str, Any] = {
        "subject": record.subject,
        "run": record.run,
        "source_edf": str(record.edf),
        "output_fif": str(output_path),
        "sampling_rate_hz": sfreq,
        "n_channels": len(names),
        "n_times": raw.shape[-1],
        "trials_total": len(decisions),
        "trials_kept": kept,
        "trials_rejected": int(len(decisions) - kept),
        "retained_fraction": kept / len(decisions),
    }
    for label, name in enumerate(CLASS_NAMES):
        selected = decisions[decisions["label"] == label]
        summary[f"{name}_total"] = len(selected)
        summary[f"{name}_kept"] = int(selected["keep"].sum())
    return summary, decisions


def run_pipeline(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
    subjects: Iterable[str | int] | str | None = None,
    runs: Iterable[str | int] | str | None = None,
    overwrite: bool = False,
) -> Path:
    config = load_config(config_path)
    data_cfg = config["dataset"]
    source = Path(dataset_root or data_cfg["root"]).expanduser().resolve()
    destination = Path(
        output_root
        or config["output"].get("root")
        or source / "derivatives" / "paper4" / "bci2020_standard"
    ).expanduser().resolve()
    records = discover_records(
        source,
        subjects if subjects is not None else data_cfg.get("subjects", "auto"),
        runs if runs is not None else data_cfg.get("runs", "auto"),
    )
    selected_subjects = (
        subjects if subjects is not None else data_cfg.get("subjects", "auto")
    )
    selected_runs = runs if runs is not None else data_cfg.get("runs", "auto")
    if selected_subjects != "auto" and selected_runs != "auto":
        expected_pairs = {
            (_subject(subject), _run(run))
            for subject in selected_subjects
            for run in selected_runs
        }
        actual_pairs = {(record.subject, record.run) for record in records}
        missing_pairs = sorted(expected_pairs - actual_pairs)
        unexpected_pairs = sorted(actual_pairs - expected_pairs)
        if missing_pairs or unexpected_pairs:
            raise RuntimeError(
                "BCI2020 record grid is incomplete before preprocessing: "
                f"expected={len(expected_pairs)}, found={len(actual_pairs)}, "
                f"missing={missing_pairs}, unexpected={unexpected_pairs}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    summaries, decisions = [], []
    for index, record in enumerate(records, start=1):
        print(f"[{index:02d}/{len(records):02d}] {record.subject} run-{record.run}")
        summary, table = process_record(
            record, config, destination, overwrite=overwrite
        )
        summaries.append(summary)
        decisions.append(table)
        print(
            f"  kept={summary['trials_kept']}/{summary['trials_total']} "
            f"({summary['retained_fraction']:.1%})"
        )
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(
        destination / "preprocessing_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(decisions, ignore_index=True).to_csv(
        destination / "trial_qc_all.csv.gz", index=False, compression="gzip"
    )
    manifest = {
        "dataset": "BCI Competition 2020 Track 3",
        "source_root": str(source),
        "output_root": str(destination),
        "records": len(records),
        "subjects": sorted({item.subject for item in records}),
        "runs": sorted({item.run for item in records}),
        "class_names": list(CLASS_NAMES),
        "config": config,
        "operations": [
            "restore 795-sample trials from BIDS events",
            "trial-wise zero-phase Butterworth bandpass",
            "common-average reference",
            "automatic peak-to-peak and flatness trial rejection",
            "native 256-Hz output; no resampling, ICA or baseline correction",
        ],
    }
    (destination / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + summary_table[
        ["subject", "run", "trials_total", "trials_kept", "retained_fraction"]
    ].to_string(index=False))
    print(f"\nOutput: {destination}")
    return destination


def _selection(value: str | None) -> list[str] | str | None:
    if value is None:
        return None
    if value.strip().casefold() == "auto":
        return "auto"
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--subjects")
    parser.add_argument("--runs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_pipeline(
        args.config,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        subjects=_selection(args.subjects),
        runs=_selection(args.runs),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
