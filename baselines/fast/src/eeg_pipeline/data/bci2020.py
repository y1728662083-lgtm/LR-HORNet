"""Paper4 adapter for standard-preprocessed BCI2020 Track-3 epochs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .ds003626 import DS003626TaskData


CLASS_NAMES = ("Hello", "Help me", "Stop", "Thank you", "Yes")
EVENT_ID = {name: index + 1 for index, name in enumerate(CLASS_NAMES)}
TASK_ALIASES = {
    "imagined_speech": "imagined_speech",
    "imaginedspeech": "imagined_speech",
    "inner_speech": "imagined_speech",
    "track3": "imagined_speech",
}


def _normalize_subject(value: str | int) -> str:
    text = str(value).strip().casefold().removeprefix("sub-")
    if not text.isdecimal():
        raise ValueError(f"Invalid BCI2020 subject: {value!r}")
    subject = int(text)
    if not 1 <= subject <= 15:
        raise ValueError(f"BCI2020 subject must be 1..15, got {value!r}")
    return f"sub-{subject:02d}"


def _normalize_run(value: str | int) -> str:
    text = str(value).strip().casefold().removeprefix("run-")
    if not text.isdecimal():
        raise ValueError(f"Invalid BCI2020 run: {value!r}")
    run = int(text)
    if run not in (0, 1, 2):
        raise ValueError(f"BCI2020 run must be 0, 1 or 2, got {value!r}")
    return f"{run:02d}"


def _normalize_task(value: str) -> str:
    try:
        return TASK_ALIASES[str(value).strip().casefold()]
    except KeyError:
        raise ValueError(
            "BCI2020 task must be 'imagined_speech' (alias: inner_speech)"
        ) from None


def _derivatives_root(dataset_root: Path, options: Mapping | None) -> Path:
    configured = (options or {}).get(
        "derivatives_root", "derivatives/paper4/bci2020_standard"
    )
    candidate = Path(str(configured)).expanduser()
    return candidate if candidate.is_absolute() else dataset_root / candidate


def load_task_sessions(
    dataset_root: str | Path,
    task: str,
    subjects: Sequence[str | int],
    sessions: Sequence[str | int] | str = "auto",
    *,
    preload: bool = False,
    options: Mapping | None = None,
) -> tuple[DS003626TaskData, ...]:
    """Load model-ready records without changing the saved preprocessing."""
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required to load BCI2020 epochs") from exc

    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"BCI2020 root does not exist: {root}")
    normalized_task = _normalize_task(task)
    normalized_subjects = tuple(_normalize_subject(value) for value in subjects)
    if not normalized_subjects or len(set(normalized_subjects)) != len(
        normalized_subjects
    ):
        raise ValueError("subjects must contain unique BCI2020 subjects")
    normalized_runs = (
        ("00", "01", "02")
        if sessions == "auto"
        else tuple(_normalize_run(value) for value in sessions)
    )
    if not normalized_runs or len(set(normalized_runs)) != len(normalized_runs):
        raise ValueError("sessions must contain unique BCI2020 runs")

    derivative_root = _derivatives_root(root, options)
    records: list[DS003626TaskData] = []
    expected_event_id = EVENT_ID
    for subject in normalized_subjects:
        for run in normalized_runs:
            path = derivative_root / subject / "eeg" / (
                f"{subject}_task-imaginedSpeech_run-{run}_desc-preproc_epo.fif"
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing standard-preprocessed BCI2020 file: {path}. "
                    "Run addition/bci2020_preprocessing/run_preprocess_bci2020.sh first."
                )
            epochs = mne.read_epochs(path, preload=preload, verbose="ERROR")
            if len(epochs.ch_names) != 64 or not np.isclose(
                epochs.info["sfreq"], 256.0
            ):
                raise RuntimeError(f"Unexpected EEG dimensions in {path}")
            if len(epochs.times) != 795:
                raise RuntimeError(f"Expected 795 samples in {path}")
            for name, code in epochs.event_id.items():
                if expected_event_id.get(name) != int(code):
                    raise RuntimeError(f"Unexpected event mapping in {path}")
            codes = epochs.events[:, 2].astype(np.int64, copy=True)
            if not set(codes.tolist()).issubset(expected_event_id.values()):
                raise RuntimeError(f"Unexpected event code in {path}")
            labels = codes - 1
            trial_indices = np.arange(len(epochs), dtype=np.int64)
            records.append(
                DS003626TaskData(
                    task=normalized_task,
                    subject=subject,
                    session=f"run-{run}",
                    epochs=epochs,
                    labels=labels,
                    trial_indices=trial_indices,
                    sample_indices=epochs.events[:, 0].astype(
                        np.int64, copy=True
                    ),
                    class_names=CLASS_NAMES,
                    sampling_rate_hz=256.0,
                )
            )
    return tuple(records)
