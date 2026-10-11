"""Read every FAST dataset and prepare missing Bimodal official-code epochs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import yaml


ADDITION_ROOT = Path(__file__).resolve().parent
PAPER4_ROOT = ADDITION_ROOT.parent
CONFIG_ROOT = ADDITION_ROOT / "configs"
REPORT_PATH = PAPER4_ROOT / "outputs" / "addition" / "fast_native_dim32_input_report.json"
BIMODAL_MEMBER = "Inner_Speech_EEG_fMRI-main/EEG_preprocessing/preprocess.py"
CONFIGS = (
    ("ds003626", "fast_native_ds003626.yaml", 256.0, 1153, 4),
    ("russian", "fast_native_russian.yaml", 250.0, 250, 6),
    ("spanish", "fast_native_spanish.yaml", 500.0, 750, 6),
    ("bimodal", "fast_native_bimodal.yaml", 512.0, 359, 8),
)
EXPECTED_DIRECTIONAL_GRID = {
    "russian": {
        "channels": 34,
        "tmin": 0.0,
        "tmax": 0.996,
        "time_window_seconds": [0.0, 1.0],
    },
    "spanish": {
        "channels": 38,
        "tmin": -0.5,
        "tmax": 0.998,
        "time_window_seconds": [-0.5, 1.0],
    },
}

sys.path.insert(0, str(PAPER4_ROOT / "src"))
sys.path.insert(0, str(PAPER4_ROOT))
sys.path.insert(0, str(ADDITION_ROOT))

from eeg_pipeline.data.dataset import EEGTaskDataset
from eeg_pipeline.data.preprocessing import preprocess_from_config
from eeg_pipeline.data.registry import load_task_sessions
from eeg_pipeline.data.splits import (
    build_adapted_fold_datasets,
    build_loso_subject_folds,
)
from fast_native_adapter import build_fast_native_model


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_paper4_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PAPER4_ROOT / path


def bimodal_outputs(config: dict) -> tuple[Path, ...]:
    root = Path(config["data"]["root"])
    configured = config["data"].get("options", {}).get(
        "derivatives_root", "derivatives/paper4/bimodal"
    )
    derivatives = Path(configured)
    if not derivatives.is_absolute():
        derivatives = root / derivatives
    return tuple(
        derivatives / f"sub-{int(subject):02d}" / f"sub-{int(subject):02d}_task-inner_epo.fif"
        for subject in config["data"]["subjects"]
    )


def validate_official_archive(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Bundled Bimodal official-code archive is missing: {path}")
    with ZipFile(path) as archive:
        if BIMODAL_MEMBER not in archive.namelist():
            raise FileNotFoundError(
                f"Official archive {path} does not contain {BIMODAL_MEMBER}"
            )


def prepare_bimodal(config: dict, *, allow_preprocess: bool) -> dict:
    outputs = bimodal_outputs(config)
    missing_before = [path for path in outputs if not path.is_file()]
    preprocessing = config.get("preprocessing", {})
    official_zip = resolve_paper4_path(preprocessing["official_code_zip"])
    validate_official_archive(official_zip)
    mode = "reused_existing"
    if missing_before:
        if not allow_preprocess:
            raise FileNotFoundError(
                "Bimodal derivative epochs are missing and preprocessing was disabled:\n"
                + "\n".join(map(str, missing_before))
            )
        prepared = deepcopy(config)
        prepared.setdefault("preprocessing", {})["official_code_zip"] = str(official_zip)
        print(
            f"BIMODAL PREPROCESS start missing={len(missing_before)} "
            f"official_archive={official_zip}",
            flush=True,
        )
        preprocess_from_config(prepared, dry_run=True)
        generated = preprocess_from_config(prepared, dry_run=False)
        print(f"BIMODAL PREPROCESS done outputs={len(generated)}", flush=True)
        mode = "official_code_generated_missing_derivatives"
    missing_after = [path for path in outputs if not path.is_file()]
    if missing_after:
        raise FileNotFoundError(
            "Bimodal preprocessing did not produce all derivatives:\n"
            + "\n".join(map(str, missing_after))
        )
    return {
        "mode": mode,
        "official_archive": str(official_zip),
        "official_member": BIMODAL_MEMBER,
        "missing_before": [str(path) for path in missing_before],
        "outputs": [str(path) for path in outputs],
    }


def source_paths(label: str, config: dict) -> list[str]:
    data = config["data"]
    root = Path(data["root"])
    if label in {"russian", "spanish"}:
        result = []
        for subject in data["subjects"]:
            language, name = str(subject).replace("\\", "/").split("/", 1)
            result.append(str(root / "preprocessed" / language / name / f"{name}_epochs.fif"))
        return result
    if label == "bimodal":
        return [str(path) for path in bimodal_outputs(config)]
    return []


def canonical_subject_id(value: object) -> str:
    """Match numeric subject spellings without altering dataset-qualified IDs."""
    normalized = str(value).replace("\\", "/").strip().casefold()
    match = re.fullmatch(r"(?:sub-?)?0*(\d+)", normalized)
    if match:
        return str(int(match.group(1)))
    return normalized


def inspect_dataset(
    label: str,
    config: dict,
    expected_rate: float,
    expected_times: int,
    expected_classes: int,
) -> dict:
    data = config["data"]
    root = Path(data["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"{label} dataset root is missing: {root}")
    sessions = load_task_sessions(
        data["dataset"],
        root,
        data["task"],
        subjects=tuple(data["subjects"]),
        sessions=data["sessions"],
        preload=False,
        options=data.get("options"),
    )
    expected_subjects = {
        canonical_subject_id(subject) for subject in data["subjects"]
    }
    observed_subjects = {
        canonical_subject_id(record.subject) for record in sessions
    }
    if observed_subjects != expected_subjects:
        raise RuntimeError(
            f"{label}: subject coverage mismatch; "
            f"missing={sorted(expected_subjects - observed_subjects)}, "
            f"unexpected={sorted(observed_subjects - expected_subjects)}"
        )
    configured_sessions = data["sessions"]
    if configured_sessions != "auto":
        expected_records = len(expected_subjects) * len(configured_sessions)
        if len(sessions) != expected_records:
            raise RuntimeError(
                f"{label}: expected {expected_records} subject-session records "
                f"({len(expected_subjects)} subjects x "
                f"{len(configured_sessions)} sessions), got {len(sessions)}"
            )
    for record in sessions:
        observed = set(map(int, np.unique(record.labels)))
        expected = set(range(expected_classes))
        if observed != expected:
            raise ValueError(
                f"{label} {record.subject}: class ids {sorted(observed)} != {sorted(expected)}"
            )
    dataset = EEGTaskDataset(
        sessions,
        standardize=bool(data["standardize"]),
        n_electrode_groups=int(data["n_electrode_groups"]),
    )
    if not np.isclose(dataset.sampling_rate_hz, expected_rate):
        raise ValueError(
            f"{label}: sfreq={dataset.sampling_rate_hz}, expected {expected_rate}"
        )
    if dataset.num_times != expected_times:
        raise ValueError(
            f"{label}: n_times={dataset.num_times}, expected {expected_times}; "
            "do not start FAST until the preprocessing/version mismatch is resolved"
        )
    if len(dataset.class_names) != expected_classes:
        raise ValueError(
            f"{label}: classes={dataset.class_names}, expected {expected_classes}"
        )
    if label in EXPECTED_DIRECTIONAL_GRID:
        grid = EXPECTED_DIRECTIONAL_GRID[label]
        options = data.get("options", {})
        if dataset.num_channels != grid["channels"]:
            raise ValueError(
                f"{label}: channels={dataset.num_channels}, "
                f"expected audited intersection={grid['channels']}"
            )
        if not np.isclose(dataset.tmin, grid["tmin"]) or not np.isclose(
            dataset.tmax, grid["tmax"]
        ):
            raise ValueError(
                f"{label}: time=[{dataset.tmin}, {dataset.tmax}], "
                f"expected [{grid['tmin']}, {grid['tmax']}]"
            )
        if options.get("time_window_seconds") != grid["time_window_seconds"]:
            raise ValueError(
                f"{label}: configured half-open time window changed: "
                f"{options.get('time_window_seconds')}"
            )
        if int(options.get("expected_common_channels", -1)) != grid["channels"]:
            raise ValueError(
                f"{label}: expected_common_channels does not match audit"
            )
    first_x, first_y = dataset[0]
    last_x, last_y = dataset[-1]
    if first_x.shape != last_x.shape or tuple(first_x.shape) != (
        dataset.num_channels,
        dataset.num_times,
    ):
        raise RuntimeError(f"{label}: inconsistent first/last sample shapes")
    if not torch.isfinite(first_x).all() or not torch.isfinite(last_x).all():
        raise ValueError(f"{label}: non-finite values after trial-wise standardization")

    folds = build_loso_subject_folds(tuple(data["subjects"]))
    train, validation, test = build_adapted_fold_datasets(
        dataset_root=root,
        task=data["task"],
        fold=folds[0],
        dataset_name=data["dataset"],
        sessions=data["sessions"],
        dataset_options=data.get("options"),
        seed=int(config["seed"]),
        standardize=bool(data["standardize"]),
        n_electrode_groups=int(data["n_electrode_groups"]),
        adapt_held_out_subjects=bool(config["split"]["adapt_held_out_subjects"]),
    )

    fast_config = config["models"]["fast"]
    model = build_fast_native_model(
        "fast",
        fast_config,
        channel_names=dataset.channel_names,
        n_times=dataset.num_times,
        n_classes=expected_classes,
        input_config=config["input_adapter"],
    ).eval()
    with torch.no_grad():
        logits = model(first_x.unsqueeze(0))
    if tuple(logits.shape) != (1, expected_classes):
        raise RuntimeError(f"{label}: FAST smoke output shape is {tuple(logits.shape)}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    metadata = dict(model.adapter_metadata)

    files = source_paths(label, config)
    missing_sources = [path for path in files if not Path(path).is_file()]
    if missing_sources:
        raise FileNotFoundError(
            f"{label}: source files disappeared during inspection:\n"
            + "\n".join(missing_sources)
        )
    result = {
        "dataset": data["dataset"],
        "root": str(root),
        "subjects": [str(item) for item in data["subjects"]],
        "records": len(sessions),
        "trials": len(dataset),
        "channels": dataset.num_channels,
        "times": dataset.num_times,
        "sampling_rate_hz": dataset.sampling_rate_hz,
        "tmin": dataset.tmin,
        "tmax": dataset.tmax,
        "channel_names_before_spatial_reorder": list(
            dataset.original_channel_names
        ),
        "channel_names_after_spatial_reorder": list(dataset.channel_names),
        "class_names": list(dataset.class_names),
        "first_label": int(first_y),
        "last_label": int(last_y),
        "fold1": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "validation_subject": str(folds[0].val_subject),
            "test_subject": str(folds[0].test_subject),
        },
        "fast": {
            "parameters": parameters,
            "output_shape": list(logits.shape),
            "adapter_metadata": metadata,
        },
        "source_files": files,
        "dataset_input_adapter": dict(data.get("options", {})),
    }
    print(
        f"DATA_OK name={label} subjects={len(expected_subjects)} "
        f"records={len(sessions)} trials={len(dataset)} "
        f"shape=({dataset.num_channels},{dataset.num_times}) "
        f"sfreq={dataset.sampling_rate_hz:g} classes={len(dataset.class_names)} "
        f"fold1={len(train)}/{len(validation)}/{len(test)} "
        f"fast_params={parameters:,}",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-bimodal-preprocess",
        action="store_true",
        help="Fail instead of invoking official code when Bimodal derivative FIFs are absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = {
        label: load_yaml(CONFIG_ROOT / filename)
        for label, filename, *_ in CONFIGS
    }
    bimodal = prepare_bimodal(
        configs["bimodal"],
        allow_preprocess=not args.skip_bimodal_preprocess,
    )
    report = {
        "status": "ok",
        "protocol": "calibrated_loso",
        "bimodal_preprocessing": bimodal,
        "datasets": {},
    }
    for label, _filename, rate, times, classes in CONFIGS:
        report["datasets"][label] = inspect_dataset(
            label, configs[label], rate, times, classes
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"FULL INPUT CHECK OK report={REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
