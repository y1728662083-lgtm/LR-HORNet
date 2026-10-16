"""Audit configs, data, dependencies and official weights before training."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PAPER4_ROOT = Path(__file__).resolve().parents[2]
SUBJECTS = list(range(1, 16))
FOLDS = list(range(15))
ARCHIVED_DATA_ROOT = "/root/autodl-tmp/BCI_2020"
DEFAULT_DATA_ROOT = PAPER4_ROOT / "data" / "bci2020"
DEFAULT_OUTPUT_ROOT = PAPER4_ROOT / "outputs" / "bci2020_all_models"
EXPECTED_WEIGHT_SHA256 = {
    "pretrained_EEGMamba.pth": "b452bb29ecf1d6131ba82a50c6e13823ec1d660d9009d013e691d19b2916f4fe",
    "labram-base-official.safetensors": "f41dcf13b2ad84bfca4f0ce24da041608c8ca197ff174b2e16a2f19dd245b476",
}

CONFIGS = {
    "lr_hornet": PAPER4_ROOT / "configs/main/bci2020.yaml",
    "eegnet": PAPER4_ROOT / "baselines/eegnet/configs/bci2020.yaml",
    "atcnet": PAPER4_ROOT / "baselines/atcnet/configs/bci2020.yaml",
    "eegmamba": PAPER4_ROOT / "baselines/eegmamba/addition/configs/bci2020_official_baselines.yaml",
    "fast": PAPER4_ROOT / "baselines/fast/addition/configs/fast_native_bci2020.yaml",
    "labram": PAPER4_ROOT / "baselines/labram/addition/configs/bci2020_official_baselines.yaml",
}

# These are the architecture/training values copied from the archived 4.4.1
# repository. Dataset identity, channel/time/class metadata and output paths are
# intentionally not part of this table.
EXPECTED = {
    "lr_hornet": {
        "loader.batch_size": 32,
        "model.name": "group4_patchLstm_deformableLstm",
        "model.kwargs.patch_size": 128,
        "model.kwargs.patch_stride": 64,
        "model.kwargs.pad_incomplete_patch": True,
        "model.kwargs.n_electrode_groups": 4,
        "model.kwargs.offset_lstm_hidden_size": 8,
        "model.kwargs.offset_lstm_layers": 1,
        "model.kwargs.offset_lstm_dropout": 0.0,
        "training.learning_rate": 0.001,
        "training.weight_decay": 0.0001,
        "training.label_smoothing": 0.05,
        "training.epochs": 100,
        "training.patience": 40,
    },
    "eegnet": {
        "loader.batch_size": 128,
        "model.name": "braindecode_eegnet",
        "model.kwargs": {},
        "training.learning_rate": 0.001,
        "training.weight_decay": 0.0001,
        "training.label_smoothing": 0.0,
        "training.epochs": 100,
        "training.patience": 40,
    },
    "atcnet": {
        "loader.batch_size": 128,
        "model.name": "braindecode_atcnet",
        "model.kwargs": {},
        "training.learning_rate": 0.001,
        "training.weight_decay": 0.0001,
        "training.label_smoothing": 0.0,
        "training.epochs": 100,
        "training.patience": 40,
    },
    "eegmamba": {
        "models.eegmamba.classifier": "avgpooling_patch_reps",
        "models.eegmamba.training.batch_size": 32,
        "models.eegmamba.training.epochs": 50,
        "models.eegmamba.training.seed": 3407,
        "models.eegmamba.training.learning_rate": 0.0001,
        "models.eegmamba.training.head_learning_rate_multiplier": 5.0,
        "models.eegmamba.training.weight_decay": 0.05,
        "models.eegmamba.training.label_smoothing": 0.1,
        "models.eegmamba.training.warmup_epochs": 0,
        "models.eegmamba.training.gradient_accumulation_steps": 1,
        "models.eegmamba.training.gradient_clip_norm": 1.0,
        "models.eegmamba.training.amp": False,
        "models.eegmamba.training.scheduler": "eegmamba_official_cosine",
        "models.eegmamba.training.selection_metric": "cohen_kappa",
        "models.eegmamba.training.early_stopping": False,
    },
    "fast": {
        "models.fast.dim_cnn": 32,
        "models.fast.dim_token": 32,
        "models.fast.num_layers": 4,
        "models.fast.num_heads": 8,
        "models.fast.dropout": 0.1,
        "models.fast.head": "Conv4Layers",
        "models.fast.window_samples": 256,
        "models.fast.stride_samples": 128,
        "models.fast.window_remainder": "nearest",
        "models.fast.training.batch_size": 16,
        "models.fast.training.epochs": 100,
        "models.fast.training.patience": 30,
        "models.fast.training.learning_rate": 0.0005,
        "models.fast.training.weight_decay": 0.01,
        "models.fast.training.warmup_epochs": 10,
        "models.fast.training.amp": True,
    },
    "labram": {
        "models.labram.init_values": 0.1,
        "models.labram.drop_path_rate": 0.1,
        "models.labram.layer_decay": 0.65,
        "models.labram.training.batch_size": 64,
        "models.labram.training.epochs": 50,
        "models.labram.training.seed": 0,
        "models.labram.training.learning_rate": 0.0005,
        "models.labram.training.head_learning_rate_multiplier": 1.0,
        "models.labram.training.weight_decay": 0.05,
        "models.labram.training.label_smoothing": 0.1,
        "models.labram.training.warmup_epochs": 5,
        "models.labram.training.gradient_accumulation_steps": 1,
        "models.labram.training.gradient_clip_norm": 0.0,
        "models.labram.training.amp": True,
        "models.labram.training.scheduler": "labram_official_warmup_cosine",
        "models.labram.training.selection_metric": "accuracy",
        "models.labram.training.early_stopping": False,
    },
}


def read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def nested(payload: dict, dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        value = value[part]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_configs() -> dict[str, dict]:
    configs = {name: read_yaml(path) for name, path in CONFIGS.items()}
    failures: list[str] = []
    for name, config in configs.items():
        data = config["data"]
        if data["dataset"] != "bci2020":
            failures.append(f"{name}: data.dataset={data['dataset']!r}")
        if data["root"] != ARCHIVED_DATA_ROOT:
            failures.append(f"{name}: data.root={data['root']!r}")
        if data["task"] != "imagined_speech" or data["subjects"] != SUBJECTS or data["sessions"] != [0, 1, 2]:
            failures.append(f"{name}: dataset split metadata differs")
        if config["split"]["fold_indices"] != FOLDS:
            failures.append(f"{name}: not all 15 LOSO folds are listed")
        if config["split"].get("adapt_held_out_subjects") is not True:
            failures.append(f"{name}: calibrated LOSO is disabled")
        for dotted, expected in EXPECTED[name].items():
            actual = nested(config, dotted)
            if actual != expected:
                failures.append(f"{name}: {dotted}={actual!r}, expected {expected!r}")
    lr = configs["lr_hornet"]
    timing = lr["patch_timing"]
    if timing != {
        "sampling_rate_hz": 256,
        "patch_seconds": 0.5,
        "stride_seconds": 0.25,
        "rounding": "exact",
        "right_pad_incomplete_patch": True,
    }:
        failures.append(f"lr_hornet: invalid patch timing contract: {timing}")
    for name in ("eegmamba", "labram"):
        adapter = configs[name]["input_adapter"]
        if adapter != {
            "source_sampling_rate_hz": 256,
            "target_sampling_rate_hz": 200,
            "patch_size_samples": 200,
            "patch_remainder": "zero_pad",
        }:
            failures.append(f"{name}: official input adapter differs: {adapter}")
    fast_adapter = configs["fast"]["input_adapter"]
    if fast_adapter != {
        "source_sampling_rate_hz": 256,
        "target_sampling_rate_hz": 256,
        "patch_size_samples": 256,
        "patch_remainder": "zero_pad",
    }:
        failures.append(f"fast: native-rate input adapter differs: {fast_adapter}")
    if failures:
        raise RuntimeError("Parameter/data contract failed:\n- " + "\n- ".join(failures))
    print("OK configs: six model/data contracts match the archived settings")
    return configs


def _find_weight(filename: str, baseline: str) -> Path | None:
    candidates = (
        PAPER4_ROOT / "baselines" / baseline / "addition" / "weights" / filename,
        PAPER4_ROOT / "addition" / "weights" / filename,
        PAPER4_ROOT / "weights" / filename,
    )
    return next((path for path in candidates if path.is_file()), None)


def validate_weights() -> None:
    for baseline, filename in (
        ("eegmamba", "pretrained_EEGMamba.pth"),
        ("labram", "labram-base-official.safetensors"),
    ):
        path = _find_weight(filename, baseline)
        if path is None:
            raise FileNotFoundError(
                f"Missing {filename}; place it under baselines/{baseline}/addition/weights/ "
                "or reuse paper4/addition/weights/."
            )
        actual = sha256(path)
        expected = EXPECTED_WEIGHT_SHA256[filename]
        if actual != expected:
            raise RuntimeError(f"{filename} SHA256 mismatch: {actual} != {expected}")
        print(f"OK weight {filename} sha256={actual}")


def validate_packaging_compatibility() -> None:
    """Catch the exact cross-version/path failures seen in the first launch."""

    for baseline in ("eegmamba", "labram"):
        root_manifest = PAPER4_ROOT / f"baselines/{baseline}/source_manifest.json"
        addition_manifest = (
            PAPER4_ROOT / f"baselines/{baseline}/addition/source_manifest.json"
        )
        if not root_manifest.is_file() or not addition_manifest.is_file():
            raise FileNotFoundError(
                f"{baseline}: source_manifest.json must exist at root and addition level"
            )
        if sha256(root_manifest) != sha256(addition_manifest):
            raise RuntimeError(f"{baseline}: source manifest copies differ")

    fast_data = PAPER4_ROOT / "baselines/fast/src/eeg_pipeline/data"
    schema = (fast_data / "ds003626.py").read_text(encoding="utf-8")
    dataset = (fast_data / "dataset.py").read_text(encoding="utf-8")
    for token in (
        "channel_indices:",
        "channel_names:",
        "time_slice:",
        "sampling_rate_hz:",
    ):
        if token not in schema:
            raise RuntimeError(f"FAST data schema lacks {token}")
    for helper in ("_task_channel_names", "_task_times", "_task_sfreq"):
        if f"def {helper}" not in dataset:
            raise RuntimeError(f"FAST dataset adapter lacks {helper}")
    print("OK packaging: official manifests and FAST BCI2020 adapter are complete")


def validate_dependencies() -> None:
    required = (
        "torch", "braindecode", "mne", "numpy", "scipy", "pandas", "yaml",
        "timm", "einops", "safetensors", "ncps", "mamba_ssm", "causal_conv1d",
    )
    versions = {}
    for name in required:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "installed")
    print("OK dependencies " + json.dumps(versions, ensure_ascii=False, sort_keys=True))


def validate_data(dataset_root: Path) -> None:
    derivative = dataset_root / "derivatives/paper4/bci2020_standard"
    command = [
        sys.executable,
        "-u",
        str(PAPER4_ROOT / "addition/bci2020_preprocessing/verify_outputs.py"),
        "--root",
        str(derivative),
        "--expected-records",
        "45",
    ]
    subprocess.run(command, cwd=PAPER4_ROOT, check=True)


def write_audit(
    configs: dict[str, dict],
    *,
    dataset_root: Path,
    output_root: Path,
) -> Path:
    output = output_root / "preflight_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed",
        "models": list(CONFIGS),
        "fits": len(CONFIGS) * 15,
        "protocol": "calibrated_loso",
        "subjects": SUBJECTS,
        "dataset_root": str(dataset_root),
        "config_sha256": {name: sha256(CONFIGS[name]) for name in configs},
        "expected_parameter_contract": EXPECTED,
        "note": "Parameter counts are recorded from each instantiated model in its per-fold result files.",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-only", action="store_true")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    configs = validate_configs()
    validate_packaging_compatibility()
    if not args.configs_only:
        validate_dependencies()
        validate_weights()
        validate_data(dataset_root)
    audit = write_audit(
        configs,
        dataset_root=dataset_root,
        output_root=output_root,
    )
    print(f"PREFLIGHT PASSED audit={audit}")


if __name__ == "__main__":
    main()
