#!/usr/bin/env python3
"""Validate every formal Section 4.8 checkpoint against the Git model code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from eeg_pipeline.models.factory import build_model  # noqa: E402


N_CHANNELS = 128
N_TIMES = 1153
N_CLASSES = 4
MAMBA_MODELS = {
    "group4_fullMamba_deformableLstm",
    "group4_patchLstm_deformableMamba",
}
CONFIGS = {
    "group4_patchLstm_deformableLstm": (
        "4.8.1/configs/00_patchlstm_deformablelstm.yaml"
    ),
    "group4_patchGRU_deformableGRU": (
        "4.8.1/configs/01_patchgru_deformablegru.yaml"
    ),
    "group4_patchRNN_deformableRNN": (
        "4.8.1/configs/02_patchrnn_deformablernn.yaml"
    ),
    "group4_patchLstm_deformableCfC": (
        "4.8.1/configs/03_patchlstm_deformablecfc.yaml"
    ),
    "group4_patchCfC_deformableCfC": (
        "4.8.1/configs/04_patchcfc_deformablecfc.yaml"
    ),
    "group4_patchLstm_deformableLstm_satt": (
        "4.8.2/configs/attention/group4_patchLstm_deformableLstm_satt.yaml"
    ),
    "group4_patchLstm_deformableLstm_statt": (
        "4.8.2/configs/attention/group4_patchLstm_deformableLstm_statt.yaml"
    ),
    "group4_fullMamba_deformableLstm": (
        "4.8.2/configs/mamba/full_mamba_patch_replaced.yaml"
    ),
    "group4_patchLstm_deformableMamba": (
        "4.8.2/configs/mamba/mamba_offset_only.yaml"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration is not a mapping: {path}")
    return payload


def _read_manifest(section: str) -> list[dict[str, str]]:
    path = ROOT / "results" / section / "checkpoint_manifest.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {"4.8.1": 25, "4.8.2": 48}[section]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows in {path}, found {len(rows)}")
    return rows


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _build(
    model_name: str,
    *,
    device: torch.device,
    attention_backend: str,
) -> torch.nn.Module:
    relative_config = CONFIGS.get(model_name)
    if relative_config is None:
        raise ValueError(f"No formal Section 4.8 config for {model_name!r}")
    config = _read_yaml(ROOT / "experiments" / relative_config)
    model_config = config["model"]
    if model_config["name"] != model_name:
        raise ValueError(
            f"Config/model mismatch: {relative_config} != {model_name}"
        )
    kwargs = dict(model_config.get("kwargs", {}))
    kwargs.pop("n_channels", None)
    kwargs.pop("n_times", None)
    if model_name.endswith(("_satt", "_statt")):
        kwargs["attention_backend"] = attention_backend
        # Checkpointing changes memory use only; disabling it keeps inference
        # deterministic and does not alter state_dict names or shapes.
        kwargs["attention_checkpoint"] = False
    return build_model(
        model_name,
        n_classes=N_CLASSES,
        n_channels=N_CHANNELS,
        n_times=N_TIMES,
        **kwargs,
    ).to(device).eval()


def validate(
    experiment_root: Path,
    *,
    sections: tuple[str, ...],
    device_name: str,
    attention_backend: str,
    skip_mamba: bool,
    load_only: bool,
) -> tuple[int, int]:
    experiment_root = experiment_root.expanduser().resolve()
    if not experiment_root.is_dir():
        raise FileNotFoundError(
            f"Section 4.8 experiment root does not exist: {experiment_root}"
        )
    device = _device(device_name)
    torch.set_num_threads(1)
    synthetic = torch.zeros(1, N_CHANNELS, N_TIMES, device=device)
    models: dict[str, torch.nn.Module] = {}
    passed = 0
    skipped = 0

    for section in sections:
        for row in _read_manifest(section):
            model_name = row["model"]
            if skip_mamba and model_name in MAMBA_MODELS:
                skipped += 1
                continue
            checkpoint = experiment_root / section / row["source_checkpoint"]
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            if checkpoint.stat().st_size != int(row["bytes"]):
                raise ValueError(f"Checkpoint size mismatch: {checkpoint}")
            if _sha256(checkpoint) != row["sha256"].strip().casefold():
                raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint}")

            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(state, Mapping) or not state:
                raise TypeError(f"Checkpoint is not a non-empty state_dict: {checkpoint}")
            if any(
                not isinstance(key, str) or not torch.is_tensor(value)
                for key, value in state.items()
            ):
                raise TypeError(f"Checkpoint contains non-tensor state: {checkpoint}")
            if any(not bool(torch.isfinite(value).all()) for value in state.values()):
                raise ValueError(f"Checkpoint contains NaN or infinity: {checkpoint}")

            if model_name not in models:
                models[model_name] = _build(
                    model_name,
                    device=device,
                    attention_backend=attention_backend,
                )
            model = models[model_name]
            model.load_state_dict(state, strict=True)
            if not load_only:
                with torch.inference_mode():
                    logits = model(synthetic)
                if tuple(logits.shape) != (1, N_CLASSES):
                    raise RuntimeError(
                        f"Unexpected output {tuple(logits.shape)}: {checkpoint}"
                    )
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError(f"Non-finite output: {checkpoint}")
            passed += 1
            print(
                f"[PASS] section={section} model={model_name} "
                f"fold={row['fold']} repeat={row['repeat']}",
                flush=True,
            )

    print(
        f"[SUMMARY] passed={passed} skipped={skipped} "
        f"device={device} attention_backend={attention_backend}",
        flush=True,
    )
    return passed, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Directory containing the original 4.8.1 and 4.8.2 result trees.",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("4.8.1", "4.8.2"),
        default=("4.8.1", "4.8.2"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--attention-backend",
        choices=("pytorch", "cuda"),
        default="pytorch",
    )
    parser.add_argument(
        "--skip-mamba",
        action="store_true",
        help="Explicitly omit the 28 Mamba checkpoints when mamba_ssm is unavailable.",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Perform hash and strict-load checks without a forward per checkpoint.",
    )
    args = parser.parse_args()
    validate(
        args.experiment_root,
        sections=tuple(args.sections),
        device_name=args.device,
        attention_backend=args.attention_backend,
        skip_mamba=args.skip_mamba,
        load_only=args.load_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
