#!/usr/bin/env python3
"""Strictly restore and execute all released cross-dataset LR-HORNet weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from eeg_pipeline.models.factory_lr_hornet import build_model  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    config: str
    channels: int
    times: int
    classes: int
    folds: int


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    bytes: int
    sha256: str


DATASETS = {
    "ds003626": DatasetSpec("ds003626.yaml", 128, 1153, 4, 10),
    "bimodal_inner_speech": DatasetSpec(
        "bimodal_inner_speech.yaml", 64, 359, 8, 4
    ),
    "directional_words_russian": DatasetSpec(
        "directional_words_russian.yaml", 34, 250, 6, 12
    ),
    "directional_words_spanish": DatasetSpec(
        "directional_words_spanish.yaml", 38, 750, 6, 10
    ),
    "bci2020": DatasetSpec("bci2020.yaml", 64, 795, 5, 15),
}


def _read_config(filename: str) -> dict:
    path = ROOT / "configs" / "main" / filename
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration is not a mapping: {path}")
    return config


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _load_manifest(
    checkpoint_root: Path,
) -> dict[tuple[str, int], CheckpointRecord]:
    manifest_path = checkpoint_root / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    index: dict[tuple[str, int], CheckpointRecord] = {}
    for row in rows:
        key = (row["dataset"], int(row["fold"]))
        if key in index:
            raise ValueError(f"Duplicate checkpoint manifest key: {key}")
        path = ROOT / row["path"]
        if checkpoint_root.resolve() != (ROOT / "checkpoints" / "main").resolve():
            path = checkpoint_root / key[0] / f"fold_{key[1]:02d}.pt"
        digest = str(row["sha256"]).strip().casefold()
        if len(digest) != 64:
            raise ValueError(f"Invalid SHA-256 in manifest for {key}: {digest!r}")
        index[key] = CheckpointRecord(
            path=path,
            bytes=int(row["bytes"]),
            sha256=digest,
        )
    return index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoke_check(
    checkpoint_root: Path = ROOT / "checkpoints" / "main",
    *,
    device: str = "cpu",
    datasets: tuple[str, ...] | None = None,
    forward_each_checkpoint: bool = True,
    backward_per_dataset: bool = True,
) -> dict[str, int]:
    """Validate every state and return per-dataset checkpoint pass counts.

    Every checkpoint is hash-checked, loaded with ``weights_only=True`` and
    restored with ``strict=True``.  Full release smoke tests also execute every
    checkpoint.  Fast unit-test mode executes the final restored state once per
    dataset.  The backward pass can be disabled for dependency-light CI and
    retained for an explicit release smoke test.
    """
    checkpoint_root = checkpoint_root.expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")
    manifest = _load_manifest(checkpoint_root)
    target_device = _device(device)
    torch.set_num_threads(1)
    passed: dict[str, int] = {}
    selected_datasets = tuple(DATASETS) if datasets is None else datasets
    unknown = set(selected_datasets).difference(DATASETS)
    if unknown:
        raise ValueError(f"Unknown dataset selection: {sorted(unknown)}")

    for dataset in selected_datasets:
        spec = DATASETS[dataset]
        config = _read_config(spec.config)
        model_config = config["model"]
        kwargs = dict(model_config.get("kwargs", {}))
        kwargs.pop("n_channels", None)
        kwargs.pop("n_times", None)
        model = build_model(
            str(model_config["name"]),
            n_classes=spec.classes,
            n_channels=spec.channels,
            n_times=spec.times,
            **kwargs,
        ).to(target_device)
        synthetic = torch.zeros(
            1, spec.channels, spec.times, device=target_device
        )

        dataset_passed = 0
        for fold in range(1, spec.folds + 1):
            record = manifest.get((dataset, fold))
            path = None if record is None else record.path
            if record is None or not path.is_file():
                raise FileNotFoundError(
                    f"Missing released checkpoint for {dataset} fold {fold}: {path}"
                )
            if path.stat().st_size != record.bytes:
                raise ValueError(
                    f"Checkpoint byte count does not match manifest: {path}"
                )
            if _sha256(path) != record.sha256:
                raise ValueError(f"Checkpoint SHA-256 does not match manifest: {path}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, Mapping) or not state:
                raise TypeError(f"Checkpoint is not a non-empty state_dict: {path}")
            if any(not isinstance(key, str) for key in state):
                raise TypeError(f"Checkpoint contains a non-string key: {path}")
            if any(not torch.is_tensor(value) for value in state.values()):
                raise TypeError(f"Checkpoint contains a non-tensor value: {path}")
            if any(not bool(torch.isfinite(value).all()) for value in state.values()):
                raise ValueError(f"Checkpoint contains NaN or infinity: {path}")
            model.load_state_dict(state, strict=True)
            if forward_each_checkpoint:
                model.eval()
                with torch.inference_mode():
                    logits = model(synthetic)
                if tuple(logits.shape) != (1, spec.classes):
                    raise RuntimeError(
                        f"{path}: expected logits (1, {spec.classes}), "
                        f"got {tuple(logits.shape)}"
                    )
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError(f"{path}: non-finite logits")
            dataset_passed += 1

        # Execute one representative state when the per-checkpoint loop is
        # disabled, or when the training graph is explicitly requested.
        representative_batch = 2 if backward_per_dataset else 1
        if not forward_each_checkpoint or backward_per_dataset:
            model.train(mode=backward_per_dataset)
            model.zero_grad(set_to_none=True)
            batch = torch.randn(
                representative_batch,
                spec.channels,
                spec.times,
                device=target_device,
            )
            with torch.set_grad_enabled(backward_per_dataset):
                logits = model(batch)
            if tuple(logits.shape) != (representative_batch, spec.classes):
                raise RuntimeError(
                    f"{dataset}: expected representative logits "
                    f"({representative_batch}, {spec.classes}), "
                    f"got {tuple(logits.shape)}"
                )
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"{dataset}: non-finite representative logits")
            if backward_per_dataset:
                targets = torch.tensor(
                    [0, min(1, spec.classes - 1)], device=target_device
                )
                loss = torch.nn.functional.cross_entropy(logits, targets)
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if not gradients or any(
                    not bool(torch.isfinite(gradient).all())
                    for gradient in gradients
                ):
                    raise RuntimeError(f"{dataset}: invalid backward gradients")

        passed[dataset] = dataset_passed
        print(
            f"[PASS] dataset={dataset} checkpoints={dataset_passed} "
            f"input=({representative_batch},{spec.channels},{spec.times}) "
            f"classes={spec.classes}"
        )

    expected_total = sum(DATASETS[name].folds for name in selected_datasets)
    if sum(passed.values()) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} checkpoints, passed {sum(passed.values())}"
        )
    execution = (
        "per-checkpoint forward"
        if forward_each_checkpoint
        else "one representative forward per dataset"
    )
    if backward_per_dataset:
        execution += " plus one dataset-shape backward per dataset"
    print(
        f"[PASS] All {expected_total} released cross-dataset checkpoints passed "
        f"weights-only load, strict restoration, and {execution}."
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=ROOT / "checkpoints" / "main",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=None,
        help="Optional subset; omission validates all five datasets.",
    )
    parser.add_argument(
        "--representative-forward",
        action="store_true",
        help=(
            "Strictly restore every checkpoint but execute only the final "
            "state for each dataset. The default executes every checkpoint."
        ),
    )
    parser.add_argument(
        "--skip-backward",
        action="store_true",
        help=(
            "Skip the real-shape backward pass. Hash, finite-value, strict "
            "restore and representative inference checks still run."
        ),
    )
    args = parser.parse_args()
    smoke_check(
        args.checkpoints_root,
        device=args.device,
        datasets=None if args.datasets is None else tuple(args.datasets),
        forward_each_checkpoint=not args.representative_forward,
        backward_per_dataset=not args.skip_backward,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
