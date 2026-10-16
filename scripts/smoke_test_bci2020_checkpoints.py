#!/usr/bin/env python3
"""Strictly restore and execute every BCI2020 experiment checkpoint.

The results tree is intentionally external to this repository.  Each
``best_model.pt`` is loaded with PyTorch's restricted ``weights_only`` loader,
restored into the repository's existing model builder with ``strict=True``,
and evaluated on one synthetic ``(1, 64, 795)`` BCI2020 input.

The six model families are checked in separate child interpreters.  This
keeps the archived baseline bundles isolated from one another because several
of them intentionally use top-level helper-module names such as ``adapters``
and ``common``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
MODEL_ORDER = (
    "lr_hornet",
    "eegnet",
    "atcnet",
    "eegmamba",
    "fast",
    "labram",
)
N_CHANNELS = 64
N_TIMES = 795
N_CLASSES = 5
INPUT_SHAPE = (1, N_CHANNELS, N_TIMES)
OUTPUT_SHAPE = (1, N_CLASSES)


@dataclass(frozen=True)
class Failure:
    model: str
    checkpoint: Path
    stage: str
    detail: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="BCI2020 result directory containing the six model families.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Forward-pass device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--_worker-model",
        choices=MODEL_ORDER,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _classify_checkpoint(path: Path, results_root: Path) -> str | None:
    parts = tuple(part.casefold() for part in path.relative_to(results_root).parts)
    joined = "/".join(parts)
    if "lr_hornet" in joined or "lr-hornet" in joined:
        return "lr_hornet"
    for model in MODEL_ORDER[1:]:
        if model in parts:
            return model
    return None


def _discover(results_root: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    groups = {model: [] for model in MODEL_ORDER}
    unknown: list[Path] = []
    for checkpoint in sorted(results_root.rglob("best_model.pt")):
        model = _classify_checkpoint(checkpoint, results_root)
        if model is None:
            unknown.append(checkpoint)
        else:
            groups[model].append(checkpoint)
    return groups, unknown


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise FileNotFoundError(f"Required configuration is missing: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration is not a mapping: {path}")
    return payload


def _schema_digest(state: Mapping[str, Any]) -> str:
    import torch

    if not state:
        raise ValueError("Checkpoint state dictionary is empty")
    digest = hashlib.sha256()
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise TypeError(
                "Checkpoint must be a raw state_dict containing only "
                "string-to-tensor entries"
            )
        descriptor = f"{key}|{tuple(value.shape)}|{value.dtype}\n"
        digest.update(descriptor.encode("utf-8"))
    return digest.hexdigest()


def _resolved_channel_names(checkpoint: Path) -> tuple[str, ...]:
    resolved = _read_yaml(checkpoint.parent / "resolved_config.yaml")
    names = resolved.get("channel_names")
    if not isinstance(names, list) or len(names) != N_CHANNELS:
        raise ValueError(
            f"Expected {N_CHANNELS} channel names in "
            f"{checkpoint.parent / 'resolved_config.yaml'}"
        )
    channel_names = tuple(str(name) for name in names)
    if len(set(channel_names)) != N_CHANNELS:
        raise ValueError("BCI2020 channel names must be unique")
    class_names = resolved.get("class_names")
    if class_names is not None and len(class_names) != N_CLASSES:
        raise ValueError(f"Expected {N_CLASSES} BCI2020 classes")
    return channel_names


def _prepend_import_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _build_lr_hornet(_: Path):
    _prepend_import_path(SOURCE_ROOT)
    from eeg_pipeline.models.factory_lr_hornet import build_model

    config = _read_yaml(REPOSITORY_ROOT / "configs" / "main" / "bci2020.yaml")
    model_config = config["model"]
    kwargs = dict(model_config.get("kwargs", {}))
    kwargs.pop("n_channels", None)
    kwargs.pop("n_times", None)
    return build_model(
        str(model_config["name"]),
        n_classes=N_CLASSES,
        n_channels=N_CHANNELS,
        n_times=N_TIMES,
        **kwargs,
    )


def _build_braindecode(model: str, _: Path):
    _prepend_import_path(SOURCE_ROOT)
    from eeg_pipeline.models.baseline_model import build_braindecode_baseline

    config = _read_yaml(
        REPOSITORY_ROOT / "baselines" / model / "configs" / "bci2020.yaml"
    )
    baseline = config["baseline"]
    architecture = {"eegnet": "EEGNet", "atcnet": "ATCNet"}[model]
    return build_braindecode_baseline(
        n_classes=N_CLASSES,
        architecture=architecture,
        n_channels=int(baseline["n_channels"]),
        n_times=int(baseline["n_times"]),
        sfreq=float(baseline["sfreq"]),
        n_electrode_groups=int(baseline["n_electrode_groups"]),
    )


def _official_config_path(model: str) -> Path:
    filename = {
        "eegmamba": "bci2020_official_baselines.yaml",
        "fast": "fast_native_bci2020.yaml",
        "labram": "bci2020_official_baselines.yaml",
    }[model]
    return (
        REPOSITORY_ROOT
        / "baselines"
        / model
        / "addition"
        / "configs"
        / filename
    )


def _build_official_baseline(model: str, checkpoint: Path):
    addition_root = REPOSITORY_ROOT / "baselines" / model / "addition"
    _prepend_import_path(addition_root)
    config = _read_yaml(_official_config_path(model))
    model_config = copy.deepcopy(config["models"][model])
    input_config = copy.deepcopy(config["input_adapter"])
    channel_names = _resolved_channel_names(checkpoint)

    # The final experiment state_dict replaces every parameter and persistent
    # buffer.  Avoid requiring the initialization-only pretrained checkpoint,
    # which is intentionally not distributed with this repository.
    if model in {"eegmamba", "labram"}:
        model_config["checkpoint"] = None
        model_config["checkpoint_required"] = False

    if model == "fast":
        builder = importlib.import_module(
            "fast_native_adapter"
        ).build_fast_native_model
    else:
        builder = importlib.import_module("adapters").build_official_model
    return builder(
        model,
        model_config,
        channel_names=channel_names,
        n_times=N_TIMES,
        n_classes=N_CLASSES,
        input_config=input_config,
    )


def _build_model(model: str, checkpoint: Path):
    if model == "lr_hornet":
        return _build_lr_hornet(checkpoint)
    if model in {"eegnet", "atcnet"}:
        return _build_braindecode(model, checkpoint)
    return _build_official_baseline(model, checkpoint)


def _resolve_device(requested: str, torch) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return requested


def _worker(results_root: Path, model_name: str, requested_device: str) -> int:
    import torch

    groups, _ = _discover(results_root)
    checkpoints = groups[model_name]
    if not checkpoints:
        print(f"[FAIL] model={model_name}: no best_model.pt files found", flush=True)
        return 1

    device = _resolve_device(requested_device, torch)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    synthetic = torch.zeros(INPUT_SHAPE, dtype=torch.float32, device=device)
    model_cache: dict[tuple[str, tuple[str, ...] | None], Any] = {}
    build_failures: dict[tuple[str, tuple[str, ...] | None], str] = {}
    failures: list[Failure] = []
    passed = 0

    print(
        f"[GROUP] model={model_name} checkpoints={len(checkpoints)} "
        f"device={device}",
        flush=True,
    )
    for index, checkpoint in enumerate(checkpoints, start=1):
        relative = checkpoint.relative_to(results_root)
        try:
            state = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state, Mapping):
                raise TypeError(
                    f"Expected a raw state_dict, got {type(state).__name__}"
                )
            schema = _schema_digest(state)
        except Exception as error:  # noqa: BLE001 - aggregate every bad file
            failures.append(
                Failure(model_name, relative, "weights_only_load", repr(error))
            )
            print(f"[FAIL] {relative} stage=weights_only_load: {error}", flush=True)
            continue

        try:
            channel_key = (
                _resolved_channel_names(checkpoint)
                if model_name in {"eegmamba", "fast", "labram"}
                else None
            )
            cache_key = (schema, channel_key)
        except Exception as error:  # noqa: BLE001
            failures.append(Failure(model_name, relative, "metadata", repr(error)))
            print(f"[FAIL] {relative} stage=metadata: {error}", flush=True)
            continue

        if cache_key not in model_cache and cache_key not in build_failures:
            try:
                built = _build_model(model_name, checkpoint).to(device).eval()
                model_cache[cache_key] = built
            except Exception as error:  # noqa: BLE001
                build_failures[cache_key] = repr(error)

        if cache_key in build_failures:
            detail = build_failures[cache_key]
            failures.append(Failure(model_name, relative, "build", detail))
            print(f"[FAIL] {relative} stage=build: {detail}", flush=True)
            continue

        model = model_cache[cache_key]
        try:
            model.load_state_dict(state, strict=True)
        except Exception as error:  # noqa: BLE001
            failures.append(Failure(model_name, relative, "strict_load", repr(error)))
            print(f"[FAIL] {relative} stage=strict_load: {error}", flush=True)
            continue

        try:
            model.eval()
            with torch.inference_mode():
                output = model(synthetic)
            if not torch.is_tensor(output):
                raise TypeError(f"Forward returned {type(output).__name__}, not Tensor")
            if tuple(output.shape) != OUTPUT_SHAPE:
                raise RuntimeError(
                    f"Expected output {OUTPUT_SHAPE}, got {tuple(output.shape)}"
                )
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError("Forward output contains NaN or infinity")
        except Exception as error:  # noqa: BLE001
            failures.append(Failure(model_name, relative, "forward", repr(error)))
            print(f"[FAIL] {relative} stage=forward: {error}", flush=True)
            continue

        passed += 1
        print(
            f"[PASS] {index:03d}/{len(checkpoints):03d} {relative}",
            flush=True,
        )

    print(
        f"[SUMMARY] model={model_name} discovered={len(checkpoints)} "
        f"passed={passed} failed={len(failures)} schemas={len(model_cache)}",
        flush=True,
    )
    if failures:
        print(f"[FAILURE DETAILS] model={model_name}", flush=True)
        for failure in failures:
            print(
                f"  {failure.checkpoint} | stage={failure.stage} | "
                f"{failure.detail}",
                flush=True,
            )
        return 1
    return 0


def _controller(results_root: Path, requested_device: str) -> int:
    results_root = results_root.expanduser().resolve()
    if not results_root.is_dir():
        print(f"[FAIL] Results root is not a directory: {results_root}")
        return 2

    groups, unknown = _discover(results_root)
    counts = Counter({model: len(paths) for model, paths in groups.items()})
    print(f"BCI2020 checkpoint root: {results_root}")
    print("Discovered model groups:")
    for model in MODEL_ORDER:
        print(f"  {model}: {counts[model]}")
    if unknown:
        print("Unclassified best_model.pt files:")
        for path in unknown:
            print(f"  {path.relative_to(results_root)}")

    missing = [model for model in MODEL_ORDER if not groups[model]]
    if missing:
        print("[FAIL] Missing required model groups: " + ", ".join(missing))

    failed_models: list[str] = []
    for model in MODEL_ORDER:
        if not groups[model]:
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--results-root",
            str(results_root),
            "--device",
            requested_device,
            "--_worker-model",
            model,
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        if completed.returncode != 0:
            failed_models.append(model)

    if unknown or missing or failed_models:
        if failed_models:
            print("[FAIL] Failed model groups: " + ", ".join(failed_models))
        return 1
    print(
        f"[PASS] All {sum(counts.values())} BCI2020 checkpoints passed "
        "weights-only load, strict restoration, and forward execution."
    )
    return 0


def main() -> int:
    args = _parse_args()
    results_root = args.results_root.expanduser().resolve()
    if args._worker_model:
        return _worker(results_root, args._worker_model, args.device)
    return _controller(results_root, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
