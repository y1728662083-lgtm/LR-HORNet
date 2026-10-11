"""Prepare and run a small, explicit model suite through the main pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: Path) -> tuple[Path, dict]:
    from main import load_config

    resolved = _resolve(path)
    return resolved, load_config(resolved)


def _output_root(config: dict) -> Path:
    raw_path = str(config["training"]["output_root"])
    path = Path(raw_path)
    # Preserve Linux absolute paths during local Windows validation.
    if path.is_absolute() or raw_path.startswith("/"):
        return path
    return PROJECT_ROOT / path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_catalog(config_paths: list[Path]) -> tuple[Path, dict]:
    """Validate a suite and build a summary-compatible catalog."""
    if not config_paths:
        raise ValueError("At least one suite config is required")
    loaded = [_load_config(path) for path in config_paths]
    reference = loaded[0][1]
    task = str(reference["data"]["task"])
    fold_indices = [int(value) for value in reference["split"]["fold_indices"]]
    root = _output_root(reference)
    model_keys: set[str] = set()
    models: list[dict] = []

    for path, config in loaded:
        current_task = str(config["data"]["task"])
        current_folds = [
            int(value) for value in config["split"]["fold_indices"]
        ]
        current_root = _output_root(config)
        if current_task != task:
            raise ValueError(
                f"Suite task mismatch in {path}: {current_task!r} != {task!r}"
            )
        if current_folds != fold_indices:
            raise ValueError(
                f"Suite fold mismatch in {path}: "
                f"{current_folds} != {fold_indices}"
            )
        if current_root != root:
            raise ValueError(
                f"Suite output_root mismatch in {path}: "
                f"{current_root} != {root}"
            )
        model_key = str(config["model"]["name"])
        if model_key in model_keys:
            raise ValueError(f"Duplicate suite model name: {model_key}")
        model_keys.add(model_key)
        models.append(
            {
                "architecture": model_key,
                "model_key": model_key,
                "paper_year": None,
                "runnable": True,
                "reason": "",
                "config": str(path),
            }
        )

    catalog = {
        "catalog_rule": "explicit two-model full-LOSO suite",
        "task": task,
        "fold_indices": fold_indices,
        "expected_folds": [index + 1 for index in fold_indices],
        "training_monitor": str(reference["training"]["monitor"]),
        "models": models,
    }
    return root, catalog


def prepare_suite(config_paths: list[Path]) -> Path:
    root, catalog = build_catalog(config_paths)
    _write_json(root / "_status" / "catalog.json", catalog)
    return root


def _n_classes(task: str) -> int:
    task_classes = {"imagine": 4, "speak": 4, "visual": 4, "pattern": 3}
    try:
        return task_classes[task]
    except KeyError:
        raise ValueError(f"Unknown task: {task!r}") from None


def run_config(config_path: Path) -> None:
    """Run all configured folds and persist size/progress for summary.py."""
    from eeg_pipeline.models.factory import build_model
    from main import run_experiment

    resolved, loaded_config = _load_config(config_path)
    config = deepcopy(loaded_config)
    model_key = str(config["model"]["name"])
    task = str(config["data"]["task"])
    fold_indices = [int(value) for value in config["split"]["fold_indices"]]
    status_path = _output_root(config) / "_status" / f"{model_key}.json"
    status = {
        "architecture": model_key,
        "model_key": model_key,
        "status": "running",
        "task": task,
        "fold_indices": fold_indices,
        "expected_folds": [index + 1 for index in fold_indices],
        "config": str(resolved),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(status_path, status)

    try:
        model = build_model(
            name=model_key,
            n_classes=_n_classes(task),
            **config["model"].get("kwargs", {}),
        )
        total_parameters = sum(
            parameter.numel() for parameter in model.parameters()
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        status.update(
            total_parameters=total_parameters,
            trainable_parameters=trainable_parameters,
            model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        )
        _write_json(status_path, status)
        del model
        gc.collect()

        results = run_experiment(config)
    except Exception as error:
        status.update(
            status="failed",
            reason=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(status_path, status)
        raise

    status.update(
        status="completed",
        completed_folds=[int(result["fold"]) for result in results],
        results=results,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(status_path, status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run-config", type=Path)
    parser.add_argument("--configs", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare:
        if not args.configs:
            raise SystemExit("--prepare requires --configs")
        print(prepare_suite(args.configs))
        return
    run_config(args.run_config)


if __name__ == "__main__":
    main()
