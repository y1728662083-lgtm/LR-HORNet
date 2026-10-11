#!/usr/bin/env python3
"""Re-evaluate Section 4.6.1 checkpoints on their held-out test subjects."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import torch
import yaml
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "configs"
MODEL_CONFIGS = {
    "braindecode_eegnet": "00_braindecode_eegnet.yaml",
    "causal1_eegnet": "01_causal1_eegnet.yaml",
    "group4_patchLstm_noDeformation": "02_group4_patchLstm_noDeformation.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--model", choices=tuple(MODEL_CONFIGS))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_project_main(project_root: Path) -> ModuleType:
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root))
    path = project_root / "main.py"
    spec = importlib.util.spec_from_file_location("paper4_main", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer(path: Path) -> tuple[str, int] | None:
    model = next((part for part in path.parts if part in MODEL_CONFIGS), None)
    folds = [
        int(part[5:])
        for part in path.parts
        if part.startswith("fold_") and part[5:].isdigit()
    ]
    return (model, folds[-1]) if model is not None and folds else None


def checkpoints(args: argparse.Namespace) -> list[tuple[Path, str, int]]:
    paths = (
        [args.checkpoint.resolve()]
        if args.checkpoint is not None
        else sorted(args.checkpoint_root.resolve().rglob("best_model.pt"))
    )
    if not paths:
        raise FileNotFoundError("No best_model.pt checkpoint was found")
    if len(paths) > 1 and (args.model is not None or args.fold is not None):
        raise ValueError("Model/fold overrides only apply to one checkpoint")
    result = []
    for path in paths:
        parsed = infer(path)
        model = args.model or (parsed[0] if parsed else None)
        fold = args.fold or (parsed[1] if parsed else None)
        if model is None or fold is None:
            raise ValueError(f"Cannot infer model/fold from {path}")
        if not 1 <= int(fold) <= 10:
            raise ValueError(f"Fold must be in 1..10: {fold}")
        result.append((path, model, int(fold)))
    return result


def state_dict(path: Path, device: torch.device):
    try:
        value = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        value = torch.load(path, map_location=device)
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(value, dict) and isinstance(value.get(key), dict):
            value = value[key]
            break
    if value and all(str(key).startswith("module.") for key in value):
        value = {str(key)[7:]: tensor for key, tensor in value.items()}
    return value


def reported(path: Path) -> dict:
    metric_path = path.with_name("test_metrics.json")
    if not metric_path.is_file():
        return {}
    value = json.loads(metric_path.read_text(encoding="utf-8"))
    return {
        f"reported_{key}": value[key]
        for key in ("loss", "accuracy", "balanced_accuracy", "macro_f1")
        if key in value
    }


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["confusion_matrix"] = json.dumps(encoded["confusion_matrix"])
            writer.writerow(encoded)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    pipeline = load_project_main(project_root)
    from eeg_pipeline.evaluation.evaluator import evaluate_model

    rows = []
    items = checkpoints(args)
    for index, (path, label, fold_number) in enumerate(items, 1):
        config = copy.deepcopy(load_yaml(CONFIG_DIR / MODEL_CONFIGS[label]))
        config["data"]["root"] = str(data_root)
        config["training"]["device"] = args.device
        fold = pipeline.build_loso_subject_folds(tuple(config["data"]["subjects"]))[
            fold_number - 1
        ]
        train_loader, _, test_loader = pipeline.build_fold_loaders(config, fold)
        model, _, _, _, device, _ = pipeline.build_training_components(
            config, train_loader.dataset
        )
        model.load_state_dict(state_dict(path, device), strict=True)
        result = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
        row = {
            "checkpoint": str(path),
            "model": label,
            "fold": fold_number,
            "test_subject": fold.test_subject,
            "val_subject": fold.val_subject,
            **result,
            **reported(path),
        }
        if "reported_accuracy" in row:
            row["accuracy_abs_delta"] = abs(
                float(row["accuracy"]) - float(row["reported_accuracy"])
            )
        rows.append(row)
        print(
            f"[{index}/{len(items)}] {label} fold={fold_number:02d} "
            f"accuracy={float(result['accuracy']):.6f}",
            flush=True,
        )
    if args.output is not None:
        write(args.output.resolve(), rows)
        print(f"evaluation_results={args.output.resolve()}")
    elif len(rows) == 1:
        print(json.dumps(rows[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
