#!/usr/bin/env python3
"""Re-evaluate one or all Section 4.5 best_model.pt checkpoints."""

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
PACKAGE_DIR = SCRIPT_DIR.parent
CONFIG_DIR = PACKAGE_DIR / "configs"
MODEL_CONFIGS = {
    "eegnet": "00_eegnet.yaml",
    "causal1_eegnet": "01_causal1_eegnet.yaml",
    "group4_causal1_deformableLstm": (
        "02_group4_causal1_deformableLstm.yaml"
    ),
    "group4_patchLstm_noDeformation": (
        "03_group4_patchLstm_noDeformation.yaml"
    ),
    "group4_patchLstm_deformableLstm": (
        "04_group4_patchLstm_deformableLstm.yaml"
    ),
}
MODEL_ALIASES = {"braindecode_eegnet": "eegnet"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--model", choices=tuple(MODEL_CONFIGS))
    parser.add_argument("--fold", type=int, help="One-based fold number, 1-10.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config


def load_project_main(project_root: Path) -> ModuleType:
    main_path = project_root / "main.py"
    source_path = project_root / "src"
    if not main_path.is_file() or not source_path.is_dir():
        raise FileNotFoundError(
            f"Expected main.py and src/ under project root: {project_root}"
        )
    sys.path.insert(0, str(source_path))
    sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location("paper4_main", main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity_from_path(path: Path) -> tuple[str, int] | None:
    model = None
    fold = None
    for part in path.parts:
        normalized = MODEL_ALIASES.get(part, part)
        if normalized in MODEL_CONFIGS:
            model = normalized
        if part.startswith("fold_") and part[5:].isdigit():
            fold = int(part[5:])
    if model is None or fold is None:
        return None
    return model, fold


def discover_checkpoints(args: argparse.Namespace) -> list[dict]:
    if args.checkpoint is not None:
        paths = [args.checkpoint.resolve()]
    else:
        root = args.checkpoint_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
        paths = sorted(root.rglob("best_model.pt"))
    if not paths:
        raise FileNotFoundError("No best_model.pt checkpoint was found")
    if len(paths) > 1 and (args.model is not None or args.fold is not None):
        raise ValueError("--model and --fold overrides are only valid for one checkpoint")

    items = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        identity = identity_from_path(path)
        model = args.model or (identity[0] if identity else None)
        fold = args.fold or (identity[1] if identity else None)
        if model is None or fold is None:
            raise ValueError(
                f"Cannot infer model/fold from {path}; provide both --model and --fold"
            )
        if not 1 <= int(fold) <= 10:
            raise ValueError(f"Fold must be between 1 and 10: {fold}")
        items.append({"path": path, "model": model, "fold": int(fold)})
    return items


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a PyTorch state dictionary")
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break
    if checkpoint and all(str(key).startswith("module.") for key in checkpoint):
        checkpoint = {str(key)[7:]: value for key, value in checkpoint.items()}
    return checkpoint


def load_checkpoint(path: Path, device: torch.device):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    return unwrap_state_dict(payload)


def reported_metrics(checkpoint: Path) -> dict:
    path = checkpoint.with_name("test_metrics.json")
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return {
        f"reported_{name}": payload[name]
        for name in ("loss", "accuracy", "balanced_accuracy", "macro_f1")
        if name in payload
    }


def write_results(path: Path, rows: list[dict]) -> None:
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
            encoded["confusion_matrix"] = json.dumps(
                encoded.get("confusion_matrix", []), separators=(",", ":")
            )
            writer.writerow(encoded)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")
    pipeline_main = load_project_main(project_root)
    from eeg_pipeline.evaluation.evaluator import evaluate_model

    rows: list[dict] = []
    checkpoints = discover_checkpoints(args)
    for index, item in enumerate(checkpoints, 1):
        model_label = str(item["model"])
        fold_number = int(item["fold"])
        config = copy.deepcopy(
            load_yaml(CONFIG_DIR / MODEL_CONFIGS[model_label])
        )
        config["data"]["root"] = str(data_root)
        config["training"]["device"] = args.device
        config["split"]["fold_indices"] = [fold_number - 1]
        fold = pipeline_main.build_loso_subject_folds(
            tuple(config["data"]["subjects"])
        )[fold_number - 1]
        train_loader, _, test_loader = pipeline_main.build_fold_loaders(
            config, fold
        )
        model, _, _, _, device, _ = pipeline_main.build_training_components(
            config, train_loader.dataset
        )
        model.load_state_dict(
            load_checkpoint(item["path"], device), strict=True
        )
        result = evaluate_model(
            model, test_loader, nn.CrossEntropyLoss(), device
        )
        reported = reported_metrics(item["path"])
        row = {
            "checkpoint": str(item["path"]),
            "model": model_label,
            "fold": fold_number,
            "test_subject": fold.test_subject,
            "val_subject": fold.val_subject,
            **result,
            **reported,
        }
        if "reported_accuracy" in row:
            row["accuracy_abs_delta"] = abs(
                float(row["accuracy"]) - float(row["reported_accuracy"])
            )
        rows.append(row)
        print(
            f"[{index}/{len(checkpoints)}] model={model_label} "
            f"fold={fold_number:02d} accuracy={float(result['accuracy']):.6f}",
            flush=True,
        )
        del model, train_loader, test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.output is not None:
        output_path = args.output.resolve()
        write_results(output_path, rows)
        print(f"evaluation_results={output_path}")
    elif len(rows) == 1:
        print(json.dumps(rows[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
