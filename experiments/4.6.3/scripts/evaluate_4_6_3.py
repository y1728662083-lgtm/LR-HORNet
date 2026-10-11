#!/usr/bin/env python3
"""Re-evaluate Section 4.6.3 fixed-seed repetition checkpoints."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import torch
import yaml
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "configs"
GROUPS = (1, 2, 4, 8, 16)
MODEL_NAMES = {
    1: "group1_patchLstm_deformableLstm",
    2: "group2_patchLstm_deformableLstm",
    4: "group4_patchLstm_deformableLstm",
    8: "group8_patchLstm_deformableLstm",
    16: "group16_patchLstm_deformableLstm",
}
MODEL_TO_GROUP = {name: groups for groups, name in MODEL_NAMES.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--groups", type=int, choices=GROUPS)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--repeat", type=int)
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


def infer(path: Path) -> tuple[int, int, int] | None:
    groups = next((MODEL_TO_GROUP[part] for part in path.parts if part in MODEL_TO_GROUP), None)
    if groups is None:
        labels = [
            part for part in path.parts
            if re.fullmatch(r"group_?0?(1|2|4|8)|group_?16", part)
        ]
        if labels:
            groups = int(labels[-1].replace("group", "").replace("_", ""))
    folds = [int(part[5:]) for part in path.parts if part.startswith("fold_") and part[5:].isdigit()]
    repeats = []
    for part in path.parts:
        match = re.fullmatch(r"repeat_(\d+)_seed_\d+", part)
        if match:
            repeats.append(int(match.group(1)))
    return (groups, folds[-1], repeats[-1]) if groups and folds and repeats else None


def checkpoints(args: argparse.Namespace) -> list[tuple[Path, int, int, int]]:
    paths = [args.checkpoint.resolve()] if args.checkpoint else sorted(
        args.checkpoint_root.resolve().rglob("best_model.pt")
    )
    if not paths:
        raise FileNotFoundError("No best_model.pt checkpoint was found")
    if len(paths) > 1 and any(x is not None for x in (args.groups, args.fold, args.repeat)):
        raise ValueError("Group/fold/repeat overrides only apply to one checkpoint")
    result = []
    for path in paths:
        parsed = infer(path)
        groups = args.groups or (parsed[0] if parsed else None)
        fold = args.fold or (parsed[1] if parsed else None)
        repeat = args.repeat or (parsed[2] if parsed else None)
        if groups is None or fold is None or repeat is None:
            raise ValueError(f"Cannot infer group/fold/repeat from {path}")
        if groups not in GROUPS or not 1 <= int(fold) <= 10 or int(repeat) < 1:
            raise ValueError(f"Invalid group/fold/repeat for {path}: {groups}/{fold}/{repeat}")
        result.append((path, int(groups), int(fold), int(repeat)))
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
    return {f"reported_{key}": value[key] for key in ("loss", "accuracy", "balanced_accuracy", "macro_f1") if key in value}


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

    items = checkpoints(args)
    cache = {}
    rows = []
    for index, (path, groups, fold_number, repeat) in enumerate(items, 1):
        key = (groups, fold_number)
        if key not in cache:
            config = load_yaml(CONFIG_DIR / f"group_{groups:02d}.yaml")
            config["data"]["root"] = str(data_root)
            config["training"]["device"] = args.device
            fold = pipeline.build_loso_subject_folds(tuple(config["data"]["subjects"]))[fold_number - 1]
            train_loader, _, test_loader = pipeline.build_fold_loaders(config, fold)
            model, _, _, _, device, _ = pipeline.build_training_components(config, train_loader.dataset)
            cache[key] = (fold, test_loader, model, device)
        fold, test_loader, model, device = cache[key]
        model.load_state_dict(state_dict(path, device), strict=True)
        result = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
        row = {
            "checkpoint": str(path), "model": MODEL_NAMES[groups],
            "groups": groups, "fold": fold_number, "repeat": repeat,
            "test_subject": fold.test_subject, "val_subject": fold.val_subject,
            **result, **reported(path),
        }
        if "reported_accuracy" in row:
            row["accuracy_abs_delta"] = abs(float(row["accuracy"]) - float(row["reported_accuracy"]))
        rows.append(row)
        print(f"[{index}/{len(items)}] groups={groups:02d} repeat={repeat:02d} accuracy={float(result['accuracy']):.6f}", flush=True)
    if args.output:
        write(args.output.resolve(), rows)
        print(f"evaluation_results={args.output.resolve()}")
    elif len(rows) == 1:
        print(json.dumps(rows[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
