#!/usr/bin/env python3
"""Re-evaluate one or all Section 4.4.2 best_model.pt checkpoints."""

from __future__ import annotations

import argparse
import copy
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
CONDITION_PATTERN = re.compile(r"^test_(\d+)_val_(\d+)$")
REPEAT_PATTERN = re.compile(r"^repeat_(\d+)_seed_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--test-subject", type=int)
    parser.add_argument("--val-subject", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    config.pop("sweep", None)
    return config


def load_project_main(project_root: Path) -> ModuleType:
    main_path = project_root / "main.py"
    src_path = project_root / "src"
    if not main_path.is_file() or not src_path.is_dir():
        raise FileNotFoundError(
            f"Expected main.py and src/ under project root: {project_root}"
        )
    sys.path.insert(0, str(src_path))
    sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location("paper4_main", main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_checkpoint_identity(path: Path) -> tuple[int, int, int | None, int | None] | None:
    test_subject = None
    val_subject = None
    repeat = None
    seed = None
    for part in path.parts:
        condition = CONDITION_PATTERN.fullmatch(part)
        if condition:
            test_subject, val_subject = map(int, condition.groups())
        repetition = REPEAT_PATTERN.fullmatch(part)
        if repetition:
            repeat, seed = map(int, repetition.groups())
    if test_subject is None or val_subject is None:
        return None
    return test_subject, val_subject, repeat, seed


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

    items = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed = parse_checkpoint_identity(path)
        if parsed is None:
            if args.test_subject is None or args.val_subject is None:
                raise ValueError(
                    "Checkpoint path does not contain test_XX_val_YY; provide "
                    "--test-subject and --val-subject"
                )
            parsed = (args.test_subject, args.val_subject, None, None)
        test_subject, val_subject, repeat, seed = parsed
        items.append(
            {
                "path": path,
                "test_subject": test_subject,
                "val_subject": val_subject,
                "repeat": repeat,
                "seed": seed,
            }
        )
    return items


def make_fold(pipeline_main: ModuleType, config: dict, test_subject: int, val_subject: int):
    subjects = tuple(int(subject) for subject in config["data"]["subjects"])
    if test_subject not in subjects or val_subject not in subjects:
        raise ValueError("Test and validation subjects must be in data.subjects")
    if test_subject == val_subject:
        raise ValueError("Test and validation subjects must differ")
    fold_index = subjects.index(test_subject)
    original = pipeline_main.build_loso_subject_folds(subjects)[fold_index]
    train_subjects = tuple(
        subject
        for subject in subjects
        if subject not in (test_subject, val_subject)
    )
    return pipeline_main.SubjectFold(
        fold_index,
        train_subjects,
        val_subject,
        original.test_subject,
    )


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
    config = load_yaml(args.config.resolve())
    data_root = args.data_root or Path(str(config["data"]["root"]))
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")
    config["data"]["root"] = str(data_root)
    config["training"]["device"] = args.device
    pipeline_main = load_project_main(project_root)
    from eeg_pipeline.evaluation.evaluator import evaluate_model

    checkpoints = discover_checkpoints(args)
    condition_cache: dict[tuple[int, int, int], tuple] = {}
    rows: list[dict] = []
    for index, item in enumerate(checkpoints, 1):
        run_config = copy.deepcopy(config)
        if item["seed"] is not None:
            run_config["seed"] = int(item["seed"])
        test_subject = int(item["test_subject"])
        val_subject = int(item["val_subject"])
        cache_key = (test_subject, val_subject, int(run_config["seed"]))
        if cache_key not in condition_cache:
            fold = make_fold(pipeline_main, run_config, test_subject, val_subject)
            train_loader, _, test_loader = pipeline_main.build_fold_loaders(
                run_config, fold
            )
            model, _, _, _, device, _ = pipeline_main.build_training_components(
                run_config, train_loader.dataset
            )
            condition_cache[cache_key] = (
                model,
                test_loader,
                nn.CrossEntropyLoss(),
                device,
            )
        model, test_loader, criterion, device = condition_cache[cache_key]
        model.load_state_dict(load_checkpoint(item["path"], device), strict=True)
        result = evaluate_model(model, test_loader, criterion, device)
        row = {
            "checkpoint": str(item["path"]),
            "test_subject": test_subject,
            "val_subject": val_subject,
            "repeat": item["repeat"] if item["repeat"] is not None else "",
            "seed": int(run_config["seed"]),
            **result,
        }
        rows.append(row)
        print(
            f"[{index}/{len(checkpoints)}] test={test_subject:02d} "
            f"val={val_subject:02d} repeat={row['repeat']} "
            f"accuracy={float(result['accuracy']):.6f}",
            flush=True,
        )

    if args.output is not None:
        output_path = args.output.resolve()
        write_results(output_path, rows)
        print(f"evaluation_results={output_path}")
    elif len(rows) == 1:
        print(json.dumps(rows[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

