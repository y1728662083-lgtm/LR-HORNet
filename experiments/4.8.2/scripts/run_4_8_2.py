#!/usr/bin/env python3
"""Run the formal Attention and Mamba compatibility experiments in Section 4.8.2."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent.parent
ATTENTION = {
    "satt": ("group4_patchLstm_deformableLstm_satt", "attention/group4_patchLstm_deformableLstm_satt.yaml"),
    "statt": ("group4_patchLstm_deformableLstm_statt", "attention/group4_patchLstm_deformableLstm_statt.yaml"),
}
MAMBA = {
    "full_mamba_patch_replaced": ("group4_fullMamba_deformableLstm", "mamba/full_mamba_patch_replaced.yaml"),
    "mamba_offset_only": ("group4_patchLstm_deformableMamba", "mamba/mamba_offset_only.yaml"),
}


@dataclass(frozen=True)
class Task:
    family: str
    variant: str
    model: str
    template: str
    fold: int | None = None
    repeat: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--family", choices=("all", "attention", "mamba"), default="all")
    parser.add_argument("--attention-variants", default=",".join(ATTENTION))
    parser.add_argument("--mamba-variants", default=",".join(MAMBA))
    parser.add_argument("--mamba-mode", choices=("all", "primary", "fold1-repeats"), default="all")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def choices(value: str, available: dict, label: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in values if name not in available]
    if unknown or not values or len(values) != len(set(values)):
        raise ValueError(f"Invalid {label} {values}; choices: {list(available)}")
    return values


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def build_tasks(args: argparse.Namespace) -> list[Task]:
    tasks: list[Task] = []
    if args.family in {"all", "attention"}:
        for variant in choices(args.attention_variants, ATTENTION, "attention variants"):
            model, template = ATTENTION[variant]
            tasks.append(Task("attention", variant, model, template))
    if args.family in {"all", "mamba"}:
        if args.mamba_mode == "primary":
            fold_repeats = [(fold, 1) for fold in range(1, 11)]
        elif args.mamba_mode == "fold1-repeats":
            fold_repeats = [(1, repeat) for repeat in range(1, 6)]
        else:
            fold_repeats = [(fold, 1) for fold in range(1, 11)] + [(1, repeat) for repeat in range(2, 6)]
        for variant in choices(args.mamba_variants, MAMBA, "Mamba variants"):
            model, template = MAMBA[variant]
            tasks.extend(Task("mamba", variant, model, template, fold, repeat) for fold, repeat in fold_repeats)
    return tasks


def task_root(output_root: Path, task: Task) -> Path:
    if task.family == "attention":
        return output_root / "attention" / task.variant
    assert task.fold is not None and task.repeat is not None
    return output_root / "mamba" / task.variant / f"fold_{task.fold:02d}" / f"repeat_{task.repeat:02d}_seed_42"


def complete(root: Path, task: Task) -> bool:
    if task.family == "attention":
        return all(
            (root / "imagine" / task.model / f"fold_{fold:02d}" / name).is_file()
            for fold in range(1, 11)
            for name in ("best_model.pt", "test_metrics.json")
        )
    assert task.fold is not None
    fold_root = root / "imagine" / task.model / f"fold_{task.fold:02d}"
    return all((fold_root / name).is_file() for name in ("best_model.pt", "test_metrics.json"))


def run_and_log(command: list[str], project_root: Path, log_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def generated_path(output_root: Path, task: Task) -> Path:
    if task.family == "attention":
        return output_root / "_generated_configs" / "attention" / f"{task.variant}.yaml"
    assert task.fold is not None and task.repeat is not None
    return output_root / "_generated_configs" / "mamba" / task.variant / f"fold_{task.fold:02d}" / f"repeat_{task.repeat:02d}_seed_42.yaml"


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if not (project_root / "main.py").is_file():
        raise FileNotFoundError(f"main.py is missing under {project_root}")
    if not args.prepare_only and not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")

    tasks = [task for index, task in enumerate(build_tasks(args)) if index % args.shard_count == args.shard_index]
    prepared: list[tuple[Task, Path, Path]] = []
    for task in tasks:
        root = task_root(output_root, task)
        config = load_yaml(PACKAGE_DIR / "configs" / task.template)
        config["seed"] = 42
        config["data"]["root"] = str(data_root)
        config["training"]["device"] = args.device
        config["training"]["output_root"] = str(root)
        if task.family == "mamba":
            assert task.fold is not None
            config["split"]["fold_indices"] = [task.fold - 1]
        path = generated_path(output_root, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        prepared.append((task, root, path))
        suffix = "10 folds" if task.family == "attention" else f"fold={task.fold:02d} repeat={task.repeat:02d}"
        print(f"prepared {task.family}/{task.variant} {suffix}: {path}")
    if args.prepare_only:
        return 0

    for task, root, config_path in prepared:
        if complete(root, task):
            print(f"skip complete task: {task.family}/{task.variant}")
            continue
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"Partial output exists; move it aside before retry: {root}")
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, root / "config.yaml")
        run_and_log(
            [args.python, "-u", str(project_root / "main.py"), "--config", str(config_path)],
            project_root,
            root / "console.log",
        )
        if not complete(root, task):
            raise RuntimeError(f"Missing outputs after training: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
