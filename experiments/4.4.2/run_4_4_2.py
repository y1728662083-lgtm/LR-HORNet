#!/usr/bin/env python3
"""Prepare and run the complete Section 4.4.2 validation-subject sweep."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Task:
    test_subject: int
    val_subject: int
    fold_index: int
    repeat: int
    seed: int

    @property
    def variant(self) -> str:
        return f"test_{self.test_subject:02d}_val_{self.val_subject:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/4.4.2"))
    parser.add_argument("--gpu-id", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    return parser.parse_args()


def resolve_from_project(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_template(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    required = {"seed", "data", "split", "loader", "model", "training", "sweep"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return config


def build_tasks(config: dict, repeats_override: int | None) -> list[Task]:
    subjects = [int(subject) for subject in config["data"]["subjects"]]
    sweep = config["sweep"]
    test_subjects = [int(subject) for subject in sweep["test_subjects"]]
    val_subjects = [int(subject) for subject in sweep["validation_subjects"]]
    repeats = repeats_override or int(sweep["repeats"])
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    unknown = set(test_subjects + val_subjects).difference(subjects)
    if unknown:
        raise ValueError(f"Sweep contains subjects absent from data.subjects: {sorted(unknown)}")

    tasks: list[Task] = []
    seed = int(config["seed"])
    for test_subject in test_subjects:
        fold_index = subjects.index(test_subject)
        for val_subject in val_subjects:
            if val_subject == test_subject:
                continue
            for repeat in range(1, repeats + 1):
                tasks.append(Task(test_subject, val_subject, fold_index, repeat, seed))
    return tasks


def select_shard(tasks: list[Task], shard_index: int, shard_count: int) -> list[Task]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")
    conditions = sorted({(task.fold_index, task.val_subject) for task in tasks})
    assigned = {
        condition
        for index, condition in enumerate(conditions)
        if index % shard_count == shard_index
    }
    return [
        task
        for task in tasks
        if (task.fold_index, task.val_subject) in assigned
    ]


def run_root(output_root: Path, task: Task) -> Path:
    return (
        output_root
        / "group_05"
        / task.variant
        / f"fold_{task.fold_index + 1:02d}"
        / f"repeat_{task.repeat:02d}_seed_{task.seed}"
    )


def generated_config_path(output_root: Path, task: Task) -> Path:
    return (
        output_root
        / "_generated_configs"
        / "group_05"
        / task.variant
        / f"fold_{task.fold_index + 1:02d}"
        / f"repeat_{task.repeat:02d}_seed_{task.seed}.yaml"
    )


def artifact_dir(output_root: Path, task: Task, model_name: str) -> Path:
    return (
        run_root(output_root, task)
        / f"val_subject_{task.val_subject:02d}"
        / "imagine"
        / model_name
        / f"fold_{task.fold_index + 1:02d}"
    )


def make_config(template: dict, task: Task, data_root: Path, output_root: Path) -> dict:
    config = copy.deepcopy(template)
    config.pop("sweep", None)
    config["seed"] = task.seed
    config["data"]["root"] = str(data_root)
    config["split"]["fold_indices"] = [task.fold_index]
    config["split"]["val_subject"] = task.val_subject
    config["training"]["output_root"] = str(run_root(output_root, task))
    return config


def prepare_task(
    template: dict,
    task: Task,
    data_root: Path,
    output_root: Path,
) -> None:
    config_path = generated_config_path(output_root, task)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(
            make_config(template, task, data_root, output_root),
            file,
            sort_keys=False,
        )

    root = run_root(output_root, task)
    root.mkdir(parents=True, exist_ok=True)
    model_name = str(template["model"]["name"])
    metadata = {
        "group": 5,
        "variant": task.variant,
        "model": model_name,
        "fold_index": task.fold_index,
        "test_subject": task.test_subject,
        "val_subject": task.val_subject,
        "repeat": task.repeat,
        "seed": task.seed,
        "config_path": str(config_path),
        "artifact_dir": str(artifact_dir(output_root, task, model_name)),
    }
    with (root / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def completed(output_root: Path, task: Task, model_name: str) -> bool:
    directory = artifact_dir(output_root, task, model_name)
    try:
        with (directory / "training_log.csv").open(
            newline="", encoding="utf-8"
        ) as file:
            if not any(row.get("is_best") == "1" for row in csv.DictReader(file)):
                return False
        with (directory / "test_metrics.json").open(encoding="utf-8") as file:
            metrics = json.load(file)
        return "accuracy" in metrics and (directory / "best_model.pt").is_file()
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    template = load_template(config_path)
    data_root_value = args.data_root or Path(str(template["data"]["root"]))
    data_root = resolve_from_project(data_root_value, project_root)
    output_root = resolve_from_project(args.output_root, project_root)
    main_path = project_root / "main.py"
    src_path = project_root / "src"
    if not main_path.is_file() or not src_path.is_dir():
        raise FileNotFoundError(
            f"Expected main.py and src/ under project root: {project_root}"
        )

    all_tasks = build_tasks(template, args.repeats)
    tasks = select_shard(all_tasks, args.shard_index, args.shard_count)
    if not args.skip_prepare:
        for task in tasks:
            prepare_task(template, task, data_root, output_root)
    missing_configs = [
        generated_config_path(output_root, task)
        for task in tasks
        if not generated_config_path(output_root, task).is_file()
    ]
    if missing_configs:
        raise FileNotFoundError(
            "Generated configs are missing; run once without --skip-prepare: "
            f"{missing_configs[0]}"
        )

    print(
        f"group=5 planned={len(tasks)} total_sweep={len(all_tasks)} "
        f"shard={args.shard_index}/{args.shard_count} seed={template['seed']}"
    )
    print(f"project_root={project_root}")
    print(f"data_root={data_root}")
    print(f"output_root={output_root}")
    if args.prepare_only:
        return 0
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    visible = environment["CUDA_VISIBLE_DEVICES"]
    if not visible or len(visible.split(",")) != 1:
        raise ValueError("--gpu-id must select exactly one physical GPU")
    environment["PYTHONPATH"] = str(src_path) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )

    model_name = str(template["model"]["name"])
    failures = 0
    for index, task in enumerate(tasks, 1):
        root = run_root(output_root, task)
        if completed(output_root, task, model_name):
            print(f"[{index}/{len(tasks)}] SKIP completed {root}", flush=True)
            continue
        print(
            f"[{index}/{len(tasks)}] GPU={visible} "
            f"test={task.test_subject} val={task.val_subject} "
            f"repeat={task.repeat}",
            flush=True,
        )
        log_path = root / "console.log"
        with log_path.open("w", encoding="utf-8") as log:
            outcome = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(main_path),
                    "--config",
                    str(generated_config_path(output_root, task)),
                ],
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        if outcome.returncode or not completed(output_root, task, model_name):
            failures += 1
            print(f"FAILED {root}; inspect {log_path}", flush=True)
        else:
            print(f"COMPLETE {root}", flush=True)

    print(f"group=5 failures={failures}/{len(tasks)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

