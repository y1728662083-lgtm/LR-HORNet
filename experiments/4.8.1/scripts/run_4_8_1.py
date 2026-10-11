#!/usr/bin/env python3
"""Run the five fold-1 recurrent-operator variants, five times at seed 42."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACKAGE_DIR / "configs"
MODELS = OrderedDict(
    (
        ("group4_patchLstm_deformableLstm", "00_patchlstm_deformablelstm.yaml"),
        ("group4_patchGRU_deformableGRU", "01_patchgru_deformablegru.yaml"),
        ("group4_patchRNN_deformableRNN", "02_patchrnn_deformablernn.yaml"),
        ("group4_patchLstm_deformableCfC", "03_patchlstm_deformablecfc.yaml"),
        ("group4_patchCfC_deformableCfC", "04_patchcfc_deformablecfc.yaml"),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--repeats", default="1-5")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def parse_models(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in MODELS]
    if unknown or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid models {names}; choices: {list(MODELS)}")
    return names


def parse_repeats(value: str) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            values.extend(range(first, last + 1))
        else:
            values.append(int(part))
    if not values or len(values) != len(set(values)) or any(value < 1 for value in values):
        raise ValueError(f"Invalid repeats: {value}")
    return values


def task_root(output_root: Path, model: str, repeat: int) -> Path:
    return output_root / model / "fold_01" / f"repeat_{repeat:02d}_seed_42"


def complete(root: Path, model: str) -> bool:
    fold = root / "imagine" / model / "fold_01"
    return all((fold / name).is_file() for name in ("best_model.pt", "test_metrics.json"))


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

    tasks = [
        (model, repeat)
        for model in parse_models(args.models)
        for repeat in parse_repeats(args.repeats)
    ]
    tasks = [
        task for index, task in enumerate(tasks)
        if index % args.shard_count == args.shard_index
    ]
    prepared: list[tuple[str, int, Path, Path]] = []
    for model, repeat in tasks:
        root = task_root(output_root, model, repeat)
        config = load_yaml(CONFIG_DIR / MODELS[model])
        config["seed"] = 42
        config["data"]["root"] = str(data_root)
        config["split"]["fold_indices"] = [0]
        config["training"]["device"] = args.device
        config["training"]["output_root"] = str(root)
        config_path = output_root / "_generated_configs" / model / f"repeat_{repeat:02d}_seed_42.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        prepared.append((model, repeat, root, config_path))
        print(f"prepared {model} repeat={repeat:02d}: {config_path}")
    if args.prepare_only:
        return 0

    for model, repeat, root, config_path in prepared:
        if complete(root, model):
            print(f"skip complete task: {model} repeat={repeat:02d}")
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
        if not complete(root, model):
            raise RuntimeError(f"Missing fold output after training: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
