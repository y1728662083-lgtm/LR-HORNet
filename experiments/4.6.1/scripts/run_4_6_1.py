#!/usr/bin/env python3
"""Run the three Section 4.6.1 local-feature extractor comparisons."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
CONFIG_DIR = PACKAGE_DIR / "configs"
MODEL_CONFIGS = OrderedDict(
    (
        ("braindecode_eegnet", "00_braindecode_eegnet.yaml"),
        ("causal1_eegnet", "01_causal1_eegnet.yaml"),
        (
            "group4_patchLstm_noDeformation",
            "02_group4_patchLstm_noDeformation.yaml",
        ),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", default=",".join(MODEL_CONFIGS))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config


def parse_models(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in MODEL_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; choose from {list(MODEL_CONFIGS)}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate models: {names}")
    return names


def model_root(output_root: Path, model: str) -> Path:
    return output_root / "imagine" / model


def complete(output_root: Path, model: str) -> bool:
    root = model_root(output_root, model)
    return all(
        (root / f"fold_{fold:02d}" / name).is_file()
        for fold in range(1, 11)
        for name in ("best_model.pt", "test_metrics.json")
    )


def run_and_log(command: list[str], project_root: Path, log_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src") + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
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

    models = parse_models(args.models)
    models = [
        model
        for index, model in enumerate(models)
        if index % args.shard_count == args.shard_index
    ]
    generated_dir = output_root / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for model in models:
        config = load_yaml(CONFIG_DIR / MODEL_CONFIGS[model])
        config["data"]["root"] = str(data_root)
        config["training"]["output_root"] = str(output_root)
        config["training"]["device"] = args.device
        path = generated_dir / MODEL_CONFIGS[model]
        with path.open("w", encoding="utf-8", newline="\n") as file:
            yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
        generated.append((model, path))
        print(f"prepared {model}: {path}")
    if args.prepare_only:
        return 0

    for model, config_path in generated:
        if complete(output_root, model):
            print(f"skip complete model: {model}")
            continue
        if model_root(output_root, model).exists():
            raise RuntimeError(
                f"Partial output exists for {model}: {model_root(output_root, model)}"
            )
        if model == "braindecode_eegnet":
            command = [
                args.python,
                "-u",
                str(project_root / "src/eeg_pipeline/models/baseline_model.py"),
                "--config",
                str(config_path),
                "--architecture",
                "EEGNet",
            ]
        else:
            command = [
                args.python,
                "-u",
                str(project_root / "main.py"),
                "--config",
                str(config_path),
            ]
        run_and_log(
            command,
            project_root,
            output_root / "_logs" / f"{model}.log",
        )
        if not complete(output_root, model):
            raise RuntimeError(f"Missing fold outputs after training {model}")
        shutil.copy2(config_path, model_root(output_root, model) / "config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
