#!/usr/bin/env python3
"""Train the four Section 4.7.1 offset-generation variants."""

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
VARIANTS = OrderedDict(
    (
        ("group4_patchLstm_deformableLstm", "00_deformable_lstm.yaml"),
        ("group4_patchLstm_pointwiseDeformation", "01_pointwise.yaml"),
        ("group4_patchLstm_staticDeformation", "02_static.yaml"),
        ("group4_patchLstm_groupedConvDeformation", "03_grouped_convolution.yaml"),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def parse_variants(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {list(VARIANTS)}")
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Variant list must be non-empty and unique: {names}")
    return names


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


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
        if environment.get("PYTHONPATH") else ""
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

    selected = parse_variants(args.variants)
    selected = [
        name for index, name in enumerate(selected)
        if index % args.shard_count == args.shard_index
    ]
    generated_dir = output_root / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[str, Path]] = []
    for model in selected:
        config = load_yaml(CONFIG_DIR / VARIANTS[model])
        config["data"]["root"] = str(data_root)
        config["training"]["output_root"] = str(output_root)
        config["training"]["device"] = args.device
        path = generated_dir / VARIANTS[model]
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        generated.append((model, path))
        print(f"prepared {model}: {path}")
    if args.prepare_only:
        return 0

    for model, config_path in generated:
        if complete(output_root, model):
            print(f"skip complete model: {model}")
            continue
        root = model_root(output_root, model)
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"Partial output exists; move it aside before retry: {root}")
        run_and_log(
            [args.python, "-u", str(project_root / "main.py"), "--config", str(config_path)],
            project_root,
            output_root / "_logs" / f"{model}.log",
        )
        if not complete(output_root, model):
            raise RuntimeError(f"Missing fold outputs after training {model}")
        shutil.copy2(config_path, root / "config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
