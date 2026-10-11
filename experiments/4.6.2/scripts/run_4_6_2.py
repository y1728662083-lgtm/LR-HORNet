#!/usr/bin/env python3
"""Run the Section 4.6.2 patch-length ablation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
BASE_CONFIG = PACKAGE_DIR / "config.yaml"


@dataclass(frozen=True)
class Variant:
    name: str
    patch_size: int
    patch_stride: int


def variants(config: dict) -> list[Variant]:
    sweep = config["sweep"]
    result = [
        Variant(
            "full_sequence_no_patch",
            int(sweep["full_sequence_no_patch"]["patch_size"]),
            int(sweep["full_sequence_no_patch"]["patch_stride"]),
        )
    ]
    result.extend(
        Variant(f"patch_{int(size):04d}", int(size), int(size) // 2)
        for size in sweep["patch_sizes"]
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variants")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def artifact_dir(run_root: Path, model: str, fold: int) -> Path:
    return run_root / "imagine" / model / f"fold_{fold:02d}"


def complete(path: Path) -> bool:
    return (path / "best_model.pt").is_file() and (
        path / "test_metrics.json"
    ).is_file()


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
        raise FileNotFoundError(data_root)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")

    base = load_yaml(BASE_CONFIG)
    sweep = base["sweep"]
    planned = variants(base)
    if args.variants:
        requested = {x.strip() for x in args.variants.split(",") if x.strip()}
        known = {x.name for x in planned}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown variants {sorted(unknown)}; choose from {sorted(known)}")
        planned = [item for item in planned if item.name in requested]
    planned = [
        item for index, item in enumerate(planned)
        if index % args.shard_count == args.shard_index
    ]
    repeat_count = args.repeats or int(sweep["repeats"])
    if not 1 <= repeat_count <= int(sweep["repeats"]):
        raise ValueError(f"repeats must be in 1..{sweep['repeats']}")
    fold = int(sweep["fold_index"]) + 1
    seed = int(sweep["seed_per_repeat"])
    model = str(base["model"]["name"])

    generated = []
    for variant in planned:
        for repeat in range(1, repeat_count + 1):
            run_root = (
                output_root / variant.name / f"fold_{fold:02d}"
                / f"repeat_{repeat:02d}_seed_{seed}"
            )
            config = copy.deepcopy(base)
            config.pop("sweep")
            config["seed"] = seed
            config["data"]["root"] = str(data_root)
            config["split"]["fold_indices"] = [fold - 1]
            config["model"]["kwargs"]["patch_size"] = variant.patch_size
            config["model"]["kwargs"]["patch_stride"] = variant.patch_stride
            config["training"]["device"] = args.device
            config["training"]["output_root"] = str(run_root)
            config_path = (
                output_root / "_generated_configs" / variant.name
                / f"fold_{fold:02d}" / f"repeat_{repeat:02d}_seed_{seed}.yaml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("w", encoding="utf-8", newline="\n") as file:
                yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
            run_root.mkdir(parents=True, exist_ok=True)
            metadata = {
                "variant": variant.name,
                "patch_size": variant.patch_size,
                "patch_stride": variant.patch_stride,
                "fold": fold,
                "repeat": repeat,
                "seed": seed,
                "config": str(config_path),
            }
            (run_root / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            generated.append((variant.name, repeat, run_root, config_path))
            print(f"prepared {variant.name} repeat={repeat:02d}: {config_path}")
    if args.prepare_only:
        return 0

    for index, (variant, repeat, run_root, config_path) in enumerate(generated, 1):
        artifacts = artifact_dir(run_root, model, fold)
        if complete(artifacts):
            print(f"[{index}/{len(generated)}] skip complete {variant} repeat={repeat:02d}")
            continue
        if artifacts.exists():
            raise RuntimeError(f"Partial output exists: {artifacts}")
        run_and_log(
            [args.python, "-u", str(project_root / "main.py"), "--config", str(config_path)],
            project_root,
            run_root / "console.log",
        )
        if not complete(artifacts):
            raise RuntimeError(f"Missing outputs after training: {artifacts}")
        print(f"[{index}/{len(generated)}] complete {variant} repeat={repeat:02d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
