#!/usr/bin/env python3
"""Run five fixed-seed repetitions of the Section 4.6.3 group ablation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
CONFIG_DIR = PACKAGE_DIR / "configs"
GROUPS = (1, 2, 4, 8, 16)
MODEL_NAMES = {
    1: "group1_patchLstm_deformableLstm",
    2: "group2_patchLstm_deformableLstm",
    4: "group4_patchLstm_deformableLstm",
    8: "group8_patchLstm_deformableLstm",
    16: "group16_patchLstm_deformableLstm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--groups", default=",".join(str(x) for x in GROUPS))
    parser.add_argument("--repeats", type=int, default=5)
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


def parse_groups(value: str) -> list[int]:
    groups = [int(x.strip()) for x in value.split(",") if x.strip()]
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"Unknown group counts {sorted(unknown)}; choose from {GROUPS}")
    if len(groups) != len(set(groups)):
        raise ValueError(f"Duplicate group counts: {groups}")
    return groups


def run_root(output_root: Path, group: int, repeat: int, seed: int) -> Path:
    return (
        output_root / MODEL_NAMES[group] / "fold_01"
        / f"repeat_{repeat:02d}_seed_{seed}"
    )


def artifact_dir(root: Path, model: str) -> Path:
    return root / "imagine" / model / "fold_01"


def complete(path: Path) -> bool:
    return (path / "best_model.pt").is_file() and (path / "test_metrics.json").is_file()


def run_and_log(command: list[str], project_root: Path, log_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command, cwd=project_root, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
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
    if not 1 <= args.repeats <= 5:
        raise ValueError("repeats must be in 1..5")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")
    groups = [
        group for index, group in enumerate(parse_groups(args.groups))
        if index % args.shard_count == args.shard_index
    ]

    tasks = []
    for group in groups:
        source = CONFIG_DIR / f"group_{group:02d}.yaml"
        base = load_yaml(source)
        seed = int(base["seed"])
        model = MODEL_NAMES[group]
        if base["model"]["name"] != model:
            raise ValueError(f"Unexpected model in {source}: {base['model']['name']}")
        for repeat in range(1, args.repeats + 1):
            root = run_root(output_root, group, repeat, seed)
            config = load_yaml(source)
            config["data"]["root"] = str(data_root)
            config["training"]["device"] = args.device
            config["training"]["output_root"] = str(root)
            path = (
                output_root / "_generated_configs" / model / "fold_01"
                / f"repeat_{repeat:02d}_seed_{seed}.yaml"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as file:
                yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
            root.mkdir(parents=True, exist_ok=True)
            (root / "metadata.json").write_text(
                json.dumps(
                    {"electrode_groups": group, "model": model, "fold": 1,
                     "repeat": repeat, "seed": seed, "config": str(path)},
                    indent=2,
                ),
                encoding="utf-8",
            )
            tasks.append((group, repeat, model, root, path))
            print(f"prepared groups={group:02d} repeat={repeat:02d}: {path}")
    if args.prepare_only:
        return 0

    failures = 0
    for index, (group, repeat, model, root, config_path) in enumerate(tasks, 1):
        artifacts = artifact_dir(root, model)
        if complete(artifacts):
            print(f"[{index}/{len(tasks)}] skip complete groups={group:02d} repeat={repeat:02d}")
            continue
        if artifacts.exists() and any(artifacts.iterdir()):
            raise RuntimeError(f"Partial output exists: {artifacts}")
        try:
            run_and_log(
                [args.python, "-u", str(project_root / "main.py"), "--config", str(config_path)],
                project_root,
                root / "console.log",
            )
        except subprocess.CalledProcessError:
            failures += 1
            print(f"[{index}/{len(tasks)}] FAILED groups={group:02d} repeat={repeat:02d}; inspect {root / 'console.log'}", flush=True)
            continue
        if not complete(artifacts):
            failures += 1
            print(f"[{index}/{len(tasks)}] FAILED missing outputs: {artifacts}", flush=True)
        else:
            print(f"[{index}/{len(tasks)}] complete groups={group:02d} repeat={repeat:02d}")
    print(f"finished tasks={len(tasks)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
