#!/usr/bin/env python3
"""Run the four independently trained Section 4.5 ablation stages.

The complete LR-HORNet stage is available as an explicit ``--models`` choice,
but the reported study reuses its Section 4.4.1 main result to avoid duplicate
ten-fold training.
"""

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
        ("eegnet", "00_eegnet.yaml"),
        ("causal1_eegnet", "01_causal1_eegnet.yaml"),
        (
            "group4_causal1_deformableLstm",
            "02_group4_causal1_deformableLstm.yaml",
        ),
        (
            "group4_patchLstm_noDeformation",
            "03_group4_patchLstm_noDeformation.yaml",
        ),
        (
            "group4_patchLstm_deformableLstm",
            "04_group4_patchLstm_deformableLstm.yaml",
        ),
    )
)
DEFAULT_ABLATION_MODELS = tuple(MODEL_CONFIGS)[:4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_ABLATION_MODELS),
        help=(
            "Comma-separated model labels. By default, run the four ablation "
            "stages; the reported complete-model row reuses Section 4.4.1."
        ),
    )
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


def selected_models(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in MODEL_CONFIGS]
    if unknown:
        raise ValueError(
            f"Unknown model labels {unknown}; choose from {list(MODEL_CONFIGS)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate model labels: {names}")
    return names


def expected_output(output_root: Path, model_label: str) -> Path:
    return output_root / "imagine" / model_label


def output_complete(output_root: Path, model_label: str) -> bool:
    model_root = expected_output(output_root, model_label)
    return all(
        (model_root / f"fold_{fold:02d}" / filename).is_file()
        for fold in range(1, 11)
        for filename in ("best_model.pt", "test_metrics.json")
    )


def stream_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    source_root = str(cwd / "src")
    environment["PYTHONPATH"] = source_root + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
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


def normalize_eegnet_output(output_root: Path) -> None:
    raw = output_root / "imagine" / "braindecode_eegnet"
    normalized = expected_output(output_root, "eegnet")
    if normalized.exists():
        raise FileExistsError(
            "Cannot normalize Braindecode EEGNet output because the target "
            f"already exists: {normalized}"
        )
    if not raw.is_dir():
        raise FileNotFoundError(f"Expected EEGNet output was not created: {raw}")
    raw.rename(normalized)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if not (project_root / "main.py").is_file():
        raise FileNotFoundError(f"main.py is missing under {project_root}")
    if not (project_root / "src" / "eeg_pipeline").is_dir():
        raise FileNotFoundError(f"src/eeg_pipeline is missing under {project_root}")
    if not args.prepare_only and not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")

    requested = selected_models(args.models)
    requested = [
        model
        for index, model in enumerate(requested)
        if index % args.shard_count == args.shard_index
    ]
    generated_dir = output_root / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    runnable: list[tuple[str, Path]] = []
    for model_label in requested:
        config = load_yaml(CONFIG_DIR / MODEL_CONFIGS[model_label])
        config["data"]["root"] = str(data_root)
        config["training"]["output_root"] = str(output_root)
        config["training"]["device"] = args.device
        generated_path = generated_dir / MODEL_CONFIGS[model_label]
        with generated_path.open("w", encoding="utf-8", newline="\n") as file:
            yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
        runnable.append((model_label, generated_path))
        print(f"prepared {model_label}: {generated_path}")

    if args.prepare_only:
        return 0

    for model_label, generated_path in runnable:
        if output_complete(output_root, model_label):
            print(f"skip complete model: {model_label}")
            continue
        model_root = expected_output(output_root, model_label)
        if model_root.exists():
            raise RuntimeError(
                f"Partial output already exists for {model_label}: {model_root}. "
                "Use a new OUTPUT_ROOT or archive the partial directory first."
            )

        print(f"\n=== training {model_label} ===", flush=True)
        if model_label == "eegnet":
            raw = output_root / "imagine" / "braindecode_eegnet"
            if raw.exists():
                raise RuntimeError(f"Partial raw EEGNet output already exists: {raw}")
            command = [
                args.python,
                "-u",
                str(project_root / "src/eeg_pipeline/models/baseline_model.py"),
                "--config",
                str(generated_path),
                "--architecture",
                "EEGNet",
            ]
        else:
            command = [
                args.python,
                "-u",
                str(project_root / "main.py"),
                "--config",
                str(generated_path),
            ]
        stream_command(
            command,
            cwd=project_root,
            log_path=output_root / "_logs" / f"{model_label}.log",
        )
        if model_label == "eegnet":
            normalize_eegnet_output(output_root)
        if not output_complete(output_root, model_label):
            raise RuntimeError(
                f"Training command finished without all 10 fold outputs: {model_label}"
            )
        shutil.copy2(
            generated_path,
            expected_output(output_root, model_label) / "config.yaml",
        )
        print(f"completed {model_label}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
