"""Repeat one YAML-selected fold five times with one fixed seed."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from eeg_pipeline.utils.repeat_single_fold import (
    EXPECTED_MODEL,
    _count_parameters,
    _load_config,
    _read_completed_metrics,
    _resolve_from_project,
    _stream_process,
    _validate_config,
    _write_reports,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPEATS = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional output root for this five-run repeat set.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = _resolve_from_project(args.config)
    base_config = _load_config(config_path)
    fold_index, fixed_seed = _validate_config(base_config)
    if fixed_seed != 42:
        raise ValueError(
            f"This fixed-repeat experiment requires YAML seed: 42; got {fixed_seed}"
        )

    if args.run_root is None:
        base_output_root = Path(base_config["training"]["output_root"])
        if not base_output_root.is_absolute():
            base_output_root = PROJECT_ROOT / base_output_root
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = base_output_root / "five_fixed_seed_repeats" / run_tag
    else:
        run_root = _resolve_from_project(args.run_root)
    run_root.mkdir(parents=True, exist_ok=False)

    total_parameters, trainable_parameters = _count_parameters(base_config)
    manifest = {
        "experiment_type": "same-fold fixed-seed repeated training",
        "source_config": str(config_path),
        "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model": base_config["model"]["name"],
        "fold_index": fold_index,
        "fold_number": fold_index + 1,
        "repeat_count": REPEATS,
        "fixed_seed": fixed_seed,
        "label_smoothing": base_config["training"]["label_smoothing"],
        "selection_rule": "validation only; test metrics are not used for selection",
        "python_executable": sys.executable,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    with (run_root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"repeat_set={run_root}")
    print(f"fold={fold_index + 1:02d}, fixed_seed={fixed_seed}, repeats={REPEATS}")
    print(
        f"parameters: total={total_parameters:,}, trainable={trainable_parameters:,}"
    )

    records: list[dict[str, Any]] = []
    for repeat in range(1, REPEATS + 1):
        repeat_root = run_root / f"repeat_{repeat:02d}_seed_{fixed_seed}"
        resolved_config = copy.deepcopy(base_config)
        resolved_config["seed"] = fixed_seed
        resolved_config["training"]["output_root"] = str(repeat_root)
        generated_config = (
            run_root / "configs" / f"repeat_{repeat:02d}_seed_{fixed_seed}.yaml"
        )
        generated_config.parent.mkdir(parents=True, exist_ok=True)
        with generated_config.open("w", encoding="utf-8", newline="\n") as file:
            yaml.safe_dump(resolved_config, file, sort_keys=False, allow_unicode=True)

        output_dir = (
            repeat_root
            / resolved_config["data"]["task"]
            / resolved_config["model"]["name"]
            / f"fold_{fold_index + 1:02d}"
        )
        log_path = run_root / "logs" / f"repeat_{repeat:02d}_seed_{fixed_seed}.log"
        record: dict[str, Any] = {
            "repeat": repeat,
            "seed": fixed_seed,
            "fold_index": fold_index,
            "fold_number": fold_index + 1,
            "status": "running",
            "exit_code": "",
            "elapsed_seconds": "",
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "best_epoch": "",
            "best_val_accuracy": "",
            "val_loss_at_best": "",
            "test_accuracy_at_best_val": "",
            "test_loss_at_best_val": "",
            "selected_by_validation": False,
            "output_dir": str(output_dir),
            "config_path": str(generated_config),
            "log_path": str(log_path),
            "error": "",
        }
        records.append(record)
        _write_reports(run_root, records)

        print(
            f"\n[fixed-seed repeat {repeat}/{REPEATS}] "
            f"seed={fixed_seed} fold={fold_index + 1:02d}"
        )
        started = time.monotonic()
        exit_code = _stream_process(
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "main.py"),
                "--config",
                str(generated_config),
            ],
            log_path,
        )
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        record["exit_code"] = exit_code
        if exit_code != 0:
            record["status"] = "failed"
            record["error"] = f"main.py exited with status {exit_code}"
        else:
            try:
                record.update(_read_completed_metrics(output_dir))
                record["status"] = "completed"
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                record["status"] = "failed"
                record["error"] = str(error)
        _write_reports(run_root, records)

    completed = [record for record in records if record["status"] == "completed"]
    selected = next(
        (record for record in records if record["selected_by_validation"]), None
    )
    print(f"\ncompleted={len(completed)}/{REPEATS}")
    print(f"runs_csv={run_root / 'repeat_runs.csv'}")
    print(f"summary_csv={run_root / 'repeat_summary.csv'}")
    if selected is not None:
        print(
            "selected_by_validation: "
            f"repeat={selected['repeat']}, seed={selected['seed']}, "
            f"best_val_accuracy={float(selected['best_val_accuracy']):.4f}, "
            "corresponding_test_accuracy="
            f"{float(selected['test_accuracy_at_best_val']):.4f}"
        )
    return 0 if len(completed) == REPEATS else 1


if __name__ == "__main__":
    raise SystemExit(main())
