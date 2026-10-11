"""Run one YAML-selected fold with predeclared seeds and summarize repeats."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_MODEL = "deformable_patchCfC_filterNFC_dualStateLSTM"
RUN_FIELDS = (
    "repeat",
    "seed",
    "fold_index",
    "fold_number",
    "status",
    "exit_code",
    "elapsed_seconds",
    "total_parameters",
    "trainable_parameters",
    "best_epoch",
    "best_val_accuracy",
    "val_loss_at_best",
    "test_accuracy_at_best_val",
    "test_loss_at_best_val",
    "selected_by_validation",
    "output_dir",
    "config_path",
    "log_path",
    "error",
)


def _resolve_from_project(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def _validate_config(config: dict[str, Any]) -> tuple[int, int]:
    model_name = config.get("model", {}).get("name")
    if model_name != EXPECTED_MODEL:
        raise ValueError(
            f"Expected model.name={EXPECTED_MODEL!r}, got {model_name!r}"
        )

    fold_indices = config.get("split", {}).get("fold_indices")
    if not isinstance(fold_indices, list) or len(fold_indices) != 1:
        raise ValueError(
            "split.fold_indices must contain exactly one fold for repeat runs"
        )
    fold_index = fold_indices[0]
    if isinstance(fold_index, bool) or not isinstance(fold_index, int):
        raise ValueError("The selected fold index must be an integer")

    monitor = config.get("training", {}).get("monitor")
    if monitor != "accuracy":
        raise ValueError(
            "training.monitor must be 'accuracy' so model selection is validation-only"
        )

    label_smoothing = float(
        config.get("training", {}).get("label_smoothing", 0.0)
    )
    if not math.isclose(label_smoothing, 0.05, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "This repeat script requires training.label_smoothing: 0.05; "
            f"got {label_smoothing}"
        )

    base_seed = config.get("seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("seed must be an integer")
    return fold_index, base_seed


def _count_parameters(config: dict[str, Any]) -> tuple[int, int]:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from eeg_pipeline.models.factory import build_model

    task_classes = {"imagine": 4, "speak": 4, "visual": 4, "pattern": 3}
    task = config["data"]["task"]
    if task not in task_classes:
        raise ValueError(f"Unknown task: {task!r}")
    model = build_model(
        name=config["model"]["name"],
        n_classes=task_classes[task],
        **config["model"].get("kwargs", {}),
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def _last_best_validation_row(training_log: Path) -> dict[str, str]:
    with training_log.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    best_rows = [
        row for row in rows if int(float(row.get("is_best", "0") or 0)) == 1
    ]
    if not best_rows:
        raise ValueError(f"No is_best=1 row found in {training_log}")
    return best_rows[-1]


def _read_completed_metrics(output_dir: Path) -> dict[str, float | int]:
    best_row = _last_best_validation_row(output_dir / "training_log.csv")
    with (output_dir / "test_metrics.json").open(encoding="utf-8") as file:
        test_metrics = json.load(file)

    values: dict[str, float | int] = {
        "best_epoch": int(best_row["epoch"]),
        "best_val_accuracy": float(best_row["val_accuracy"]),
        "val_loss_at_best": float(best_row["val_loss"]),
        "test_accuracy_at_best_val": float(test_metrics["accuracy"]),
        "test_loss_at_best_val": float(test_metrics["loss"]),
    }
    for key, value in values.items():
        if key != "best_epoch" and not math.isfinite(float(value)):
            raise ValueError(f"Non-finite metric {key}={value!r} in {output_dir}")
    return values


def _select_by_validation(records: list[dict[str, Any]]) -> None:
    for record in records:
        record["selected_by_validation"] = False
    completed = [record for record in records if record["status"] == "completed"]
    if not completed:
        return
    selected = min(
        completed,
        key=lambda record: (
            -float(record["best_val_accuracy"]),
            float(record["val_loss_at_best"]),
            int(record["repeat"]),
        ),
    )
    selected["selected_by_validation"] = True


def _atomic_write_csv(
    path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _mean_and_sample_std(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sample_std


def _write_reports(run_root: Path, records: list[dict[str, Any]]) -> None:
    _select_by_validation(records)
    _atomic_write_csv(run_root / "repeat_runs.csv", RUN_FIELDS, records)

    completed = [record for record in records if record["status"] == "completed"]
    val_values = [float(record["best_val_accuracy"]) for record in completed]
    test_values = [
        float(record["test_accuracy_at_best_val"]) for record in completed
    ]
    val_mean, val_std = _mean_and_sample_std(val_values)
    test_mean, test_std = _mean_and_sample_std(test_values)
    selected = next(
        (record for record in records if record["selected_by_validation"]), None
    )
    summary = {
        "model": EXPECTED_MODEL,
        "fold_number": records[0]["fold_number"] if records else "",
        "requested_runs": len(records),
        "completed_runs": len(completed),
        "failed_runs": len(records) - len(completed),
        "best_val_accuracy_mean": val_mean,
        "best_val_accuracy_sample_std": val_std,
        "test_accuracy_at_best_val_mean": test_mean,
        "test_accuracy_at_best_val_sample_std": test_std,
        "selected_repeat_by_validation": selected["repeat"] if selected else "",
        "selected_seed_by_validation": selected["seed"] if selected else "",
        "selected_best_val_accuracy": (
            selected["best_val_accuracy"] if selected else ""
        ),
        "selected_test_accuracy_at_best_val": (
            selected["test_accuracy_at_best_val"] if selected else ""
        ),
        "selection_rule": (
            "highest best_val_accuracy; ties: lowest val_loss_at_best, then "
            "lowest repeat; test metrics never used for selection"
        ),
    }
    _atomic_write_csv(run_root / "repeat_summary.csv", summary.keys(), [summary])


def _stream_process(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Predeclared seeds; defaults to YAML seed through seed+4.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional repeat-set output root.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = _resolve_from_project(args.config)
    base_config = _load_config(config_path)
    fold_index, base_seed = _validate_config(base_config)
    seeds = args.seeds if args.seeds is not None else list(range(base_seed, base_seed + 5))
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five distinct, predeclared seeds are required")
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("Every seed must be in [0, 2**32)")

    if args.run_root is None:
        base_output_root = Path(base_config["training"]["output_root"])
        if not base_output_root.is_absolute():
            base_output_root = PROJECT_ROOT / base_output_root
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = base_output_root / "five_seed_repeats" / run_tag
    else:
        run_root = _resolve_from_project(args.run_root)
    run_root.mkdir(parents=True, exist_ok=False)

    source_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    total_parameters, trainable_parameters = _count_parameters(base_config)
    manifest = {
        "source_config": str(config_path),
        "source_config_sha256": source_hash,
        "model": base_config["model"]["name"],
        "fold_index": fold_index,
        "fold_number": fold_index + 1,
        "seeds": seeds,
        "label_smoothing": base_config["training"]["label_smoothing"],
        "selection_rule": "validation only; test metrics are not used for selection",
        "python_executable": sys.executable,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }
    with (run_root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"repeat_set={run_root}")
    print(f"fold={fold_index + 1:02d}, seeds={seeds}")
    print(
        f"parameters: total={total_parameters:,}, trainable={trainable_parameters:,}"
    )

    records: list[dict[str, Any]] = []
    for repeat, seed in enumerate(seeds, start=1):
        repeat_root = run_root / f"repeat_{repeat:02d}_seed_{seed}"
        resolved_config = copy.deepcopy(base_config)
        resolved_config["seed"] = seed
        resolved_config["training"]["output_root"] = str(repeat_root)
        generated_config = run_root / "configs" / f"repeat_{repeat:02d}_seed_{seed}.yaml"
        generated_config.parent.mkdir(parents=True, exist_ok=True)
        with generated_config.open("w", encoding="utf-8", newline="\n") as file:
            yaml.safe_dump(resolved_config, file, sort_keys=False, allow_unicode=True)

        output_dir = (
            repeat_root
            / resolved_config["data"]["task"]
            / resolved_config["model"]["name"]
            / f"fold_{fold_index + 1:02d}"
        )
        log_path = run_root / "logs" / f"repeat_{repeat:02d}_seed_{seed}.log"
        record: dict[str, Any] = {
            "repeat": repeat,
            "seed": seed,
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

        print(f"\n[repeat {repeat}/5] seed={seed} fold={fold_index + 1:02d}")
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
    print(f"\ncompleted={len(completed)}/5")
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
    return 0 if len(completed) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
