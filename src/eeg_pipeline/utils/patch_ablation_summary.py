"""Summarize the deformable Patch-CfC patch-size/stride grid."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path


PATCH_SUMMARY_FIELDS = (
    "patch_size",
    "patch_stride",
    "patch_relation",
    "status",
    "completed_folds",
    "total_parameters",
    "best_val_accuracy_mean",
    "best_val_accuracy_std",
    "test_accuracy_at_best_val_mean",
    "test_accuracy_at_best_val_std",
    "best_epochs",
    "result_directory",
    "reason",
)


def _read_best_validation_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"training log has no epoch rows: {path}")
    marked = [row for row in rows if row.get("is_best") == "1"]
    if marked:
        return marked[-1]
    return max(rows, key=lambda row: float(row["val_accuracy"]))


def _read_test_accuracy(path: Path) -> float:
    with path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    if not isinstance(metrics, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return float(metrics["accuracy"])


def _mean_std(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.6f}", f"{std:.6f}"


def _patch_relation(patch_size: int, patch_stride: int) -> str:
    if patch_stride < patch_size:
        return "overlap"
    if patch_stride == patch_size:
        return "non_overlapping"
    return "gapped"


def _read_total_parameters(path: Path) -> str:
    if not path.is_file():
        return ""
    pattern = re.compile(r"parameters: total=([0-9,]+)")
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            match = pattern.search(line)
            if match is not None:
                return str(int(match.group(1).replace(",", "")))
    return ""


def _summarize_combination(
    output_root: Path,
    patch_size: int,
    patch_stride: int,
) -> dict[str, str | int]:
    combination_name = f"k{patch_size:03d}_s{patch_stride:03d}"
    combination_dir = output_root / combination_name
    fold_dirs = sorted(
        path
        for path in combination_dir.rglob("fold_*")
        if path.is_dir()
    ) if combination_dir.is_dir() else []

    val_accuracies: list[float] = []
    test_accuracies: list[float] = []
    best_epochs: list[int] = []
    errors: list[str] = []

    for fold_dir in fold_dirs:
        training_log = fold_dir / "training_log.csv"
        test_metrics = fold_dir / "test_metrics.json"
        if not training_log.is_file() or not test_metrics.is_file():
            errors.append(f"{fold_dir.name}: incomplete output")
            continue
        try:
            best_row = _read_best_validation_row(training_log)
            val_accuracy = float(best_row["val_accuracy"])
            test_accuracy = _read_test_accuracy(test_metrics)
            best_epoch = int(best_row["epoch"])
            if not all(math.isfinite(x) for x in (val_accuracy, test_accuracy)):
                raise ValueError("non-finite accuracy")
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            errors.append(f"{fold_dir.name}: {error}")
            continue
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        best_epochs.append(best_epoch)

    if patch_stride > patch_size:
        status = "skipped_invalid"
        errors.append("model requires patch_stride <= patch_size")
    elif val_accuracies and errors:
        status = "partial"
    elif val_accuracies:
        status = "completed"
    elif combination_dir.is_dir():
        status = "failed_or_incomplete"
        errors.append(f"see {combination_dir / 'run.log'}")
    else:
        status = "not_run"

    val_mean, val_std = _mean_std(val_accuracies)
    test_mean, test_std = _mean_std(test_accuracies)
    total_parameters = _read_total_parameters(combination_dir / "run.log")
    return {
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "patch_relation": _patch_relation(patch_size, patch_stride),
        "status": status,
        "completed_folds": len(val_accuracies),
        "total_parameters": total_parameters,
        "best_val_accuracy_mean": val_mean,
        "best_val_accuracy_std": val_std,
        "test_accuracy_at_best_val_mean": test_mean,
        "test_accuracy_at_best_val_std": test_std,
        "best_epochs": ";".join(str(epoch) for epoch in best_epochs),
        "result_directory": str(combination_dir),
        "reason": " | ".join(errors),
    }


def build_patch_ablation_summary(
    output_root: Path,
    output_path: Path,
    patch_sizes: list[int],
    patch_strides: list[int],
) -> list[dict[str, str | int]]:
    """Write one summary row for every patch-size/stride combination."""
    if not patch_sizes or not patch_strides:
        raise ValueError("patch_sizes and patch_strides must not be empty")
    if len(set(patch_sizes)) != len(patch_sizes):
        raise ValueError("patch_sizes contains duplicates")
    if len(set(patch_strides)) != len(patch_strides):
        raise ValueError("patch_strides contains duplicates")
    if any(value <= 0 for value in (*patch_sizes, *patch_strides)):
        raise ValueError("patch sizes and strides must be positive")

    rows = [
        _summarize_combination(output_root, patch_size, patch_stride)
        for patch_size in patch_sizes
        for patch_stride in patch_strides
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=PATCH_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
