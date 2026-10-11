"""Aggregate fold metrics and sample-level OOF predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "loss")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path, *, expected_folds: int = 10) -> dict[str, Path]:
    root = Path(root)
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/fold_*/test_metrics.json")):
        metadata_path = metrics_path.parent / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        fold_rows.append(
            {
                "model": metrics["model"],
                "fold": metrics["fold"],
                "test_subject": metrics["test_subject"],
                "best_epoch": metrics["best_epoch"],
                "best_validation_metric": metrics["best_validation_metric"],
                "test_accuracy": metrics["accuracy"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_loss": metrics["loss"],
                "n_test_trials": metrics["n_test_trials"],
                "parameters_total": metrics["parameters_total"],
                "parameters_trainable": metrics["parameters_trainable"],
                "metrics_file": str(metrics_path),
            }
        )
        predictions_path = metrics_path.parent / "test_predictions.csv"
        if predictions_path.is_file():
            with predictions_path.open(newline="", encoding="utf-8") as handle:
                prediction_rows.extend(csv.DictReader(handle))

    fold_rows.sort(key=lambda row: (row["model"], int(row["fold"])))
    _write_csv(root / "fold_results.csv", fold_rows)
    _write_csv(root / "oof_predictions.csv", prediction_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[str(row["model"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for model, rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "model": model,
            "folds_completed": len(rows),
            "expected_folds": expected_folds,
            "complete_10fold": int(len(rows) == expected_folds),
            "test_subjects": " ".join(str(row["test_subject"]) for row in rows),
            "parameters_total": rows[0]["parameters_total"],
        }
        for source, label in (
            ("test_accuracy", "accuracy"),
            ("test_balanced_accuracy", "balanced_accuracy"),
            ("test_macro_f1", "macro_f1"),
            ("test_loss", "loss"),
        ):
            values = np.asarray([float(row[source]) for row in rows])
            summary[f"{label}_mean"] = float(values.mean())
            summary[f"{label}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary[f"{label}_min"] = float(values.min())
            summary[f"{label}_max"] = float(values.max())
        summary_rows.append(summary)
    _write_csv(root / "cv_summary.csv", summary_rows)
    print(f"summary: folds={len(fold_rows)}, models={len(summary_rows)}, root={root}")
    return {
        "fold_results": root / "fold_results.csv",
        "cv_summary": root / "cv_summary.csv",
        "oof_predictions": root / "oof_predictions.csv",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-folds", type=int, default=10)
    args = parser.parse_args()
    summarize(args.root, expected_folds=args.expected_folds)


if __name__ == "__main__":
    main()
