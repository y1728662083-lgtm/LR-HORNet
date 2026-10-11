#!/usr/bin/env python3
"""Summarize stored Section 4.5 test_metrics.json files without loading weights."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


MODEL_SOURCES = (
    ("eegnet", "eegnet"),
    ("causal1_eegnet", "causal1_eegnet"),
    ("group4_causal1_deformableLstm", "group4_causal1_deformableLstm"),
    ("group4_patchLstm_noDeformation", "group4_patchLstm_noDeformation"),
    ("group4_patchLstm_deformableLstm", "lr_hornet_main"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.result_root.resolve()
    rows = []
    for model, source_directory in MODEL_SOURCES:
        model_root = root / source_directory
        if not model_root.is_dir():
            model_root = root / "imagine" / source_directory
        if not model_root.is_dir():
            raise FileNotFoundError(model_root)
        metric_paths = sorted(model_root.rglob("test_metrics.json"))
        if len(metric_paths) != 10:
            raise ValueError(
                f"{model}: expected 10 test_metrics.json files, "
                f"found {len(metric_paths)}"
            )
        metrics = []
        for path in metric_paths:
            with path.open(encoding="utf-8") as file:
                payload = json.load(file)
            metrics.append(payload)
        row = {
            "model": model,
            "folds": len(metrics),
            "accuracy_mean": statistics.fmean(x["accuracy"] for x in metrics),
            "accuracy_sd": statistics.stdev(x["accuracy"] for x in metrics),
            "balanced_accuracy_mean": statistics.fmean(
                x["balanced_accuracy"] for x in metrics
            ),
            "macro_f1_mean": statistics.fmean(x["macro_f1"] for x in metrics),
        }
        rows.append(row)
        print(
            f"{model}: accuracy={row['accuracy_mean']:.4f} "
            f"+/- {row['accuracy_sd']:.4f}, macro_f1={row['macro_f1_mean']:.4f}"
        )
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
