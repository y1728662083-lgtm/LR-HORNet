#!/usr/bin/env python3
"""Summarize stored Section 4.6.1 test metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


MODELS = (
    "braindecode_eegnet",
    "causal1_eegnet",
    "group4_patchLstm_noDeformation",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    if (root / "imagine").is_dir():
        root = root / "imagine"
    rows = []
    for model in MODELS:
        metrics = []
        for fold in range(1, 11):
            path = root / model / f"fold_{fold:02d}" / "test_metrics.json"
            metrics.append(json.loads(path.read_text(encoding="utf-8")))
        row = {
            "model": model,
            "folds": 10,
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
            f"+/- {row['accuracy_sd']:.4f}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
