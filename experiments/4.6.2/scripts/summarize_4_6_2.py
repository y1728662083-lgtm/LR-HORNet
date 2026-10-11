#!/usr/bin/env python3
"""Summarize stored Section 4.6.2 test metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


VARIANTS = (
    "full_sequence_no_patch", "patch_0008", "patch_0016", "patch_0032",
    "patch_0064", "patch_0128", "patch_0256", "patch_0512",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.result_root.resolve()
    rows = []
    for variant in VARIANTS:
        paths = sorted((root / variant).rglob("test_metrics.json"))
        if not paths:
            raise FileNotFoundError(f"No metrics found for {variant} under {root}")
        if not args.allow_partial and len(paths) != 5:
            raise ValueError(f"Expected 5 repeats for {variant}, found {len(paths)}")
        metrics = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        accuracies = [float(x["accuracy"]) for x in metrics]
        row = {
            "variant": variant,
            "repeats": len(metrics),
            "accuracy_mean": statistics.fmean(accuracies),
            "accuracy_sd": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "balanced_accuracy_mean": statistics.fmean(float(x["balanced_accuracy"]) for x in metrics),
            "macro_f1_mean": statistics.fmean(float(x["macro_f1"]) for x in metrics),
        }
        rows.append(row)
        print(f"{variant}: accuracy={row['accuracy_mean']:.4f} +/- {row['accuracy_sd']:.4f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
