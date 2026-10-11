#!/usr/bin/env python3
"""Summarize stored Section 4.6.3 fixed-seed repetitions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


MODELS = (
    (1, "group1_patchLstm_deformableLstm"),
    (2, "group2_patchLstm_deformableLstm"),
    (4, "group4_patchLstm_deformableLstm"),
    (8, "group8_patchLstm_deformableLstm"),
    (16, "group16_patchLstm_deformableLstm"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = args.result_root.resolve()
    rows = []
    incomplete = []
    for groups, model in MODELS:
        paths = sorted((root / model).rglob("test_metrics.json"))
        if len(paths) != 5:
            incomplete.append(f"{model}: {len(paths)}/5")
        metrics = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        accuracies = [float(x["accuracy"]) for x in metrics]
        row = {
            "electrode_groups": groups,
            "model": model,
            "fold": 1,
            "requested_repeats": 5,
            "completed_repeats": len(metrics),
            "status": "complete" if len(metrics) == 5 else "incomplete",
            "accuracy_mean": statistics.fmean(accuracies) if accuracies else "",
            "accuracy_sd": statistics.stdev(accuracies) if len(accuracies) > 1 else (0.0 if accuracies else ""),
            "balanced_accuracy_mean": statistics.fmean(float(x["balanced_accuracy"]) for x in metrics) if metrics else "",
            "macro_f1_mean": statistics.fmean(float(x["macro_f1"]) for x in metrics) if metrics else "",
        }
        rows.append(row)
        if metrics:
            print(f"groups={groups:02d}: completed={len(metrics)}/5 accuracy={float(row['accuracy_mean']):.4f} +/- {float(row['accuracy_sd']):.4f}")
        else:
            print(f"groups={groups:02d}: completed=0/5")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.require_complete and incomplete:
        raise RuntimeError("Incomplete results: " + ", ".join(incomplete))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
