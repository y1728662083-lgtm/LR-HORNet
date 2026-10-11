#!/usr/bin/env python3
"""Summarize Section 4.7.2 descriptive and mechanism results."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-diagnostics", type=Path, required=True)
    parser.add_argument("--mechanism-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    visual = read(args.visual_diagnostics.resolve())
    causal_rows, shift_rows, status = [], [], []
    rank_dirs = sorted(path for path in args.mechanism_root.resolve().iterdir() if path.is_dir())
    for rank, path in enumerate(rank_dirs, 1):
        causal_path = path / "causal_ablation_summary.csv"
        shift_path = path / "artificial_shift_summary.csv"
        status.append({"rank": rank, "result_dir": path.name, "causal_complete": int(causal_path.is_file()), "artificial_shift_complete": int(shift_path.is_file()), "included_in_statistics": int(causal_path.is_file() and shift_path.is_file())})
        for row in read(causal_path) if causal_path.is_file() else []:
            causal_rows.append({"rank": rank, "result_dir": path.name, **row})
        for row in read(shift_path) if shift_path.is_file() else []:
            shift_rows.append({"rank": rank, "result_dir": path.name, **row})
    aggregate = []
    for mode in ("learned", "zero", "shuffled", "random_matched"):
        selected = [row for row in causal_rows if row.get("mode") == mode]
        aggregate.append({
            "mode": mode,
            "checkpoints": len(selected),
            "accuracy_mean": statistics.fmean(float(row["accuracy"]) for row in selected),
            "balanced_accuracy_mean": statistics.fmean(float(row["balanced_accuracy"]) for row in selected),
            "macro_f1_mean": statistics.fmean(float(row["macro_f1"]) for row in selected),
            "mean_abs_applied_offset_samples": statistics.fmean(float(row["mean_abs_applied_offset_samples"]) for row in selected),
        })
    output = args.output_dir.resolve()
    write(output / "group04_offset_trend_diagnostics.csv", visual)
    write(output / "group06_causal_by_rank.csv", causal_rows)
    write(output / "group06_artificial_shift_by_rank.csv", shift_rows)
    write(output / "group06_causal_aggregate_rank01_04.csv", aggregate)
    write(output / "completion_status.csv", sorted(status, key=lambda row: row["rank"]))
    print(f"summary_dir={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
