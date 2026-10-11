#!/usr/bin/env python3
"""Summarize formal Attention and Mamba results with their distinct protocols."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


ATTENTION_MODELS = {
    "group4_patchLstm_deformableLstm_satt",
    "group4_patchLstm_deformableLstm_statt",
}
MAMBA_MODELS = {
    "group4_fullMamba_deformableLstm",
    "group4_patchLstm_deformableMamba",
}
MODELS = ATTENTION_MODELS | MAMBA_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def infer(path: Path) -> tuple[str, int, int] | None:
    model = next((name for name in MODELS if name in path.parts), None)
    folds = [int(value) for value in re.findall(r"fold_(\d+)", str(path))]
    repeats = [int(value) for value in re.findall(r"repeat_(\d+)", str(path))]
    if model is None or not folds:
        return None
    return model, folds[-1], repeats[-1] if repeats else 1


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary(rows: list[dict]) -> list[dict]:
    result = []
    for model in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        result.append({
            "model": model,
            "n": len(selected),
            "accuracy_mean": statistics.fmean(float(row["accuracy"]) for row in selected),
            "accuracy_sd": statistics.stdev(float(row["accuracy"]) for row in selected) if len(selected) > 1 else 0.0,
            "balanced_accuracy_mean": statistics.fmean(float(row["balanced_accuracy"]) for row in selected),
            "macro_f1_mean": statistics.fmean(float(row["macro_f1"]) for row in selected),
        })
    return result


def main() -> int:
    args = parse_args()
    root = args.result_root.resolve()
    rows = []
    seen = set()
    for path in sorted(root.rglob("test_metrics.json")):
        parsed = infer(path)
        if parsed is None:
            continue
        model, fold, repeat = parsed
        key = (model, fold, repeat)
        if key in seen:
            raise ValueError(f"Duplicate result for {key}: {path}")
        seen.add(key)
        metric = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "model": model,
            "fold": fold,
            "repeat": repeat,
            "seed": 42,
            "loss": metric["loss"],
            "accuracy": metric["accuracy"],
            "balanced_accuracy": metric["balanced_accuracy"],
            "macro_f1": metric["macro_f1"],
            "metrics_path": str(path),
        })
    if not rows:
        raise FileNotFoundError(f"No formal Section 4.8.2 metrics found under {root}")
    if args.require_complete:
        missing = []
        for model in ATTENTION_MODELS:
            missing.extend((model, fold, 1) for fold in range(1, 11) if (model, fold, 1) not in seen)
        for model in MAMBA_MODELS:
            missing.extend((model, fold, 1) for fold in range(1, 11) if (model, fold, 1) not in seen)
            missing.extend((model, 1, repeat) for repeat in range(2, 6) if (model, 1, repeat) not in seen)
        if missing:
            raise RuntimeError(f"Missing formal results: {missing}")
    primary = [row for row in rows if row["model"] in ATTENTION_MODELS or row["repeat"] == 1]
    fold1_repeats = [row for row in rows if row["model"] in MAMBA_MODELS and row["fold"] == 1]
    output = args.output_dir.resolve()
    write_csv(output / "results_by_run.csv", rows)
    write_csv(output / "primary_10fold_summary.csv", summary(primary))
    write_csv(output / "mamba_fold1_repeat5_summary.csv", summary(fold1_repeats))
    print(f"runs={len(rows)} primary={output / 'primary_10fold_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
