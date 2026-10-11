#!/usr/bin/env python3
"""Summarize stored Section 4.7.1 metrics and paired fold contrasts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


MODELS = (
    "group4_patchLstm_deformableLstm",
    "group4_patchLstm_pointwiseDeformation",
    "group4_patchLstm_staticDeformation",
    "group4_patchLstm_groupedConvDeformation",
)


def fold_from_path(path: Path) -> int | None:
    folds = []
    for part in path.parts:
        match = re.fullmatch(r"fold_?(\d+)(?:_seed_\d+)?", part)
        if match:
            folds.append(int(match.group(1)))
    return folds[-1] if folds else None


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path) -> list[dict]:
    rows = []
    seen: dict[tuple[str, int], tuple[dict, Path]] = {}
    for path in sorted(root.rglob("test_metrics.json")):
        model = next((part for part in path.parts if part in MODELS), None)
        fold = fold_from_path(path)
        if model is None or fold is None:
            continue
        key = (model, fold)
        metric = json.loads(path.read_text(encoding="utf-8"))
        if key in seen:
            previous_metric, previous_path = seen[key]
            if metric != previous_metric:
                raise RuntimeError(
                    f"Conflicting duplicate result for {model} fold {fold}: "
                    f"{previous_path} and {path}"
                )
            print(
                f"warning: identical duplicate skipped for {model} fold {fold}: "
                f"{path}"
            )
            continue
        seen[key] = (metric, path)
        rows.append({
            "model": model,
            "fold": fold,
            "loss": metric.get("loss", ""),
            "accuracy": metric["accuracy"],
            "balanced_accuracy": metric["balanced_accuracy"],
            "macro_f1": metric["macro_f1"],
            "metrics_file": str(path),
        })
    rows.sort(key=lambda row: (MODELS.index(row["model"]), row["fold"]))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        item = {"model": model, "folds": len(selected), "complete_10_folds": int(len(selected) == 10)}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = [float(row[metric]) for row in selected]
            item[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            item[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        result.append(item)
    return result


def contrasts(rows: list[dict]) -> list[dict]:
    indexed = {(row["model"], row["fold"]): row for row in rows}
    left = MODELS[0]
    result = []
    for right in MODELS[1:]:
        pairs = [
            (indexed[(left, fold)], indexed[(right, fold)])
            for fold in range(1, 11)
            if (left, fold) in indexed and (right, fold) in indexed
        ]
        item = {"left_model": left, "right_model": right, "paired_folds": len(pairs)}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = [float(a[metric]) - float(b[metric]) for a, b in pairs]
            item[f"{metric}_mean_delta"] = statistics.fmean(values) if values else ""
            item[f"{metric}_wins"] = sum(value > 0 for value in values)
            item[f"{metric}_ties"] = sum(value == 0 for value in values)
            item[f"{metric}_losses"] = sum(value < 0 for value in values)
        result.append(item)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows = collect(args.result_root.resolve())
    summary = summarize(rows)
    paired = contrasts(rows)
    if args.require_complete:
        incomplete = [item["model"] for item in summary if not item["complete_10_folds"]]
        if incomplete:
            raise RuntimeError(f"Incomplete variants: {incomplete}")
    output = args.output_dir.resolve()
    write(output / "results_by_fold.csv", rows)
    write(output / "results_summary.csv", summary)
    write(output / "paired_contrasts.csv", paired)
    for item in summary:
        print(f"{item['model']}: n={item['folds']} accuracy={item['accuracy_mean']:.6f} macro_f1={item['macro_f1_mean']:.6f}")
    print(f"summary_dir={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
