"""Summarize fixed-seed LR-HORNet repetitions across BCI2020 LOSO folds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PAPER4_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PAPER4_ROOT / "outputs" / "bci2020_lr_hornet_fixed_seed_repeat10"


def _stats(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def summarize(
    root: Path,
    *,
    expected_folds: int = 15,
    expected_repeats: int = 10,
    expected_seed: int = 42,
    include_repeats: tuple[int, ...] | None = None,
) -> Path:
    all_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("repeat_result.json"))
    ]
    all_observed = [
        (int(row["repeat"]), int(row["fold"])) for row in all_records
    ]
    if len(set(all_observed)) != len(all_observed):
        raise RuntimeError("Duplicate repeat/fold result records found")

    selected_repeats = (
        tuple(range(1, expected_repeats + 1))
        if include_repeats is None
        else tuple(dict.fromkeys(include_repeats))
    )
    if not selected_repeats or any(repeat < 1 for repeat in selected_repeats):
        raise ValueError("include_repeats must contain unique positive integers")
    records = [
        row for row in all_records if int(row["repeat"]) in selected_repeats
    ]
    expected = {
        (repeat, fold)
        for repeat in selected_repeats
        for fold in range(1, expected_folds + 1)
    }
    observed = {(int(row["repeat"]), int(row["fold"])) for row in records}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"Incomplete result grid: missing={missing}, extra={extra}")
    if {int(row["seed"]) for row in records} != {expected_seed}:
        raise RuntimeError("Results do not all use the requested fixed seed")
    parameter_counts = {int(row["parameters_total"]) for row in records}
    if len(parameter_counts) != 1:
        raise RuntimeError(f"Inconsistent parameter counts: {parameter_counts}")

    summary_root = root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    all_runs_path = summary_root / "all_runs.csv"
    fields = list(records[0])
    with all_runs_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (row["fold"], row["repeat"])))

    by_fold = []
    for fold in range(1, expected_folds + 1):
        selected = [row for row in records if int(row["fold"]) == fold]
        row = {
            "fold": fold,
            "test_subject": selected[0]["test_subject"],
            "validation_subject": selected[0]["validation_subject"],
            "seed": expected_seed,
            "repeats_complete": len(selected),
            "parameters_total": next(iter(parameter_counts)),
        }
        row.update(_stats([float(item["validation_accuracy"]) for item in selected], "validation_accuracy"))
        row.update(_stats([float(item["test_accuracy"]) for item in selected], "test_accuracy"))
        row.update(_stats([float(item["test_balanced_accuracy"]) for item in selected], "test_balanced_accuracy"))
        row.update(_stats([float(item["test_macro_f1"]) for item in selected], "test_macro_f1"))
        by_fold.append(row)
    fold_path = summary_root / "summary_by_fold.csv"
    with fold_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(by_fold[0]))
        writer.writeheader()
        writer.writerows(by_fold)

    overall = {
        "model": "lr_hornet",
        "dataset": "bci2020",
        "protocol": "calibrated_loso",
        "fixed_seed": expected_seed,
        "folds": expected_folds,
        "included_repeats": list(selected_repeats),
        "repeats_per_fold": len(selected_repeats),
        "runs_complete": len(records),
        "parameters_total": next(iter(parameter_counts)),
    }
    for key in (
        "validation_accuracy",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
    ):
        overall.update(_stats([float(row[key]) for row in records], key))
        fold_means = [float(row[f"{key}_mean"]) for row in by_fold]
        overall[f"{key}_mean_across_fold_means"] = statistics.fmean(fold_means)
        overall[f"{key}_std_across_fold_means"] = (
            statistics.stdev(fold_means) if len(fold_means) > 1 else 0.0
        )
    overall_path = summary_root / "overall_summary.json"
    overall_path.write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = [
        "# BCI2020 LR-HORNet fixed-seed repeats",
        "",
        f"- Protocol: calibrated LOSO ({expected_folds} folds)",
        f"- Fixed seed: {expected_seed}",
        f"- Included repeats: {', '.join(map(str, selected_repeats))}",
        f"- Repeats per fold: {len(selected_repeats)}",
        f"- Completed fits: {len(records)}",
        f"- Parameters: {next(iter(parameter_counts)):,}",
        f"- Test accuracy over all fits: {overall['test_accuracy_mean']:.6f} ± {overall['test_accuracy_std']:.6f}",
        f"- Mean of the {expected_folds} fold means: {overall['test_accuracy_mean_across_fold_means']:.6f} ± {overall['test_accuracy_std_across_fold_means']:.6f}",
        "",
        "All repetitions use the same split seed and training seed. Any variation therefore reflects non-deterministic runtime behavior rather than seed variation.",
    ]
    (summary_root / "summary_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {all_runs_path}")
    print(f"WROTE {fold_path}")
    print(f"WROTE {overall_path}")
    print(f"COMPLETE runs={len(records)} seed={expected_seed}")
    return summary_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--expected-folds", type=int, default=15)
    parser.add_argument("--expected-repeats", type=int, default=10)
    parser.add_argument(
        "--include-repeats",
        nargs="+",
        type=int,
        help=(
            "Explicit repeat identifiers to summarize. Records from other "
            "repeats are ignored; every selected repeat must contain all folds."
        ),
    )
    parser.add_argument("--expected-seed", type=int, default=42)
    args = parser.parse_args()
    summarize(
        args.root,
        expected_folds=args.expected_folds,
        expected_repeats=args.expected_repeats,
        expected_seed=args.expected_seed,
        include_repeats=(
            tuple(args.include_repeats) if args.include_repeats else None
        ),
    )


if __name__ == "__main__":
    main()
