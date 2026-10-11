"""Summarize fixed-seed repeated LOSO runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "best_val_accuracy",
    "test_accuracy_at_best_val",
    "test_balanced_accuracy_at_best_val",
    "test_macro_f1_at_best_val",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--expected-folds",
        type=int,
        default=10,
        help="Number of LOSO folds expected for every summarized model.",
    )
    return parser.parse_args()


def _atomic_csv(
    path: Path, fields: Iterable[str], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _mean_std(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    return (
        f"{statistics.fmean(values):.6f}",
        f"{statistics.stdev(values) if len(values) > 1 else 0.0:.6f}",
    )


def _float(row: dict[str, Any], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field}={value!r}")
    return value


def _collect(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("*/fold_*/repeat_runs.csv")):
        fold_root = path.parent
        manifest_path = fold_root / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)
        with path.open(newline="", encoding="utf-8") as file:
            raw_rows = list(csv.DictReader(file))
            repeat_count = manifest.get("repeat_count", len(raw_rows))
            for raw in raw_rows:
                row: dict[str, Any] = dict(raw)
                row["model"] = manifest.get(
                    "model", fold_root.parent.name
                )
                row["yaml_path"] = manifest.get("source_config", "")
                row["fixed_seed"] = manifest.get(
                    "fixed_seed", raw.get("seed", "")
                )
                row["repeat_count"] = repeat_count
                if raw.get("status") == "completed":
                    metrics_path = (
                        Path(raw["output_dir"]) / "test_metrics.json"
                    )
                    try:
                        with metrics_path.open(encoding="utf-8") as metrics_file:
                            metrics = json.load(metrics_file)
                        accuracy = float(raw["test_accuracy_at_best_val"])
                        row["test_balanced_accuracy_at_best_val"] = float(
                            metrics.get("balanced_accuracy", accuracy)
                        )
                        row["test_macro_f1_at_best_val"] = float(
                            metrics.get("macro_f1", accuracy)
                        )
                    except (OSError, TypeError, ValueError, json.JSONDecodeError):
                        row["test_balanced_accuracy_at_best_val"] = ""
                        row["test_macro_f1_at_best_val"] = ""
                else:
                    row["test_balanced_accuracy_at_best_val"] = ""
                    row["test_macro_f1_at_best_val"] = ""
                rows.append(row)
    return rows


def _selected(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        return None
    return min(
        completed,
        key=lambda row: (
            -_float(row, "best_val_accuracy"),
            _float(row, "val_loss_at_best"),
            int(row["repeat"]),
        ),
    )


def _fold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), int(row["fold_number"]))].append(row)

    result: list[dict[str, Any]] = []
    for (model, fold), group in sorted(grouped.items()):
        completed = [row for row in group if row.get("status") == "completed"]
        requested_runs = max(
            (int(row.get("repeat_count") or 0) for row in group),
            default=len(group),
        )
        seeds = sorted({str(row.get("fixed_seed", "")) for row in group})
        item: dict[str, Any] = {
            "model": model,
            "fold_number": fold,
            "fixed_seed": seeds[0] if len(seeds) == 1 else "MIXED",
            "yaml_path": group[0].get("yaml_path", ""),
            "total_parameters": group[0].get("total_parameters", ""),
            "trainable_parameters": group[0].get(
                "trainable_parameters", ""
            ),
            "requested_runs": requested_runs,
            "completed_runs": len(completed),
            "failed_runs": sum(
                row.get("status") == "failed" for row in group
            ),
        }
        item["pending_runs"] = max(
            requested_runs - item["completed_runs"] - item["failed_runs"],
            0,
        )
        for metric in METRICS:
            values = [
                _float(row, metric)
                for row in completed
                if row.get(metric) not in (None, "")
            ]
            mean, std = _mean_std(values)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_sample_std"] = std
            item[f"{metric}_min"] = (
                f"{min(values):.6f}" if values else ""
            )
            item[f"{metric}_max"] = (
                f"{max(values):.6f}" if values else ""
            )
        selected = _selected(group)
        item["selected_repeat_by_validation"] = (
            selected["repeat"] if selected else ""
        )
        item["selected_best_val_accuracy"] = (
            selected["best_val_accuracy"] if selected else ""
        )
        item["selected_test_accuracy_at_best_val"] = (
            selected["test_accuracy_at_best_val"] if selected else ""
        )
        if completed:
            diagnostic = max(
                completed,
                key=lambda row: _float(
                    row, "test_accuracy_at_best_val"
                ),
            )
            item["diagnostic_max_test_accuracy"] = diagnostic[
                "test_accuracy_at_best_val"
            ]
            item["diagnostic_max_test_repeat"] = diagnostic["repeat"]
        else:
            item["diagnostic_max_test_accuracy"] = ""
            item["diagnostic_max_test_repeat"] = ""
        result.append(item)
    return result


def _model_rows(
    rows: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    *,
    expected_folds: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    fold_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in folds:
        fold_groups[str(row["model"])].append(row)

    result: list[dict[str, Any]] = []
    for model, group in sorted(grouped.items()):
        completed = [row for row in group if row.get("status") == "completed"]
        model_folds = fold_groups[model]
        seeds = sorted({str(row.get("fixed_seed", "")) for row in group})
        repeats_per_fold = max(
            (int(row.get("repeat_count") or 0) for row in group),
            default=10,
        )
        requested_runs = expected_folds * repeats_per_fold
        item: dict[str, Any] = {
            "model": model,
            "fixed_seed": seeds[0] if len(seeds) == 1 else "MIXED",
            "yaml_path": group[0].get("yaml_path", ""),
            "total_parameters": group[0].get("total_parameters", ""),
            "trainable_parameters": group[0].get(
                "trainable_parameters", ""
            ),
            "expected_folds": expected_folds,
            "discovered_folds": len(model_folds),
            "completed_folds": sum(
                int(row["completed_runs"]) == int(row["requested_runs"])
                for row in model_folds
            ),
            "requested_runs": requested_runs,
            "completed_runs": len(completed),
            "failed_runs": sum(
                row.get("status") == "failed" for row in group
            ),
        }
        item["pending_runs"] = max(
            requested_runs - item["completed_runs"] - item["failed_runs"],
            0,
        )
        for metric in METRICS:
            run_values = [
                _float(row, metric)
                for row in completed
                if row.get(metric) not in (None, "")
            ]
            run_mean, run_std = _mean_std(run_values)
            fold_values = [
                float(row[f"{metric}_mean"])
                for row in model_folds
                if row.get(f"{metric}_mean") not in (None, "")
            ]
            fold_mean, fold_std = _mean_std(fold_values)
            item[f"{metric}_all_runs_mean"] = run_mean
            item[f"{metric}_all_runs_sample_std"] = run_std
            item[f"{metric}_fold_mean"] = fold_mean
            item[f"{metric}_fold_sample_std"] = fold_std

        selected_tests = [
            float(row["selected_test_accuracy_at_best_val"])
            for row in model_folds
            if row.get("selected_test_accuracy_at_best_val")
            not in (None, "")
        ]
        selected_mean, selected_std = _mean_std(selected_tests)
        item["validation_selected_test_accuracy_fold_mean"] = selected_mean
        item[
            "validation_selected_test_accuracy_fold_sample_std"
        ] = selected_std
        test_values = [
            _float(row, "test_accuracy_at_best_val") for row in completed
        ]
        item["diagnostic_max_test_accuracy"] = (
            f"{max(test_values):.6f}" if test_values else ""
        )
        result.append(item)
    return result


def _pair(row: dict[str, Any], mean: str, std: str) -> str:
    if row.get(mean) in (None, ""):
        return "N/A"
    return (
        f"{100 * float(row[mean]):.2f}% ± "
        f"{100 * float(row[std]):.2f}%"
    )


def _markdown(
    path: Path,
    models: list[dict[str, Any]],
    folds: list[dict[str, Any]],
) -> None:
    lines = [
        "# Group4 Deformable Fixed-Seed LOSO Summary",
        "",
        "Every repetition within a fold uses its recorded fixed seed. "
        "Variation therefore measures runtime non-determinism, not "
        "independent-seed uncertainty. Test maxima are diagnostic only.",
        "",
        "## Model summary",
        "",
        "| Model | Parameters | Completed | Best val | Test ACC | "
        "Balanced ACC | Macro-F1 | Validation-selected test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in models:
        lines.append(
            "| {model} | {params} | {done}/{requested} | {val} | "
            "{test} | {balanced} | {f1} | {selected} |".format(
                model=row["model"],
                params=row["total_parameters"],
                done=row["completed_runs"],
                requested=row["requested_runs"],
                val=_pair(
                    row,
                    "best_val_accuracy_all_runs_mean",
                    "best_val_accuracy_all_runs_sample_std",
                ),
                test=_pair(
                    row,
                    "test_accuracy_at_best_val_all_runs_mean",
                    "test_accuracy_at_best_val_all_runs_sample_std",
                ),
                balanced=_pair(
                    row,
                    "test_balanced_accuracy_at_best_val_all_runs_mean",
                    "test_balanced_accuracy_at_best_val_all_runs_sample_std",
                ),
                f1=_pair(
                    row,
                    "test_macro_f1_at_best_val_all_runs_mean",
                    "test_macro_f1_at_best_val_all_runs_sample_std",
                ),
                selected=_pair(
                    row,
                    "validation_selected_test_accuracy_fold_mean",
                    "validation_selected_test_accuracy_fold_sample_std",
                ),
            )
        )
    lines.extend([
        "",
        "## Fold summary",
        "",
        "| Model | Fold | Completed | Best val | Test ACC | "
        "Validation-selected test | Diagnostic max test |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in folds:
        selected = row.get("selected_test_accuracy_at_best_val")
        maximum = row.get("diagnostic_max_test_accuracy")
        lines.append(
            "| {model} | {fold} | {done}/{requested} | {val} | "
            "{test} | {selected} | {maximum} |".format(
                model=row["model"],
                fold=row["fold_number"],
                done=row["completed_runs"],
                requested=row["requested_runs"],
                val=_pair(
                    row,
                    "best_val_accuracy_mean",
                    "best_val_accuracy_sample_std",
                ),
                test=_pair(
                    row,
                    "test_accuracy_at_best_val_mean",
                    "test_accuracy_at_best_val_sample_std",
                ),
                selected=(
                    f"{100 * float(selected):.2f}%"
                    if selected not in (None, "") else "N/A"
                ),
                maximum=(
                    f"{100 * float(maximum):.2f}%"
                    if maximum not in (None, "") else "N/A"
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _args()
    if args.expected_folds < 1:
        raise ValueError("--expected-folds must be a positive integer")
    run_root = args.run_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_root / "summary"
    )
    rows = _collect(run_root)
    if not rows:
        raise FileNotFoundError(
            f"No repeat_runs.csv files found under {run_root}"
        )
    folds = _fold_rows(rows)
    models = _model_rows(
        rows,
        folds,
        expected_folds=args.expected_folds,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "all_runs.csv", rows[0].keys(), rows)
    _atomic_csv(
        output_dir / "summary_by_fold.csv", folds[0].keys(), folds
    )
    _atomic_csv(
        output_dir / "summary_by_model.csv", models[0].keys(), models
    )
    _markdown(output_dir / "summary_report.md", models, folds)
    print(
        f"summary: runs={len(rows)}, folds={len(folds)}, "
        f"models={len(models)}, expected_folds={args.expected_folds}"
    )
    print(f"summary_dir={output_dir}")


if __name__ == "__main__":
    main()
