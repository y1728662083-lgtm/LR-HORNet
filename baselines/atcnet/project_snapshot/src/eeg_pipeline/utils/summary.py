"""Summarize Braindecode baselines or deformable-patch ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path


BASE_SUMMARY_FIELDS = (
    "architecture",
    "model_key",
    "paper_year",
    "status",
    "total_parameters",
    "trainable_parameters",
    "expected_folds",
    "completed_folds",
)
AGGREGATE_SUMMARY_FIELDS = (
    "best_val_accuracy_mean",
    "best_val_accuracy_std",
    "test_accuracy_at_best_val_mean",
    "test_accuracy_at_best_val_std",
    "reason",
)
PARAMETER_LINE = re.compile(
    r"parameters:\s*total=([\d,]+),\s*trainable=([\d,]+)"
)


def _fold_fields(fold: int) -> tuple[str, str, str]:
    prefix = f"fold_{fold:02d}"
    return (
        f"{prefix}_best_epoch",
        f"{prefix}_best_val_accuracy",
        f"{prefix}_test_accuracy_at_best_val",
    )


def _summary_fields(expected_folds: list[int]) -> tuple[str, ...]:
    per_fold = tuple(
        field
        for fold in expected_folds
        for field in _fold_fields(fold)
    )
    return BASE_SUMMARY_FIELDS + per_fold + AGGREGATE_SUMMARY_FIELDS


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _best_validation_row(path: Path) -> dict[str, str]:
    """Return the row whose checkpoint was selected by validation."""
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Training log has no epoch rows: {path}")

    marked = [row for row in rows if row.get("is_best") == "1"]
    if marked:
        # Each improvement is marked; the last improvement is the global best
        # checkpoint for the configured validation monitor.
        return marked[-1]
    return max(rows, key=lambda row: float(row["val_accuracy"]))


def _mean_std(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.6f}", f"{std:.6f}"


def _status_payload(root: Path, model_key: str) -> dict:
    path = root / "_status" / f"{model_key}.json"
    if not path.is_file():
        return {}
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"status": "invalid_status", "reason": str(error)}


def _log_parameter_counts(
    root: Path,
    architecture: str,
) -> tuple[tuple[int, int] | None, str]:
    """Recover exact counts emitted by the real training model."""
    log_path = root / "_logs" / f"{architecture}.log"
    if not log_path.is_file():
        return None, ""

    counts: set[tuple[int, int]] = set()
    try:
        with log_path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = PARAMETER_LINE.search(line)
                if match is None:
                    continue
                counts.add(
                    tuple(
                        int(value.replace(",", ""))
                        for value in match.groups()
                    )
                )
    except OSError as error:
        return None, f"could not read parameter log {log_path}: {error}"

    if not counts:
        return None, ""
    if len(counts) != 1:
        return None, (
            f"conflicting parameter counts in {log_path}: {sorted(counts)}"
        )
    return next(iter(counts)), ""


def _started_timestamp(status_payload: dict) -> float | None:
    started_at = status_payload.get("started_at_utc")
    if not isinstance(started_at, str):
        return None
    try:
        return datetime.fromisoformat(started_at).timestamp()
    except ValueError:
        return None


def _empty_row(
    spec: dict,
    expected_folds: list[int],
    *,
    status: str,
    reason: str,
) -> dict[str, str | int]:
    row: dict[str, str | int] = {
        field: "" for field in _summary_fields(expected_folds)
    }
    row.update(
        architecture=str(spec["architecture"]),
        model_key=str(spec["model_key"]),
        paper_year=spec.get("paper_year") or "",
        status=status,
        expected_folds=len(expected_folds),
        completed_folds=0,
        reason=reason,
    )
    return row


def _summarize_model(
    root: Path,
    spec: dict,
    *,
    task: str,
    expected_folds: list[int],
) -> dict[str, str | int]:
    architecture = str(spec["architecture"])
    model_key = str(spec["model_key"])
    if not bool(spec.get("runnable", False)):
        return _empty_row(
            spec,
            expected_folds,
            status="skipped_incompatible",
            reason=str(spec.get("reason", "")),
        )

    status_payload = _status_payload(root, model_key)
    has_status_payload = bool(status_payload)

    row = _empty_row(
        spec,
        expected_folds,
        status=str(status_payload.get("status", "not_run")),
        reason="",
    )
    problems: list[str] = []
    row["total_parameters"] = status_payload.get("total_parameters", "")
    row["trainable_parameters"] = status_payload.get(
        "trainable_parameters", ""
    )
    log_counts, log_problem = _log_parameter_counts(root, architecture)
    if log_problem:
        problems.append(log_problem)
    if log_counts is not None:
        status_counts = (
            row["total_parameters"],
            row["trainable_parameters"],
        )
        if all(value not in ("", None) for value in status_counts):
            normalized_status_counts = tuple(int(value) for value in status_counts)
            if normalized_status_counts != log_counts:
                problems.append(
                    "status/log parameter mismatch: "
                    f"status={normalized_status_counts}, log={log_counts}"
                )
        else:
            row["total_parameters"], row["trainable_parameters"] = log_counts

    status_task = status_payload.get("task")
    if has_status_payload and status_task != task:
        problems.append(
            f"status task mismatch: expected {task!r}, got {status_task!r}"
        )
    status_folds = status_payload.get("expected_folds")
    if status_folds is not None and list(status_folds) != expected_folds:
        problems.append(
            "status expected_folds mismatch: "
            f"expected {expected_folds}, got {status_folds}"
        )

    started_timestamp = _started_timestamp(status_payload)
    val_accuracies: list[float] = []
    test_accuracies: list[float] = []
    completed_fold_ids: list[int] = []

    for fold in expected_folds:
        fold_dir = root / task / model_key / f"fold_{fold:02d}"
        training_log = fold_dir / "training_log.csv"
        test_metrics = fold_dir / "test_metrics.json"
        if not training_log.is_file() or not test_metrics.is_file():
            problems.append(f"fold_{fold:02d}: incomplete output")
            continue
        if started_timestamp is not None and (
            training_log.stat().st_mtime < started_timestamp
            or test_metrics.stat().st_mtime < started_timestamp
        ):
            problems.append(
                f"fold_{fold:02d}: output predates the current run"
            )
            continue

        try:
            best_row = _best_validation_row(training_log)
            metrics = _read_json(test_metrics)
            val_accuracy = float(best_row["val_accuracy"])
            test_accuracy = float(metrics["accuracy"])
            best_epoch = int(best_row["epoch"])
            if not all(
                math.isfinite(value)
                for value in (val_accuracy, test_accuracy)
            ):
                raise ValueError("non-finite accuracy")
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
        ) as error:
            problems.append(f"fold_{fold:02d}: {error}")
            continue

        epoch_field, val_field, test_field = _fold_fields(fold)
        row[epoch_field] = best_epoch
        row[val_field] = f"{val_accuracy:.6f}"
        row[test_field] = f"{test_accuracy:.6f}"
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        completed_fold_ids.append(fold)

    row["completed_folds"] = len(completed_fold_ids)
    val_mean, val_std = _mean_std(val_accuracies)
    test_mean, test_std = _mean_std(test_accuracies)
    row["best_val_accuracy_mean"] = val_mean
    row["best_val_accuracy_std"] = val_std
    row["test_accuracy_at_best_val_mean"] = test_mean
    row["test_accuracy_at_best_val_std"] = test_std

    run_status = str(status_payload.get("status", "not_run"))
    all_folds_complete = completed_fold_ids == expected_folds
    if has_status_payload:
        if run_status == "completed" and all_folds_complete and not problems:
            row["status"] = "completed"
        elif run_status == "failed" and completed_fold_ids:
            row["status"] = "partial_failed"
        elif run_status == "failed":
            row["status"] = "failed"
        elif completed_fold_ids:
            row["status"] = "partial"
        else:
            row["status"] = run_status
    elif all_folds_complete and not problems:
        # Older launchers wrote the fold artifacts but no per-model status
        # JSON. Recover those completed runs directly from their outputs.
        row["status"] = "completed"
    elif completed_fold_ids:
        row["status"] = "partial"
    else:
        row["status"] = "not_run"

    status_reason = str(status_payload.get("reason", ""))
    if status_reason:
        problems.append(status_reason)
    if not has_status_payload and problems:
        problems.append("No per-model status file; recovered fold outputs")
    row["reason"] = " | ".join(problems)
    return row


def _catalog_expected_folds(catalog: dict) -> list[int]:
    raw = catalog.get("expected_folds")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Catalog expected_folds must be a non-empty list")
    folds: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Invalid catalog fold id: {value!r}")
        folds.append(value)
    if len(folds) != len(set(folds)):
        raise ValueError(f"Catalog expected_folds contains duplicates: {folds}")
    return folds


def build_summary(output_root: Path, output_path: Path) -> list[dict]:
    """Build a wide CSV with every fold and cross-fold aggregates."""
    catalog_path = output_root / "_status" / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"Baseline catalog does not exist: {catalog_path}. "
            "Run run_braindecode_baselines.sh first."
        )
    catalog = _read_json(catalog_path)
    specs = catalog.get("models")
    if not isinstance(specs, list):
        raise ValueError(f"Invalid model catalog: {catalog_path}")
    task = catalog.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError(f"Catalog task is missing: {catalog_path}")
    expected_folds = _catalog_expected_folds(catalog)

    rows = [
        _summarize_model(
            output_root,
            spec,
            task=task,
            expected_folds=expected_folds,
        )
        for spec in specs
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=_summary_fields(expected_folds),
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("braindecode", "patch-ablation"),
        default="braindecode",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--patch-sizes", type=int, nargs="+", default=(8, 16, 32, 64, 128)
    )
    parser.add_argument(
        "--patch-strides", type=int, nargs="+", default=(4, 8, 16, 32)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser()

    if args.mode == "patch-ablation":
        from patch_ablation_summary import build_patch_ablation_summary

        output_path = args.output or (
            output_root / "deformable_patchCfC_patch_ablation_summary.csv"
        )
        rows = build_patch_ablation_summary(
            output_root,
            output_path,
            args.patch_sizes,
            args.patch_strides,
        )
        completed = sum(row["status"] == "completed" for row in rows)
        skipped = sum(
            row["status"] == "skipped_invalid" for row in rows
        )
        failed = len(rows) - completed - skipped
        print(
            f"patch-ablation summary: completed={completed}, "
            f"skipped_invalid={skipped}, failed_or_pending={failed}"
        )
        print(f"summary saved to: {output_path}")
        return

    output_path = args.output or output_root / "braindecode_baseline_summary.csv"
    rows = build_summary(output_root, output_path)
    completed = sum(row["status"] == "completed" for row in rows)
    skipped = sum(row["status"] == "skipped_incompatible" for row in rows)
    failed = len(rows) - completed - skipped
    print(
        f"baseline summary: completed={completed}, skipped={skipped}, "
        f"failed_or_pending={failed}"
    )
    print(f"summary saved to: {output_path}")


if __name__ == "__main__":
    main()
