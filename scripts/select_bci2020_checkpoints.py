#!/usr/bin/env python3
"""Select and release one BCI2020 LR-HORNet checkpoint per fold.

The archived experiment contains nine complete fixed-seed repeats for each of
15 folds.  This utility implements the paper-specific, explicitly post-hoc
rule requested for the released artifacts: maximize test accuracy within each
fold over repeats 1--9 and, when several repeats tie, choose the lowest repeat
number.  Repeat 10 is excluded in full because it is incomplete.

The command copies the 15 selected state dictionaries into
``checkpoints/main/bci2020`` and writes a portable provenance table.  It never
modifies the external result tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INCLUDED_REPEATS = tuple(range(1, 10))
EXPECTED_FOLDS = tuple(range(1, 16))


@dataclass(frozen=True)
class Candidate:
    repeat: int
    fold: int
    selected_epoch: int
    validation_accuracy: float
    test_loss: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_macro_f1: float
    parameters_total: int
    directory: Path
    metrics_path: Path
    training_log_path: Path
    checkpoint_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help=(
            "Path to bci2020_lr_hornet_fixed_seed_repeat10 containing "
            "repeat_01 ... repeat_10 and summary/all_runs.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "checkpoints" / "main" / "bci2020",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "4.4.1" / "bci2020",
    )
    return parser.parse_args()


def _read_candidates(results_root: Path) -> list[Candidate]:
    source = results_root / "summary" / "all_runs.csv"
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    candidates: list[Candidate] = []
    for row in rows:
        if row.get("status") != "complete":
            continue
        repeat = int(row["repeat"])
        fold = int(row["fold"])
        if repeat not in INCLUDED_REPEATS:
            continue
        fold_root = (
            results_root
            / f"repeat_{repeat:02d}"
            / "bci2020"
            / "imagined_speech"
            / "group4_patchLstm_deformableLstm"
            / f"fold_{fold:02d}"
        )
        metrics_path = fold_root / "test_metrics.json"
        training_log_path = fold_root / "training_log.csv"
        checkpoint_path = fold_root / "best_model.pt"
        for path in (metrics_path, training_log_path, checkpoint_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed-run artifact: {path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        reported_accuracy = float(row["test_accuracy"])
        if not math.isclose(
            float(metrics["accuracy"]), reported_accuracy, abs_tol=1e-15
        ):
            raise ValueError(
                f"Metric mismatch for repeat {repeat}, fold {fold}: "
                f"{metrics['accuracy']} != {reported_accuracy}"
            )
        candidates.append(
            Candidate(
                repeat=repeat,
                fold=fold,
                selected_epoch=int(row["best_epoch"]),
                validation_accuracy=float(row["validation_accuracy"]),
                test_loss=float(row["test_loss"]),
                test_accuracy=reported_accuracy,
                test_balanced_accuracy=float(row["test_balanced_accuracy"]),
                test_macro_f1=float(row["test_macro_f1"]),
                parameters_total=int(row["parameters_total"]),
                directory=fold_root,
                metrics_path=metrics_path,
                training_log_path=training_log_path,
                checkpoint_path=checkpoint_path,
            )
        )

    expected = len(INCLUDED_REPEATS) * len(EXPECTED_FOLDS)
    if len(candidates) != expected:
        raise ValueError(
            f"Expected {expected} complete candidates from repeats 1-9, "
            f"found {len(candidates)}"
        )
    observed = {(item.repeat, item.fold) for item in candidates}
    required = {
        (repeat, fold)
        for repeat in INCLUDED_REPEATS
        for fold in EXPECTED_FOLDS
    }
    if observed != required:
        raise ValueError("Complete-repeat coverage is not exactly 9 x 15")
    return candidates


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_outputs(
    candidates: list[Candidate],
    *,
    results_root: Path,
    checkpoint_root: Path,
    output_root: Path,
) -> None:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    by_fold = {
        fold: sorted(
            (item for item in candidates if item.fold == fold),
            key=lambda item: (-item.test_accuracy, item.repeat),
        )
        for fold in EXPECTED_FOLDS
    }
    selected = [by_fold[fold][0] for fold in EXPECTED_FOLDS]
    rows: list[dict[str, object]] = []
    for item in selected:
        tied = [
            candidate.repeat
            for candidate in by_fold[item.fold]
            if math.isclose(
                candidate.test_accuracy,
                item.test_accuracy,
                abs_tol=1e-15,
            )
        ]
        destination = checkpoint_root / f"fold_{item.fold:02d}.pt"
        shutil.copy2(item.checkpoint_path, destination)
        rows.append(
            {
                "fold": item.fold,
                "selected_repeat": item.repeat,
                "tied_best_repeats": ";".join(str(value) for value in tied),
                "selected_epoch": item.selected_epoch,
                "validation_accuracy": item.validation_accuracy,
                "test_loss": item.test_loss,
                "test_accuracy": item.test_accuracy,
                "test_balanced_accuracy": item.test_balanced_accuracy,
                "test_macro_f1": item.test_macro_f1,
                "parameters_total": item.parameters_total,
                "source_checkpoint_path": _relative(
                    item.checkpoint_path, results_root
                ),
                "source_checkpoint_sha256": _sha256(item.checkpoint_path),
                "source_metrics_path": _relative(item.metrics_path, results_root),
                "source_metrics_sha256": _sha256(item.metrics_path),
                "source_training_log_path": _relative(
                    item.training_log_path, results_root
                ),
                "source_training_log_sha256": _sha256(item.training_log_path),
                "released_checkpoint_path": destination.relative_to(ROOT).as_posix(),
                "released_checkpoint_bytes": destination.stat().st_size,
                "released_checkpoint_sha256": _sha256(destination),
            }
        )

    selection_path = output_root / "selected_checkpoints.csv"
    with selection_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    accuracies = [item.test_accuracy for item in selected]
    validations = [item.validation_accuracy for item in selected]
    balanced = [item.test_balanced_accuracy for item in selected]
    macro_f1 = [item.test_macro_f1 for item in selected]
    summary = {
        "model": "lr_hornet",
        "dataset": "bci2020",
        "protocol": "calibrated_loso",
        "fixed_seed": 42,
        "selection_policy": "maximum_test_accuracy_within_each_fold",
        "tie_break": "lowest_repeat_number",
        "included_repeats": list(INCLUDED_REPEATS),
        "excluded_repeats": [10],
        "folds": len(EXPECTED_FOLDS),
        "candidate_runs": len(candidates),
        "selected_checkpoints": len(selected),
        "parameters_total": selected[0].parameters_total,
        "validation_accuracy_mean": statistics.fmean(validations),
        "validation_accuracy_std": statistics.stdev(validations),
        "test_accuracy_mean": statistics.fmean(accuracies),
        "test_accuracy_std": statistics.stdev(accuracies),
        "test_accuracy_min": min(accuracies),
        "test_accuracy_max": max(accuracies),
        "test_balanced_accuracy_mean": statistics.fmean(balanced),
        "test_balanced_accuracy_std": statistics.stdev(balanced),
        "test_macro_f1_mean": statistics.fmean(macro_f1),
        "test_macro_f1_std": statistics.stdev(macro_f1),
    }
    (output_root / "overall_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    all_accuracies = [item.test_accuracy for item in candidates]
    repeat_summary = {
        "included_repeats": list(INCLUDED_REPEATS),
        "runs_complete": len(candidates),
        "test_accuracy_mean_all_runs": statistics.fmean(all_accuracies),
        "test_accuracy_std_all_runs": statistics.stdev(all_accuracies),
        "note": "Descriptive only; not the released per-fold selection summary.",
    }
    (output_root / "all_repeats_summary.json").write_text(
        json.dumps(repeat_summary, indent=2) + "\n", encoding="utf-8"
    )

    baseline_path = output_root / "model_summary.csv"
    with baseline_path.open(newline="", encoding="utf-8-sig") as stream:
        baseline_rows = [
            row for row in csv.DictReader(stream) if row["model"] != "lr_hornet"
        ]
    model_rows = baseline_rows + [
        {
            "model": "lr_hornet",
            "folds_complete": len(selected),
            "test_accuracy_mean": summary["test_accuracy_mean"],
            "test_accuracy_std": summary["test_accuracy_std"],
            "validation_accuracy_mean": summary["validation_accuracy_mean"],
            "parameters_total": summary["parameters_total"],
        }
    ]
    model_rows.sort(key=lambda row: float(row["test_accuracy_mean"]), reverse=True)
    with baseline_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)

    report = (
        "# BCI2020 released LR-HORNet checkpoints\n\n"
        "- Protocol: calibrated LOSO (15 folds)\n"
        "- Candidate pool: complete fixed-seed repeats 1-9 (135 runs)\n"
        "- Repeat 10: excluded in full because it is incomplete\n"
        "- Selection: maximum test accuracy within each fold; lowest repeat "
        "number breaks exact ties\n"
        f"- Released checkpoints: {len(selected)}\n"
        f"- Test accuracy: {summary['test_accuracy_mean']:.6f} ± "
        f"{summary['test_accuracy_std']:.6f}\n\n"
        "This is a post-hoc test-set selection rule and therefore is not an "
        "unbiased estimate of generalization. The 135-run descriptive mean is "
        "retained separately in `all_repeats_summary.json`.\n"
    )
    (output_root / "summary_report.md").write_text(report, encoding="utf-8")

    print(
        f"Selected {len(selected)} checkpoints from {len(candidates)} "
        f"candidates; test accuracy={summary['test_accuracy_mean']:.6f} "
        f"± {summary['test_accuracy_std']:.6f}"
    )


def main() -> int:
    args = _parse_args()
    results_root = args.results_root.expanduser().resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"BCI2020 result root does not exist: {results_root}")
    candidates = _read_candidates(results_root)
    _write_outputs(
        candidates,
        results_root=results_root,
        checkpoint_root=args.checkpoint_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
