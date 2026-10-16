"""Collect BCI2020 fold metrics from all six model output layouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PAPER4_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PAPER4_ROOT / "outputs" / "bci2020_all_models"
MODELS = ("lr_hornet", "eegnet", "atcnet", "eegmamba", "fast", "labram")


def _model_for_path(path: Path) -> str | None:
    lowered = tuple(part.casefold() for part in path.parts)
    if "group4_patchlstm_deformablelstm" in lowered:
        return "lr_hornet"
    if "braindecode_eegnet" in lowered:
        return "eegnet"
    if "braindecode_atcnet" in lowered:
        return "atcnet"
    for model in ("eegmamba", "fast", "labram"):
        if model in lowered:
            return model
    return None


def _best_val(path: Path, payload: dict) -> float | None:
    if "best_validation_metric" in payload:
        return float(payload["best_validation_metric"])
    csv_path = path.parent / "training_log.csv"
    if not csv_path.is_file():
        return None
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    best = [row for row in rows if str(row.get("is_best", "")).casefold() in {"1", "true"}]
    row = best[-1] if best else (rows[-1] if rows else None)
    if row is None:
        return None
    for key in ("val_accuracy", "validation_accuracy"):
        if key in row and row[key] != "":
            return float(row[key])
    return None


def summarize(
    root: Path,
    *,
    expected_folds: int,
    allow_incomplete: bool,
    models: tuple[str, ...] = MODELS,
) -> tuple[Path, Path]:
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for path in sorted(root.rglob("test_metrics.json")):
        model = _model_for_path(path)
        if model is None or model not in models:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold = int(payload.get("fold") or path.parent.name.removeprefix("fold_"))
        key = (model, fold)
        if key in seen:
            raise RuntimeError(f"Duplicate result for {key}: {path}")
        seen.add(key)
        rows.append(
            {
                "model": model,
                "fold": fold,
                "test_subject": payload.get("test_subject", ""),
                "best_epoch": payload.get("best_epoch", ""),
                "validation_accuracy": _best_val(path, payload),
                "test_accuracy": payload.get("accuracy"),
                "test_balanced_accuracy": payload.get("balanced_accuracy", ""),
                "test_macro_f1": payload.get("macro_f1", ""),
                "parameters_total": payload.get("parameters_total", ""),
                "parameters_trainable": payload.get("parameters_trainable", ""),
                "metrics_path": str(path),
            }
        )
    counts = {model: sum(row["model"] == model for row in rows) for model in models}
    if not allow_incomplete:
        bad = {model: count for model, count in counts.items() if count != expected_folds}
        if bad:
            raise RuntimeError(f"Incomplete results: {bad}; expected {expected_folds} each")

    summary_root = root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    fold_path = summary_root / "fold_results.csv"
    fields = list(rows[0]) if rows else ["model", "fold"]
    with fold_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (models.index(row["model"]), row["fold"])))

    aggregate: list[dict] = []
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        accuracies = [float(row["test_accuracy"]) for row in selected if row["test_accuracy"] not in (None, "")]
        validation = [float(row["validation_accuracy"]) for row in selected if row["validation_accuracy"] not in (None, "")]
        parameter_values = {int(row["parameters_total"]) for row in selected if row["parameters_total"] not in (None, "")}
        aggregate.append(
            {
                "model": model,
                "folds_complete": len(selected),
                "test_accuracy_mean": statistics.fmean(accuracies) if accuracies else math.nan,
                "test_accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "validation_accuracy_mean": statistics.fmean(validation) if validation else math.nan,
                "parameters_total": next(iter(parameter_values)) if len(parameter_values) == 1 else "inconsistent_or_missing",
            }
        )
    model_path = summary_root / "model_summary.csv"
    with model_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    print(f"WROTE {fold_path}")
    print(f"WROTE {model_path}")
    print(f"COUNTS {counts}")
    return fold_path, model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--expected-folds", type=int, default=15)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    summarize(
        args.root,
        expected_folds=args.expected_folds,
        allow_incomplete=args.allow_incomplete,
        models=tuple(args.models),
    )


if __name__ == "__main__":
    main()
