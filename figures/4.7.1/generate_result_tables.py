"""Regenerate Section 4.7.1 tables from archived experiment outputs.

The script reads four offset-generation models across LOSO folds 1--10,
validates the raw test metrics and training logs, and writes auditable
run-level and model-level CSV files. No result is entered manually.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS_ROOT = (
    REPOSITORY_ROOT / "experiments" / "4.7.1" / "reference_results"
)
EXPECTED_FOLDS = list(range(1, 11))


@dataclass(frozen=True)
class ModelSpec:
    source_directory: str
    model_key: str
    display_label: str
    short_label: str
    model_order: int


MODEL_SPECS = (
    ModelSpec(
        "group4_patchLstm_deformableLstm",
        "group4_patchLstm_deformableLstm",
        "LSTM",
        "LSTM",
        1,
    ),
    ModelSpec(
        "group4_patchLstm_pointwiseDeformation",
        "group4_patchLstm_pointwiseDeformation",
        "Pointwise",
        "Pointwise",
        2,
    ),
    ModelSpec(
        "group4_patchLstm_staticDeformation",
        "group4_patchLstm_staticDeformation",
        "Static",
        "Static",
        3,
    ),
    ModelSpec(
        "group4_patchLstm_groupedConvDeformation",
        "group4_patchLstm_groupedConvDeformation",
        "Grouped Conv",
        "Grouped\nConv",
        4,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_fold(metric_path: Path) -> int:
    match = re.fullmatch(r"fold[_-]?(\d+)", metric_path.parent.name)
    if match is None:
        raise ValueError(f"Cannot parse fold from {metric_path}")
    return int(match.group(1))


def relative_source_path(path: Path, results_root: Path) -> str:
    return path.resolve().relative_to(results_root.resolve()).as_posix()


def load_run_level_results(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for spec in MODEL_SPECS:
        model_root = results_root / spec.source_directory
        if not model_root.is_dir():
            raise FileNotFoundError(f"Missing model directory: {model_root}")
        metric_paths = sorted(model_root.rglob("test_metrics.json"))
        if len(metric_paths) != len(EXPECTED_FOLDS):
            raise ValueError(
                f"{spec.model_key}: expected 10 metric files, "
                f"found {len(metric_paths)}"
            )

        for metric_path in metric_paths:
            fold = parse_fold(metric_path)
            with metric_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)

            required_keys = {
                "loss",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "confusion_matrix",
            }
            missing = required_keys.difference(metrics)
            if missing:
                raise ValueError(f"Missing {sorted(missing)} in {metric_path}")

            confusion = np.asarray(metrics["confusion_matrix"], dtype=int)
            if (
                confusion.ndim != 2
                or confusion.shape[0] != confusion.shape[1]
                or np.any(confusion < 0)
            ):
                raise ValueError(f"Invalid confusion matrix in {metric_path}")
            sample_count = int(confusion.sum())
            if sample_count <= 0:
                raise ValueError(f"Empty confusion matrix in {metric_path}")

            accuracy = float(metrics["accuracy"])
            recomputed_accuracy = float(np.trace(confusion) / sample_count)
            if not np.isclose(accuracy, recomputed_accuracy, atol=1e-12):
                raise ValueError(
                    f"Accuracy does not match confusion matrix in {metric_path}"
                )

            training_log_path = metric_path.with_name("training_log.csv")
            if not training_log_path.is_file():
                raise FileNotFoundError(
                    f"Missing training log for {metric_path}"
                )
            training_log = pd.read_csv(training_log_path)
            if "val_accuracy" not in training_log.columns:
                raise ValueError(
                    f"val_accuracy is absent from {training_log_path}"
                )
            validation_values = pd.to_numeric(
                training_log["val_accuracy"], errors="coerce"
            ).dropna()
            if validation_values.empty:
                raise ValueError(
                    f"No numeric val_accuracy values in {training_log_path}"
                )

            rows.append(
                {
                    "model_order": spec.model_order,
                    "model_key": spec.model_key,
                    "model_label": spec.display_label,
                    "short_label": spec.short_label.replace("\n", " "),
                    "fold": fold,
                    "test_loss": float(metrics["loss"]),
                    "accuracy": accuracy,
                    "best_validation_accuracy": float(validation_values.max()),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "test_sample_count": sample_count,
                    "metrics_path": relative_source_path(metric_path, results_root),
                    "training_log_path": relative_source_path(
                        training_log_path, results_root
                    ),
                    "metrics_sha256": sha256_file(metric_path),
                    "training_log_sha256": sha256_file(training_log_path),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["model_order", "fold"]
    ).reset_index(drop=True)


def validate_design(run_df: pd.DataFrame) -> None:
    expected_rows = len(MODEL_SPECS) * len(EXPECTED_FOLDS)
    if len(run_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(run_df)}")
    if run_df.duplicated(["model_key", "fold"]).any():
        raise ValueError("Duplicate model-fold rows were found")

    for spec in MODEL_SPECS:
        observed = sorted(
            run_df.loc[run_df["model_key"] == spec.model_key, "fold"].tolist()
        )
        if observed != EXPECTED_FOLDS:
            raise ValueError(
                f"{spec.model_key}: expected folds {EXPECTED_FOLDS}, "
                f"found {observed}"
            )

    for column in (
        "accuracy",
        "best_validation_accuracy",
        "balanced_accuracy",
        "macro_f1",
    ):
        if not run_df[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} contains a value outside [0, 1]")


def summarize_models(run_df: pd.DataFrame) -> pd.DataFrame:
    return (
        run_df.groupby(
            ["model_order", "model_key", "model_label", "short_label"],
            as_index=False,
            sort=False,
        )
        .agg(
            n_folds=("accuracy", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            accuracy_sem=("accuracy", "sem"),
            accuracy_min=("accuracy", "min"),
            accuracy_max=("accuracy", "max"),
            validation_accuracy_mean=("best_validation_accuracy", "mean"),
            validation_accuracy_sd=("best_validation_accuracy", "std"),
            validation_accuracy_sem=("best_validation_accuracy", "sem"),
            validation_accuracy_min=("best_validation_accuracy", "min"),
            validation_accuracy_max=("best_validation_accuracy", "max"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            test_sample_count_min=("test_sample_count", "min"),
            test_sample_count_max=("test_sample_count", "max"),
        )
        .sort_values("model_order")
        .reset_index(drop=True)
    )


def export_csv(dataframe: pd.DataFrame, output_path: Path) -> bool:
    try:
        dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        print(
            f"Warning: {output_path.name} is open elsewhere; "
            "the existing file was retained."
        )
        return False
    return True


def generate_result_tables(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    output_dir: Path = SCRIPT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path], dict[str, bool]]:
    results_root = results_root.resolve()
    output_dir = output_dir.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {results_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_df = load_run_level_results(results_root)
    validate_design(run_df)
    summary_df = summarize_models(run_df)

    paths = {
        "run_level": output_dir / "run_level_results.csv",
        "model_summary": output_dir / "model_summary.csv",
    }
    status = {
        "run_level": export_csv(run_df, paths["run_level"]),
        "model_summary": export_csv(summary_df, paths["model_summary"]),
    }
    return run_df, summary_df, paths, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Section 4.7.1 CSV tables from raw results."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_df, summary_df, paths, status = generate_result_tables(
        args.results_root, args.output_dir
    )
    print(
        f"Validated {len(run_df)} results: "
        f"{len(summary_df)} models x {len(EXPECTED_FOLDS)} folds."
    )
    print(summary_df.to_string(index=False, float_format="%.6f"))
    print("\nGenerated tables:")
    for name, path in paths.items():
        suffix = "" if status[name] else " (existing file retained)"
        print(f"  {path}{suffix}")


if __name__ == "__main__":
    main()
