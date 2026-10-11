"""Regenerate Section 4.5 result tables from the archived experiment outputs.

No metric is entered manually. The script reads every ``test_metrics.json``
and its corresponding ``training_log.csv``, validates the five-model by
ten-fold design, and writes run-level and model-level CSV files. To avoid an
unnecessary duplicate training campaign, the final LR-HORNet stage reuses the
ten-fold ds003626 main-experiment results from Section 4.4.1; their hashes and
metrics are checked against the main result table.

Run from any working directory:

    python generate_result_tables.py
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
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT / "experiments" / "4.5" / "reference_results"
)
MAIN_RESULT_TABLE = (
    PROJECT_ROOT / "results" / "4.4.1" / "fold_level_results_with_val_acc.csv"
)
EXPECTED_FOLDS = list(range(1, 11))


@dataclass(frozen=True)
class ModelSpec:
    source_directory: str
    model_key: str
    display_label: str
    stage_order: int


MODEL_SPECS = (
    ModelSpec("eegnet", "eegnet", "EEGNet", 1),
    ModelSpec("causal1_eegnet", "causal1_eegnet", "Causal EEGNet", 2),
    ModelSpec(
        "group4_causal1_deformableLstm",
        "group4_causal1_deformableLstm",
        "Causal EEGNet + HOR",
        3,
    ),
    ModelSpec(
        "group4_patchLstm_noDeformation",
        "group4_patchLstm_noDeformation",
        "LR (without HOR)",
        4,
    ),
    ModelSpec(
        "lr_hornet_main",
        "group4_patchLstm_deformableLstm",
        "LR-HORNet",
        5,
    ),
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_fold(metric_path: Path) -> int:
    """Read the one-based fold number from the directory containing a metric."""
    match = re.fullmatch(r"fold[_-]?(\d+)", metric_path.parent.name)
    if match is None:
        raise ValueError(f"Cannot parse fold from: {metric_path}")
    return int(match.group(1))


def relative_source_path(path: Path, results_root: Path) -> str:
    """Return a portable path relative to the Section 4.5 result root."""
    return path.resolve().relative_to(results_root.resolve()).as_posix()


def load_run_level_results(results_root: Path) -> pd.DataFrame:
    """Read all five models and ten folds directly from archived result files."""
    rows: list[dict[str, float | int | str]] = []

    for spec in MODEL_SPECS:
        model_root = results_root / spec.source_directory
        if not model_root.is_dir():
            raise FileNotFoundError(f"Missing model result directory: {model_root}")

        metric_paths = sorted(model_root.rglob("test_metrics.json"))
        if len(metric_paths) != len(EXPECTED_FOLDS):
            raise ValueError(
                f"{spec.model_key}: expected 10 test_metrics.json files, "
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
            missing_keys = required_keys.difference(metrics)
            if missing_keys:
                raise ValueError(
                    f"Missing {sorted(missing_keys)} in: {metric_path}"
                )

            confusion_matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
            if (
                confusion_matrix.ndim != 2
                or confusion_matrix.shape[0] != confusion_matrix.shape[1]
                or np.any(confusion_matrix < 0)
            ):
                raise ValueError(f"Invalid confusion matrix in: {metric_path}")
            sample_count = int(confusion_matrix.sum())
            if sample_count <= 0:
                raise ValueError(f"Empty confusion matrix in: {metric_path}")

            accuracy = float(metrics["accuracy"])
            accuracy_from_confusion = float(
                np.trace(confusion_matrix) / sample_count
            )
            if not np.isclose(accuracy, accuracy_from_confusion, atol=1e-12):
                raise ValueError(
                    f"Accuracy does not match confusion matrix in: {metric_path}"
                )

            training_log_path = metric_path.with_name("training_log.csv")
            if not training_log_path.is_file():
                raise FileNotFoundError(
                    f"Missing training log for: {metric_path}"
                )
            training_log = pd.read_csv(training_log_path)
            if "val_accuracy" not in training_log.columns:
                raise ValueError(
                    f"val_accuracy is absent from: {training_log_path}"
                )
            validation_values = pd.to_numeric(
                training_log["val_accuracy"], errors="coerce"
            ).dropna()
            if validation_values.empty:
                raise ValueError(
                    f"No numeric val_accuracy values in: {training_log_path}"
                )

            rows.append(
                {
                    "stage_order": spec.stage_order,
                    "model_key": spec.model_key,
                    "model_label": spec.display_label,
                    "fold": fold,
                    "test_loss": float(metrics["loss"]),
                    "accuracy": accuracy,
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "best_validation_accuracy": float(validation_values.max()),
                    "test_sample_count": sample_count,
                    "metrics_path": relative_source_path(
                        metric_path, results_root
                    ),
                    "training_log_path": relative_source_path(
                        training_log_path, results_root
                    ),
                    "metrics_sha256": sha256_file(metric_path),
                    "training_log_sha256": sha256_file(training_log_path),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["stage_order", "fold"]
    ).reset_index(drop=True)


def validate_design(run_df: pd.DataFrame) -> None:
    """Fail loudly when the archived results do not match the stated design."""
    expected_rows = len(MODEL_SPECS) * len(EXPECTED_FOLDS)
    if len(run_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(run_df)}")

    if run_df.duplicated(["model_key", "fold"]).any():
        duplicates = run_df.loc[
            run_df.duplicated(["model_key", "fold"], keep=False),
            ["model_key", "fold", "metrics_path"],
        ]
        raise ValueError(
            "Duplicate model-fold results were found:\n"
            + duplicates.to_string(index=False)
        )

    for spec in MODEL_SPECS:
        model_rows = run_df.loc[run_df["model_key"] == spec.model_key]
        observed_folds = sorted(model_rows["fold"].tolist())
        if observed_folds != EXPECTED_FOLDS:
            raise ValueError(
                f"{spec.model_key}: expected folds {EXPECTED_FOLDS}, "
                f"found {observed_folds}"
            )

    metric_columns = ["accuracy", "balanced_accuracy", "macro_f1"]
    for column in metric_columns:
        if not run_df[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} contains a value outside [0, 1]")


def validate_main_result_provenance(run_df: pd.DataFrame) -> None:
    """Confirm that the final stage is the released ds003626 main result."""
    main_df = pd.read_csv(MAIN_RESULT_TABLE)
    expected = main_df.loc[
        (main_df["dataset_key"] == "ds003626")
        & (main_df["model"] == "LR-HORNet")
    ].copy()
    observed = run_df.loc[
        run_df["model_key"] == "group4_patchLstm_deformableLstm"
    ].copy()
    if sorted(expected["fold"].astype(int)) != EXPECTED_FOLDS:
        raise ValueError("The Section 4.4.1 LR-HORNet fold set is incomplete")
    if sorted(observed["fold"].astype(int)) != EXPECTED_FOLDS:
        raise ValueError("The reused LR-HORNet fold set is incomplete")

    expected = expected.set_index("fold")
    observed = observed.set_index("fold")
    string_pairs = (
        ("metrics_sha256", "metrics_sha256"),
        ("training_log_sha256", "training_log_sha256"),
    )
    numeric_pairs = (
        ("accuracy", "accuracy"),
        ("balanced_accuracy", "balanced_accuracy"),
        ("macro_f1", "macro_f1"),
        ("best_validation_accuracy", "val_accuracy"),
    )
    for fold in EXPECTED_FOLDS:
        for observed_column, expected_column in string_pairs:
            if str(observed.at[fold, observed_column]) != str(
                expected.at[fold, expected_column]
            ):
                raise ValueError(
                    f"Main-result hash mismatch in fold {fold}: "
                    f"{observed_column}"
                )
        for observed_column, expected_column in numeric_pairs:
            if not np.isclose(
                float(observed.at[fold, observed_column]),
                float(expected.at[fold, expected_column]),
                atol=1e-12,
            ):
                raise ValueError(
                    f"Main-result metric mismatch in fold {fold}: "
                    f"{observed_column}"
                )


def summarize_models(run_df: pd.DataFrame) -> pd.DataFrame:
    """Compute unweighted LOSO-fold means and sample standard deviations."""
    summary = (
        run_df.groupby(
            ["stage_order", "model_key", "model_label"],
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
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            test_sample_count_min=("test_sample_count", "min"),
            test_sample_count_max=("test_sample_count", "max"),
        )
        .sort_values("stage_order")
        .reset_index(drop=True)
    )
    return summary


def export_csv(dataframe: pd.DataFrame, output_path: Path) -> bool:
    """Write a CSV while reporting a clear warning if it is open elsewhere."""
    try:
        dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        print(
            f"Warning: could not update {output_path.name} because it is open "
            "in another application; the existing file was left unchanged."
        )
        return False
    return True


def generate_result_tables(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    output_dir: Path = SCRIPT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path], dict[str, bool]]:
    """Build, validate, and export run-level and model-level result tables."""
    results_root = results_root.resolve()
    output_dir = output_dir.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {results_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_df = load_run_level_results(results_root)
    validate_design(run_df)
    validate_main_result_provenance(run_df)
    summary_df = summarize_models(run_df)
    reported_df = summary_df[
        [
            "model_key",
            "n_folds",
            "accuracy_mean",
            "accuracy_sd",
            "balanced_accuracy_mean",
            "macro_f1_mean",
        ]
    ].rename(columns={"model_key": "model", "n_folds": "folds"})

    output_paths = {
        "run_level": output_dir / "run_level_results.csv",
        "model_summary": output_dir / "model_summary.csv",
        "reported_summary": (
            PROJECT_ROOT / "results" / "4.5" / "reported_results_summary.csv"
        ),
    }
    write_status = {
        "run_level": export_csv(run_df, output_paths["run_level"]),
        "model_summary": export_csv(summary_df, output_paths["model_summary"]),
        "reported_summary": export_csv(
            reported_df, output_paths["reported_summary"]
        ),
    }
    return run_df, summary_df, output_paths, write_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Section 4.5 CSV tables from archived results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root containing the five model result directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory receiving run_level_results.csv and model_summary.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_df, summary_df, output_paths, write_status = generate_result_tables(
        args.results_root, args.output_dir
    )
    print(
        f"Validated {len(run_df)} archived results: "
        f"{len(summary_df)} models x {len(EXPECTED_FOLDS)} folds."
    )
    print(summary_df.to_string(index=False, float_format="%.6f"))
    print("\nGenerated tables:")
    for name, output_path in output_paths.items():
        suffix = "" if write_status[name] else " (existing file retained)"
        print(f"  {output_path}{suffix}")


if __name__ == "__main__":
    main()
