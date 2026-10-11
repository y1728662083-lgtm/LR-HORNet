"""Extract and validate the selected 10-fold LSTM-CfC results."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = (
    SCRIPT_DIR.parents[2]
    / "实验"
    / "group4_patchLstm_deformableCfC _selected"
).resolve()
EXPECTED_FOLDS = tuple(range(1, 11))
EXPECTED_SEED = 42
MODEL_NAME = "group4_patchLstm_deformableCfC"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _parse_identity(path: Path) -> tuple[int, int, int]:
    path_text = path.as_posix()
    fold_values = {int(value) for value in re.findall(r"fold_(\d+)", path_text)}
    if len(fold_values) != 1:
        raise ValueError(f"Cannot identify one fold from {path}")
    fold = next(iter(fold_values))

    repeat_matches = re.findall(r"repeat_(\d+)_seed_(\d+)", path_text)
    if len(repeat_matches) != 1:
        raise ValueError(f"Cannot identify one repeat and seed from {path}")
    repeat, seed = (int(value) for value in repeat_matches[0])
    return fold, repeat, seed


def _validate_metrics(metric_path: Path, metrics: dict) -> tuple[int, float]:
    required = {
        "loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "confusion_matrix",
    }
    missing = sorted(required.difference(metrics))
    if missing:
        raise ValueError(f"Missing keys {missing} in {metric_path}")

    confusion = np.asarray(metrics["confusion_matrix"], dtype=int)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(f"Invalid confusion matrix in {metric_path}")
    sample_count = int(confusion.sum())
    if sample_count <= 0:
        raise ValueError(f"Empty confusion matrix in {metric_path}")

    recomputed_accuracy = float(np.trace(confusion) / sample_count)
    stored_accuracy = float(metrics["accuracy"])
    if not np.isclose(recomputed_accuracy, stored_accuracy, atol=1e-12):
        raise ValueError(
            f"Accuracy mismatch in {metric_path}: "
            f"stored={stored_accuracy}, recomputed={recomputed_accuracy}"
        )
    return sample_count, recomputed_accuracy


def read_fold_results(results_root: Path = DEFAULT_RESULTS_ROOT) -> pd.DataFrame:
    if not results_root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {results_root}")

    metric_paths = sorted(results_root.rglob("test_metrics.json"))
    if len(metric_paths) != len(EXPECTED_FOLDS):
        raise ValueError(
            f"Expected {len(EXPECTED_FOLDS)} test_metrics.json files, "
            f"found {len(metric_paths)}"
        )

    rows: list[dict] = []
    for metric_path in metric_paths:
        fold, repeat, seed = _parse_identity(metric_path)
        if seed != EXPECTED_SEED:
            raise ValueError(f"Unexpected seed {seed} in {metric_path}")

        with metric_path.open("r", encoding="utf-8") as stream:
            metrics = json.load(stream)
        sample_count, accuracy = _validate_metrics(metric_path, metrics)

        training_log_path = metric_path.with_name("training_log.csv")
        if not training_log_path.is_file():
            raise FileNotFoundError(f"Missing training log for {metric_path}")
        training_log = pd.read_csv(training_log_path)
        if "val_accuracy" not in training_log.columns:
            raise ValueError(f"val_accuracy is absent from {training_log_path}")
        validation_values = pd.to_numeric(
            training_log["val_accuracy"], errors="coerce"
        ).dropna()
        if validation_values.empty:
            raise ValueError(
                f"No numeric val_accuracy values in {training_log_path}"
            )

        rows.append(
            {
                "model": MODEL_NAME,
                "fold": fold,
                "selected_repeat": repeat,
                "seed": seed,
                "test_loss": float(metrics["loss"]),
                "test_accuracy": accuracy,
                "best_validation_accuracy": float(validation_values.max()),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "test_sample_count": sample_count,
                "metrics_path": _relative_path(metric_path, results_root),
                "training_log_path": _relative_path(
                    training_log_path, results_root
                ),
                "metrics_sha256": _sha256(metric_path),
                "training_log_sha256": _sha256(training_log_path),
            }
        )

    fold_df = pd.DataFrame(rows).sort_values("fold", ignore_index=True)
    observed_folds = tuple(fold_df["fold"].tolist())
    if observed_folds != EXPECTED_FOLDS:
        raise ValueError(
            f"Expected folds {EXPECTED_FOLDS}, found {observed_folds}"
        )
    if fold_df["fold"].nunique() != len(EXPECTED_FOLDS):
        raise ValueError("Fold IDs are not unique")

    for column in (
        "test_accuracy",
        "best_validation_accuracy",
        "balanced_accuracy",
        "macro_f1",
    ):
        if not fold_df[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} contains a value outside [0, 1]")
    return fold_df


def summarize_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "n_folds": len(fold_df),
                "test_accuracy_mean": fold_df["test_accuracy"].mean(),
                "test_accuracy_sd": fold_df["test_accuracy"].std(ddof=1),
                "test_accuracy_sem": fold_df["test_accuracy"].sem(ddof=1),
                "test_accuracy_min": fold_df["test_accuracy"].min(),
                "test_accuracy_max": fold_df["test_accuracy"].max(),
                "validation_accuracy_mean": fold_df[
                    "best_validation_accuracy"
                ].mean(),
                "validation_accuracy_sd": fold_df[
                    "best_validation_accuracy"
                ].std(ddof=1),
                "validation_accuracy_sem": fold_df[
                    "best_validation_accuracy"
                ].sem(ddof=1),
                "validation_accuracy_min": fold_df[
                    "best_validation_accuracy"
                ].min(),
                "validation_accuracy_max": fold_df[
                    "best_validation_accuracy"
                ].max(),
                "macro_f1_mean": fold_df["macro_f1"].mean(),
            }
        ]
    )


def generate_tables(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    output_dir: Path = SCRIPT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    fold_df = read_fold_results(results_root)
    summary_df = summarize_results(fold_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fold_results": output_dir / "fold_results.csv",
        "summary": output_dir / "summary.csv",
    }
    fold_df.to_csv(paths["fold_results"], index=False, encoding="utf-8-sig")
    summary_df.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    return fold_df, summary_df, paths


def main() -> None:
    fold_df, summary_df, paths = generate_tables()
    print("Validated design: 10 selected LOSO-fold results.")
    print("\nFold-wise test accuracy:")
    printable = fold_df[["fold", "selected_repeat", "test_accuracy"]].copy()
    printable["test_accuracy"] *= 100.0
    print(printable.to_string(index=False, float_format="%.2f"))
    print("\nSummary:")
    summary_print = summary_df[
        ["n_folds", "test_accuracy_mean", "test_accuracy_sd"]
    ].copy()
    summary_print[["test_accuracy_mean", "test_accuracy_sd"]] *= 100.0
    print(summary_print.to_string(index=False, float_format="%.4f"))
    print("\nGenerated files:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
