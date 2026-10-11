"""Regenerate the Section 4.4.2 result tables from archived run outputs.

No numerical value in the two output tables is entered manually.  For each
archived run, this script reads ``test_metrics.json`` for test accuracy and
``training_log.csv`` for the maximum validation accuracy, validates the
3 x 9 x 5 experimental design, and writes:

* ``run_level_results.csv``: one row per run, with relative source paths;
* ``condition_summary.csv``: mean and sample standard deviation over five runs.

Run from any working directory:

    python generate_result_tables.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "最终结果"
    / "实验"
    / "4.4"
    / "4.4.2"
    / "group_05"
)

TEST_SUBJECTS = [6, 7, 8]
ALL_SUBJECTS = list(range(1, 11))
EXPECTED_REPEATS = 5
EXPECTED_SEED = 42
EXPECTED_RUNS = 135
EXPECTED_CONDITIONS = 27


def parse_identifiers(path: Path) -> tuple[int, int, int, int]:
    """Extract test subject, validation subject, repeat, and seed from a path."""
    normalized = path.as_posix()
    pair_match = re.search(r"test_(\d{2})_val_(\d{2})", normalized)
    repeat_match = re.search(r"repeat_(\d{2})_seed_(\d+)", normalized)
    if pair_match is None or repeat_match is None:
        raise ValueError(f"Cannot parse experiment identifiers from: {path}")
    return (
        int(pair_match.group(1)),
        int(pair_match.group(2)),
        int(repeat_match.group(1)),
        int(repeat_match.group(2)),
    )


def relative_source_path(path: Path, results_root: Path) -> str:
    """Return a portable POSIX path relative to the archived-results root."""
    return path.resolve().relative_to(results_root.resolve()).as_posix()


def load_run_level_results(results_root: Path) -> pd.DataFrame:
    """Read raw test and validation metrics for every archived run."""
    rows: list[dict[str, float | int | str]] = []
    metric_paths = sorted(results_root.rglob("test_metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(
            f"No test_metrics.json files were found below: {results_root}"
        )

    for metric_path in metric_paths:
        test_subject, validation_subject, repeat, seed = parse_identifiers(
            metric_path
        )
        with metric_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if "accuracy" not in metrics or "confusion_matrix" not in metrics:
            raise ValueError(
                f"accuracy or confusion_matrix is absent from: {metric_path}"
            )

        training_log_path = metric_path.with_name("training_log.csv")
        if not training_log_path.exists():
            raise FileNotFoundError(f"Missing training log: {training_log_path}")

        training_log = pd.read_csv(training_log_path)
        if "val_accuracy" not in training_log.columns:
            raise ValueError(f"val_accuracy is absent from: {training_log_path}")
        validation_values = pd.to_numeric(
            training_log["val_accuracy"], errors="coerce"
        ).dropna()
        if validation_values.empty:
            raise ValueError(
                f"No numeric val_accuracy values were found in: {training_log_path}"
            )

        confusion_matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
        if confusion_matrix.ndim != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
            raise ValueError(f"Invalid confusion matrix in: {metric_path}")

        rows.append(
            {
                "test_subject": test_subject,
                "validation_subject": validation_subject,
                "repeat": repeat,
                "seed": seed,
                "test_accuracy": float(metrics["accuracy"]),
                "best_validation_accuracy": float(validation_values.max()),
                "test_sample_count": int(confusion_matrix.sum()),
                "metrics_path": relative_source_path(metric_path, results_root),
                "training_log_path": relative_source_path(
                    training_log_path, results_root
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["test_subject", "validation_subject", "repeat", "seed"]
    ).reset_index(drop=True)


def validate_design(run_df: pd.DataFrame) -> None:
    """Fail if the archived outputs do not match the stated protocol."""
    if len(run_df) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} runs, found {len(run_df)}")

    observed_tests = sorted(run_df["test_subject"].unique().tolist())
    if observed_tests != TEST_SUBJECTS:
        raise ValueError(
            f"Expected test subjects {TEST_SUBJECTS}, found {observed_tests}"
        )

    if run_df.duplicated(
        ["test_subject", "validation_subject", "repeat", "seed"]
    ).any():
        duplicates = run_df.loc[
            run_df.duplicated(
                ["test_subject", "validation_subject", "repeat", "seed"],
                keep=False,
            )
        ]
        raise ValueError(
            "Duplicate run identifiers were found:\n"
            + duplicates.to_string(index=False)
        )

    observed_seeds = sorted(run_df["seed"].unique().tolist())
    if observed_seeds != [EXPECTED_SEED]:
        raise ValueError(
            f"Expected seed {EXPECTED_SEED}, found seeds {observed_seeds}"
        )

    counts = run_df.groupby(["test_subject", "validation_subject"]).size()
    if len(counts) != EXPECTED_CONDITIONS or not bool(
        (counts == EXPECTED_REPEATS).all()
    ):
        raise ValueError(
            f"Expected {EXPECTED_CONDITIONS} conditions with "
            f"{EXPECTED_REPEATS} runs each; observed counts were:\n"
            f"{counts.to_string()}"
        )

    for test_subject in TEST_SUBJECTS:
        expected_validations = sorted(set(ALL_SUBJECTS) - {test_subject})
        observed_validations = sorted(
            run_df.loc[
                run_df["test_subject"] == test_subject, "validation_subject"
            ].unique()
        )
        if observed_validations != expected_validations:
            raise ValueError(
                f"Test subject {test_subject}: expected validation subjects "
                f"{expected_validations}, found {observed_validations}"
            )

        for validation_subject in expected_validations:
            observed_repeats = sorted(
                run_df.loc[
                    (run_df["test_subject"] == test_subject)
                    & (run_df["validation_subject"] == validation_subject),
                    "repeat",
                ].tolist()
            )
            expected_repeat_ids = list(range(1, EXPECTED_REPEATS + 1))
            if observed_repeats != expected_repeat_ids:
                raise ValueError(
                    f"Test {test_subject}, validation {validation_subject}: "
                    f"expected repeats {expected_repeat_ids}, found "
                    f"{observed_repeats}"
                )


def summarize_conditions(run_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate five repeated runs within each test-validation condition."""
    return (
        run_df.groupby(["test_subject", "validation_subject"], as_index=False)
        .agg(
            n_runs=("test_accuracy", "size"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_sd=("test_accuracy", "std"),
            validation_accuracy_mean=("best_validation_accuracy", "mean"),
            validation_accuracy_sd=("best_validation_accuracy", "std"),
            test_sample_count=("test_sample_count", "first"),
        )
        .sort_values(["test_subject", "validation_subject"])
        .reset_index(drop=True)
    )


def export_csv(dataframe: pd.DataFrame, output_path: Path) -> bool:
    """Write a table, while reporting a clear warning if the file is open."""
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
    """Build, validate, and export both result tables from raw files."""
    results_root = results_root.resolve()
    output_dir = output_dir.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Result directory does not exist: {results_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_df = load_run_level_results(results_root)
    validate_design(run_df)
    summary_df = summarize_conditions(run_df)

    output_paths = {
        "run_level": output_dir / "run_level_results.csv",
        "condition_summary": output_dir / "condition_summary.csv",
    }
    write_status = {
        "run_level": export_csv(run_df, output_paths["run_level"]),
        "condition_summary": export_csv(
            summary_df, output_paths["condition_summary"]
        ),
    }
    return run_df, summary_df, output_paths, write_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Section 4.4.2 CSV tables from archived results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root containing the test-validation experiment directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory receiving the two generated CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_df, summary_df, output_paths, write_status = generate_result_tables(
        args.results_root, args.output_dir
    )

    print(
        f"Validated {len(run_df)} runs and {len(summary_df)} conditions "
        f"from: {args.results_root.resolve()}"
    )
    for name, output_path in output_paths.items():
        suffix = "" if write_status[name] else " (existing file retained)"
        print(f"  {output_path}{suffix}")


if __name__ == "__main__":
    main()
