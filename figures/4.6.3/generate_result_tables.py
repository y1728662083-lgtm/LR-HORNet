"""Regenerate Section 4.6.3 tables from archived raw experiment outputs.

The script reads five electrode-group conditions with five repeated fold-1
runs per condition. It validates the test confusion matrices, derives the best
validation accuracy from each training log, and writes auditable CSV tables.
No plotted value is entered manually.
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
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "最终结果"
    / "实验"
    / "4.6"
    / "4.6.3"
    / "fold1_recurrent_group_ablation"
    / "fixed_seed42_repeat5"
)
EXPECTED_RUNS_PER_CONDITION = 5
EXPECTED_SEED = 42


@dataclass(frozen=True)
class ConditionSpec:
    group_count: int
    source_directory: str
    condition_order: int

    @property
    def condition_key(self) -> str:
        return f"group_{self.group_count}"


CONDITION_SPECS = (
    ConditionSpec(1, "group1_patchLstm_deformableLstm", 1),
    ConditionSpec(2, "group2_patchLstm_deformableLstm", 2),
    ConditionSpec(4, "group4_patchLstm_deformableLstm", 3),
    ConditionSpec(8, "group8_patchLstm_deformableLstm", 4),
    ConditionSpec(16, "group16_patchLstm_deformableLstm", 5),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_source_path(path: Path, results_root: Path) -> str:
    return path.resolve().relative_to(results_root.resolve()).as_posix()


def find_named_parent(path: Path, pattern: str, description: str) -> re.Match[str]:
    compiled = re.compile(pattern)
    for parent in path.parents:
        match = compiled.fullmatch(parent.name)
        if match is not None:
            return match
    raise ValueError(f"Cannot identify {description} from {path}")


def read_run(metric_path: Path, spec: ConditionSpec, results_root: Path) -> dict:
    repeat_match = find_named_parent(
        metric_path, r"repeat_(\d+)_seed_(\d+)", "repeat and seed"
    )
    fold_match = find_named_parent(metric_path, r"fold_(\d+)", "fold")
    repeat = int(repeat_match.group(1))
    seed = int(repeat_match.group(2))
    fold = int(fold_match.group(1))
    if seed != EXPECTED_SEED:
        raise ValueError(f"Expected seed {EXPECTED_SEED}: {metric_path}")
    if fold != 1:
        raise ValueError(f"Expected fold 1: {metric_path}")

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
    test_sample_count = int(confusion.sum())
    if test_sample_count <= 0:
        raise ValueError(f"Empty confusion matrix in {metric_path}")

    accuracy = float(metrics["accuracy"])
    recomputed_accuracy = float(np.trace(confusion) / test_sample_count)
    if not np.isclose(accuracy, recomputed_accuracy, atol=1e-12):
        raise ValueError(
            f"Accuracy/confusion-matrix mismatch in {metric_path}: "
            f"{accuracy} versus {recomputed_accuracy}"
        )

    training_log_path = metric_path.with_name("training_log.csv")
    if not training_log_path.is_file():
        raise FileNotFoundError(f"Missing training log: {training_log_path}")
    training_log = pd.read_csv(training_log_path)
    if "val_accuracy" not in training_log.columns:
        raise ValueError(f"val_accuracy is absent from {training_log_path}")
    validation_values = pd.to_numeric(
        training_log["val_accuracy"], errors="coerce"
    ).dropna()
    if validation_values.empty:
        raise ValueError(f"No numeric validation accuracy in {training_log_path}")

    return {
        "condition_order": spec.condition_order,
        "condition_key": spec.condition_key,
        "group_count": spec.group_count,
        "fold": fold,
        "repeat": repeat,
        "seed": seed,
        "test_loss": float(metrics["loss"]),
        "accuracy": accuracy,
        "best_validation_accuracy": float(validation_values.max()),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "test_sample_count": test_sample_count,
        "metrics_path": relative_source_path(metric_path, results_root),
        "training_log_path": relative_source_path(training_log_path, results_root),
        "metrics_sha256": sha256_file(metric_path),
        "training_log_sha256": sha256_file(training_log_path),
    }


def load_run_level_results(results_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in CONDITION_SPECS:
        condition_root = results_root / spec.source_directory
        if not condition_root.is_dir():
            raise FileNotFoundError(f"Missing condition directory: {condition_root}")
        metric_paths = sorted(condition_root.rglob("test_metrics.json"))
        if len(metric_paths) != EXPECTED_RUNS_PER_CONDITION:
            raise ValueError(
                f"{spec.condition_key}: expected {EXPECTED_RUNS_PER_CONDITION} "
                f"test metric files, found {len(metric_paths)}"
            )
        rows.extend(read_run(path, spec, results_root) for path in metric_paths)

    run_df = pd.DataFrame(rows).sort_values(
        ["condition_order", "repeat"]
    ).reset_index(drop=True)
    if run_df.duplicated(["condition_key", "repeat", "seed"]).any():
        raise ValueError("Duplicate condition-repeat-seed rows were found")

    for spec in CONDITION_SPECS:
        observed = run_df.loc[run_df["condition_key"] == spec.condition_key]
        if len(observed) != EXPECTED_RUNS_PER_CONDITION:
            raise ValueError(
                f"{spec.condition_key}: expected {EXPECTED_RUNS_PER_CONDITION} runs"
            )
        if observed["repeat"].nunique() != EXPECTED_RUNS_PER_CONDITION:
            raise ValueError(f"{spec.condition_key}: repeat identifiers are not unique")

    for column in (
        "accuracy",
        "best_validation_accuracy",
        "balanced_accuracy",
        "macro_f1",
    ):
        if not run_df[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} contains a value outside [0, 1]")
    return run_df


def summarize_conditions(run_df: pd.DataFrame) -> pd.DataFrame:
    return (
        run_df.groupby(
            ["condition_order", "condition_key", "group_count"],
            as_index=False,
            sort=False,
        )
        .agg(
            n_runs=("accuracy", "size"),
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
        .sort_values("condition_order")
        .reset_index(drop=True)
    )


def export_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")


def generate_result_tables(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    output_dir: Path = SCRIPT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    results_root = results_root.resolve()
    output_dir = output_dir.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {results_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_df = load_run_level_results(results_root)
    summary_df = summarize_conditions(run_df)
    paths = {
        "run_level": output_dir / "run_level_results.csv",
        "condition_summary": output_dir / "condition_summary.csv",
    }
    export_csv(run_df, paths["run_level"])
    export_csv(summary_df, paths["condition_summary"])
    return run_df, summary_df, paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Section 4.6.3 CSV tables from raw results."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_df, summary_df, paths = generate_result_tables(
        args.results_root, args.output_dir
    )
    print(
        f"Validated {len(run_df)} results: {len(summary_df)} conditions x "
        f"{EXPECTED_RUNS_PER_CONDITION} repeated fold-1 runs."
    )
    print(summary_df.to_string(index=False, float_format="%.6f"))
    print("\nGenerated tables:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
