"""Regenerate Section 4.6.2 result tables from archived raw outputs.

No result is entered manually. The script reads 8 conditions x 5 repeated
fold-1 runs, validates metadata and confusion matrices, and exports auditable
run-level and condition-level CSV files.
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
    PROJECT_ROOT / "最终结果" / "实验" / "4.6" / "4.6.2"
)
EXPECTED_REPEATS = list(range(1, 6))


@dataclass(frozen=True)
class ConditionSpec:
    source_directory: str
    condition_key: str
    display_label: str
    expected_patch_size: int
    condition_type: str
    condition_order: int


CONDITION_SPECS = (
    ConditionSpec("patch_0008", "patch_0008", "8", 8, "patch", 1),
    ConditionSpec("patch_0016", "patch_0016", "16", 16, "patch", 2),
    ConditionSpec("patch_0032", "patch_0032", "32", 32, "patch", 3),
    ConditionSpec("patch_0064", "patch_0064", "64", 64, "patch", 4),
    ConditionSpec("patch_0128", "patch_0128", "128", 128, "patch", 5),
    ConditionSpec("patch_0256", "patch_0256", "256", 256, "patch", 6),
    ConditionSpec("patch_0512", "patch_0512", "512", 512, "patch", 7),
    ConditionSpec(
        "full_sequence_no_patch",
        "full_sequence_no_patch",
        "1153\n(No Patch)",
        1153,
        "no_patch",
        8,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_source_path(path: Path, results_root: Path) -> str:
    return path.resolve().relative_to(results_root.resolve()).as_posix()


def find_repeat_directory(metric_path: Path) -> tuple[Path, int, int]:
    pattern = re.compile(r"repeat_(\d+)_seed_(\d+)")
    for parent in metric_path.parents:
        match = pattern.fullmatch(parent.name)
        if match is not None:
            return parent, int(match.group(1)), int(match.group(2))
    raise ValueError(f"Cannot find repeat directory for {metric_path}")


def load_run_level_results(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for spec in CONDITION_SPECS:
        condition_root = results_root / spec.source_directory
        if not condition_root.is_dir():
            raise FileNotFoundError(
                f"Missing condition directory: {condition_root}"
            )
        metric_paths = sorted(condition_root.rglob("test_metrics.json"))
        if len(metric_paths) != len(EXPECTED_REPEATS):
            raise ValueError(
                f"{spec.condition_key}: expected 5 test metric files, "
                f"found {len(metric_paths)}"
            )

        for metric_path in metric_paths:
            repeat_dir, repeat, seed = find_repeat_directory(metric_path)
            metadata_path = repeat_dir / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Missing metadata: {metadata_path}")

            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            with metric_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)

            if metadata.get("variant") != spec.condition_key:
                raise ValueError(f"Variant mismatch in {metadata_path}")
            if int(metadata.get("repeat", -1)) != repeat:
                raise ValueError(f"Repeat mismatch in {metadata_path}")
            if int(metadata.get("seed", -1)) != seed or seed != 42:
                raise ValueError(f"Expected seed 42 in {metadata_path}")
            if int(metadata.get("fold_index", -1)) != 0:
                raise ValueError(f"Expected fold_index 0 in {metadata_path}")
            patch_size = int(metadata.get("patch_size", -1))
            patch_stride = int(metadata.get("patch_stride", -1))
            if patch_size != spec.expected_patch_size:
                raise ValueError(
                    f"Patch size mismatch in {metadata_path}: "
                    f"expected {spec.expected_patch_size}, found {patch_size}"
                )

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
                    "condition_order": spec.condition_order,
                    "condition_key": spec.condition_key,
                    "condition_type": spec.condition_type,
                    "display_label": spec.display_label.replace("\n", " "),
                    "fold": 1,
                    "repeat": repeat,
                    "seed": seed,
                    "patch_size": patch_size,
                    "patch_stride": patch_stride,
                    "test_loss": float(metrics["loss"]),
                    "accuracy": accuracy,
                    "best_validation_accuracy": float(validation_values.max()),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "test_sample_count": sample_count,
                    "metrics_path": relative_source_path(metric_path, results_root),
                    "metadata_path": relative_source_path(
                        metadata_path, results_root
                    ),
                    "training_log_path": relative_source_path(
                        training_log_path, results_root
                    ),
                    "metrics_sha256": sha256_file(metric_path),
                    "metadata_sha256": sha256_file(metadata_path),
                    "training_log_sha256": sha256_file(training_log_path),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["condition_order", "repeat"]
    ).reset_index(drop=True)


def validate_design(run_df: pd.DataFrame) -> None:
    expected_rows = len(CONDITION_SPECS) * len(EXPECTED_REPEATS)
    if len(run_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(run_df)}")
    if run_df.duplicated(["condition_key", "repeat"]).any():
        raise ValueError("Duplicate condition-repeat rows were found")

    for spec in CONDITION_SPECS:
        rows = run_df.loc[run_df["condition_key"] == spec.condition_key]
        observed_repeats = sorted(rows["repeat"].tolist())
        if observed_repeats != EXPECTED_REPEATS:
            raise ValueError(
                f"{spec.condition_key}: expected repeats {EXPECTED_REPEATS}, "
                f"found {observed_repeats}"
            )

    for column in (
        "accuracy",
        "best_validation_accuracy",
        "balanced_accuracy",
        "macro_f1",
    ):
        if not run_df[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} contains a value outside [0, 1]")


def summarize_conditions(run_df: pd.DataFrame) -> pd.DataFrame:
    return (
        run_df.groupby(
            [
                "condition_order",
                "condition_key",
                "condition_type",
                "display_label",
                "patch_size",
                "patch_stride",
            ],
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
    summary_df = summarize_conditions(run_df)

    paths = {
        "run_level": output_dir / "run_level_results.csv",
        "condition_summary": output_dir / "condition_summary.csv",
    }
    status = {
        "run_level": export_csv(run_df, paths["run_level"]),
        "condition_summary": export_csv(
            summary_df, paths["condition_summary"]
        ),
    }
    return run_df, summary_df, paths, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Section 4.6.2 CSV tables from raw results."
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
        f"{len(summary_df)} conditions x {len(EXPECTED_REPEATS)} runs."
    )
    print(summary_df.to_string(index=False, float_format="%.6f"))
    print("\nGenerated tables:")
    for name, path in paths.items():
        suffix = "" if status[name] else " (existing file retained)"
        print(f"  {path}{suffix}")


if __name__ == "__main__":
    main()
