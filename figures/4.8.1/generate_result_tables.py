"""Read and validate the raw results used by the Section 4.8.1 figure."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = (
    SCRIPT_DIR.parent.parent / "实验" / "4.8" / "4.8.1"
).resolve()
EXPECTED_RUNS_PER_MODEL = 5
EXPECTED_FOLD = 1
EXPECTED_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    order: int
    result_dir: str
    model_key: str
    display_label: str
    patch_operator: str
    offset_operator: str


MODEL_SPECS = (
    ModelSpec(
        1,
        "group4_patchLstm_deformableLstm",
        "group4_patchLstm_deformableLstm",
        "LSTM-LSTM",
        "LSTM",
        "LSTM",
    ),
    ModelSpec(
        2,
        "patch_gru_offset_gru",
        "group4_patchGRU_deformableGRU",
        "GRU-GRU",
        "GRU",
        "GRU",
    ),
    ModelSpec(
        3,
        "patch_rnn_offset_rnn",
        "group4_patchRNN_deformableRNN",
        "RNN-RNN",
        "RNN",
        "RNN",
    ),
    ModelSpec(
        4,
        "group4_patchLstm_deformableCfC",
        "group4_patchLstm_deformableCfC",
        "LSTM-CfC",
        "LSTM",
        "CfC",
    ),
    ModelSpec(
        5,
        "group4_patchCfC_deformableCfC",
        "group4_patchCfC_deformableCfC",
        "CfC-CfC",
        "CfC",
        "CfC",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _extract_run_identity(path: Path) -> tuple[int, int, int]:
    path_text = path.as_posix()
    repeat_matches = re.findall(r"repeat_(\d+)_seed_(\d+)", path_text)
    if len(repeat_matches) != 1:
        raise ValueError(f"Cannot identify one repeat and seed from {path}")
    repeat, seed = (int(value) for value in repeat_matches[0])

    fold_values = {int(value) for value in re.findall(r"fold_(\d+)", path_text)}
    if fold_values != {EXPECTED_FOLD}:
        raise ValueError(f"Unexpected fold identity in {path}: {fold_values}")
    return EXPECTED_FOLD, repeat, seed


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


def read_run_results(results_root: Path = DEFAULT_RESULTS_ROOT) -> pd.DataFrame:
    if not results_root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {results_root}")

    rows: list[dict] = []
    for spec in MODEL_SPECS:
        model_root = results_root / spec.result_dir
        metric_paths = sorted(model_root.rglob("test_metrics.json"))
        if len(metric_paths) != EXPECTED_RUNS_PER_MODEL:
            raise ValueError(
                f"{spec.display_label}: expected {EXPECTED_RUNS_PER_MODEL} "
                f"test_metrics.json files, found {len(metric_paths)}"
            )

        seen_repeats: set[int] = set()
        for metric_path in metric_paths:
            fold, repeat, seed = _extract_run_identity(metric_path)
            if seed != EXPECTED_SEED:
                raise ValueError(f"Unexpected seed {seed} in {metric_path}")
            if repeat in seen_repeats:
                raise ValueError(
                    f"Duplicate repeat {repeat} for {spec.display_label}"
                )
            seen_repeats.add(repeat)

            with metric_path.open("r", encoding="utf-8") as stream:
                metrics = json.load(stream)
            sample_count, recomputed_accuracy = _validate_metrics(
                metric_path, metrics
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
                    "model_order": spec.order,
                    "model_key": spec.model_key,
                    "model_label": spec.display_label,
                    "patch_operator": spec.patch_operator,
                    "offset_operator": spec.offset_operator,
                    "fold": fold,
                    "repeat": repeat,
                    "seed": seed,
                    "test_loss": float(metrics["loss"]),
                    "accuracy": recomputed_accuracy,
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

    run_df = pd.DataFrame(rows).sort_values(
        ["model_order", "repeat"], ignore_index=True
    )
    validate_design(run_df)
    return run_df


def validate_design(run_df: pd.DataFrame) -> None:
    expected_rows = len(MODEL_SPECS) * EXPECTED_RUNS_PER_MODEL
    if len(run_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(run_df)}")

    for spec in MODEL_SPECS:
        subset = run_df.loc[run_df["model_key"] == spec.model_key]
        if len(subset) != EXPECTED_RUNS_PER_MODEL:
            raise ValueError(
                f"{spec.display_label}: expected {EXPECTED_RUNS_PER_MODEL} rows"
            )
        if subset["repeat"].nunique() != EXPECTED_RUNS_PER_MODEL:
            raise ValueError(f"{spec.display_label}: repeat IDs are not unique")
        if set(subset["fold"]) != {EXPECTED_FOLD}:
            raise ValueError(f"{spec.display_label}: unexpected fold")
        if set(subset["seed"]) != {EXPECTED_SEED}:
            raise ValueError(f"{spec.display_label}: unexpected seed")

    for column in (
        "accuracy",
        "best_validation_accuracy",
        "balanced_accuracy",
        "macro_f1",
    ):
        if not run_df[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} contains a value outside [0, 1]")


def summarize_results(run_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        run_df.groupby(
            [
                "model_order",
                "model_key",
                "model_label",
                "patch_operator",
                "offset_operator",
            ],
            as_index=False,
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
            macro_f1_mean=("macro_f1", "mean"),
            test_sample_count=("test_sample_count", "first"),
        )
        .sort_values("model_order", ignore_index=True)
    )
    return summary_df


def write_tables(
    run_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path = SCRIPT_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_level": output_dir / "run_level_results.csv",
        "summary": output_dir / "model_summary.csv",
    }
    run_df.to_csv(paths["run_level"], index=False, encoding="utf-8-sig")
    summary_df.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    return paths


def generate_result_tables(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    output_dir: Path = SCRIPT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    run_df = read_run_results(results_root)
    summary_df = summarize_results(run_df)
    paths = write_tables(run_df, summary_df, output_dir)
    return run_df, summary_df, paths


def main() -> None:
    run_df, summary_df, paths = generate_result_tables()
    print(
        f"Validated design: {len(MODEL_SPECS)} models x "
        f"{EXPECTED_RUNS_PER_MODEL} runs = {len(run_df)} results."
    )
    print("\nTest-accuracy summary:")
    print(
        summary_df[
            ["model_label", "n_runs", "accuracy_mean", "accuracy_sd"]
        ].to_string(index=False, float_format="%.6f")
    )
    print("\nGenerated files:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
