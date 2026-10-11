"""Regenerate the Section 4.8.2 comparison tables from raw outputs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "最终结果" / "实验"
EXPECTED_FOLDS = tuple(range(1, 11))
PROTOCOL_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    source_root: Path
    model_key: str
    raw_model_name: str
    display_label: str
    short_label: str
    model_order: int
    repeat_01_only: bool = False


MODEL_SPECS = (
    ModelSpec(
        EXPERIMENT_ROOT / "4.5" / "imagine" / "本文模型",
        "lr_hornet",
        "group4_patchLstm_deformableLstm",
        "LR-HORNet",
        "LR-HORNet",
        1,
    ),
    ModelSpec(
        EXPERIMENT_ROOT
        / "4.8"
        / "4.8.2"
        / "mamba"
        / "full_mamba_patch_replaced",
        "mamba_patch",
        "group4_fullMamba_deformableLstm",
        "M-Patch",
        "M-Patch",
        2,
        True,
    ),
    ModelSpec(
        EXPERIMENT_ROOT
        / "4.8"
        / "4.8.2"
        / "mamba"
        / "mamba_offset_only",
        "mamba_deformable",
        "group4_patchLstm_deformableMamba",
        "M-Deform",
        "M-Deform",
        3,
        True,
    ),
    ModelSpec(
        EXPERIMENT_ROOT
        / "4.8"
        / "4.8.2"
        / "attention"
        / "group4_patchlstm_deformable_attention_vector_token"
        / "statt",
        "statt",
        "group4_patchLstm_deformableLstm_statt",
        "statt",
        "STATT",
        4,
    ),
    ModelSpec(
        EXPERIMENT_ROOT
        / "4.8"
        / "4.8.2"
        / "attention"
        / "group4_patchlstm_deformable_attention_vector_token"
        / "satt",
        "satt",
        "group4_patchLstm_deformableLstm_satt",
        "satt",
        "SATT",
        5,
    ),
)

# Published snapshot means are used only as a selection-integrity check.
EXPECTED_ACCURACY_MEANS = {
    # The corrected fold-9 file yields the current formal LR-HORNet mean.
    "lr_hornet": 0.30903703703703705,
    "mamba_patch": 0.28494444444444444,
    "mamba_deformable": 0.29544444444444445,
    "statt": 0.28903703703703704,
    "satt": 0.289,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _fold_from_path(path: Path) -> int:
    match = re.fullmatch(r"fold[_-]?(\d+)", path.parent.name)
    if match is None:
        raise ValueError(f"Cannot parse fold from {path}")
    return int(match.group(1))


def _repeat_from_path(path: Path) -> int | None:
    match = re.search(r"repeat_(\d+)_seed_(\d+)", path.as_posix())
    if match is None:
        return None
    repeat, seed = (int(value) for value in match.groups())
    if seed != PROTOCOL_SEED:
        raise ValueError(f"Unexpected seed {seed} in {path}")
    return repeat


def _metric_paths(spec: ModelSpec) -> list[Path]:
    if not spec.source_root.is_dir():
        raise FileNotFoundError(f"Missing result directory: {spec.source_root}")
    paths = sorted(spec.source_root.rglob("test_metrics.json"))
    if spec.repeat_01_only:
        paths = [
            path
            for path in paths
            if "repeat_01_seed_42" in path.as_posix()
        ]
    if len(paths) != len(EXPECTED_FOLDS):
        raise ValueError(
            f"{spec.model_key}: expected 10 selected metric files, "
            f"found {len(paths)}"
        )
    return paths


def _validate_metrics(path: Path, metrics: dict) -> tuple[int, float]:
    required = {
        "loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "confusion_matrix",
    }
    missing = sorted(required.difference(metrics))
    if missing:
        raise ValueError(f"Missing keys {missing} in {path}")

    confusion = np.asarray(metrics["confusion_matrix"], dtype=int)
    if (
        confusion.ndim != 2
        or confusion.shape[0] != confusion.shape[1]
        or np.any(confusion < 0)
    ):
        raise ValueError(f"Invalid confusion matrix in {path}")
    sample_count = int(confusion.sum())
    if sample_count <= 0:
        raise ValueError(f"Empty confusion matrix in {path}")

    recomputed = float(np.trace(confusion) / sample_count)
    stored = float(metrics["accuracy"])
    if not np.isclose(stored, recomputed, atol=1e-12):
        raise ValueError(
            f"Accuracy mismatch in {path}: stored={stored}, "
            f"recomputed={recomputed}"
        )
    return sample_count, recomputed


def load_run_level_results() -> pd.DataFrame:
    rows: list[dict] = []
    for spec in MODEL_SPECS:
        for metric_path in _metric_paths(spec):
            fold = _fold_from_path(metric_path)
            repeat = _repeat_from_path(metric_path)
            if spec.repeat_01_only and repeat != 1:
                raise ValueError(
                    f"{spec.model_key}: selected a non-primary repeat in "
                    f"{metric_path}"
                )

            with metric_path.open("r", encoding="utf-8") as stream:
                metrics = json.load(stream)
            sample_count, accuracy = _validate_metrics(metric_path, metrics)

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
                    "short_label": spec.short_label,
                    "raw_model_name": spec.raw_model_name,
                    "fold": fold,
                    "repeat": repeat,
                    "protocol_seed": PROTOCOL_SEED,
                    "test_loss": float(metrics["loss"]),
                    "accuracy": accuracy,
                    "best_validation_accuracy": float(
                        validation_values.max()
                    ),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "test_sample_count": sample_count,
                    "metrics_path": _relative_path(metric_path),
                    "training_log_path": _relative_path(training_log_path),
                    "metrics_sha256": _sha256(metric_path),
                    "training_log_sha256": _sha256(training_log_path),
                }
            )

    frame = pd.DataFrame(rows).sort_values(
        ["model_order", "fold"], ignore_index=True
    )
    _validate_design(frame)
    return frame


def _validate_design(frame: pd.DataFrame) -> None:
    expected_rows = len(MODEL_SPECS) * len(EXPECTED_FOLDS)
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}")
    if frame.duplicated(["model_key", "fold"]).any():
        raise ValueError("Duplicate model-fold rows were found")

    for spec in MODEL_SPECS:
        observed = tuple(
            frame.loc[frame["model_key"] == spec.model_key, "fold"].tolist()
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
        if not frame[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} contains a value outside [0, 1]")


def summarize_models(run_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
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
        )
        .sort_values("model_order", ignore_index=True)
    )

    for row in summary.itertuples(index=False):
        expected = EXPECTED_ACCURACY_MEANS[row.model_key]
        if not np.isclose(row.accuracy_mean, expected, atol=1e-12):
            raise ValueError(
                f"{row.model_key}: mean {row.accuracy_mean} does not match "
                f"the archived snapshot {expected}"
            )
    return summary


def generate_result_tables(
    output_dir: Path = SCRIPT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_df = load_run_level_results()
    summary_df = summarize_models(run_df)
    paths = {
        "run_level": output_dir / "run_level_results.csv",
        "model_summary": output_dir / "model_summary.csv",
    }
    run_df.to_csv(paths["run_level"], index=False, encoding="utf-8-sig")
    summary_df.to_csv(
        paths["model_summary"], index=False, encoding="utf-8-sig"
    )
    return run_df, summary_df, paths


def main() -> None:
    run_df, summary_df, paths = generate_result_tables()
    print("Validated design: 5 models x 10 LOSO folds = 50 results.")
    print(
        summary_df[
            ["model_label", "n_folds", "accuracy_mean", "accuracy_sd"]
        ].to_string(index=False, float_format="%.6f")
    )
    print("\nGenerated tables:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
