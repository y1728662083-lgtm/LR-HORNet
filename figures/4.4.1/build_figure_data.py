"""Build the reproducible data tables used by the Section 4.4.1 figures.

Accuracy and parameter counts are read from the audited cross-dataset summary
tables.  FLOPs are recomputed for one-sample inference with 1 MAC = 2 FLOPs.
The EEGMamba estimate follows the linear token complexity of the released
Mamba2 backbone and is normalized to the previously audited ds003626 value.
ATCNet and FAST use architecture-level operator formulas normalized to their
audited ds003626 forward-profile values; this preserves the original counting
convention while adapting the count to each dataset's input geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
SUMMARY_ROOT = (
    PROJECT_ROOT
    / "最终结果"
    / "实验"
    / "4.4"
    / "4.4.1"
    / "dataset_result"
    / "summary_by_dataset"
)

MODEL_ORDER = (
    "LR-HORNet",
    "EEGNet",
    "ATCNet",
    "EEGMamba",
    "FAST",
    "LaBraM",
)

DATASET_ORDER = (
    "ds003626",
    "bimodal_inner_speech",
    "directional_words_russian",
    "directional_words_spanish",
)

DATASET_LABELS = {
    "ds003626": "ds003626",
    "bimodal_inner_speech": "Bimodal",
    "directional_words_russian": "Directional Words-Russian",
    "directional_words_spanish": "Directional Words-Spanish",
}

EXPECTED_FOLD_COUNTS = {
    "ds003626": 10,
    "bimodal_inner_speech": 4,
    "directional_words_russian": 12,
    "directional_words_spanish": 10,
}


@dataclass(frozen=True)
class DatasetGeometry:
    channels: int
    times: int
    classes: int
    sample_rate: int
    patch_size: int
    patch_stride: int
    resampled_times_200hz: int
    baseline_patches: int
    fast_window: int
    fast_stride: int


GEOMETRY = {
    "ds003626": DatasetGeometry(128, 1153, 4, 256, 128, 64, 901, 5, 256, 128),
    "bimodal_inner_speech": DatasetGeometry(64, 359, 8, 512, 256, 128, 140, 1, 256, 128),
    "directional_words_russian": DatasetGeometry(34, 250, 6, 250, 125, 63, 200, 1, 250, 125),
    "directional_words_spanish": DatasetGeometry(38, 750, 6, 500, 250, 125, 300, 2, 500, 250),
}


def _downstream_eegnet_flops(
    channels: int,
    effective_times: int,
    classes: int,
) -> int:
    f1, f2 = 8, 16
    spatial = 2 * f2 * effective_times * channels
    after_pool1 = effective_times // 4
    after_depthwise = after_pool1 + 1
    depthwise = 2 * f2 * after_depthwise * 16
    pointwise = 2 * f2 * f2 * after_depthwise
    final_times = after_depthwise // 8
    classifier = 2 * classes * f2 * final_times
    return spatial + depthwise + pointwise + classifier


def eegnet_flops(g: DatasetGeometry) -> int:
    output_times = g.times + 1
    temporal = 2 * 8 * g.channels * output_times * 64
    return temporal + _downstream_eegnet_flops(
        g.channels, output_times, g.classes
    )


def lr_hornet_flops(g: DatasetGeometry) -> int:
    n_patches = (g.times - g.patch_size) // g.patch_stride + 1
    effective_times = g.patch_size + (n_patches - 1) * g.patch_stride
    recurrent_steps = g.channels * n_patches * g.patch_size
    recurrent_per_step = 2 * 4 * 8 * (1 + 8)
    patch_encoder = recurrent_steps * recurrent_per_step

    tokens = 16 * effective_times
    history_resampler = (
        tokens * recurrent_per_step
        + tokens * (2 * 8 * 2)
        + tokens
        + 7 * tokens
    )
    return (
        patch_encoder
        + _downstream_eegnet_flops(
            g.channels, effective_times, g.classes
        )
        + history_resampler
    )


def _atcnet_raw_flops(g: DatasetGeometry) -> int:
    f1, f2 = 16, 32
    kernel_1, kernel_2 = 64, 16
    pool_1, pool_2 = 8, 7
    n_windows, tcn_kernel = 5, 4
    tcn_depth = 2

    min_len_tcn = (tcn_kernel - 1) * (2 ** (tcn_depth - 1)) + 1
    minimum_times = (n_windows + min_len_tcn - 1) * pool_1 * pool_2
    if g.times < minimum_times:
        scale = g.times / minimum_times
        kernel_1 = max(1, int(kernel_1 * scale))
        kernel_2 = max(1, int(kernel_2 * scale))
        pool_1 = max(1, int(pool_1 * scale))
        pool_2 = max(1, int(pool_2 * scale))
        n_windows = max(1, int(n_windows * scale))
        tcn_kernel = max(2, int(tcn_kernel * scale))

    condensed_times = int(g.times / (pool_1 * pool_2))
    window_times = condensed_times - n_windows + 1
    stem = (
        2 * f1 * g.times * g.channels * kernel_1
        + 2 * f2 * g.times * g.channels
        + 2 * f2 * (g.times // pool_1) * f2 * kernel_2
    )

    embed_dim = 16
    attention = (
        3 * 2 * window_times * f2 * embed_dim
        + 2 * window_times * embed_dim * f2
        + 2 * 2 * window_times * window_times * embed_dim
    )
    tcn = 4 * (2 * f2 * window_times * f2 * tcn_kernel)
    classifier = 2 * f2 * g.classes
    return stem + n_windows * (attention + tcn + classifier)


def atcnet_flops(g: DatasetGeometry) -> int:
    # Normalize to the audited forward-profile count on 1x128x1153.
    reference = 320_190_000
    raw_reference = _atcnet_raw_flops(GEOMETRY["ds003626"])
    return round(_atcnet_raw_flops(g) * reference / raw_reference)


def _nearest_fast_grid(times: int, window: int, stride: int) -> int:
    if times <= window:
        return window
    offset = times - window
    floor_count = offset // stride + 1
    floor_length = window + (floor_count - 1) * stride
    ceil_count = int(np.ceil(offset / stride)) + 1
    ceil_length = window + (ceil_count - 1) * stride
    return floor_length if times - floor_length <= ceil_length - times else ceil_length


def _fast_raw_flops(g: DatasetGeometry) -> int:
    zones = 8
    dim = 32
    adjusted_times = _nearest_fast_grid(
        g.times, g.fast_window, g.fast_stride
    )
    n_tokens = (
        (adjusted_times - g.fast_window) // g.fast_stride + 1
    )
    conv_times = g.fast_window - 4
    conv_1 = 2 * dim * conv_times * g.channels * 5
    conv_2 = 2 * dim * conv_times * dim * g.channels
    conv_3_4 = 2 * (2 * dim * conv_times * dim * 5) * zones
    spatial_temporal_head = n_tokens * (conv_1 + conv_2 + conv_3_4)

    input_projection = 2 * n_tokens * (dim * zones) * dim
    sequence_length = n_tokens + 1
    transformer_layer = (
        4 * 2 * sequence_length * dim * dim
        + 4 * dim * sequence_length * sequence_length
        + 2 * sequence_length * dim * (2 * dim) * 2
    )
    transformer = 4 * transformer_layer
    classifier = 2 * dim * g.classes
    return spatial_temporal_head + input_projection + transformer + classifier


def fast_flops(g: DatasetGeometry) -> int:
    # Normalize to the audited official-implementation profile on ds003626.
    reference = 941_790_000
    raw_reference = _fast_raw_flops(GEOMETRY["ds003626"])
    return round(_fast_raw_flops(g) * reference / raw_reference)


def labram_flops(g: DatasetGeometry) -> int:
    channel_patches = g.channels * g.baseline_patches
    token_count = channel_patches + 1
    patch_embedding = (
        2 * 8 * channel_patches * 25 * 15
        + 2 * (2 * 8 * channel_patches * 25 * 8 * 3)
    )
    block = (
        2 * token_count * 200 * 600
        + 2 * token_count * 200 * 200
        + 2 * token_count * 200 * 800
        + 2 * token_count * 800 * 200
        + 4 * token_count * token_count * 200
    )
    classifier = 2 * 200 * g.classes
    return patch_embedding + 12 * block + classifier


def eegmamba_flops(g: DatasetGeometry) -> int:
    # The released Mamba2 encoder is linear in channel-patch token count.
    # The constant is fixed by the audited ds003626 theoretical estimate.
    reference = 5_960_000_000
    reference_tokens = (
        GEOMETRY["ds003626"].channels
        * GEOMETRY["ds003626"].baseline_patches
    )
    reference_classifier = 2 * 200 * GEOMETRY["ds003626"].classes
    tokens = g.channels * g.baseline_patches
    classifier = 2 * 200 * g.classes
    return round(
        (reference - reference_classifier) * tokens / reference_tokens
        + classifier
    )


FLOP_FUNCTIONS = {
    "LR-HORNet": lr_hornet_flops,
    "EEGNet": eegnet_flops,
    "ATCNet": atcnet_flops,
    "EEGMamba": eegmamba_flops,
    "FAST": fast_flops,
    "LaBraM": labram_flops,
}

FLOP_METHODS = {
    "LR-HORNet": "analytical operator count",
    "EEGNet": "analytical operator count",
    "ATCNet": "architecture formula normalized to audited ds003626 profile",
    "EEGMamba": "linear-token theoretical estimate normalized to audited ds003626 estimate",
    "FAST": "architecture formula normalized to audited ds003626 profile",
    "LaBraM": "analytical operator count",
}


def load_source_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = pd.read_csv(SUMMARY_ROOT / "fold_level_results.csv")
    accuracy = pd.read_csv(SUMMARY_ROOT / "model_summary_all_datasets.csv")
    parameters = pd.read_csv(SUMMARY_ROOT / "parameter_counts_by_dataset.csv")
    return folds, accuracy, parameters


def validate_sources(
    folds: pd.DataFrame,
    accuracy: pd.DataFrame,
    parameters: pd.DataFrame,
) -> None:
    expected_pairs = {
        (dataset, model)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
    }
    accuracy_pairs = set(zip(accuracy["dataset_key"], accuracy["model"]))
    parameter_pairs = set(zip(parameters["dataset_key"], parameters["model"]))
    if accuracy_pairs != expected_pairs:
        raise ValueError("Accuracy summary does not contain the expected 4x6 design")
    if parameter_pairs != expected_pairs:
        raise ValueError("Parameter summary does not contain the expected 4x6 design")

    for dataset_key, expected_count in EXPECTED_FOLD_COUNTS.items():
        dataset_folds = folds.loc[folds["dataset_key"] == dataset_key]
        counts = dataset_folds.groupby("model")["fold"].nunique().to_dict()
        expected = {model: expected_count for model in MODEL_ORDER}
        if counts != expected:
            raise ValueError(
                f"Unexpected {dataset_key} fold coverage: {counts}; "
                f"expected {expected}"
            )
    if not bool(accuracy["complete"].all()):
        raise ValueError("At least one cross-dataset model summary is incomplete")


def build_fold_table(folds: pd.DataFrame) -> pd.DataFrame:
    table = folds.loc[
        folds["dataset_key"].isin(DATASET_ORDER),
        [
            "dataset_order",
            "dataset_key",
            "model_order",
            "model",
            "fold",
            "accuracy",
        ],
    ].copy()
    table["dataset_label"] = table["dataset_key"].map(DATASET_LABELS)
    table["test_accuracy_pct"] = table["accuracy"] * 100.0
    return table.drop(columns="accuracy").sort_values(
        ["dataset_order", "model_order", "fold"]
    ).reset_index(drop=True)


def build_cross_dataset_table(
    accuracy: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    accuracy_table = accuracy[
        ["dataset_key", "model", "accuracy_mean", "accuracy_sd"]
    ].copy()
    accuracy_table["mean_test_accuracy_pct"] = (
        accuracy_table.pop("accuracy_mean") * 100.0
    )
    accuracy_table["test_accuracy_sd_pct"] = (
        accuracy_table.pop("accuracy_sd") * 100.0
    )
    parameter_table = parameters[
        ["dataset_key", "model", "total_parameters"]
    ].copy()
    table = accuracy_table.merge(
        parameter_table,
        on=["dataset_key", "model"],
        validate="one_to_one",
    )

    rows: list[dict[str, object]] = []
    for dataset_key in DATASET_ORDER:
        geometry = GEOMETRY[dataset_key]
        for model in MODEL_ORDER:
            row = table.loc[
                (table["dataset_key"] == dataset_key)
                & (table["model"] == model)
            ].iloc[0]
            rows.append(
                {
                    "dataset_order": DATASET_ORDER.index(dataset_key) + 1,
                    "dataset_key": dataset_key,
                    "dataset_label": DATASET_LABELS[dataset_key],
                    "model_order": MODEL_ORDER.index(model) + 1,
                    "model": model,
                    "mean_test_accuracy_pct": row["mean_test_accuracy_pct"],
                    "test_accuracy_sd_pct": row["test_accuracy_sd_pct"],
                    "total_parameters": int(row["total_parameters"]),
                    "flops": FLOP_FUNCTIONS[model](geometry),
                    "flops_m": FLOP_FUNCTIONS[model](geometry) / 1e6,
                    "flop_method": FLOP_METHODS[model],
                    "input_channels": geometry.channels,
                    "input_times": geometry.times,
                    "n_classes": geometry.classes,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    folds, accuracy, parameters = load_source_tables()
    validate_sources(folds, accuracy, parameters)
    fold_table = build_fold_table(folds)
    cross_dataset = build_cross_dataset_table(accuracy, parameters)

    fold_path = SCRIPT_DIR / "fold_test_accuracy_all_datasets.csv"
    metric_path = SCRIPT_DIR / "cross_dataset_figure_data.csv"
    fold_table.to_csv(fold_path, index=False, encoding="utf-8-sig")
    cross_dataset.to_csv(metric_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(fold_table)} fold rows to {fold_path}")
    print(f"Wrote {len(cross_dataset)} cross-dataset rows to {metric_path}")
    print("\nFLOPs (MFLOPs):")
    print(
        cross_dataset.pivot(
            index="model", columns="dataset_label", values="flops_m"
        ).reindex(index=MODEL_ORDER, columns=[DATASET_LABELS[key] for key in DATASET_ORDER])
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
