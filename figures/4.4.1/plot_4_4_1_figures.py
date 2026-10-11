"""Generate independent publication figures for Section 4.4.1.

Outputs are PNG (600 dpi) and editable SVG.  All panels use the same model
colors, markers, and hatches so the figures can be assembled later without
changing their visual encoding.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from build_figure_data import (
    DATASET_LABELS,
    DATASET_ORDER,
    MODEL_ORDER,
    SCRIPT_DIR,
    build_cross_dataset_table,
    build_fold_table,
    load_source_tables,
    validate_sources,
)


MODEL_STYLES = {
    "LR-HORNet": {"color": "#B2182B", "marker": "o", "linestyle": "-", "hatch": ""},
    "EEGNet": {"color": "#2166AC", "marker": "s", "linestyle": "--", "hatch": "///"},
    "ATCNet": {"color": "#1B9E77", "marker": "D", "linestyle": "-.", "hatch": "\\\\"},
    "EEGMamba": {"color": "#756BB1", "marker": "^", "linestyle": ":", "hatch": "xx"},
    "FAST": {"color": "#E08214", "marker": "P", "linestyle": (0, (5, 1.4)), "hatch": ".."},
    "LaBraM": {"color": "#5B5B5B", "marker": "X", "linestyle": (0, (3, 1, 1, 1)), "hatch": "++"},
}

BAR_STEP = 0.0625
DATASET_GROUP_SPACING = 0.50


def configure_style() -> None:
    sns.set_style("ticks")
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "font.size": 17,
            "axes.labelsize": 23,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.45,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "xtick.major.width": 1.30,
            "ytick.major.width": 1.30,
            "legend.fontsize": 15,
            "text.color": "#202020",
            "axes.labelcolor": "#202020",
            "axes.edgecolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "svg.fonttype": "none",
        }
    )


def _bold_ticks(axis: plt.Axes) -> None:
    plt.setp(axis.get_xticklabels(), fontweight="bold")
    plt.setp(axis.get_yticklabels(), fontweight="bold")


def _clean_axis(axis: plt.Axes, *, grid: str = "y") -> None:
    axis.grid(axis=grid, color="#D9E0E6", linewidth=0.9, alpha=0.9, zorder=0)
    if grid == "y":
        axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", length=5.0, pad=7)
    _bold_ticks(axis)


def _legend_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=MODEL_STYLES[model]["color"],
            edgecolor="#303030",
            linewidth=0.8,
            hatch=MODEL_STYLES[model]["hatch"],
            label=model,
        )
        for model in MODEL_ORDER
    ]


def _add_bar_legend(axis: plt.Axes) -> None:
    legend = axis.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#AEB7C2",
        facecolor="white",
        columnspacing=1.5,
        handletextpad=0.55,
        borderpad=0.65,
    )
    legend.get_frame().set_linewidth(0.9)
    for text in legend.get_texts():
        text.set_fontweight("bold")


def _save(
    figure: plt.Figure,
    stem: str,
    *,
    preserve_canvas: bool = False,
) -> dict[str, Path]:
    paths = {
        "png": SCRIPT_DIR / f"{stem}.png",
        "svg": SCRIPT_DIR / f"{stem}.svg",
    }
    bbox = None if preserve_canvas else "tight"
    figure.savefig(paths["png"], dpi=600, bbox_inches=bbox)
    figure.savefig(paths["svg"], bbox_inches=bbox)
    plt.close(figure)
    return paths


def _fold_axis_limits(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    data_min = float(np.nanmin(values))
    data_max = float(np.nanmax(values))
    data_span = data_max - data_min
    tick_step = 5.0 if data_span <= 35.0 else 10.0
    margin = max(2.0, data_span * 0.06)
    lower = max(0.0, np.floor((data_min - margin) / tick_step) * tick_step)
    upper = np.ceil((data_max + margin) / tick_step) * tick_step
    if upper - lower < 4.0 * tick_step:
        upper = lower + 4.0 * tick_step
    if upper - data_max < 0.5 * tick_step:
        upper += tick_step
    upper = min(110.0, upper)
    ticks = np.arange(lower, upper + 0.1, tick_step)
    return lower, upper, ticks


def _spread_fold_labels(
    values: list[tuple[str, float]],
    lower: float,
    upper: float,
) -> dict[str, float]:
    """Spread same-fold labels vertically while keeping them close to points."""
    sorted_values = sorted(values, key=lambda item: item[1])
    gap = (upper - lower) * 0.032
    lower_bound = lower + gap * 0.75
    upper_bound = upper - gap * 0.75
    positions: list[float] = []
    for _, value in sorted_values:
        position = max(value, positions[-1] + gap) if positions else value
        positions.append(position)

    if positions[-1] > upper_bound:
        shift = positions[-1] - upper_bound
        positions = [position - shift for position in positions]
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(positions[index], positions[index + 1] - gap)
    if positions[0] < lower_bound:
        shift = lower_bound - positions[0]
        positions = [position + shift for position in positions]

    return {
        model: position
        for (model, _), position in zip(sorted_values, positions, strict=True)
    }


def _annotate_fold_values(
    axis: plt.Axes,
    dataset_rows: pd.DataFrame,
    folds: np.ndarray,
    y_min: float,
    y_max: float,
) -> None:
    x_offsets = {
        "LR-HORNet": -0.12,
        "EEGNet": -0.07,
        "ATCNet": -0.02,
        "EEGMamba": 0.03,
        "FAST": 0.08,
        "LaBraM": 0.13,
    }
    for fold in folds:
        fold_rows = dataset_rows.loc[dataset_rows["fold"] == fold].set_index("model")
        values = [
            (model, float(fold_rows.loc[model, "test_accuracy_pct"]))
            for model in MODEL_ORDER
        ]
        label_positions = _spread_fold_labels(values, y_min, y_max)
        for model, value in values:
            label_y = label_positions[model]
            label_x = float(fold) + x_offsets[model]
            color = MODEL_STYLES[model]["color"]
            if abs(label_y - value) > (y_max - y_min) * 0.010:
                axis.plot(
                    [float(fold), label_x],
                    [value, label_y],
                    color=color,
                    linewidth=0.75,
                    alpha=0.62,
                    zorder=4,
                )
            axis.text(
                label_x,
                label_y,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=12.5,
                fontweight="bold",
                color=color,
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )


def plot_fold_accuracy(
    fold_table: pd.DataFrame,
    dataset_key: str,
) -> dict[str, Path]:
    dataset_rows = fold_table.loc[fold_table["dataset_key"] == dataset_key].copy()
    if dataset_rows.empty:
        raise ValueError(f"No fold-level results found for {dataset_key}")

    figure, axis = plt.subplots(figsize=(17.8, 10.4))
    figure.subplots_adjust(left=0.090, right=0.988, top=0.90, bottom=0.14)
    folds = np.sort(dataset_rows["fold"].unique()).astype(int)

    for model in MODEL_ORDER:
        model_rows = dataset_rows.loc[dataset_rows["model"] == model].sort_values(
            "fold"
        )
        if model_rows["fold"].tolist() != folds.tolist():
            raise ValueError(f"Incomplete {dataset_key} trajectory for {model}")
        style = MODEL_STYLES[model]
        axis.plot(
            folds,
            model_rows["test_accuracy_pct"].to_numpy(dtype=float),
            label=model,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=3.1,
            markersize=10.5,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=3,
        )

    y_min, y_max, y_ticks = _fold_axis_limits(
        dataset_rows["test_accuracy_pct"].to_numpy(dtype=float)
    )
    axis.set_xlim(float(folds.min()) - 0.35, float(folds.max()) + 0.35)
    axis.set_ylim(y_min, y_max)
    axis.set_xticks(folds)
    axis.set_xticklabels([f"Fold {value}" for value in folds])
    axis.set_yticks(y_ticks)
    axis.set_xlabel("Fold", labelpad=12)
    axis.set_ylabel("Test Acc (%)", labelpad=12)
    _clean_axis(axis)
    _annotate_fold_values(axis, dataset_rows, folds, y_min, y_max)

    legend = axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.012, 1.015),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=0.98,
        edgecolor="#AEB7C2",
        facecolor="white",
        handlelength=2.8,
        columnspacing=1.45,
        handletextpad=0.60,
        borderpad=0.65,
    )
    legend.get_frame().set_linewidth(0.9)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    axis.text(
        0.982,
        0.965,
        DATASET_LABELS[dataset_key],
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=19,
        fontweight="bold",
        color="#4A4A4A",
    )
    output_suffix = {
        "ds003626": "ds003626",
        "bimodal_inner_speech": "bimodal",
        "directional_words_russian": "directional_words_russian",
        "directional_words_spanish": "directional_words_spanish",
    }[dataset_key]
    return _save(
        figure,
        f"figure_4_4_1_fold_test_accuracy_{output_suffix}",
    )


def _dataset_tick_labels() -> list[str]:
    return [
        "ds003626",
        "Bimodal",
        "Directional Words\nRussian",
        "Directional Words\nSpanish",
    ]


def _draw_grouped_bars(
    axis: plt.Axes,
    data: pd.DataFrame,
    value_column: str,
    *,
    error_column: str | None = None,
) -> None:
    x_positions = (
        np.arange(len(DATASET_ORDER), dtype=float) * DATASET_GROUP_SPACING
    )
    width = BAR_STEP
    offsets = (np.arange(len(MODEL_ORDER)) - (len(MODEL_ORDER) - 1) / 2) * width

    for model_index, model in enumerate(MODEL_ORDER):
        subset = (
            data.loc[data["model"] == model]
            .set_index("dataset_key")
            .reindex(DATASET_ORDER)
        )
        values = subset[value_column].to_numpy(dtype=float)
        errors = (
            subset[error_column].to_numpy(dtype=float)
            if error_column is not None
            else None
        )
        style = MODEL_STYLES[model]
        axis.bar(
            x_positions + offsets[model_index],
            values,
            width=width * 0.90,
            yerr=errors,
            color=style["color"],
            edgecolor="#303030",
            linewidth=0.75,
            hatch=style["hatch"],
            error_kw={
                "ecolor": style["color"],
                "elinewidth": 1.55,
                "capsize": 3.2,
                "capthick": 1.55,
            },
            zorder=3,
        )

    axis.set_xlim(
        float(x_positions[0]) - 0.27,
        float(x_positions[-1]) + 0.27,
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(_dataset_tick_labels(), linespacing=1.05)
    _bold_ticks(axis)


def _bar_locations() -> tuple[np.ndarray, float, np.ndarray]:
    x_positions = (
        np.arange(len(DATASET_ORDER), dtype=float) * DATASET_GROUP_SPACING
    )
    width = BAR_STEP
    offsets = (np.arange(len(MODEL_ORDER)) - (len(MODEL_ORDER) - 1) / 2) * width
    return x_positions, width, offsets


def _format_parameter_count(value: float) -> str:
    if value >= 1.0e6:
        return f"{value / 1.0e6:.3f} M"
    return f"{value / 1.0e3:.2f} K"


def _format_flops(value: float) -> str:
    if value >= 1.0e9:
        return f"{value / 1.0e9:.2f} G"
    return f"{value / 1.0e6:.2f} M"


def _annotate_grouped_bars(
    axis: plt.Axes,
    data: pd.DataFrame,
    value_column: str,
    formatter,
    *,
    error_column: str | None = None,
    logarithmic: bool = False,
) -> None:
    x_positions, _, offsets = _bar_locations()
    for model_index, model in enumerate(MODEL_ORDER):
        subset = (
            data.loc[data["model"] == model]
            .set_index("dataset_key")
            .reindex(DATASET_ORDER)
        )
        values = subset[value_column].to_numpy(dtype=float)
        errors = (
            subset[error_column].to_numpy(dtype=float)
            if error_column is not None
            else np.zeros_like(values)
        )
        for dataset_index, (value, error) in enumerate(zip(values, errors, strict=True)):
            label = formatter(value, error) if error_column is not None else formatter(value)
            label_y = value * 1.12 if logarithmic else value + error + 1.2
            axis.text(
                x_positions[dataset_index] + offsets[model_index],
                label_y,
                label,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=13.2,
                fontweight="bold",
                color=MODEL_STYLES[model]["color"],
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )


def plot_mean_accuracy(data: pd.DataFrame) -> dict[str, Path]:
    figure, axis = plt.subplots(figsize=(12.0, 12.0))
    figure.subplots_adjust(left=0.125, right=0.975, top=0.82, bottom=0.14)
    _draw_grouped_bars(
        axis,
        data,
        "mean_test_accuracy_pct",
        error_column="test_accuracy_sd_pct",
    )
    axis.set_ylim(0.0, 105.0)
    axis.set_yticks(np.arange(0, 101, 10))
    axis.set_ylabel("Mean Test Acc (%)", labelpad=12)
    _clean_axis(axis)
    _annotate_grouped_bars(
        axis,
        data,
        "mean_test_accuracy_pct",
        lambda value, error: f"{value:.2f}±{error:.2f}",
        error_column="test_accuracy_sd_pct",
    )
    _add_bar_legend(axis)
    return _save(
        figure,
        "figure_4_4_1_mean_test_accuracy",
        preserve_canvas=True,
    )


def plot_parameters(data: pd.DataFrame) -> dict[str, Path]:
    figure, axis = plt.subplots(figsize=(12.0, 12.0))
    figure.subplots_adjust(left=0.135, right=0.975, top=0.82, bottom=0.14)
    _draw_grouped_bars(axis, data, "total_parameters")
    axis.set_yscale("log")
    axis.set_ylim(1.0e3, 1.0e8)
    axis.set_yticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    axis.set_yticklabels(["1 K", "10 K", "100 K", "1 M", "10 M", "100 M"])
    axis.set_ylabel("Parameters", labelpad=12)
    _clean_axis(axis)
    _annotate_grouped_bars(
        axis,
        data,
        "total_parameters",
        _format_parameter_count,
        logarithmic=True,
    )
    _add_bar_legend(axis)
    return _save(
        figure,
        "figure_4_4_1_parameters",
        preserve_canvas=True,
    )


def plot_flops(data: pd.DataFrame) -> dict[str, Path]:
    figure, axis = plt.subplots(figsize=(12.0, 12.0))
    figure.subplots_adjust(left=0.135, right=0.975, top=0.82, bottom=0.14)
    _draw_grouped_bars(axis, data, "flops")
    axis.set_yscale("log")
    axis.set_ylim(1.0e6, 1.0e11)
    axis.set_yticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11])
    axis.set_yticklabels(["1 M", "10 M", "100 M", "1 G", "10 G", "100 G"])
    axis.set_ylabel("FLOPs", labelpad=12)
    _clean_axis(axis)
    _annotate_grouped_bars(
        axis,
        data,
        "flops",
        _format_flops,
        logarithmic=True,
    )
    _add_bar_legend(axis)
    return _save(
        figure,
        "figure_4_4_1_flops",
        preserve_canvas=True,
    )


def main() -> None:
    configure_style()
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    fold_table = pd.read_csv(SCRIPT_DIR / "fold_test_accuracy_all_datasets.csv")
    cross_dataset = pd.read_csv(SCRIPT_DIR / "cross_dataset_figure_data.csv")

    outputs: dict[str, dict[str, Path]] = {
        **{
            f"fold accuracy ({DATASET_LABELS[dataset_key]})": plot_fold_accuracy(
                fold_table, dataset_key
            )
            for dataset_key in DATASET_ORDER
        },
        "mean accuracy": plot_mean_accuracy(cross_dataset),
        "parameters": plot_parameters(cross_dataset),
        "FLOPs": plot_flops(cross_dataset),
    }
    print("Generated seven independent figures with a shared model palette:")
    for figure_name, paths in outputs.items():
        print(f"  {figure_name}:")
        for path in paths.values():
            print(f"    {path}")


if __name__ == "__main__":
    main()
