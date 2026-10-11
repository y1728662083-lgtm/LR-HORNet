"""Plot the selected 10-fold test accuracy of the LSTM-CfC model."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from generate_lstm_cfc_10fold_tables import (
    DEFAULT_RESULTS_ROOT,
    generate_tables,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = DEFAULT_RESULTS_ROOT

# Ordered, non-identical blue-teal shades keep the ten folds harmonious while
# preserving visible category boundaries.
FOLD_COLORS = [
    "#DCEAF5",
    "#C7DDEC",
    "#B2D0E2",
    "#9AC2D6",
    "#82B4C9",
    "#69A5BB",
    "#5595AA",
    "#43849A",
    "#347286",
    "#295E72",
]
MEAN_COLOR = "#A23B4A"


def configure_style() -> None:
    sns.set_style("ticks")
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "font.size": 15,
            "axes.labelsize": 19,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.35,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14.5,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
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


def make_figure(fold_df, summary_df, output_dir: Path) -> dict[str, Path]:
    configure_style()
    figure, axis = plt.subplots(figsize=(13.2, 7.4))
    figure.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.155)

    folds = fold_df["fold"].to_numpy(dtype=float)
    accuracies = fold_df["test_accuracy"].to_numpy(dtype=float) * 100.0
    mean_value = float(summary_df.loc[0, "test_accuracy_mean"] * 100.0)
    sd_value = float(summary_df.loc[0, "test_accuracy_sd"] * 100.0)

    bars = axis.bar(
        folds,
        accuracies,
        width=0.42,
        color=FOLD_COLORS,
        edgecolor="#355769",
        linewidth=1.15,
        zorder=3,
    )
    for bar, fold, accuracy in zip(bars, folds, accuracies):
        label_y = accuracy + 0.55
        if int(fold) in {2, 9}:
            label_y = mean_value + 0.25
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            label_y,
            f"{accuracy:.2f}",
            ha="center",
            va="bottom",
            fontsize=12.8,
            fontweight="bold",
            color="#263D4A",
            zorder=6,
        )

    line_left, line_right = 0.48, 10.52
    axis.plot(
        [line_left, line_right],
        [mean_value, mean_value],
        color=MEAN_COLOR,
        linewidth=3.0,
        linestyle=(0, (8, 4)),
        dash_capstyle="round",
        zorder=5,
    )
    axis.text(
        line_right + 0.16,
        mean_value,
        f"{mean_value:.2f} ± {sd_value:.2f}%",
        ha="left",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=MEAN_COLOR,
        bbox={
            "boxstyle": "square,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.92,
        },
        zorder=6,
    )

    axis.set_xlim(0.35, 12.25)
    axis.set_ylim(0.0, 42.0)
    axis.set_xticks(folds)
    axis.set_xticklabels([str(int(fold)) for fold in folds])
    axis.set_yticks(np.arange(0.0, 40.1, 5.0))
    axis.set_xlabel("Fold", labelpad=12)
    axis.set_ylabel("Test Acc (%)", labelpad=14)
    axis.grid(axis="y", color="#D9DEE4", linewidth=0.9, alpha=0.9, zorder=0)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", length=5.2, pad=8)
    _bold_ticks(axis)

    output_paths = {
        "png": output_dir / "figure_4_8_1_lstm_cfc_10fold_test_accuracy.png",
        "svg": output_dir / "figure_4_8_1_lstm_cfc_10fold_test_accuracy.svg",
    }
    figure.savefig(output_paths["png"], dpi=600, bbox_inches="tight")
    figure.savefig(output_paths["svg"], bbox_inches="tight")
    plt.close(figure)
    return output_paths


def main() -> None:
    fold_df, summary_df, table_paths = generate_tables(
        RESULTS_ROOT, SCRIPT_DIR
    )
    figure_paths = make_figure(fold_df, summary_df, SCRIPT_DIR)

    mean_value = summary_df.loc[0, "test_accuracy_mean"] * 100.0
    sd_value = summary_df.loc[0, "test_accuracy_sd"] * 100.0
    print("Validated design: 10 selected LOSO-fold results.")
    print(f"Mean test accuracy: {mean_value:.4f} +/- {sd_value:.4f}%")
    print("\nGenerated files:")
    for path in table_paths.values():
        print(f"  {path}")
    for path in figure_paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
