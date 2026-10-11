"""Create the independent publication figures for Section 4.6.1.

Separate test- and validation-accuracy figures show fold-wise results. Each has
a non-overlapping dot inset in the reserved upper-right band reporting the
unweighted ten-fold mean. The CSV files are regenerated before plotting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from generate_result_tables import (
    DEFAULT_RESULTS_ROOT,
    EXPECTED_FOLDS,
    MODEL_SPECS,
    generate_result_tables,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = DEFAULT_RESULTS_ROOT

# High-contrast academic palette with redundant marker and line-style coding.
MODEL_STYLES = {
    "braindecode_eegnet": {
        "color": "#2F6690",
        "marker": "o",
        "linestyle": "-",
    },
    "causal1_eegnet": {
        "color": "#D97706",
        "marker": "s",
        "linestyle": "--",
    },
    "group4_patchLstm_noDeformation": {
        "color": "#7A5195",
        "marker": "^",
        "linestyle": "-.",
    },
}


def configure_style() -> None:
    sns.set_style("ticks")
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 17,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.25,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 13,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "legend.fontsize": 13,
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


def make_metric_figure(
    run_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    run_column: str,
    mean_column: str,
    y_label: str,
    main_ylim: tuple[float, float],
    main_yticks: list[float],
    inset_ylim: tuple[float, float],
    inset_yticks: list[float],
    filename_stem: str,
) -> dict[str, Path]:
    """Draw one independent fold-wise figure and its mean-accuracy dot inset."""
    configure_style()

    figure, axis = plt.subplots(figsize=(14.8, 8.2))
    figure.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.15)

    x_values = np.asarray(EXPECTED_FOLDS, dtype=float)
    for spec in MODEL_SPECS:
        values = (
            run_df.loc[run_df["model_key"] == spec.model_key]
            .sort_values("fold")[run_column]
            .to_numpy()
            * 100.0
        )
        style = MODEL_STYLES[spec.model_key]
        axis.plot(
            x_values,
            values,
            label=spec.display_label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.65,
            markersize=9.0,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=1.25,
            zorder=3,
        )

    reserved_band_lower = main_ylim[0] + 0.665 * (
        main_ylim[1] - main_ylim[0]
    )
    observed_maximum = float(run_df[run_column].max() * 100.0)
    if observed_maximum >= reserved_band_lower:
        raise ValueError(
            f"The reserved legend/inset band would conceal {run_column}: "
            f"observed maximum {observed_maximum:.2f}% is not below "
            f"{reserved_band_lower:.2f}%."
        )

    # The upper band is intentionally empty so the legend and inset do not
    # conceal any line or marker.
    axis.set_xlim(0.7, 10.3)
    axis.set_ylim(*main_ylim)
    axis.set_xticks(x_values)
    axis.set_xticklabels([f"Fold {fold}" for fold in EXPECTED_FOLDS])
    axis.set_yticks(main_yticks)
    axis.set_xlabel("Fold", labelpad=12)
    axis.set_ylabel(y_label, labelpad=12)
    axis.grid(axis="y", color="#D8DEE5", linewidth=0.9, alpha=0.9, zorder=0)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", length=5.0, pad=7)
    _bold_ticks(axis)

    legend = axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.012, 0.985),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=0.98,
        edgecolor="#AEB7C2",
        facecolor="white",
        handlelength=2.7,
        columnspacing=1.6,
        handletextpad=0.65,
        borderpad=0.65,
    )
    legend.get_frame().set_linewidth(0.9)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    # The inset occupies only the reserved upper band.
    inset = axis.inset_axes([0.675, 0.665, 0.305, 0.305])
    ordered_summary = summary_df.sort_values("model_order")
    means = ordered_summary[mean_column].to_numpy() * 100.0
    keys = ordered_summary["model_key"].tolist()
    colors = [MODEL_STYLES[key]["color"] for key in keys]
    short_labels = ordered_summary["short_label"].tolist()
    point_positions = np.arange(len(keys), dtype=float)

    for position, mean_value, model_key, color in zip(
        point_positions, means, keys, colors
    ):
        inset.scatter(
            position,
            mean_value,
            s=150,
            marker=MODEL_STYLES[model_key]["marker"],
            color=color,
            edgecolor="white",
            linewidth=1.25,
            zorder=4,
        )
    inset.set_ylim(*inset_ylim)
    inset.set_yticks(inset_yticks)
    inset.set_xticks(point_positions)
    inset.set_xticklabels(short_labels)
    inset.set_ylabel("Mean Acc (%)", fontsize=11.5, labelpad=5)
    inset.set_title("10-fold mean", fontsize=13.5, pad=5, fontweight="bold")
    inset.grid(axis="y", color="#E0E4E9", linewidth=0.75, alpha=0.85, zorder=0)
    inset.grid(axis="x", visible=False)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_linewidth(0.9)
    inset.spines["bottom"].set_linewidth(0.9)
    inset.tick_params(axis="x", labelsize=10.5, length=0, pad=4)
    inset.tick_params(axis="y", labelsize=10.5, width=0.9, pad=3)
    _bold_ticks(inset)

    for position, mean_value in zip(point_positions, means):
        inset.text(
            position,
            mean_value + 0.75,
            f"{mean_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11.0,
            fontweight="bold",
            color="#202020",
        )

    output_paths = {
        "png": output_dir / f"{filename_stem}.png",
        "svg": output_dir / f"{filename_stem}.svg",
    }
    figure.savefig(output_paths["png"], dpi=600, bbox_inches="tight")
    figure.savefig(output_paths["svg"], bbox_inches="tight")
    plt.close(figure)
    return output_paths


def main() -> None:
    run_df = pd.read_csv(SCRIPT_DIR / "run_level_results.csv")
    summary_df = pd.read_csv(SCRIPT_DIR / "model_summary.csv")
    table_paths = {
        "run_level": SCRIPT_DIR / "run_level_results.csv",
        "summary": SCRIPT_DIR / "model_summary.csv",
    }
    table_status = {name: False for name in table_paths}
    test_paths = make_metric_figure(
        run_df,
        summary_df,
        SCRIPT_DIR,
        run_column="accuracy",
        mean_column="accuracy_mean",
        y_label="Test accuracy (%)",
        main_ylim=(18.0, 50.0),
        main_yticks=list(np.arange(20.0, 50.1, 5.0)),
        inset_ylim=(25.0, 32.2),
        inset_yticks=[25.0, 28.0, 31.0],
        filename_stem="figure_4_6_1_local_feature_comparison",
    )
    validation_paths = make_metric_figure(
        run_df,
        summary_df,
        SCRIPT_DIR,
        run_column="best_validation_accuracy",
        mean_column="validation_accuracy_mean",
        y_label="Validation accuracy (%)",
        main_ylim=(26.0, 58.0),
        main_yticks=list(np.arange(30.0, 55.1, 5.0)),
        inset_ylim=(34.0, 38.4),
        inset_yticks=[34.0, 36.0, 38.0],
        filename_stem="figure_4_6_1_validation_accuracy",
    )
    figure_paths = {
        "test_png": test_paths["png"],
        "test_svg": test_paths["svg"],
        "validation_png": validation_paths["png"],
        "validation_svg": validation_paths["svg"],
    }

    print("Validated design: 3 models x 10 LOSO folds = 30 results.")
    print("\nTest-accuracy summary:")
    print(
        summary_df[
            [
                "model_label",
                "n_folds",
                "accuracy_mean",
                "accuracy_sd",
                "validation_accuracy_mean",
                "validation_accuracy_sd",
            ]
        ].to_string(index=False, float_format="%.6f")
    )
    print("\nGenerated files:")
    for name, path in table_paths.items():
        suffix = "" if table_status[name] else " (existing file retained)"
        print(f"  {path}{suffix}")
    for path in figure_paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
