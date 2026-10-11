"""Create the Section 4.7.1 validation-accuracy comparison figure.

The main chart reports the best validation accuracy for every LOSO fold. A
lollipop inset in the reserved upper-right band reports the unweighted
ten-fold mean. Source CSV files are regenerated from the raw experiment files
before plotting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_result_tables import (
    DEFAULT_RESULTS_ROOT,
    EXPECTED_FOLDS,
    MODEL_SPECS,
    generate_result_tables,
)
from plot_offset_strategy_comparison import (
    MODEL_STYLES,
    _bold_ticks,
    configure_style,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = DEFAULT_RESULTS_ROOT
INSET_BASELINE = 35.5


def make_figure(
    run_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    configure_style()
    figure, axis = plt.subplots(figsize=(14.8, 8.2))
    figure.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.15)

    x_values = np.asarray(EXPECTED_FOLDS, dtype=float)
    for spec in MODEL_SPECS:
        values = (
            run_df.loc[run_df["model_key"] == spec.model_key]
            .sort_values("fold")["best_validation_accuracy"]
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
            markersize=8.8,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=3,
        )

    axis.set_xlim(0.7, 10.3)
    axis.set_ylim(25.0, 55.0)
    axis.set_xticks(x_values)
    axis.set_xticklabels([f"Fold {fold}" for fold in EXPECTED_FOLDS])
    axis.set_yticks(np.arange(25.0, 55.1, 5.0))
    axis.set_xlabel("Fold", labelpad=12)
    axis.set_ylabel("Validation accuracy (%)", labelpad=12)
    axis.grid(axis="y", color="#D8DEE5", linewidth=0.9, alpha=0.9, zorder=0)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", length=5.0, pad=7)
    _bold_ticks(axis)

    legend = axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.012, 0.985),
        ncol=4,
        frameon=True,
        fancybox=False,
        framealpha=0.98,
        edgecolor="#AEB7C2",
        facecolor="white",
        handlelength=2.4,
        columnspacing=1.25,
        handletextpad=0.55,
        borderpad=0.65,
    )
    legend.get_frame().set_linewidth(0.9)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    # The expanded main-axis upper range reserves a clear band for the inset.
    inset = axis.inset_axes([0.645, 0.665, 0.335, 0.305])
    ordered_summary = summary_df.sort_values("model_order")
    means = ordered_summary["validation_accuracy_mean"].to_numpy() * 100.0
    keys = ordered_summary["model_key"].tolist()
    labels = [
        next(spec.short_label for spec in MODEL_SPECS if spec.model_key == key)
        for key in keys
    ]
    point_positions = np.arange(len(keys), dtype=float)

    inset.axhline(
        INSET_BASELINE,
        color="#B7BEC5",
        linewidth=0.85,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    for position, mean_value, model_key in zip(
        point_positions, means, keys
    ):
        style = MODEL_STYLES[model_key]
        inset.vlines(
            position,
            INSET_BASELINE,
            mean_value,
            color=style["color"],
            linewidth=2.25,
            zorder=2,
        )
        inset.scatter(
            position,
            mean_value,
            s=125,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=1.15,
            zorder=4,
        )
        inset.text(
            position,
            mean_value + 0.09,
            f"{mean_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10.2,
            fontweight="bold",
            color="#202020",
        )

    inset.set_xlim(-0.45, len(keys) - 0.55)
    inset.set_ylim(35.4, 37.0)
    inset.set_yticks([35.5, 36.0, 36.5, 37.0])
    inset.set_xticks(point_positions)
    inset.set_xticklabels(labels)
    inset.set_ylabel("Mean Acc (%)", fontsize=11.2, labelpad=4)
    inset.set_title("10-fold mean", fontsize=13.5, pad=5, fontweight="bold")
    inset.grid(axis="y", color="#E0E4E9", linewidth=0.75, alpha=0.85, zorder=0)
    inset.grid(axis="x", visible=False)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_linewidth(0.9)
    inset.spines["bottom"].set_linewidth(0.9)
    inset.tick_params(axis="x", labelsize=9.8, length=0, pad=4)
    inset.tick_params(axis="y", labelsize=10.3, width=0.9, pad=3)
    _bold_ticks(inset)

    output_paths = {
        "png": output_dir
        / "figure_4_7_1_offset_strategy_validation_comparison.png",
        "svg": output_dir
        / "figure_4_7_1_offset_strategy_validation_comparison.svg",
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
    figure_paths = make_figure(run_df, summary_df, SCRIPT_DIR)

    print("Validated design: 4 models x 10 LOSO folds = 40 results.")
    print("\nValidation-accuracy summary:")
    print(
        summary_df[
            [
                "model_label",
                "n_folds",
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
