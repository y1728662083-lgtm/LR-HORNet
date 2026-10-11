"""Plot the Section 4.8.1 validation-accuracy operator comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_result_tables import DEFAULT_RESULTS_ROOT, generate_result_tables
from plot_operator_comparison import (
    CHANCE_ACCURACY,
    MODEL_STYLES,
    _bold_ticks,
    configure_style,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = DEFAULT_RESULTS_ROOT


def make_figure(
    run_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    configure_style()
    figure, axis = plt.subplots(figsize=(12.8, 7.6))
    figure.subplots_adjust(left=0.105, right=0.98, top=0.95, bottom=0.25)

    ordered_summary = summary_df.sort_values("model_order")
    x_positions = np.arange(len(ordered_summary), dtype=float)

    axis.axhline(
        CHANCE_ACCURACY,
        color="#89939E",
        linewidth=1.25,
        linestyle=(0, (5, 4)),
        alpha=0.9,
        zorder=0,
    )

    for x_position, row in zip(
        x_positions, ordered_summary.itertuples(index=False)
    ):
        style = MODEL_STYLES[row.model_key]
        mean_value = row.validation_accuracy_mean * 100.0
        sd_value = row.validation_accuracy_sd * 100.0
        run_values = (
            run_df.loc[run_df["model_key"] == row.model_key]
            .sort_values("repeat")["best_validation_accuracy"]
            .to_numpy()
            * 100.0
        )
        jitter = np.linspace(-0.14, 0.14, len(run_values))

        axis.vlines(
            x_position,
            CHANCE_ACCURACY,
            mean_value,
            color=style["color"],
            linewidth=4.2,
            alpha=0.82,
            zorder=1,
        )
        axis.scatter(
            x_position + jitter,
            run_values,
            s=55,
            color=style["color"],
            alpha=0.32,
            edgecolor="white",
            linewidth=0.75,
            zorder=2,
        )
        axis.errorbar(
            x_position,
            mean_value,
            yerr=sd_value,
            fmt="none",
            ecolor=style["color"],
            elinewidth=2.25,
            capsize=7,
            capthick=2.25,
            zorder=3,
        )
        axis.scatter(
            x_position,
            mean_value,
            s=235,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=1.7,
            zorder=4,
        )
        axis.text(
            x_position,
            mean_value + sd_value + 0.62,
            f"{mean_value:.2f} ± {sd_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=style["color"],
            zorder=5,
        )

    axis.set_xlim(-0.55, len(x_positions) - 0.45)
    axis.set_ylim(23.5, 39.2)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([""] * len(x_positions))
    axis.set_yticks(np.arange(24.0, 38.1, 2.0))
    axis.set_xlabel("Model")
    axis.xaxis.set_label_coords(0.5, -0.205)
    axis.set_ylabel("Mean Val Acc (%)", labelpad=14)
    axis.grid(axis="y", color="#D9DEE4", linewidth=0.9, alpha=0.9, zorder=0)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", length=5.2, pad=8)
    _bold_ticks(axis)

    label_transform = axis.get_xaxis_transform()
    for x_position, row in zip(
        x_positions, ordered_summary.itertuples(index=False)
    ):
        for y_position, field_name, operator in (
            (-0.052, "patch", row.patch_operator),
            (-0.105, "deformable", row.offset_operator),
        ):
            axis.text(
                x_position - 0.11,
                y_position,
                field_name,
                transform=label_transform,
                ha="center",
                va="top",
                fontsize=13.2,
                fontweight="bold",
                clip_on=False,
            )
            axis.text(
                x_position + 0.13,
                y_position,
                ":",
                transform=label_transform,
                ha="center",
                va="top",
                fontsize=13.2,
                fontweight="bold",
                clip_on=False,
            )
            axis.text(
                x_position + 0.19,
                y_position,
                operator,
                transform=label_transform,
                ha="left",
                va="top",
                fontsize=13.2,
                fontweight="bold",
                clip_on=False,
            )

    output_paths = {
        "png": output_dir
        / "figure_4_8_1_operator_validation_comparison.png",
        "svg": output_dir
        / "figure_4_8_1_operator_validation_comparison.svg",
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
    figure_paths = make_figure(run_df, summary_df, SCRIPT_DIR)

    print("Validated design: 5 models x 5 repeated runs = 25 results.")
    print("\nMean validation accuracy (mean +/- sample SD):")
    printable = summary_df[
        [
            "model_label",
            "n_runs",
            "validation_accuracy_mean",
            "validation_accuracy_sd",
        ]
    ].copy()
    printable["validation_accuracy_mean"] *= 100.0
    printable["validation_accuracy_sd"] *= 100.0
    print(printable.to_string(index=False, float_format="%.2f"))
    print("\nGenerated files:")
    for path in table_paths.values():
        print(f"  {path}")
    for path in figure_paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
