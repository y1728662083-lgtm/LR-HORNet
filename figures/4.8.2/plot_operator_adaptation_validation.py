"""Create the Section 4.8.2 fold-wise validation-accuracy figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_result_tables import (
    EXPECTED_FOLDS,
    MODEL_SPECS,
    generate_result_tables,
)
from plot_operator_adaptation import (
    MODEL_STYLES,
    _bold_ticks,
    configure_style,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def make_figure(run_df, summary_df, output_dir: Path) -> dict[str, Path]:
    configure_style()
    figure, axis = plt.subplots(figsize=(14.8, 8.2))
    figure.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.15)

    x_values = np.asarray(EXPECTED_FOLDS, dtype=float)
    for spec in MODEL_SPECS:
        values = (
            run_df.loc[run_df["model_key"] == spec.model_key]
            .sort_values("fold")["best_validation_accuracy"]
            .to_numpy(dtype=float)
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
            linewidth=2.6,
            markersize=8.7,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=3,
        )

    main_ylim = (26.0, 58.0)
    reserved_band_lower = main_ylim[0] + 0.665 * (
        main_ylim[1] - main_ylim[0]
    )
    observed_maximum = float(
        run_df["best_validation_accuracy"].max() * 100.0
    )
    if observed_maximum >= reserved_band_lower:
        raise ValueError(
            "Legend/inset band would conceal a fold result: "
            f"maximum={observed_maximum:.2f}%, "
            f"reserved band starts at {reserved_band_lower:.2f}%."
        )

    axis.set_xlim(0.7, 10.3)
    axis.set_ylim(*main_ylim)
    axis.set_xticks(x_values)
    axis.set_xticklabels([f"Fold {fold}" for fold in EXPECTED_FOLDS])
    axis.set_yticks(np.arange(30.0, 55.1, 5.0))
    axis.set_xlabel("Fold", labelpad=12)
    axis.set_ylabel("Val Acc (%)", labelpad=12)
    axis.grid(axis="y", color="#D8DEE5", linewidth=0.9, alpha=0.9, zorder=0)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", length=5.0, pad=7)
    _bold_ticks(axis)

    legend = axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.012, 0.985),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=0.98,
        edgecolor="#AEB7C2",
        facecolor="white",
        handlelength=2.8,
        columnspacing=1.5,
        handletextpad=0.65,
        borderpad=0.65,
    )
    legend.get_frame().set_linewidth(0.9)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    inset = axis.inset_axes([0.605, 0.665, 0.375, 0.305])
    ordered_summary = summary_df.sort_values("model_order")
    means = (
        ordered_summary["validation_accuracy_mean"].to_numpy(dtype=float)
        * 100.0
    )
    keys = ordered_summary["model_key"].tolist()
    short_labels = ordered_summary["short_label"].tolist()
    positions = np.arange(len(keys), dtype=float)

    for position, mean_value, model_key in zip(positions, means, keys):
        style = MODEL_STYLES[model_key]
        inset.scatter(
            position,
            mean_value,
            s=135,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )
        inset.text(
            position,
            mean_value + 0.14,
            f"{mean_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#202020",
            zorder=5,
        )

    inset.set_xlim(-0.62, 4.25)
    inset.set_ylim(34.0, 37.0)
    inset.set_yticks([34.0, 35.0, 36.0, 37.0])
    inset.set_xticks(positions)
    inset.set_xticklabels(short_labels)
    inset.set_ylabel("Mean Acc (%)", fontsize=11.5, labelpad=5)
    inset.set_title("10-fold mean", fontsize=13.5, pad=5, fontweight="bold")
    inset.grid(axis="y", color="#E0E4E9", linewidth=0.75, alpha=0.85, zorder=0)
    inset.grid(axis="x", visible=False)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_linewidth(0.9)
    inset.spines["bottom"].set_linewidth(0.9)
    inset.tick_params(axis="x", labelsize=9.2, length=0, pad=4)
    inset.tick_params(axis="y", labelsize=10.5, width=0.9, pad=3)
    _bold_ticks(inset)

    output_paths = {
        "png": output_dir
        / "figure_4_8_2_operator_adaptation_validation_accuracy.png",
        "svg": output_dir
        / "figure_4_8_2_operator_adaptation_validation_accuracy.svg",
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

    print("Validated design: 5 models x 10 LOSO folds = 50 results.")
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
    for path in table_paths.values():
        print(f"  {path}")
    for path in figure_paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
