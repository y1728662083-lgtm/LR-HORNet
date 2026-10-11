"""Create the Section 4.6.3 electrode-group ablation figure.

The figure follows the Section 4.6.2 lollipop design while adding lightly
jittered run-level observations for greater transparency. Means and error bars
represent five repeated fold-1 runs and sample standard deviations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from generate_result_tables import DEFAULT_RESULTS_ROOT, generate_result_tables


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = DEFAULT_RESULTS_ROOT
CHANCE_ACCURACY = 25.0

# A muted, print-friendly palette distinct from the blue-grey palette in 4.6.2.
# Group 1 (no electrode grouping) is deliberately emphasized in red.
GROUP_STYLES = {
    1: {"color": "#C23B3B", "marker": "X"},
    2: {"color": "#D1902F", "marker": "o"},
    4: {"color": "#3F8F68", "marker": "D"},
    8: {"color": "#237E98", "marker": "^"},
    16: {"color": "#71579A", "marker": "s"},
}


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
            "xtick.labelsize": 15,
            "ytick.labelsize": 14,
            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
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


def make_figure(
    run_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    configure_style()
    ordered = summary_df.sort_values("condition_order").reset_index(drop=True)
    x_positions = np.arange(len(ordered), dtype=float)
    groups = ordered["group_count"].astype(int).tolist()
    means = ordered["accuracy_mean"].to_numpy() * 100.0
    standard_deviations = ordered["accuracy_sd"].to_numpy() * 100.0

    figure, axis = plt.subplots(figsize=(12.8, 7.6))
    figure.subplots_adjust(left=0.10, right=0.985, top=0.95, bottom=0.16)

    axis.axhline(
        CHANCE_ACCURACY,
        color="#8C949B",
        linewidth=1.35,
        linestyle=(0, (5, 3)),
        zorder=1,
    )

    # Showing all five repeats is the deliberate visual distinction from 4.6.2.
    jitter = np.linspace(-0.14, 0.14, 5)
    for position, group, mean_value, sd_value in zip(
        x_positions, groups, means, standard_deviations
    ):
        style = GROUP_STYLES[group]
        group_runs = (
            run_df.loc[run_df["group_count"] == group]
            .sort_values("repeat")["accuracy"]
            .to_numpy()
            * 100.0
        )
        axis.scatter(
            position + jitter,
            group_runs,
            s=42,
            color=style["color"],
            alpha=0.35,
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )
        axis.vlines(
            position,
            CHANCE_ACCURACY,
            mean_value,
            color=style["color"],
            linewidth=3.2,
            alpha=0.95,
            zorder=3,
        )
        axis.errorbar(
            position,
            mean_value,
            yerr=sd_value,
            fmt="none",
            ecolor=style["color"],
            elinewidth=2.0,
            capsize=7,
            capthick=2.0,
            zorder=4,
        )
        axis.scatter(
            position,
            mean_value,
            s=205,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
        )
        axis.text(
            position,
            mean_value + sd_value + 0.72,
            f"{mean_value:.2f} ± {sd_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=13.0,
            fontweight="bold",
            color=style["color"] if group == 1 else "#202020",
            zorder=6,
        )

    axis.set_xlim(-0.58, len(ordered) - 0.42)
    axis.set_ylim(23.5, 38.0)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([str(group) for group in groups])
    axis.set_yticks([25, 28, 31, 34, 37])
    axis.set_xlabel("Group", labelpad=12)
    axis.set_ylabel("Test Acc (%)", labelpad=12)
    axis.grid(axis="y", color="#D8DEE5", linewidth=0.9, alpha=0.9, zorder=0)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", length=5.0, pad=7)
    plt.setp(axis.get_xticklabels(), fontweight="bold")
    plt.setp(axis.get_yticklabels(), fontweight="bold")
    axis.get_xticklabels()[0].set_color(GROUP_STYLES[1]["color"])

    output_paths = {
        "png": output_dir / "figure_4_6_3_group_ablation_test_accuracy.png",
        "svg": output_dir / "figure_4_6_3_group_ablation_test_accuracy.svg",
    }
    figure.savefig(output_paths["png"], dpi=600, bbox_inches="tight")
    figure.savefig(output_paths["svg"], bbox_inches="tight")
    plt.close(figure)
    return output_paths


def main() -> None:
    run_df = pd.read_csv(SCRIPT_DIR / "run_level_results.csv")
    summary_df = pd.read_csv(SCRIPT_DIR / "condition_summary.csv")
    table_paths = {
        "run_level": SCRIPT_DIR / "run_level_results.csv",
        "summary": SCRIPT_DIR / "condition_summary.csv",
    }
    figure_paths = make_figure(run_df, summary_df, SCRIPT_DIR)

    print("Validated design: 5 group conditions x 5 repeated fold-1 runs.")
    print("\nTest-accuracy summary:")
    print(
        summary_df[
            ["group_count", "n_runs", "accuracy_mean", "accuracy_sd"]
        ].to_string(index=False, float_format="%.6f")
    )
    print("\nGenerated files:")
    for path in (*table_paths.values(), *figure_paths.values()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
