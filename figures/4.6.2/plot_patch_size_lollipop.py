"""Create the Section 4.6.2 patch-size lollipop figures.

Independent test- and validation-accuracy figures report fold-1 means across
five repeated runs; error bars are sample standard deviations. CSV files are
regenerated from the archived raw results before PNG and SVG export.
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

# Derived from the blue-grey-to-navy palette used in the Section 4.5 figure.
CONDITION_STYLES = {
    "patch_0008": {"color": "#8798A5", "marker": "o"},
    "patch_0016": {"color": "#7893A9", "marker": "s"},
    "patch_0032": {"color": "#6F8EAA", "marker": "D"},
    "patch_0064": {"color": "#527FA6", "marker": "^"},
    "patch_0128": {"color": "#3F75A8", "marker": "v"},
    "patch_0256": {"color": "#2F6FA9", "marker": "P"},
    "patch_0512": {"color": "#0B3C6F", "marker": "h"},
    "full_sequence_no_patch": {"color": "#B2182B", "marker": "X"},
}


def configure_style() -> None:
    sns.set_style("ticks")
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "font.size": 14,
            "axes.labelsize": 17,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.25,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
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


def make_metric_figure(
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    mean_column: str,
    sd_column: str,
    y_label: str,
    y_limits: tuple[float, float],
    y_ticks: list[float],
    filename_stem: str,
) -> dict[str, Path]:
    """Draw one independent lollipop figure for the requested metric."""
    configure_style()
    ordered = summary_df.sort_values("condition_order").reset_index(drop=True)
    x_positions = np.arange(len(ordered), dtype=float)
    means = ordered[mean_column].to_numpy() * 100.0
    standard_deviations = ordered[sd_column].to_numpy() * 100.0
    condition_keys = ordered["condition_key"].tolist()
    labels = [
        "1153\n(No Patch)" if key == "full_sequence_no_patch" else str(size)
        for key, size in zip(condition_keys, ordered["patch_size"])
    ]

    figure, axis = plt.subplots(figsize=(14.8, 7.8))
    figure.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.18)

    axis.axhline(
        CHANCE_ACCURACY,
        color="#8A939B",
        linewidth=1.35,
        linestyle=(0, (5, 3)),
        zorder=1,
    )

    for index, (position, mean_value, sd_value, condition_key) in enumerate(
        zip(x_positions, means, standard_deviations, condition_keys)
    ):
        style = CONDITION_STYLES[condition_key]
        axis.vlines(
            position,
            CHANCE_ACCURACY,
            mean_value,
            color=style["color"],
            linewidth=3.0,
            alpha=0.92,
            zorder=2,
        )
        axis.errorbar(
            position,
            mean_value,
            yerr=sd_value,
            fmt="none",
            ecolor=style["color"],
            elinewidth=1.9,
            capsize=7,
            capthick=1.9,
            zorder=3,
        )
        axis.scatter(
            position,
            mean_value,
            s=190,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=1.45,
            zorder=4,
        )
        label_height = mean_value + sd_value + 0.85
        axis.text(
            position,
            label_height,
            f"{mean_value:.2f} ± {sd_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11.2,
            fontweight="bold",
            color=(
                "#B2182B"
                if condition_key == "full_sequence_no_patch"
                else "#202020"
            ),
        )

    axis.set_xlim(-0.55, len(ordered) - 0.45)
    axis.set_ylim(*y_limits)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_yticks(y_ticks)
    axis.set_xlabel("Patch size", labelpad=12)
    axis.set_ylabel(y_label, labelpad=12)
    axis.grid(axis="y", color="#D8DEE5", linewidth=0.9, alpha=0.9, zorder=0)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", length=5.0, pad=7)
    plt.setp(axis.get_xticklabels(), fontweight="bold")
    plt.setp(axis.get_yticklabels(), fontweight="bold")
    axis.get_xticklabels()[-1].set_color("#B2182B")

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
    summary_df = pd.read_csv(SCRIPT_DIR / "condition_summary.csv")
    table_paths = {
        "run_level": SCRIPT_DIR / "run_level_results.csv",
        "summary": SCRIPT_DIR / "condition_summary.csv",
    }
    table_status = {name: False for name in table_paths}
    test_paths = make_metric_figure(
        summary_df,
        SCRIPT_DIR,
        mean_column="accuracy_mean",
        sd_column="accuracy_sd",
        y_label="Test accuracy (%)",
        y_limits=(20.0, 45.0),
        y_ticks=list(np.arange(20.0, 45.1, 5.0)),
        filename_stem="figure_4_6_2_patch_size_lollipop",
    )
    validation_paths = make_metric_figure(
        summary_df,
        SCRIPT_DIR,
        mean_column="validation_accuracy_mean",
        sd_column="validation_accuracy_sd",
        y_label="Validation accuracy (%)",
        y_limits=(20.0, 40.0),
        y_ticks=list(np.arange(20.0, 40.1, 5.0)),
        filename_stem="figure_4_6_2_validation_accuracy_lollipop",
    )
    figure_paths = {
        "test_png": test_paths["png"],
        "test_svg": test_paths["svg"],
        "validation_png": validation_paths["png"],
        "validation_svg": validation_paths["svg"],
    }

    print("Validated design: 8 conditions x 5 repeated fold-1 runs = 40 results.")
    print("\nTest-accuracy summary:")
    print(
        summary_df[
            [
                "condition_key",
                "patch_size",
                "n_runs",
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
