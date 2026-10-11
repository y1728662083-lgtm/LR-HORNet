"""Create the publication figure for Section 4.4.2.

The raw-result parsing, protocol validation, and CSV generation are provided by
``generate_result_tables.py``.  Running this script regenerates those tables
from the archived experiment outputs before drawing the figure.

Run from any working directory:

    python plot_validation_subject_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from generate_result_tables import (
    ALL_SUBJECTS,
    DEFAULT_RESULTS_ROOT,
    TEST_SUBJECTS,
    generate_result_tables,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = DEFAULT_RESULTS_ROOT

def configure_style() -> None:
    """Apply a restrained, journal-friendly visual style."""
    sns.set_style("ticks")
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def make_figure(summary_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Generate paired heatmaps with a common accuracy scale."""
    configure_style()

    figure = plt.figure(figsize=(11.8, 5.35), constrained_layout=False)
    grid = figure.add_gridspec(
        nrows=2,
        ncols=24,
        height_ratios=[1.0, 1.0],
        hspace=0.48,
        wspace=0.22,
        left=0.080,
        right=0.955,
        top=0.925,
        bottom=0.135,
    )

    test_heatmap = (
        summary_df.pivot(index="test_subject", columns="validation_subject", values="test_accuracy_mean")
        .reindex(index=TEST_SUBJECTS, columns=ALL_SUBJECTS)
        .mul(100.0)
    )
    validation_heatmap = (
        summary_df.pivot(
            index="test_subject",
            columns="validation_subject",
            values="validation_accuracy_mean",
        )
        .reindex(index=TEST_SUBJECTS, columns=ALL_SUBJECTS)
        .mul(100.0)
    )
    shared_values = np.concatenate(
        [
            test_heatmap.to_numpy()[~test_heatmap.isna().to_numpy()],
            validation_heatmap.to_numpy()[~validation_heatmap.isna().to_numpy()],
        ]
    )
    color_min = np.floor(shared_values.min() / 2.5) * 2.5
    color_max = np.ceil(shared_values.max() / 2.5) * 2.5

    blue_map = LinearSegmentedColormap.from_list(
        "academic_blues",
        ["#F7FBFF", "#DCEAF6", "#9ECAE1", "#4292C6", "#08519C", "#08306B"],
    )
    axes = [figure.add_subplot(grid[0, :23]), figure.add_subplot(grid[1, :23])]
    colorbar_ax = figure.add_subplot(grid[:, 23])
    panel_specs = [
        (
            axes[0],
            test_heatmap,
            "(a)",
        ),
        (
            axes[1],
            validation_heatmap,
            "(b)",
        ),
    ]

    for ax, heatmap_data, panel_label in panel_specs:
        heatmap_mask = heatmap_data.isna()
        sns.heatmap(
            heatmap_data,
            mask=heatmap_mask,
            cmap=blue_map,
            vmin=color_min,
            vmax=color_max,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 10, "fontweight": "bold"},
            linewidths=0.8,
            linecolor="white",
            cbar=False,
            ax=ax,
        )
        ax.set_ylabel("test subject")
        ax.set_xticklabels([str(subject) for subject in ALL_SUBJECTS], rotation=0)
        ax.set_yticklabels([str(subject) for subject in TEST_SUBJECTS], rotation=0)
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        ax.tick_params(length=0)
        add_panel_label(ax, panel_label)

        for row_index, test_subject in enumerate(TEST_SUBJECTS):
            column_index = test_subject - 1
            ax.add_patch(
                Rectangle(
                    (column_index, row_index),
                    1,
                    1,
                    facecolor="#F1F1F1",
                    edgecolor="#B8B8B8",
                    linewidth=0.8,
                    hatch="////",
                    zorder=4,
                )
            )
            ax.text(
                column_index + 0.5,
                row_index + 0.5,
                "—",
                ha="center",
                va="center",
                color="#666666",
                fontsize=11,
                fontweight="bold",
                zorder=5,
            )

    axes[0].set_xlabel("")
    axes[1].set_xlabel("Validation subject")
    shared_norm = mpl.colors.Normalize(vmin=color_min, vmax=color_max)
    scalar_map = mpl.cm.ScalarMappable(norm=shared_norm, cmap=blue_map)
    scalar_map.set_array([])
    colorbar = figure.colorbar(scalar_map, cax=colorbar_ax)
    colorbar.set_label(
        "Accuracy (%)", labelpad=8, fontsize=12, fontweight="bold"
    )
    plt.setp(colorbar.ax.get_yticklabels(), fontsize=10.5, fontweight="bold")
    colorbar.outline.set_linewidth(0.7)

    output_paths = {
        "png": output_dir / "figure_4_4_2_validation_subject_sensitivity.png",
        "svg": output_dir / "figure_4_4_2_validation_subject_sensitivity.svg",
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
    table_write_status = {name: False for name in table_paths}

    output_paths = make_figure(summary_df, SCRIPT_DIR)

    print("Validated design: 135 runs, 27 conditions, 5 repeats per condition.")
    print("\nCondition-level summary:")
    display_columns = [
        "test_subject",
        "validation_subject",
        "test_accuracy_mean",
        "test_accuracy_sd",
        "validation_accuracy_mean",
        "validation_accuracy_sd",
    ]
    print(summary_df[display_columns].to_string(index=False, float_format="%.4f"))
    print("\nAccuracy ranges across validation-subject conditions:")
    for test_subject in TEST_SUBJECTS:
        subset = summary_df.loc[summary_df["test_subject"] == test_subject]
        print(
            f"  Test subject {test_subject}: test="
            f"{subset['test_accuracy_mean'].min():.4f}–"
            f"{subset['test_accuracy_mean'].max():.4f}; validation="
            f"{subset['validation_accuracy_mean'].min():.4f}–"
            f"{subset['validation_accuracy_mean'].max():.4f}"
        )

    print("\nGenerated files:")
    for name, table_path in table_paths.items():
        suffix = "" if table_write_status[name] else " (existing file retained)"
        print(f"  {table_path}{suffix}")
    for output_path in output_paths.values():
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
