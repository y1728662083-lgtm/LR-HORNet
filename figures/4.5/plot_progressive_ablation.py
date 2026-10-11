"""Generate the publication figure for the Section 4.5 ablation study.

The script first regenerates both CSV tables from the archived raw results via
``generate_result_tables.py`` and then exports a 600-DPI PNG and an SVG.

Run from any working directory:

    python plot_progressive_ablation.py
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

MODEL_COLORS = {
    "eegnet": "#8798A5",
    "causal1_eegnet": "#6F8EAA",
    "group4_causal1_deformableLstm": "#527FA6",
    "group4_patchLstm_noDeformation": "#2F6FA9",
    "group4_patchLstm_deformableLstm": "#0B3C6F",
}
LR_HORNET_KEY = "group4_patchLstm_deformableLstm"
HIGHLIGHT_COLOR = "#B2182B"


def configure_style() -> None:
    """Apply a bold, high-legibility journal style."""
    sns.set_style("ticks")
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "font.size": 11.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "axes.linewidth": 1.0,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
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
    """Draw matched box-and-point panels for test and validation accuracy."""
    configure_style()

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14.6, 6.3),
        sharey=True,
        constrained_layout=False,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.895,
        bottom=0.235,
        wspace=0.10,
    )
    model_keys = [spec.model_key for spec in MODEL_SPECS]
    model_labels = [
        "EEGNet",
        "Causal\nEEGNet",
        "Causal EEGNet\n+ HOR",
        "LR\n(without HOR)",
        "LR-HORNet",
    ]
    x_positions = np.arange(len(MODEL_SPECS), dtype=float)
    jitter = np.linspace(-0.17, 0.17, len(EXPECTED_FOLDS))

    panel_specs = [
        (
            axes[0],
            "accuracy",
            "accuracy_mean",
            "accuracy_sd",
            "Test accuracy",
            "(a)",
        ),
        (
            axes[1],
            "best_validation_accuracy",
            "validation_accuracy_mean",
            "validation_accuracy_sd",
            "Validation accuracy",
            "(b)",
        ),
    ]

    combined_values = np.concatenate(
        [
            run_df["accuracy"].to_numpy() * 100.0,
            run_df["best_validation_accuracy"].to_numpy() * 100.0,
        ]
    )
    upper_estimates = np.concatenate(
        [
            (
                summary_df["accuracy_mean"] + summary_df["accuracy_sd"]
            ).to_numpy()
            * 100.0,
            (
                summary_df["validation_accuracy_mean"]
                + summary_df["validation_accuracy_sd"]
            ).to_numpy()
            * 100.0,
        ]
    )
    y_min = np.floor(np.nanmin(combined_values) / 5.0) * 5.0 - 2.5
    y_max = (
        np.ceil(
            max(np.nanmax(combined_values), np.nanmax(upper_estimates) + 2.0)
            / 5.0
        )
        * 5.0
        + 5.0
    )

    for (
        axis,
        run_column,
        mean_column,
        sd_column,
        title,
        panel_label,
    ) in panel_specs:
        for model_index, model_key in enumerate(model_keys):
            model_runs = (
                run_df.loc[run_df["model_key"] == model_key]
                .sort_values("fold")
                .reset_index(drop=True)
            )
            model_summary = summary_df.loc[
                summary_df["model_key"] == model_key
            ].iloc[0]
            values = model_runs[run_column].to_numpy() * 100.0
            mean_value = float(model_summary[mean_column] * 100.0)
            sd_value = float(model_summary[sd_column] * 100.0)
            color = MODEL_COLORS[model_key]

            axis.boxplot(
                values,
                positions=[x_positions[model_index]],
                widths=0.56,
                patch_artist=True,
                showfliers=False,
                boxprops={
                    "facecolor": mpl.colors.to_rgba(color, 0.22),
                    "edgecolor": color,
                    "linewidth": 1.7,
                    "zorder": 1,
                },
                medianprops={
                    "color": "#202020",
                    "linewidth": 2.0,
                    "zorder": 3,
                },
                whiskerprops={
                    "color": color,
                    "linewidth": 1.5,
                    "zorder": 1,
                },
                capprops={
                    "color": color,
                    "linewidth": 1.5,
                    "zorder": 1,
                },
            )
            axis.scatter(
                x_positions[model_index] + jitter,
                values,
                s=48,
                color=color,
                alpha=0.68,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            axis.scatter(
                [x_positions[model_index]],
                [mean_value],
                marker="D",
                s=82,
                color=color,
                edgecolor="white",
                linewidth=1.1,
                zorder=5,
            )
            axis.text(
                x_positions[model_index],
                max(float(np.max(values)), mean_value + sd_value) + 0.65,
                f"{mean_value:.2f} ± {sd_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                color=(
                    HIGHLIGHT_COLOR
                    if model_key == LR_HORNET_KEY
                    else "#202020"
                ),
            )

        axis.set_xlim(-0.48, len(MODEL_SPECS) - 0.52)
        axis.set_ylim(y_min, y_max)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(model_labels)
        axis.set_title(title, pad=12)
        axis.set_xlabel("")
        axis.grid(axis="y", color="#D8DEE5", linewidth=0.85, alpha=0.9)
        axis.grid(axis="x", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", length=0, pad=9)
        axis.text(
            -0.10,
            1.06,
            panel_label,
            transform=axis.transAxes,
            fontsize=15,
            fontweight="bold",
            ha="left",
            va="top",
        )
        plt.setp(axis.get_xticklabels(), fontsize=10.8, fontweight="bold")
        for tick_label, model_key in zip(axis.get_xticklabels(), model_keys):
            if model_key == LR_HORNET_KEY:
                tick_label.set_color(HIGHLIGHT_COLOR)
        plt.setp(axis.get_yticklabels(), fontsize=11.5, fontweight="bold")

    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_ylabel("")

    output_paths = {
        "png": output_dir / "figure_4_5_progressive_ablation.png",
        "svg": output_dir / "figure_4_5_progressive_ablation.svg",
    }
    figure.savefig(output_paths["png"], dpi=600, bbox_inches="tight")
    figure.savefig(output_paths["svg"], bbox_inches="tight")
    plt.close(figure)
    return output_paths


def main() -> None:
    run_df, summary_df, table_paths, table_status = generate_result_tables()
    figure_paths = make_figure(run_df, summary_df, SCRIPT_DIR)

    print("Validated design: 5 models x 10 LOSO folds = 50 results.")
    print("\nAccuracy summary:")
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
    for name, table_path in table_paths.items():
        suffix = "" if table_status[name] else " (existing file retained)"
        print(f"  {table_path}{suffix}")
    for figure_path in figure_paths.values():
        print(f"  {figure_path}")


if __name__ == "__main__":
    main()
