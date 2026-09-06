from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "mean_glucose_mgdl": "Mean glucose (mg/dL)",
    "cv_percent": "CV (%)",
    "tir_percent": "TIR (%)",
    "tbr_percent": "TBR (%)",
    "tar_percent": "TAR (%)",
    "lbgi": "LBGI",
    "hbgi": "HBGI",
    "gmi_percent": "GMI (%)",
}


def _setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            # Grayscale-friendly defaults
            "axes.grid": True,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
        }
    )


def _save(fig, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_matching_overview(
    matches: pd.DataFrame, output: Path, excellent: float, good: float
) -> None:
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.4))

    # (A) Distance histogram — grayscale: dark gray fill, black edge
    axes[0].hist(matches["distance"], bins=30, color="#404040", edgecolor="white",
                 linewidth=0.5)
    axes[0].axvline(excellent, color="black", linestyle="--", linewidth=1.2,
                    label=f"Excellent ({excellent:.2f})")
    axes[0].axvline(good, color="black", linestyle=":", linewidth=1.2,
                    label=f"Good ({good:.2f})")
    axes[0].set_xlabel("Normalized Euclidean distance")
    axes[0].set_ylabel("Participants")
    axes[0].set_title("(A) 3D matching distance")
    axes[0].legend(fontsize=8)

    order = sorted(matches["scenario"].unique())
    counts = matches["scenario"].value_counts().reindex(order)
    short = [str(value).split("_", 1)[0] for value in order]
    max_count = int(counts.values.max())
    # grayscale: medium gray bars
    axes[1].bar(short, counts.values, color="#707070", edgecolor="black", linewidth=0.7)
    for index, value in enumerate(counts.values):
        axes[1].text(index, value + max_count * 0.04, str(value),
                     ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Nearest-member assignments")
    axes[1].set_title("(B) Scenario utilization")
    # Extra headroom so bar labels don't clip the top border
    axes[1].set_ylim(0, max_count * 1.20)
    _save(fig, output, "fig1_matching_overview")


def plot_agreement_grid(
    matches: pd.DataFrame, metrics: list[str], output: Path, name: str
) -> None:
    _setup()
    fig, axes = plt.subplots(2, len(metrics), figsize=(7.16, 5.2))
    if len(metrics) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, metric in enumerate(metrics):
        real = matches[f"real_{metric}"].to_numpy(float)
        virtual = matches[f"virtual_{metric}"].to_numpy(float)
        low = min(real.min(), virtual.min())
        high = max(real.max(), virtual.max())
        # Scatter: filled circles, grayscale
        axes[0, column].scatter(real, virtual, s=8, alpha=0.40,
                                color="#404040", marker="o")
        axes[0, column].plot([low, high], [low, high], "k--", linewidth=1.0)
        axes[0, column].set_xlabel(f"Real {LABELS[metric]}")
        axes[0, column].set_ylabel(f"Virtual {LABELS[metric]}")
        axes[0, column].set_title(f"({chr(65 + column)}) {LABELS[metric]}")

        average = (real + virtual) / 2
        difference = real - virtual
        bias = difference.mean()
        sd = difference.std(ddof=1)
        # Bland-Altman: open triangles to distinguish from identity scatter
        axes[1, column].scatter(
            average, difference, s=8, alpha=0.40,
            color="#606060", marker="^"
        )
        axes[1, column].axhline(bias, color="black", linewidth=1.2,
                                label=f"Bias {bias:+.2f}")
        axes[1, column].axhline(
            bias + 1.96 * sd, color="black", linestyle="--", linewidth=0.9,
            label=f"+1.96SD {bias + 1.96*sd:+.2f}"
        )
        axes[1, column].axhline(
            bias - 1.96 * sd, color="black", linestyle="--", linewidth=0.9,
            label=f"−1.96SD {bias - 1.96*sd:+.2f}"
        )
        axes[1, column].set_xlabel("Pair mean")
        axes[1, column].set_ylabel("Real − virtual")
        if column == 0:
            axes[1, column].legend(fontsize=7, loc="upper right")
    _save(fig, output, name)


def plot_diurnal(diurnal: pd.DataFrame, output: Path) -> None:
    _setup()
    qualified = diurnal[diurnal["qualified"]].copy()
    definitions = [
        ("nocturnal_lbgi", "Nocturnal LBGI"),
        ("nocturnal_hbgi", "Nocturnal HBGI"),
        ("dawn_lbgi", "Dawn LBGI"),
        ("dawn_hbgi", "Dawn HBGI"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.8))
    for index, (key, title) in enumerate(definitions):
        axis = axes.flat[index]
        real = qualified[f"real_{key}"].to_numpy(float)
        virtual = qualified[f"virtual_{key}"].to_numpy(float)
        average = (real + virtual) / 2
        difference = real - virtual
        bias = difference.mean()
        sd = difference.std(ddof=1)
        # Open squares, grayscale-safe
        axis.scatter(average, difference, s=8, alpha=0.40,
                     color="#505050", marker="s")
        axis.axhline(bias, color="black", linewidth=1.4,
                     label=f"Bias {bias:+.2f}")
        axis.axhline(
            bias + 1.96 * sd, color="black", linestyle="--", linewidth=1.0,
            label=f"+1.96SD"
        )
        axis.axhline(
            bias - 1.96 * sd, color="black", linestyle="--", linewidth=1.0,
            label=f"−1.96SD"
        )
        axis.set_title(f"({chr(65 + index)}) {title}")
        axis.set_xlabel("Pair mean")
        axis.set_ylabel("Real − virtual")
        axis.legend(fontsize=8, loc="upper right")
    _save(fig, output, "fig4_diurnal_agreement")


def plot_ablation(ablation: pd.DataFrame, output: Path) -> None:
    _setup()
    fig, axis = plt.subplots(figsize=(4.0, 3.0))
    labels = [
        LABELS[value].replace(" (", "\n(") for value in ablation["removed_feature"]
    ]
    values = ablation["reassigned_percent"].to_numpy()
    # Grayscale bar with hatch pattern for print clarity
    axis.bar(labels, values, color="#606060", edgecolor="black", linewidth=0.7,
             hatch="//")
    for index, value in enumerate(values):
        axis.text(index, value + 2.0, f"{value:.1f}%",
                  ha="center", fontsize=9, fontweight="bold")
    axis.set_ylabel("Nearest-member assignments changed (%)")
    axis.set_ylim(0, 110)   # headroom for labels at top of bars
    axis.tick_params(axis="x", rotation=0)
    axis.set_title("Feature-ablation sensitivity")
    _save(fig, output, "fig3_feature_ablation")

