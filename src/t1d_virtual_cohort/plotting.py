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

    # (A) Distance histogram — solid blue, no hatch (histogram shape carries info)
    axes[0].hist(matches["distance"], bins=30, color="#4472C4", edgecolor="white",
                 linewidth=0.5)
    axes[0].axvline(excellent, color="#C00000", linestyle="--", linewidth=1.4,
                    label=f"Excellent ≤{excellent:.2f}")
    axes[0].axvline(good, color="#7F6000", linestyle=":", linewidth=1.4,
                    label=f"Good ≤{good:.2f}")
    axes[0].set_xlabel("Normalized Euclidean distance")
    axes[0].set_ylabel("Participants")
    axes[0].set_title("(A) 3D matching distance")
    axes[0].legend(fontsize=8)

    order = sorted(matches["scenario"].unique())
    counts = matches["scenario"].value_counts().reindex(order)
    short = [str(value).split("_", 1)[0] for value in order]
    max_count = int(counts.values.max())
    # green bars + hatch for grayscale legibility
    axes[1].bar(short, counts.values, color="#70AD47", edgecolor="black",
                linewidth=0.8, hatch="..")
    for index, value in enumerate(counts.values):
        axes[1].text(index, value + max_count * 0.05, str(value),
                     ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Nearest-member assignments")
    axes[1].set_title("(B) Scenario utilization")
    axes[1].set_ylim(0, max_count * 1.22)
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
        # Scatter: blue filled circles + identity line
        axes[0, column].scatter(real, virtual, s=8, alpha=0.45,
                                color="#4472C4", marker="o", linewidths=0)
        axes[0, column].plot([low, high], [low, high], "k--", linewidth=1.0)
        axes[0, column].set_xlabel(f"Real {LABELS[metric]}")
        axes[0, column].set_ylabel(f"Virtual {LABELS[metric]}")
        axes[0, column].set_title(f"({chr(65 + column)}) {LABELS[metric]}")

        average = (real + virtual) / 2
        difference = real - virtual
        bias = difference.mean()
        sd = difference.std(ddof=1)
        # Bland-Altman: orange open triangles (colour + shape for grayscale)
        axes[1, column].scatter(
            average, difference, s=10, alpha=0.45,
            color="#ED7D31", marker="^", linewidths=0
        )
        axes[1, column].axhline(bias, color="#C00000", linewidth=1.3,
                                label=f"Bias {bias:+.2f}")
        axes[1, column].axhline(
            bias + 1.96 * sd, color="#404040", linestyle="--", linewidth=1.0,
            label=f"+1.96 SD"
        )
        axes[1, column].axhline(
            bias - 1.96 * sd, color="#404040", linestyle="--", linewidth=1.0,
            label=f"−1.96 SD"
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
        # Brown open squares for diurnal (colour + shape)
        axis.scatter(average, difference, s=10, alpha=0.45,
                     color="#A0522D", marker="s", linewidths=0)
        axis.axhline(bias, color="#C00000", linewidth=1.3,
                     label=f"Bias {bias:+.2f}")
        axis.axhline(
            bias + 1.96 * sd, color="#404040", linestyle="--", linewidth=1.0,
            label="+1.96 SD"
        )
        axis.axhline(
            bias - 1.96 * sd, color="#404040", linestyle="--", linewidth=1.0,
            label="−1.96 SD"
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
    # Blue bars + hatch for grayscale legibility
    axis.bar(labels, values, color="#5B9BD5", edgecolor="black", linewidth=0.8,
             hatch="//")
    for index, value in enumerate(values):
        axis.text(index, value + 2.5, f"{value:.1f}%",
                  ha="center", fontsize=9, fontweight="bold")
    axis.set_ylabel("Nearest-member assignments changed (%)")
    axis.set_ylim(0, 110)
    axis.tick_params(axis="x", rotation=0)
    axis.set_title("Feature-ablation sensitivity")
    _save(fig, output, "fig3_feature_ablation")

