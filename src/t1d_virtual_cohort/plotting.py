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
}


def _setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 300,
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
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    axes[0].hist(matches["distance"], bins=30, color="#4472C4", edgecolor="white")
    axes[0].axvline(excellent, color="#C00000", linestyle="--", linewidth=1)
    axes[0].axvline(good, color="#7F6000", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Normalized Euclidean distance")
    axes[0].set_ylabel("Participants")
    axes[0].set_title("(A) 3D matching distance")

    order = sorted(matches["scenario"].unique())
    counts = matches["scenario"].value_counts().reindex(order)
    short = [str(value).split("_", 1)[0] for value in order]
    axes[1].bar(short, counts.values, color="#70AD47")
    for index, value in enumerate(counts.values):
        axes[1].text(index, value + 3, str(value), ha="center", fontsize=7)
    axes[1].set_ylabel("Top-1 assignments")
    axes[1].set_title("(B) Scenario utilization")
    _save(fig, output, "fig1_matching_overview")


def plot_agreement_grid(
    matches: pd.DataFrame, metrics: list[str], output: Path, name: str
) -> None:
    _setup()
    fig, axes = plt.subplots(2, len(metrics), figsize=(7.16, 4.4))
    if len(metrics) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, metric in enumerate(metrics):
        real = matches[f"real_{metric}"].to_numpy(float)
        virtual = matches[f"virtual_{metric}"].to_numpy(float)
        low = min(real.min(), virtual.min())
        high = max(real.max(), virtual.max())
        axes[0, column].scatter(real, virtual, s=6, alpha=0.45, color="#4472C4")
        axes[0, column].plot([low, high], [low, high], "k--", linewidth=0.8)
        axes[0, column].set_xlabel(f"Real {LABELS[metric]}")
        axes[0, column].set_ylabel(f"Virtual {LABELS[metric]}")
        axes[0, column].set_title(f"({chr(65 + column)}) {LABELS[metric]}")

        average = (real + virtual) / 2
        difference = real - virtual
        bias = difference.mean()
        sd = difference.std(ddof=1)
        axes[1, column].scatter(
            average, difference, s=6, alpha=0.45, color="#ED7D31"
        )
        axes[1, column].axhline(bias, color="#C00000", linewidth=0.9)
        axes[1, column].axhline(
            bias + 1.96 * sd, color="#7F7F7F", linestyle="--", linewidth=0.8
        )
        axes[1, column].axhline(
            bias - 1.96 * sd, color="#7F7F7F", linestyle="--", linewidth=0.8
        )
        axes[1, column].set_xlabel("Pair mean")
        axes[1, column].set_ylabel("Real - virtual")
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
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0))
    for index, (key, title) in enumerate(definitions):
        axis = axes.flat[index]
        real = qualified[f"real_{key}"].to_numpy(float)
        virtual = qualified[f"virtual_{key}"].to_numpy(float)
        average = (real + virtual) / 2
        difference = real - virtual
        bias = difference.mean()
        sd = difference.std(ddof=1)
        axis.scatter(average, difference, s=6, alpha=0.4, color="#A0522D")
        axis.axhline(bias, color="#C00000", linewidth=1)
        axis.axhline(
            bias + 1.96 * sd, color="#7F7F7F", linestyle="--", linewidth=0.8
        )
        axis.axhline(
            bias - 1.96 * sd, color="#7F7F7F", linestyle="--", linewidth=0.8
        )
        axis.set_title(f"({chr(65 + index)}) {title}")
        axis.set_xlabel("Pair mean")
        axis.set_ylabel("Real - virtual")
    _save(fig, output, "fig4_diurnal_agreement")


def plot_ablation(ablation: pd.DataFrame, output: Path) -> None:
    _setup()
    fig, axis = plt.subplots(figsize=(3.5, 2.8))
    labels = [LABELS[value] for value in ablation["removed_feature"]]
    values = ablation["reassigned_percent"].to_numpy()
    axis.bar(labels, values, color="#5B9BD5")
    for index, value in enumerate(values):
        axis.text(index, value + 1, f"{value:.1f}%", ha="center", fontsize=7)
    axis.set_ylabel("Top-1 assignments changed (%)")
    axis.set_ylim(0, 100)
    axis.tick_params(axis="x", rotation=20)
    axis.set_title("Feature-ablation sensitivity")
    _save(fig, output, "fig3_feature_ablation")

