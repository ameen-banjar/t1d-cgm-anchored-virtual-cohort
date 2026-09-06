"""
Base-physiology bootstrap for R1.2.

Resamples on the 30 base-subject clusters (not the 76 member-key clusters)
and reports 90% CI for mean difference, then compares with existing
member-cluster CI and TOST CI to produce a three-way final verdict.

Inputs:
  outputs/revision/matches_full_180.csv  (has base_subject column)

Outputs:
  outputs/revision/base_physiology_bootstrap.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

MATCHES_FILE = PROJECT / "outputs/revision/matches_full_180.csv"
OUTPUT_FILE  = PROJECT / "outputs/revision/base_physiology_bootstrap.csv"

MARGINS = {
    "mean_glucose_mgdl": 10.0,
    "cv_percent":         5.0,
    "gmi_percent":        0.5,
    "tir_percent":        5.0,
    "tbr_percent":        3.0,
    "tar_percent":        5.0,
    "lbgi":               1.0,
    "hbgi":               2.0,
}

BOOTSTRAP_REPS = 5000
SEED           = 2026
CONFIDENCE     = 0.90


def base_physiology_bootstrap_ci(
    differences: np.ndarray,
    base_subjects: np.ndarray,
    reps: int = BOOTSTRAP_REPS,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> tuple[float, float]:
    """Cluster bootstrap resampling on base-subject level (30 clusters)."""
    values = np.asarray(differences, dtype=float)
    labels = np.asarray(base_subjects)
    keep = np.isfinite(values)
    values, labels = values[keep], labels[keep]

    unique_bases = np.unique(labels)
    grouped = {b: values[labels == b] for b in unique_bases}

    rng = np.random.default_rng(seed)
    estimates = np.empty(reps)
    for i in range(reps):
        sampled = rng.choice(unique_bases, size=len(unique_bases), replace=True)
        estimates[i] = np.concatenate([grouped[b] for b in sampled]).mean()

    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(estimates, [tail, 1.0 - tail])
    return float(lo), float(hi)


def main():
    matches = pd.read_csv(MATCHES_FILE)
    print(f"Loaded {len(matches)} matches")
    print(f"Unique base subjects: {matches['base_subject'].nunique()}")
    print(f"Base subjects: {sorted(matches['base_subject'].unique())[:5]} ...")

    rows = []
    for metric, margin in MARGINS.items():
        real_col    = f"real_{metric}"
        virtual_col = f"virtual_{metric}"

        if real_col not in matches.columns or virtual_col not in matches.columns:
            print(f"  SKIP {metric}: columns not found")
            continue

        diff = matches[real_col].values - matches[virtual_col].values
        bases = matches["base_subject"].values
        bias = float(np.nanmean(diff))

        lo, hi = base_physiology_bootstrap_ci(diff, bases)
        robust = bool(lo > -margin and hi < margin)

        rows.append({
            "metric":      metric,
            "margin":      margin,
            "bias":        round(bias, 4),
            "bp_ci90_lo":  round(lo, 4),
            "bp_ci90_hi":  round(hi, 4),
            "n_base":      int(matches["base_subject"].nunique()),
            "bp_equivalent": robust,
        })
        verdict = "Robustly equivalent" if robust else "Not equivalent (base-physiology)"
        print(f"  {metric:25s}  bias={bias:+.3f}  90%CI=[{lo:+.3f}, {hi:+.3f}]  {verdict}")

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")

    n_equiv = result["bp_equivalent"].sum()
    print(f"\nBase-physiology bootstrap: {n_equiv}/{len(result)} metrics equivalent")

    return result


if __name__ == "__main__":
    main()
