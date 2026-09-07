"""
Revision analysis R1-4: margin sensitivity.

For each whole-day metric, tests equivalence at 0.50×, 0.75×, 1.00×,
1.25×, 1.50×, 2.00× of the study-specified margin.
Reports verdict (both TOST and base-subject bootstrap) at each margin.
Also outputs the "required margin" — the smallest symmetric value that
would contain both the 90% TOST CI and the base-subject bootstrap CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1d_virtual_cohort.io import read_real_summary, read_virtual_summary
from t1d_virtual_cohort.matching import match_members
from t1d_virtual_cohort.statistics import cluster_bootstrap_mean_ci, paired_tost

VIRTUAL_SUMMARY = ROOT / "data/derived/virtual_profile_summary.csv"
REAL_SUMMARY    = ROOT / "data/raw/real_summary.csv"
OUTPUT_DIR      = ROOT / "outputs/revision"

FEATURES = ["mean_glucose_mgdl", "cv_percent", "tir_percent"]

STUDY_MARGINS = {
    "mean_glucose_mgdl": 10.0,
    "cv_percent":         5.0,
    "tir_percent":        5.0,
    "gmi_percent":        0.5,
    "tbr_percent":        3.0,
    "tar_percent":        5.0,
    "lbgi":               1.0,
    "hbgi":               2.0,
}

MULTIPLIERS   = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
BOOTSTRAP_REPS = 2000
SEED           = 2026


def required_margin(ci_lo: float, ci_hi: float,
                    boot_lo: float, boot_hi: float) -> float:
    """Smallest symmetric δ that contains both CIs."""
    return float(max(abs(ci_lo), abs(ci_hi), abs(boot_lo), abs(boot_hi)))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real    = read_real_summary(REAL_SUMMARY)
    virtual = read_virtual_summary(VIRTUAL_SUMMARY)
    matches = match_members(real, virtual, FEATURES)

    rows = []
    for metric, base_margin in STUDY_MARGINS.items():
        rc, vc = f"real_{metric}", f"virtual_{metric}"
        if rc not in matches or vc not in matches:
            continue
        diff = matches[rc] - matches[vc]

        # base-subject bootstrap (R1-2 fix) — cluster = virtual_id
        boot_base = cluster_bootstrap_mean_ci(
            diff, matches["virtual_id"], replicates=BOOTSTRAP_REPS, seed=SEED
        )

        req_m = required_margin(
            *paired_tost(matches[rc], matches[vc], base_margin)["ci90_low":"ci90_high"]
             if False else (0,0),  # placeholder, computed below
            *boot_base,
        )

        for mult in MULTIPLIERS:
            margin = round(base_margin * mult, 6)
            tost   = paired_tost(matches[rc], matches[vc], margin)
            tost_eq = tost["equivalent"]
            boot_eq = boot_base[0] > -margin and boot_base[1] < margin

            if tost_eq and boot_eq:
                verdict = "Robustly equivalent"
            elif tost_eq:
                verdict = "Not robust (base-boot)"
            else:
                verdict = "Not equivalent"

            rows.append({
                "metric":          metric,
                "study_margin":    base_margin,
                "multiplier":      mult,
                "test_margin":     margin,
                "bias":            tost["bias_real_minus_virtual"],
                "ci90_lo":         tost["ci90_low"],
                "ci90_hi":         tost["ci90_high"],
                "boot_base_lo":    boot_base[0],
                "boot_base_hi":    boot_base[1],
                "tost_equiv":      tost_eq,
                "boot_equiv":      boot_eq,
                "verdict":         verdict,
            })

    df = pd.DataFrame(rows)

    # required margin per metric
    req_rows = []
    for metric, base_margin in STUDY_MARGINS.items():
        rc, vc = f"real_{metric}", f"virtual_{metric}"
        if rc not in matches or vc not in matches:
            continue
        diff  = matches[rc] - matches[vc]
        tost  = paired_tost(matches[rc], matches[vc], base_margin)
        boot  = cluster_bootstrap_mean_ci(
            diff, matches["virtual_id"], replicates=BOOTSTRAP_REPS, seed=SEED
        )
        req_m = required_margin(tost["ci90_low"], tost["ci90_high"], boot[0], boot[1])
        req_rows.append({
            "metric":          metric,
            "study_margin":    base_margin,
            "bias":            tost["bias_real_minus_virtual"],
            "ci90_lo":         tost["ci90_low"],
            "ci90_hi":         tost["ci90_high"],
            "boot_base_lo":    boot[0],
            "boot_base_hi":    boot[1],
            "required_margin": req_m,
            "margin_ratio":    round(req_m / base_margin, 3),
        })
    req_df = pd.DataFrame(req_rows)

    df.to_csv(OUTPUT_DIR / "margin_sensitivity_full.csv", index=False)
    req_df.to_csv(OUTPUT_DIR / "margin_required.csv", index=False)

    # pivot: metric × multiplier → verdict
    pivot = df.pivot(index="metric", columns="multiplier", values="verdict")

    print("MARGIN SENSITIVITY — WHOLE-DAY METRICS")
    print("=" * 78)
    print(f"{'Metric':<22} " + "  ".join(f"{m:.2f}×" for m in MULTIPLIERS))
    print("-" * 78)
    sym = {"Robustly equivalent": "✓", "Not robust (base-boot)": "~",
           "Not equivalent": "✗"}
    for metric in STUDY_MARGINS:
        if metric not in pivot.index:
            continue
        verdicts = "      ".join(sym.get(pivot.loc[metric, m], "?")
                                 for m in MULTIPLIERS)
        print(f"{metric:<22} {verdicts}")

    print("\nLegend: ✓ Robustly equivalent  ~ TOST pass but base-boot fails  "
          "✗ Not equivalent")

    print("\nREQUIRED MARGIN (smallest δ that would contain both CIs)")
    print("=" * 78)
    print(f"{'Metric':<22} {'Study δ':>8} {'Required δ':>11} {'Ratio':>7} "
          f"{'TOST CI':>16}  {'Base-boot CI':>16}")
    print("-" * 78)
    for _, row in req_df.iterrows():
        tost_ci = f"({row.ci90_lo:+.2f},{row.ci90_hi:+.2f})"
        boot_ci = f"({row.boot_base_lo:+.2f},{row.boot_base_hi:+.2f})"
        status  = "✓" if row.required_margin <= row.study_margin else "✗ EXCEEDS"
        print(f"{row.metric:<22} {row.study_margin:>8.2f} {row.required_margin:>11.3f}"
              f" {row.margin_ratio:>7.3f}  {tost_ci:>16}  {boot_ci:>16}  {status}")


if __name__ == "__main__":
    main()
