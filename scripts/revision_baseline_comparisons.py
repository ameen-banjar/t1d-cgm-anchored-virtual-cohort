"""
Revision analyses:
  R1-6 / R2-3  — baseline and alternative matching comparisons
  R1-8         — beyond-TOST agreement metrics (MAE, RMSE, LoA, quantiles)
  R1-2         — base-subject-level bootstrap (also runs in adult_only script)

Baselines implemented:
  B0  unmatched (random shuffle — repeat 50 seeds, report median)
  B1  mean-glucose-only 1D matching
  B2  mean + CV  2D matching
  B3  mean + TIR 2D matching
  B4  3D matching (proposed method)
  B5  adult-only 3D matching (from R2-7 analysis)

For each baseline, whole-day TOST + beyond-TOST metrics are computed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1d_virtual_cohort.io import read_real_summary, read_virtual_summary
from t1d_virtual_cohort.matching import match_members
from t1d_virtual_cohort.statistics import cluster_bootstrap_mean_ci, paired_tost

VIRTUAL_SUMMARY = ROOT / "data/derived/virtual_profile_summary.csv"
REAL_SUMMARY    = ROOT / "data/raw/real_summary.csv"
OUTPUT_DIR      = ROOT / "outputs/revision"

MARGINS = {
    "mean_glucose_mgdl": 10.0,
    "cv_percent":         5.0,
    "tir_percent":        5.0,
    "gmi_percent":        0.5,
    "tbr_percent":        3.0,
    "tar_percent":        5.0,
    "lbgi":               1.0,
    "hbgi":               2.0,
}

BOOTSTRAP_REPS = 2000
SEED           = 2026

ALL_METRICS = list(MARGINS.keys())


# ── helpers ──────────────────────────────────────────────────────────────────

def _random_assignment(real: pd.DataFrame, virtual: pd.DataFrame,
                       seed: int) -> pd.DataFrame:
    """Assign each real participant a uniformly random virtual member."""
    rng = np.random.default_rng(seed)
    n = len(real)
    chosen = rng.integers(0, len(virtual), size=n)
    rows = []
    for i, j in enumerate(chosen):
        row = {
            "subject_id": real.iloc[i]["subject_id"],
            "virtual_row": int(j),
            "virtual_id":  virtual.iloc[j]["virtual_id"],
            "scenario":    virtual.iloc[j]["scenario"],
            "member_key":  f"{virtual.iloc[j]['virtual_id']}|{virtual.iloc[j]['scenario']}",
            "distance":    np.nan,
        }
        for metric in ALL_METRICS:
            if metric in real and metric in virtual:
                row[f"real_{metric}"]    = real.iloc[i][metric]
                row[f"virtual_{metric}"] = virtual.iloc[j][metric]
        rows.append(row)
    return pd.DataFrame(rows)


def _beyond_tost(real_vals, virtual_vals) -> dict:
    """MAE, RMSE, LoA (Bland-Altman), quantile differences (25/50/75/95)."""
    r = np.asarray(real_vals, dtype=float)
    v = np.asarray(virtual_vals, dtype=float)
    diff = r - v
    keep = np.isfinite(diff)
    r, v, diff = r[keep], v[keep], diff[keep]
    n = len(diff)
    mean_diff = diff.mean()
    sd_diff   = diff.std(ddof=1)
    se_mean   = sd_diff / np.sqrt(n)
    t95       = stats.t.ppf(0.975, df=n - 1)
    loa_lo    = mean_diff - 1.96 * sd_diff
    loa_hi    = mean_diff + 1.96 * sd_diff
    # 95% CI around each LoA limit (Bland & Altman 1999)
    loa_se    = np.sqrt(3 * sd_diff**2 / n)
    return {
        "n":          int(n),
        "mae":        float(np.abs(diff).mean()),
        "median_ae":  float(np.abs(diff).median() if hasattr(diff, "median")
                            else np.median(np.abs(diff))),
        "rmse":       float(np.sqrt((diff**2).mean())),
        "loa_mean":   float(mean_diff),
        "loa_lo":     float(loa_lo),
        "loa_hi":     float(loa_hi),
        "loa_lo_ci95_lo": float(loa_lo - t95 * loa_se),
        "loa_lo_ci95_hi": float(loa_lo + t95 * loa_se),
        "loa_hi_ci95_lo": float(loa_hi - t95 * loa_se),
        "loa_hi_ci95_hi": float(loa_hi + t95 * loa_se),
        "q25_diff":   float(np.percentile(diff, 25)),
        "q50_diff":   float(np.percentile(diff, 50)),
        "q75_diff":   float(np.percentile(diff, 75)),
        "q95_diff":   float(np.percentile(diff, 95)),
        "q05_diff":   float(np.percentile(diff,  5)),
    }


def _full_metric_table(matches: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for metric, margin in MARGINS.items():
        rc, vc = f"real_{metric}", f"virtual_{metric}"
        if rc not in matches or vc not in matches:
            continue
        tost = paired_tost(matches[rc], matches[vc], margin)
        diff = matches[rc] - matches[vc]
        boot = cluster_bootstrap_mean_ci(
            diff, matches["member_key"], replicates=BOOTSTRAP_REPS, seed=SEED
        )
        robust = tost["equivalent"] and boot[0] > -margin and boot[1] < margin
        verdict = ("Robustly equivalent" if robust
                   else ("Not robust" if tost["equivalent"] else "Not equivalent"))
        bt = _beyond_tost(matches[rc], matches[vc])
        row = {
            "method":    label,
            "metric":    metric,
            "margin":    margin,
            "bias":      tost["bias_real_minus_virtual"],
            "ci90_lo":   tost["ci90_low"],
            "ci90_hi":   tost["ci90_high"],
            "boot_lo":   boot[0],
            "boot_hi":   boot[1],
            "verdict":   verdict,
            "n_equiv":   int(verdict == "Robustly equivalent"),
        }
        row.update({f"bt_{k}": v for k, v in bt.items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real    = read_real_summary(REAL_SUMMARY)
    virtual = read_virtual_summary(VIRTUAL_SUMMARY)
    virtual_adult = virtual[virtual["virtual_id"].str.startswith("adult")].copy()

    BASELINES: list[tuple[str, pd.DataFrame]] = [
        ("B1_mean_only",   match_members(real, virtual, ["mean_glucose_mgdl"])),
        ("B2_mean_cv",     match_members(real, virtual, ["mean_glucose_mgdl", "cv_percent"])),
        ("B3_mean_tir",    match_members(real, virtual, ["mean_glucose_mgdl", "tir_percent"])),
        ("B4_3D_proposed", match_members(real, virtual,
                                         ["mean_glucose_mgdl", "cv_percent", "tir_percent"])),
        ("B5_adult_3D",    match_members(real, virtual_adult,
                                         ["mean_glucose_mgdl", "cv_percent", "tir_percent"])),
    ]

    # B0: random — 50 seeds, pick median n_equiv across seeds per metric
    print("Computing random-assignment baseline (50 seeds)…")
    random_tables = []
    for s in range(50):
        rm = _random_assignment(real, virtual, seed=s)
        random_tables.append(_full_metric_table(rm, "B0_random"))
    random_agg = (pd.concat(random_tables)
                  .groupby("metric")[["bias", "bt_mae", "bt_rmse", "n_equiv"]]
                  .median()
                  .reset_index()
                  .rename(columns={"bt_mae": "mae", "bt_rmse": "rmse"})
                  .assign(method="B0_random_median50"))

    # full tables for B1–B5
    all_tables = []
    print("\nMethod             n_unique_profiles  mean_dist  n_equiv/8")
    print("-" * 60)
    for label, matches in BASELINES:
        t = _full_metric_table(matches, label)
        all_tables.append(t)
        nu = matches["member_key"].nunique()
        md = matches["distance"].mean() if matches["distance"].notna().any() else float("nan")
        ne = t["n_equiv"].sum()
        print(f"{label:<25} {nu:>5}              {md:>8.4f}   {ne}/8")

    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "baseline_comparisons_full.csv", index=False)
    random_agg.to_csv(OUTPUT_DIR / "baseline_random_summary.csv", index=False)

    # --- summary table for paper (TOST + beyond-TOST, 3 key metrics) ---
    key_metrics = ["mean_glucose_mgdl", "cv_percent", "tir_percent",
                   "lbgi", "hbgi", "tbr_percent"]
    summary_cols = ["method", "metric", "bias", "ci90_lo", "ci90_hi",
                    "verdict", "bt_mae", "bt_rmse", "bt_loa_lo", "bt_loa_hi",
                    "bt_q05_diff", "bt_q95_diff"]
    paper_table = combined[combined["metric"].isin(key_metrics)][summary_cols].copy()
    paper_table.to_csv(OUTPUT_DIR / "baseline_comparisons_paper.csv", index=False)

    # --- beyond-TOST for proposed method only (Table for R1-8) ---
    proposed = combined[combined["method"] == "B4_3D_proposed"].copy()
    bt_cols  = ["metric", "bt_mae", "bt_median_ae", "bt_rmse",
                "bt_loa_lo", "bt_loa_hi", "bt_q05_diff", "bt_q25_diff",
                "bt_q50_diff", "bt_q75_diff", "bt_q95_diff"]
    proposed[bt_cols].to_csv(OUTPUT_DIR / "beyond_tost_proposed.csv", index=False)

    # --- print key comparison table ---
    print("\n\nWHOLE-DAY EQUIVALENCE VERDICTS BY METHOD")
    print("=" * 70)
    pivot = (combined.groupby(["method", "metric"])["verdict"]
             .first().unstack("metric")
             .reindex(columns=ALL_METRICS))
    for m, row in pivot.iterrows():
        n_eq = (row == "Robustly equivalent").sum()
        print(f"{m:<25} {n_eq}/8   " + "  ".join(
            ("✓" if v == "Robustly equivalent"
             else ("~" if v == "Not robust" else "✗"))
            for v in row))
    print("Legend: ✓ Robustly equivalent  ~ Not robust  ✗ Not equivalent")

    print("\n\nBEYOND-TOST METRICS — PROPOSED 3D METHOD")
    print("=" * 70)
    print(f"{'Metric':<22} {'MAE':>6} {'RMSE':>6} {'LoA lo':>8} {'LoA hi':>8}"
          f" {'p05':>7} {'p50':>7} {'p95':>7}")
    for _, row in proposed[bt_cols].iterrows():
        print(f"{row.metric:<22} {row.bt_mae:>6.2f} {row.bt_rmse:>6.2f}"
              f" {row.bt_loa_lo:>8.2f} {row.bt_loa_hi:>8.2f}"
              f" {row.bt_q05_diff:>7.2f} {row.bt_q50_diff:>7.2f}"
              f" {row.bt_q95_diff:>7.2f}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
