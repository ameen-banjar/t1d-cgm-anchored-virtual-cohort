"""
Revision analysis R2-7: adult-only virtual library.

Runs matching and whole-day TOST using only the 10 adult Simglucose base
subjects (60 profiles) and compares results against the full-library run.
Also re-runs the base-subject-level bootstrap (R1-2) for both libraries.
"""
from __future__ import annotations

import sys
from pathlib import Path

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

BOOTSTRAP_REPS  = 2000
BOOTSTRAP_SEED  = 2026


def tost_table(matches: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    rows = []
    for metric, margin in MARGINS.items():
        real_col, virt_col = f"real_{metric}", f"virtual_{metric}"
        if real_col not in matches or virt_col not in matches:
            continue
        res = paired_tost(matches[real_col], matches[virt_col], margin)
        diff = matches[real_col] - matches[virt_col]
        # member-cluster bootstrap (original)
        boot_member = cluster_bootstrap_mean_ci(
            diff, matches["member_key"], replicates=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED
        )
        # base-subject bootstrap (R1-2 fix)
        boot_base = cluster_bootstrap_mean_ci(
            diff, matches[cluster_col], replicates=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED
        )
        member_equiv = boot_member[0] > -margin and boot_member[1] < margin
        base_equiv   = boot_base[0]   > -margin and boot_base[1]   < margin

        if res["equivalent"] and member_equiv:
            verdict_member = "Robustly equivalent"
        elif res["equivalent"]:
            verdict_member = "Not robust (member)"
        else:
            verdict_member = "Not equivalent"

        if res["equivalent"] and base_equiv:
            verdict_base = "Robustly equivalent"
        elif res["equivalent"]:
            verdict_base = "Not robust (base)"
        else:
            verdict_base = "Not equivalent"

        rows.append({
            "metric":              metric,
            "bias":                res["bias_real_minus_virtual"],
            "ci90_low":            res["ci90_low"],
            "ci90_high":           res["ci90_high"],
            "tost_equivalent":     res["equivalent"],
            "boot_member_low":     boot_member[0],
            "boot_member_high":    boot_member[1],
            "boot_base_low":       boot_base[0],
            "boot_base_high":      boot_base[1],
            "verdict_member_boot": verdict_member,
            "verdict_base_boot":   verdict_base,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real    = read_real_summary(REAL_SUMMARY)
    virtual = read_virtual_summary(VIRTUAL_SUMMARY)

    # add base-subject column (strip scenario suffix)
    virtual["base_subject"] = virtual["virtual_id"]

    # --- Full library (180 profiles) ---
    matches_full = match_members(real, virtual, FEATURES)
    matches_full["base_subject"] = matches_full["virtual_id"]
    table_full = tost_table(matches_full, "base_subject")
    table_full.insert(0, "library", "full_180")

    selected_full   = matches_full["member_key"].nunique()
    base_full       = matches_full["virtual_id"].nunique()
    mean_dist_full  = matches_full["distance"].mean()

    # --- Adult-only library (60 profiles) ---
    virtual_adult = virtual[virtual["virtual_id"].str.startswith("adult")].copy()
    matches_adult = match_members(real, virtual_adult, FEATURES)
    matches_adult["base_subject"] = matches_adult["virtual_id"]
    table_adult = tost_table(matches_adult, "base_subject")
    table_adult.insert(0, "library", "adult_60")

    selected_adult  = matches_adult["member_key"].nunique()
    base_adult      = matches_adult["virtual_id"].nunique()
    mean_dist_adult = matches_adult["distance"].mean()

    # --- Save results ---
    combined = pd.concat([table_full, table_adult], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "adult_vs_full_equivalence.csv", index=False)

    matches_full.to_csv(OUTPUT_DIR / "matches_full_180.csv", index=False)
    matches_adult.to_csv(OUTPUT_DIR / "matches_adult_60.csv", index=False)

    print("=" * 70)
    print("ADULT-ONLY vs FULL-LIBRARY COMPARISON (R2-7 + R1-2)")
    print("=" * 70)

    print(f"\nFull library  (180): {selected_full} unique profiles, "
          f"{base_full} unique base subjects, mean dist {mean_dist_full:.4f}")
    print(f"Adult library  (60): {selected_adult} unique profiles, "
          f"{base_adult} unique base subjects, mean dist {mean_dist_adult:.4f}")

    for library_label, table in [("FULL 180", table_full), ("ADULT-ONLY 60", table_adult)]:
        print(f"\n--- {library_label} ---")
        print(f"{'Metric':<22} {'Bias':>6}  {'TOST CI':>16}  "
              f"{'Member-boot CI':>18}  {'Base-boot CI':>18}  {'Verdict(base)':>22}")
        for _, row in table.iterrows():
            tost_ci   = f"({row.ci90_low:+.2f},{row.ci90_high:+.2f})"
            mboot_ci  = f"({row.boot_member_low:+.2f},{row.boot_member_high:+.2f})"
            bboot_ci  = f"({row.boot_base_low:+.2f},{row.boot_base_high:+.2f})"
            print(f"{row.metric:<22} {row.bias:>+6.2f}  {tost_ci:>16}  "
                  f"{mboot_ci:>18}  {bboot_ci:>18}  {row.verdict_base_boot:>22}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
