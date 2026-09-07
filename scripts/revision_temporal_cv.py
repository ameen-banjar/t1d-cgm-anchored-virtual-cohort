"""
Revision R1-10: repeated 5-fold × 10-rep cross-validation of temporal matching.

For each fold-repetition:
1. Fit MinMax scaler on training-fold real subjects + full 180-profile virtual library.
2. Select best candidate feature set on training fold (minimise sum of
   absolute nocturnal+dawn HBGI and LBGI biases after division by margins).
3. Freeze selection; evaluate on held-out fold, re-matching each held-out
   participant against the complete 180-profile library.
4. Report distributions across 50 fold-rep combinations.

Virtual-library scope: all 180 profiles (30 base subjects × 6 scenarios).
Whole-day metrics come from data/derived/virtual_profile_summary.csv.
Nocturnal/dawn metrics come from outputs/revision/virtual_temporal_all180.csv
(pre-computed from virtual traces in one pass).
"""
from __future__ import annotations

import sys, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1d_virtual_cohort.statistics import paired_tost, cluster_bootstrap_mean_ci

DIURNAL    = ROOT / "outputs/private/diurnal_subject_level.csv"
MATCHES    = ROOT / "outputs/private/matches_3d.csv"
VIRT_WDAY  = ROOT / "data/derived/virtual_profile_summary.csv"
VIRT_TEMP  = ROOT / "outputs/revision/virtual_temporal_all180.csv"
OUTPUT_DIR = ROOT / "outputs/revision"

MARGINS = {"nocturnal_lbgi": 1.0, "nocturnal_hbgi": 2.0,
           "dawn_lbgi": 1.0, "dawn_hbgi": 2.0}

N_FOLDS    = 5
N_REPS     = 10
SEED_BASE  = 2026

CANDS = {
    "baseline_3d":         ["mean_glucose_mgdl", "cv_percent", "tir_percent"],
    "plus_nocturnal_hbgi": ["mean_glucose_mgdl", "cv_percent", "tir_percent",
                            "nocturnal_hbgi"],
    "plus_nocturnal_lbgi": ["mean_glucose_mgdl", "cv_percent", "tir_percent",
                            "nocturnal_lbgi"],
    "plus_dawn_hbgi":      ["mean_glucose_mgdl", "cv_percent", "tir_percent",
                            "dawn_hbgi"],
    "plus_noct_dawn_hbgi": ["mean_glucose_mgdl", "cv_percent", "tir_percent",
                            "nocturnal_hbgi", "dawn_hbgi"],
}

BASE_FEATS = ["mean_glucose_mgdl", "cv_percent", "tir_percent"]


def fold_id(subject_id: str, rep: int, n_folds: int) -> int:
    digest = hashlib.sha256(f"{rep}|{subject_id}".encode()).hexdigest()
    return int(digest[:8], 16) % n_folds


def _match_and_score(train_real: pd.DataFrame, test_real: pd.DataFrame,
                     virtual: pd.DataFrame, features: list[str]) -> dict:
    """Match test subjects against full virtual library using scaler from train+virtual."""
    avail = [f for f in features if f in train_real.columns and f in virtual.columns]
    if len(avail) < len(BASE_FEATS):
        return {}

    combined = pd.concat([train_real[avail], virtual[avail]], ignore_index=True)
    scaler = MinMaxScaler().fit(combined)
    v_scaled  = scaler.transform(virtual[avail])
    te_scaled = scaler.transform(test_real[avail])
    distances = np.sqrt(((te_scaled[:, None, :] - v_scaled[None, :, :]) ** 2).sum(2))
    nearest   = distances.argmin(axis=1)

    results = {}
    for metric in MARGINS:
        real_col = f"real_{metric}"
        if real_col not in test_real.columns or metric not in virtual.columns:
            continue
        virt_vals = virtual.iloc[nearest][metric].to_numpy()
        bias = float(np.mean(test_real[real_col].to_numpy() - virt_vals))
        results[metric] = bias
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Stage-2 real-subject diurnal data ──────────────────────────
    diurnal = pd.read_csv(DIURNAL)
    qual = diurnal[diurnal["qualified"]].reset_index(drop=True)
    print(f"Stage-2 qualified subjects: {len(qual)}")

    # Merge real whole-day features from matches file
    matches = pd.read_csv(MATCHES)
    wday_real = matches[["subject_id", "real_mean_glucose_mgdl",
                         "real_cv_percent", "real_tir_percent"]].drop_duplicates("subject_id")
    qual = qual.merge(wday_real, on="subject_id", how="left")

    # Promote real_ columns to plain names for matching
    for col in BASE_FEATS:
        if f"real_{col}" in qual.columns:
            qual[col] = qual[f"real_{col}"]
    for col in list(MARGINS):
        real_col = f"real_{col}"
        if real_col in qual.columns:
            qual[col] = qual[real_col]

    # ── 2. Build virtual library from all 180 profiles ─────────────────────
    virt_wday = pd.read_csv(VIRT_WDAY)
    virt_wday["member_key"] = (virt_wday["virtual_id"].astype(str)
                               + "|"
                               + virt_wday["scenario"].astype(str))

    virt_temp = pd.read_csv(VIRT_TEMP)

    # Merge whole-day + temporal metrics
    virtual = virt_wday.merge(virt_temp[["member_key",
                                         "nocturnal_lbgi", "nocturnal_hbgi",
                                         "dawn_lbgi", "dawn_hbgi"]],
                              on="member_key", how="left")

    print(f"Virtual library size: {len(virtual)} profiles")
    print(f"Virtual profiles with temporal metrics: {virtual['nocturnal_hbgi'].notna().sum()}")

    # ── 3. Cross-validation ────────────────────────────────────────────────
    rows = []
    selected_counts = {k: 0 for k in CANDS}

    for rep in range(N_REPS):
        qual["fold"] = qual["subject_id"].apply(
            lambda s: fold_id(s, rep, N_FOLDS))

        for fold in range(N_FOLDS):
            train = qual[qual["fold"] != fold].copy()
            test  = qual[qual["fold"] == fold].copy()

            # Model selection on training fold
            best_name, best_score = None, np.inf
            for cand_name, cand_feats in CANDS.items():
                res = _match_and_score(train, train, virtual, cand_feats)
                if not res:
                    continue
                score = sum(abs(res.get(m, 99)) / MARGINS[m] for m in MARGINS)
                if score < best_score:
                    best_score = score
                    best_name  = cand_name

            if best_name is None:
                best_name = "baseline_3d"
            selected_counts[best_name] += 1

            # Evaluate on held-out fold (re-match against full 180-profile library)
            heldout = _match_and_score(train, test, virtual, CANDS[best_name])

            row = {"rep": rep, "fold": fold, "selected_model": best_name,
                   "n_test": len(test)}
            row.update({f"bias_{m}": heldout.get(m, np.nan) for m in MARGINS})
            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "temporal_cv_results.csv", index=False)

    # ── 4. Summary ─────────────────────────────────────────────────────────
    print("\nREPEATED 5-FOLD CV — TEMPORAL MATCHING (10 reps = 50 evaluations)")
    print("Virtual-library scope: all 180 profiles")
    print("=" * 70)

    print("\nModel selection frequency (across 50 fold-reps):")
    for name, count in sorted(selected_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:<28} {count:>3}/50 ({count/50*100:.0f}%)")

    print("\nHeld-out bias distributions (re-matched against full 180-profile library):")
    print(f"{'Metric':<22} {'Median':>8} {'IQR':>16} {'p5–p95':>18} {'Margin':>8}")
    for metric, margin in MARGINS.items():
        col = f"bias_{metric}"
        vals = results[col].dropna()
        p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
        print(f"{metric:<22} {p50:>+8.3f}  [{p25:+.3f},{p75:+.3f}]  "
              f"[{p5:+.3f},{p95:+.3f}]  ±{margin}")

    # Check if any fold achieved equivalence (|bias| < margin)
    print("\nEquivalence check (|held-out bias| < margin):")
    for metric, margin in MARGINS.items():
        col = f"bias_{metric}"
        vals = results[col].dropna()
        n_equiv = (vals.abs() < margin).sum()
        print(f"  {metric:<22} {n_equiv:>3}/50 folds within margin ±{margin}")

    print(f"\nResults saved to {OUTPUT_DIR}/temporal_cv_results.csv")
    return results


if __name__ == "__main__":
    main()
