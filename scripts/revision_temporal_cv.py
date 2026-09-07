"""
Revision R1-10: repeated 5-fold × 10-rep cross-validation of temporal matching.

For each fold-repetition:
1. Fit MinMax scaler on training fold only.
2. Select best candidate feature set on training fold (minimise sum of
   absolute nocturnal+dawn HBGI and LBGI biases after division by margins).
3. Freeze selection; evaluate on held-out fold.
4. Report distributions across 50 fold-rep combinations.
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

DIURNAL   = ROOT / "outputs/private/diurnal_subject_level.csv"
OUTPUT_DIR = ROOT / "outputs/revision"

MARGINS = {"nocturnal_lbgi": 1.0, "nocturnal_hbgi": 2.0,
           "dawn_lbgi": 1.0, "dawn_hbgi": 2.0}

BASE = ["mean_glucose_mgdl", "cv_percent", "tir_percent"]

# candidate feature sets evaluated in evaluate_temporal_matching.py
CANDIDATES = {
    "baseline_3d":       BASE,
    "plus_noct_mean":    BASE + ["nocturnal_mean_virtual"],
    "plus_dawn_mean":    BASE + ["dawn_mean_virtual"],
    "plus_window_means": BASE + ["nocturnal_mean_virtual", "dawn_mean_virtual"],
    "plus_noct_tar":     BASE + ["nocturnal_tar_virtual"],
    "plus_dawn_tar":     BASE + ["dawn_tar_virtual"],
}

N_FOLDS = 5
N_REPS  = 10
BOOTSTRAP_REPS = 1000   # lighter per fold
SEED_BASE = 2026


def fold_id(subject_id: str, rep: int, n_folds: int) -> int:
    digest = hashlib.sha256(f"{rep}|{subject_id}".encode()).hexdigest()
    return int(digest[:8], 16) % n_folds


def _match_and_score(train: pd.DataFrame, test: pd.DataFrame,
                     virtual: pd.DataFrame, features: list[str]) -> dict:
    """Match test subjects using scaler fitted on train, return diurnal biases."""
    # keep only features that exist in both real and virtual
    avail = [f for f in features if f in train.columns and f in virtual.columns]
    if not avail:
        return {}

    combined = pd.concat([train[avail], virtual[avail]], ignore_index=True)
    scaler = MinMaxScaler().fit(combined)
    v_scaled   = scaler.transform(virtual[avail])
    te_scaled  = scaler.transform(test[avail])
    distances  = np.sqrt(((te_scaled[:, None, :] - v_scaled[None, :, :]) ** 2).sum(2))
    nearest    = distances.argmin(axis=1)

    results = {}
    for metric in MARGINS:
        real_col = f"real_{metric}"
        # virtual_profiles columns are renamed to plain names (no "virtual_" prefix)
        virt_col = metric if metric in virtual.columns else f"virtual_{metric}"
        if real_col not in test.columns or virt_col not in virtual.columns:
            continue
        virt_vals = virtual.iloc[nearest][virt_col].to_numpy()
        bias = float(np.mean(test[real_col].to_numpy() - virt_vals))
        results[metric] = bias
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load diurnal subject-level data (qualified stage-2)
    diurnal = pd.read_csv(DIURNAL)
    qual = diurnal[diurnal["qualified"]].reset_index(drop=True)
    print(f"Stage-2 qualified subjects: {len(qual)}")

    # Check which candidate features are actually in the data
    print("Available columns:", [c for c in qual.columns if "virtual" in c or "real" in c][:10])

    # Build virtual "profile" data frame from diurnal (virtual metrics per member_key)
    # Each row in qual has real_ and virtual_ columns for the matched virtual member
    # We need virtual features for each unique member_key
    virt_cols = [c for c in qual.columns if c.startswith("virtual_")]
    real_cols  = [c for c in qual.columns if c.startswith("real_")]

    virtual_profiles = (qual.groupby("member_key")[virt_cols]
                        .mean()
                        .reset_index())
    # rename virtual_ → plain for scaler
    virt_rename = {c: c.replace("virtual_", "") for c in virt_cols}
    virtual_profiles = virtual_profiles.rename(columns=virt_rename)

    # Whole-day features available in diurnal data
    # (diurnal only has nocturnal/dawn metrics — whole-day are in matches file)
    matches = pd.read_csv(ROOT / "outputs/private/matches_3d.csv")
    # merge real whole-day into qual
    wday_real = matches[["subject_id", "real_mean_glucose_mgdl",
                          "real_cv_percent", "real_tir_percent"]].copy()
    qual = qual.merge(wday_real, on="subject_id", how="left")

    # merge virtual whole-day into virtual_profiles via matches
    wday_virt = matches[["member_key", "virtual_mean_glucose_mgdl",
                          "virtual_cv_percent", "virtual_tir_percent"]].drop_duplicates("member_key")
    virtual_profiles = virtual_profiles.merge(wday_virt, on="member_key", how="left")

    # rename virtual whole-day columns to plain for matching
    for col in ["virtual_mean_glucose_mgdl", "virtual_cv_percent", "virtual_tir_percent"]:
        plain = col.replace("virtual_", "")
        if col in virtual_profiles.columns:
            virtual_profiles[plain] = virtual_profiles[col]

    # simplified candidate sets with available features
    BASE_AVAIL = ["mean_glucose_mgdl", "cv_percent", "tir_percent"]
    CANDS_AVAIL = {
        "baseline_3d":           BASE_AVAIL,
        "plus_nocturnal_hbgi":   BASE_AVAIL + ["nocturnal_hbgi"],
        "plus_nocturnal_lbgi":   BASE_AVAIL + ["nocturnal_lbgi"],
        "plus_dawn_hbgi":        BASE_AVAIL + ["dawn_hbgi"],
        "plus_noct_dawn_hbgi":   BASE_AVAIL + ["nocturnal_hbgi", "dawn_hbgi"],
    }

    # add virtual diurnal features to virtual_profiles for candidate matching
    for col in ["nocturnal_hbgi", "nocturnal_lbgi", "dawn_hbgi", "dawn_lbgi"]:
        if col in virtual_profiles.columns:
            pass  # already renamed from virtual_nocturnal_hbgi etc.
        else:
            virt_col = f"virtual_{col}"
            if virt_col in qual.columns:
                virt_map = (qual.groupby("member_key")[virt_col]
                            .mean().reset_index()
                            .rename(columns={virt_col: col}))
                virtual_profiles = virtual_profiles.merge(virt_map, on="member_key", how="left")

    # add plain aliases for whole-day real features (needed by _match_and_score)
    for col in ["mean_glucose_mgdl", "cv_percent", "tir_percent"]:
        if f"real_{col}" in qual.columns:
            qual[col] = qual[f"real_{col}"]

    # add real diurnal features to qual
    for col in ["nocturnal_hbgi", "nocturnal_lbgi", "dawn_hbgi", "dawn_lbgi"]:
        real_col = f"real_{col}"
        if real_col in qual.columns:
            qual[col] = qual[real_col]

    rows = []
    selected_counts = {k: 0 for k in CANDS_AVAIL}

    for rep in range(N_REPS):
        qual["fold"] = qual["subject_id"].apply(lambda s: fold_id(s, rep, N_FOLDS))

        for fold in range(N_FOLDS):
            train = qual[qual["fold"] != fold].copy()
            test  = qual[qual["fold"] == fold].copy()

            # --- model selection on training fold ---
            best_name, best_score = None, np.inf
            for cand_name, cand_feats in CANDS_AVAIL.items():
                avail = [f for f in cand_feats
                         if f in train.columns and f in virtual_profiles.columns]
                if len(avail) < len(BASE_AVAIL):   # skip if base features missing
                    continue
                res = _match_and_score(train, train, virtual_profiles, avail)
                score = sum(abs(res.get(m, 99)) / MARGINS[m] for m in MARGINS)
                if score < best_score:
                    best_score = score
                    best_name  = cand_name

            if best_name is None:
                best_name = "baseline_3d"
            selected_counts[best_name] += 1

            # --- evaluate on held-out fold ---
            best_feats = [f for f in CANDS_AVAIL[best_name]
                          if f in qual.columns and f in virtual_profiles.columns]
            heldout = _match_and_score(train, test, virtual_profiles, best_feats)

            row = {"rep": rep, "fold": fold, "selected_model": best_name, "n_test": len(test)}
            row.update({f"bias_{m}": heldout.get(m, np.nan) for m in MARGINS})
            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "temporal_cv_results.csv", index=False)

    # --- summary ---
    print("\nREPEATED 5-FOLD CV — TEMPORAL MATCHING (10 repetitions = 50 evaluations)")
    print("=" * 70)

    print("\nModel selection frequency (across 50 fold-reps):")
    for name, count in sorted(selected_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:<28} {count:>3}/50 ({count/50*100:.0f}%)")

    print("\nHeld-out bias distributions:")
    print(f"{'Metric':<22} {'Median':>8} {'IQR':>16} {'p5–p95':>18} {'Equiv margin':>14}")
    for metric, margin in MARGINS.items():
        col = f"bias_{metric}"
        vals = results[col].dropna()
        p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
        print(f"{metric:<22} {p50:>+8.3f}  [{p25:+.3f},{p75:+.3f}]  "
              f"[{p5:+.3f},{p95:+.3f}]  ±{margin}")

    print(f"\nResults saved to {OUTPUT_DIR}/temporal_cv_results.csv")
    return results


if __name__ == "__main__":
    main()
