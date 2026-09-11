"""
Libre scan-weighting sensitivity analysis.

Compares LBGI/HBGI computed on:
  A) Raw 15-min traces (existing pipeline — sub-15-min readings retained)
  B) 15-min-grid regularized traces (each timestamp rounded to nearest 15-min
     slot; only one reading per slot kept, matching original drop_duplicates logic)

If the verdicts and effect sizes are stable, we report this as a passed
sensitivity check and add a Methods note. If HBGI changes materially (>0.5
shift in mean bias, or a verdict reversal), we update the primary analysis.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# --- paths ---
ROOT = Path("/Users/abanjar/Desktop/1st Papre Archive/Submission_Package/t1d_cgm_anchored_virtual_cohort_Q1")
TRACE_DIR = Path("/Users/abanjar/Here/First paper copy/RL_By_Patient")
DIURNAL_PRIVATE = ROOT / "outputs/private/diurnal_subject_level.csv"

# --- risk function (mirrors src/t1d_virtual_cohort/metrics.py) ---
def risk_indices(glucose_mgdl):
    g = np.asarray(glucose_mgdl, dtype=float)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return np.nan, np.nan
    t = 1.509 * (np.log(np.clip(g, 1e-6, None)) ** 1.084 - 5.381)
    risk = 10.0 * t**2
    lbgi = float(np.where(t < 0, risk, 0.0).mean())
    hbgi = float(np.where(t > 0, risk, 0.0).mean())
    return lbgi, hbgi


def regularize_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Snap each timestamp to nearest 15-min slot and keep last per slot."""
    df = df.copy()
    # Round to nearest 15-min
    df["ts_15"] = df["timestamp"].dt.round("15min")
    # Keep last per slot (matches original keep='last' drop_duplicates)
    df = (df.sort_values("timestamp")
           .drop_duplicates("ts_15", keep="last")
           [["ts_15", "glucose_mgdl"]]
           .rename(columns={"ts_15": "timestamp"})
           .reset_index(drop=True))
    return df


def read_trace_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Time": "timestamp", "CGM": "glucose_mgdl"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df["glucose_mgdl"] = pd.to_numeric(df["glucose_mgdl"], errors="coerce")
    return (df[["timestamp", "glucose_mgdl"]]
            .dropna()
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True))


def window(df: pd.DataFrame, sh: int, eh: int) -> pd.Series:
    h = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    return df[(h >= sh) & (h < eh)]["glucose_mgdl"]


# ── Load qualified subjects from existing analysis ──────────────────────────
diurnal = pd.read_csv(DIURNAL_PRIVATE)
qualified = diurnal[diurnal["qualified"]].copy()
print(f"Qualified subjects: {len(qualified)}")

rows = []
missing = 0
for _, row in qualified.iterrows():
    sid = str(row["subject_id"])
    trace_path = TRACE_DIR / f"{sid}.csv"
    if not trace_path.exists():
        missing += 1
        continue
    try:
        raw = read_trace_raw(trace_path)
        reg = regularize_to_15min(raw)
    except Exception as e:
        missing += 1
        continue

    # Original (raw) metrics using only 15-min-ish data (already in diurnal CSV)
    # Re-compute raw for verification
    raw_noct_lbgi, raw_noct_hbgi = risk_indices(window(raw, 0, 6))
    raw_dawn_lbgi, raw_dawn_hbgi = risk_indices(window(raw, 4, 8))

    # Regularized metrics
    reg_noct_lbgi, reg_noct_hbgi = risk_indices(window(reg, 0, 6))
    reg_dawn_lbgi, reg_dawn_hbgi = risk_indices(window(reg, 4, 8))

    # Original pipeline values (from CSV)
    orig_real_noct_hbgi = row["real_nocturnal_hbgi"]
    orig_real_dawn_hbgi = row["real_dawn_hbgi"]
    virt_noct_hbgi = row["virtual_nocturnal_hbgi"]
    virt_dawn_hbgi = row["virtual_dawn_hbgi"]

    rows.append({
        "subject_id": sid,
        # Original pipeline (from CSV) — real side
        "orig_real_noct_hbgi": orig_real_noct_hbgi,
        "orig_real_dawn_hbgi": orig_real_dawn_hbgi,
        # Re-computed from raw trace (should match orig)
        "raw_noct_hbgi": raw_noct_hbgi,
        "raw_dawn_hbgi": raw_dawn_hbgi,
        # Regularized
        "reg_noct_hbgi": reg_noct_hbgi,
        "reg_dawn_hbgi": reg_dawn_hbgi,
        # Virtual (unchanged)
        "virt_noct_hbgi": virt_noct_hbgi,
        "virt_dawn_hbgi": virt_dawn_hbgi,
        # LBGI
        "orig_real_noct_lbgi": row["real_nocturnal_lbgi"],
        "raw_noct_lbgi": raw_noct_lbgi,
        "reg_noct_lbgi": reg_noct_lbgi,
        "virt_noct_lbgi": row["virtual_nocturnal_lbgi"],
    })

df = pd.DataFrame(rows)
print(f"Subjects with traces: {len(df)}, missing: {missing}")

# ── Compute biases ──────────────────────────────────────────────────────────
df["orig_bias_noct_hbgi"] = df["orig_real_noct_hbgi"] - df["virt_noct_hbgi"]
df["reg_bias_noct_hbgi"]  = df["reg_noct_hbgi"]  - df["virt_noct_hbgi"]
df["orig_bias_dawn_hbgi"] = df["orig_real_dawn_hbgi"] - df["virt_dawn_hbgi"]
df["reg_bias_dawn_hbgi"]  = df["reg_dawn_hbgi"]  - df["virt_dawn_hbgi"]
df["orig_bias_noct_lbgi"] = df["orig_real_noct_lbgi"] - df["virt_noct_lbgi"]
df["reg_bias_noct_lbgi"]  = df["reg_noct_lbgi"]  - df["virt_noct_lbgi"]

print("\n=== Nocturnal HBGI bias (real - virtual) ===")
print(f"Original pipeline: mean={df['orig_bias_noct_hbgi'].mean():.4f}, "
      f"median={df['orig_bias_noct_hbgi'].median():.4f}, "
      f"IQR=[{df['orig_bias_noct_hbgi'].quantile(0.25):.3f}, {df['orig_bias_noct_hbgi'].quantile(0.75):.3f}]")
print(f"Regularized grid:  mean={df['reg_bias_noct_hbgi'].mean():.4f}, "
      f"median={df['reg_bias_noct_hbgi'].median():.4f}, "
      f"IQR=[{df['reg_bias_noct_hbgi'].quantile(0.25):.3f}, {df['reg_bias_noct_hbgi'].quantile(0.75):.3f}]")
print(f"Difference in median bias: {df['reg_bias_noct_hbgi'].median() - df['orig_bias_noct_hbgi'].median():.4f}")

print("\n=== Dawn HBGI bias (real - virtual) ===")
print(f"Original pipeline: mean={df['orig_bias_dawn_hbgi'].mean():.4f}, "
      f"median={df['orig_bias_dawn_hbgi'].median():.4f}")
print(f"Regularized grid:  mean={df['reg_bias_dawn_hbgi'].mean():.4f}, "
      f"median={df['reg_bias_dawn_hbgi'].median():.4f}")

print("\n=== Nocturnal LBGI bias (real - virtual) ===")
print(f"Original pipeline: mean={df['orig_bias_noct_lbgi'].mean():.4f}, "
      f"median={df['orig_bias_noct_lbgi'].median():.4f}")
print(f"Regularized grid:  mean={df['reg_bias_noct_lbgi'].mean():.4f}, "
      f"median={df['reg_bias_noct_lbgi'].median():.4f}")

# ── Equivalence check (margin = 2.0 for HBGI, 1.0 for LBGI) ────────────────
from scipy import stats

def tost_90ci(x):
    """Two one-sided 90% CI using paired t-test approach on differences."""
    n = len(x)
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.95, df=n-1)
    return mean - t_crit * se, mean + t_crit * se

print("\n=== TOST 90% CI ===")
lo, hi = tost_90ci(df['orig_bias_noct_hbgi'])
print(f"Orig noct HBGI: [{lo:.3f}, {hi:.3f}], margin=±2.0, equiv={lo > -2 and hi < 2}")
lo, hi = tost_90ci(df['reg_bias_noct_hbgi'])
print(f"Reg  noct HBGI: [{lo:.3f}, {hi:.3f}], margin=±2.0, equiv={lo > -2 and hi < 2}")

lo, hi = tost_90ci(df['orig_bias_noct_lbgi'])
print(f"Orig noct LBGI: [{lo:.3f}, {hi:.3f}], margin=±1.0, equiv={lo > -1 and hi < 1}")
lo, hi = tost_90ci(df['reg_bias_noct_lbgi'])
print(f"Reg  noct LBGI: [{lo:.3f}, {hi:.3f}], margin=±1.0, equiv={lo > -1 and hi < 1}")

# ── Save results ─────────────────────────────────────────────────────────────
out_path = ROOT / "outputs/revision/libre_scan_sensitivity.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"N subjects analysed: {len(df)}")
