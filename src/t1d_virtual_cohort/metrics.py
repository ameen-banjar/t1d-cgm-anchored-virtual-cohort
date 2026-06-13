from __future__ import annotations

import numpy as np
import pandas as pd


def gmi_percent(mean_glucose_mgdl: float) -> float:
    return 3.31 + 0.02392 * float(mean_glucose_mgdl)


def risk_indices(glucose_mgdl) -> tuple[float, float]:
    glucose = np.asarray(glucose_mgdl, dtype=float)
    glucose = glucose[np.isfinite(glucose)]
    if glucose.size == 0:
        return np.nan, np.nan
    transform = 1.509 * (
        np.log(np.clip(glucose, 1e-6, None)) ** 1.084 - 5.381
    )
    risk = 10.0 * transform**2
    lbgi = float(np.where(transform < 0, risk, 0.0).mean())
    hbgi = float(np.where(transform > 0, risk, 0.0).mean())
    return lbgi, hbgi


def summarize_glucose(glucose_mgdl) -> dict[str, float]:
    glucose = pd.Series(glucose_mgdl, dtype=float).dropna()
    if glucose.empty:
        return {
            "mean_glucose_mgdl": np.nan,
            "gmi_percent": np.nan,
            "cv_percent": np.nan,
            "tir_percent": np.nan,
            "tbr_percent": np.nan,
            "tar_percent": np.nan,
            "lbgi": np.nan,
            "hbgi": np.nan,
        }
    mean = float(glucose.mean())
    sd = float(glucose.std(ddof=1))
    lbgi, hbgi = risk_indices(glucose)
    return {
        "mean_glucose_mgdl": mean,
        "gmi_percent": gmi_percent(mean),
        "cv_percent": 100.0 * sd / mean,
        "tir_percent": float(glucose.between(70, 180).mean() * 100.0),
        "tbr_percent": float((glucose < 70).mean() * 100.0),
        "tar_percent": float((glucose > 180).mean() * 100.0),
        "lbgi": lbgi,
        "hbgi": hbgi,
    }

