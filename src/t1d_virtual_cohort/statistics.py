from __future__ import annotations

import numpy as np
from scipy import stats


def paired_tost(real, virtual, margin: float, alpha: float = 0.05) -> dict:
    difference = np.asarray(real, dtype=float) - np.asarray(virtual, dtype=float)
    difference = difference[np.isfinite(difference)]
    n = difference.size
    if n < 2:
        raise ValueError("Paired TOST requires at least two finite differences")
    mean = float(difference.mean())
    sd = float(difference.std(ddof=1))
    se = sd / np.sqrt(n)
    if se == 0:
        p_tost = 0.0 if -margin < mean < margin else 1.0
        ci = (mean, mean)
    else:
        df = n - 1
        critical = stats.t.ppf(1 - alpha, df)
        ci = (mean - critical * se, mean + critical * se)
        p_lower = 1 - stats.t.cdf((mean + margin) / se, df)
        p_upper = stats.t.cdf((mean - margin) / se, df)
        p_tost = float(max(p_lower, p_upper))
    return {
        "n": int(n),
        "margin": float(margin),
        "real_mean": float(np.nanmean(real)),
        "virtual_mean": float(np.nanmean(virtual)),
        "bias_real_minus_virtual": mean,
        "ci90_low": float(ci[0]),
        "ci90_high": float(ci[1]),
        "p_tost": p_tost,
        "equivalent": bool(ci[0] > -margin and ci[1] < margin),
    }


def cluster_bootstrap_mean_ci(
    differences,
    clusters,
    replicates: int = 2000,
    seed: int = 2026,
    confidence: float = 0.90,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    labels = np.asarray(clusters)
    keep = np.isfinite(values)
    values, labels = values[keep], labels[keep]
    unique = np.unique(labels)
    grouped = {label: values[labels == label] for label in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for index in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        estimates[index] = np.concatenate([grouped[label] for label in sampled]).mean()
    tail = (1.0 - confidence) / 2.0
    return tuple(np.quantile(estimates, [tail, 1.0 - tail]).astype(float))

