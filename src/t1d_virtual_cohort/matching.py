from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


DEFAULT_FEATURES = ["mean_glucose_mgdl", "cv_percent", "tir_percent"]


def match_members(
    real: pd.DataFrame,
    virtual: pd.DataFrame,
    features: list[str] | None = None,
    normalization: str = "minmax",
) -> pd.DataFrame:
    features = features or DEFAULT_FEATURES
    scaler = MinMaxScaler() if normalization == "minmax" else StandardScaler()
    combined = pd.concat([real[features], virtual[features]], ignore_index=True)
    scaler.fit(combined)
    real_scaled = scaler.transform(real[features])
    virtual_scaled = scaler.transform(virtual[features])
    distances = np.sqrt(
        ((real_scaled[:, None, :] - virtual_scaled[None, :, :]) ** 2).sum(axis=2)
    )
    nearest = distances.argmin(axis=1)
    rows = []
    for i, j in enumerate(nearest):
        row = {
            "subject_id": real.iloc[i]["subject_id"],
            "virtual_row": int(j),
            "virtual_id": virtual.iloc[j]["virtual_id"],
            "scenario": virtual.iloc[j]["scenario"],
            "member_key": f"{virtual.iloc[j]['virtual_id']}|{virtual.iloc[j]['scenario']}",
            "distance": float(distances[i, j]),
        }
        for metric in sorted(
            set(features)
            | {
                "gmi_percent",
                "tbr_percent",
                "tar_percent",
                "lbgi",
                "hbgi",
            }
        ):
            if metric in real and metric in virtual:
                row[f"real_{metric}"] = real.iloc[i][metric]
                row[f"virtual_{metric}"] = virtual.iloc[j][metric]
        rows.append(row)
    return pd.DataFrame(rows)


def feature_ablation(
    real: pd.DataFrame,
    virtual: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = features or DEFAULT_FEATURES
    baseline = match_members(real, virtual, features)[["subject_id", "member_key"]]
    baseline = baseline.rename(columns={"member_key": "baseline_member"})
    rows = []
    for removed in features:
        reduced = [feature for feature in features if feature != removed]
        result = match_members(real, virtual, reduced)[["subject_id", "member_key"]]
        merged = baseline.merge(result, on="subject_id")
        rows.append(
            {
                "removed_feature": removed,
                "reassigned_percent": float(
                    (merged["baseline_member"] != merged["member_key"]).mean() * 100
                ),
            }
        )
    return pd.DataFrame(rows)

