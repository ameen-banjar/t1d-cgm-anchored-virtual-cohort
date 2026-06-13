#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from t1d_virtual_cohort.diurnal import _window
from t1d_virtual_cohort.io import index_virtual_traces, read_trace
from t1d_virtual_cohort.metrics import risk_indices
from t1d_virtual_cohort.statistics import (
    cluster_bootstrap_mean_ci,
    paired_tost,
)


BASE_FEATURES = ["mean_glucose_mgdl", "cv_percent", "tir_percent"]
OUTCOMES = [
    "nocturnal_lbgi",
    "nocturnal_hbgi",
    "dawn_lbgi",
    "dawn_hbgi",
]
MARGINS = {
    "nocturnal_lbgi": 1.0,
    "nocturnal_hbgi": 2.0,
    "dawn_lbgi": 1.0,
    "dawn_hbgi": 2.0,
}
CANDIDATES = {
    "baseline_3d": BASE_FEATURES,
    "plus_nocturnal_mean": BASE_FEATURES + ["nocturnal_mean"],
    "plus_dawn_mean": BASE_FEATURES + ["dawn_mean"],
    "plus_window_means": BASE_FEATURES + ["nocturnal_mean", "dawn_mean"],
    "plus_window_tar": BASE_FEATURES + ["nocturnal_tar", "dawn_tar"],
    "plus_means_and_tar": BASE_FEATURES
    + ["nocturnal_mean", "dawn_mean", "nocturnal_tar", "dawn_tar"],
}


def _temporal_features(trace: pd.DataFrame) -> dict[str, float]:
    result = {}
    for window, hours in [("nocturnal", (0, 6)), ("dawn", (4, 8))]:
        glucose = _window(trace, *hours)["glucose_mgdl"].astype(float)
        result[f"{window}_mean"] = float(glucose.mean())
        result[f"{window}_tar"] = float((glucose > 180).mean() * 100.0)
        lbgi, hbgi = risk_indices(glucose)
        result[f"{window}_lbgi"] = lbgi
        result[f"{window}_hbgi"] = hbgi
    return result


def _split_bucket(subject_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{subject_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 10


def _match(
    development: pd.DataFrame,
    subjects: pd.DataFrame,
    virtual: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    scaler = MinMaxScaler().fit(
        pd.concat([development[features], virtual[features]], ignore_index=True)
    )
    virtual_scaled = scaler.transform(virtual[features])
    subject_scaled = scaler.transform(subjects[features])
    distances = np.sqrt(
        ((subject_scaled[:, None, :] - virtual_scaled[None, :, :]) ** 2).sum(axis=2)
    )
    nearest = distances.argmin(axis=1)
    matched = subjects[["subject_id", *OUTCOMES]].reset_index(drop=True).copy()
    matched["member_key"] = virtual.iloc[nearest]["member_key"].to_numpy()
    for outcome in OUTCOMES:
        matched[f"virtual_{outcome}"] = virtual.iloc[nearest][outcome].to_numpy()
    return matched


def _development_score(matched: pd.DataFrame) -> float:
    return float(
        sum(
            abs((matched[outcome] - matched[f"virtual_{outcome}"]).mean())
            / MARGINS[outcome]
            for outcome in OUTCOMES
        )
    )


def _validation_rows(
    model: str,
    matched: pd.DataFrame,
    replicates: int,
    seed: int,
) -> list[dict]:
    rows = []
    for outcome in OUTCOMES:
        real = matched[outcome]
        virtual = matched[f"virtual_{outcome}"]
        result = paired_tost(real, virtual, MARGINS[outcome])
        cluster_low, cluster_high = cluster_bootstrap_mean_ci(
            real - virtual,
            matched["member_key"],
            replicates=replicates,
            seed=seed,
        )
        rows.append(
            {
                "model": model,
                "metric": outcome,
                "n": len(matched),
                "real_mean": real.mean(),
                "virtual_mean": virtual.mean(),
                "bias_real_minus_virtual": result["bias_real_minus_virtual"],
                "ci90_low": result["ci90_low"],
                "ci90_high": result["ci90_high"],
                "cluster_bootstrap_ci90_low": cluster_low,
                "cluster_bootstrap_ci90_high": cluster_high,
                "paired_equivalent": result["equivalent"],
                "cluster_equivalent": (
                    cluster_low > -MARGINS[outcome]
                    and cluster_high < MARGINS[outcome]
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select temporal matching features on development subjects "
        "and evaluate the selected model on held-out subjects."
    )
    parser.add_argument("--real-summary", required=True)
    parser.add_argument("--diurnal-subjects", required=True)
    parser.add_argument("--real-traces", required=True)
    parser.add_argument("--virtual-summary", required=True)
    parser.add_argument("--virtual-traces", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    diurnal = pd.read_csv(args.diurnal_subjects)
    qualified_ids = diurnal.loc[diurnal["qualified"], "subject_id"].astype(str)
    real_summary = pd.read_csv(args.real_summary)
    real_summary["subject_id"] = real_summary["subject_id"].astype(str)
    real_summary = real_summary.set_index("subject_id")

    real_rows = []
    for subject_id in qualified_ids:
        trace = read_trace(Path(args.real_traces) / f"{subject_id}.csv")
        row = {"subject_id": subject_id, **real_summary.loc[subject_id].to_dict()}
        row.update(_temporal_features(trace))
        real_rows.append(row)
    real = pd.DataFrame(real_rows)

    virtual = pd.read_csv(args.virtual_summary)
    virtual["member_key"] = (
        virtual["virtual_id"].astype(str) + "|" + virtual["scenario"].astype(str)
    )
    virtual_index = index_virtual_traces(args.virtual_traces)
    virtual_rows = []
    for row in virtual.itertuples(index=False):
        values = row._asdict()
        trace = read_trace(
            virtual_index[(str(row.virtual_id), str(row.scenario))]
        )
        values.update(_temporal_features(trace))
        virtual_rows.append(values)
    virtual = pd.DataFrame(virtual_rows)

    real["split"] = real["subject_id"].map(
        lambda subject_id: (
            "development"
            if _split_bucket(subject_id, args.seed) < 6
            else "validation"
        )
    )
    development = real[real["split"] == "development"].reset_index(drop=True)
    validation = real[real["split"] == "validation"].reset_index(drop=True)

    development_matches = {}
    selection_rows = []
    for model, features in CANDIDATES.items():
        matched = _match(development, development, virtual, features)
        development_matches[model] = matched
        selection_rows.append(
            {
                "model": model,
                "development_n": len(development),
                "development_score": _development_score(matched),
                "features": ",".join(features),
            }
        )
    selection = pd.DataFrame(selection_rows).sort_values("development_score")
    selected_model = str(selection.iloc[0]["model"])
    selection.to_csv(output / "temporal_matching_model_selection.csv", index=False)

    validation_rows = []
    for model in ["baseline_3d", selected_model]:
        matched = _match(
            development, validation, virtual, CANDIDATES[model]
        )
        validation_rows.extend(
            _validation_rows(
                model, matched, args.bootstrap_replicates, args.seed
            )
        )
    pd.DataFrame(validation_rows).to_csv(
        output / "temporal_matching_validation.csv", index=False
    )
    metadata = {
        "seed": args.seed,
        "development_n": len(development),
        "validation_n": len(validation),
        "selected_model": selected_model,
        "selection_criterion": (
            "Sum of absolute development-set mean biases divided by "
            "metric-specific equivalence margins"
        ),
    }
    (output / "temporal_matching_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
