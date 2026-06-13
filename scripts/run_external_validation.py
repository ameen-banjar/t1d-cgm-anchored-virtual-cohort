#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

from t1d_virtual_cohort.diurnal import _window
from t1d_virtual_cohort.io import index_virtual_traces, read_trace
from t1d_virtual_cohort.metrics import risk_indices, summarize_glucose
from t1d_virtual_cohort.statistics import (
    cluster_bootstrap_mean_ci,
    paired_tost,
)


DIURNAL_MARGINS = {
    "nocturnal_lbgi": 1.0,
    "nocturnal_hbgi": 2.0,
    "dawn_lbgi": 1.0,
    "dawn_hbgi": 2.0,
}


def _clean_external_trace(
    path: Path,
    flatline_hours: float,
    glucose_min: float,
    glucose_max: float,
) -> tuple[pd.DataFrame, dict]:
    trace = read_trace(path)
    in_range = trace["glucose_mgdl"].between(
        glucose_min, glucose_max, inclusive="both"
    )
    run_id = trace["glucose_mgdl"].ne(trace["glucose_mgdl"].shift()).cumsum()
    run_size = trace.groupby(run_id).size()
    interval_minutes = float(
        trace["timestamp"].diff().dt.total_seconds().div(60).median()
    )
    long_runs = run_size.index[
        run_size * interval_minutes / 60.0 >= flatline_hours
    ]
    flatline = run_id.isin(long_runs)
    cleaned = trace[in_range & ~flatline].copy()
    metadata = {
        "raw_n": len(trace),
        "valid_n": len(cleaned),
        "excluded_range_n": int((~in_range).sum()),
        "excluded_flatline_n": int(flatline.sum()),
        "interval_minutes": interval_minutes,
    }
    return cleaned, metadata


def _window_completeness(
    trace: pd.DataFrame,
    first_timestamp: pd.Timestamp,
    last_timestamp: pd.Timestamp,
    start_hour: int,
    end_hour: int,
    interval_minutes: float,
) -> float:
    days = (last_timestamp.normalize() - first_timestamp.normalize()).days + 1
    expected_per_day = int((end_hour - start_hour) * 60 / interval_minutes)
    hour = trace["timestamp"].dt.hour + trace["timestamp"].dt.minute / 60.0
    window = trace.loc[
        (hour >= start_hour) & (hour < end_hour), "timestamp"
    ]
    observed = (
        window.groupby(window.dt.date).nunique().clip(upper=expected_per_day).sum()
    )
    return float(observed / (days * expected_per_day))


def _external_summary(path: Path, flatline_hours: float) -> dict:
    raw = read_trace(path)
    cleaned, metadata = _clean_external_trace(path, flatline_hours, 40.0, 400.0)
    first_timestamp = raw["timestamp"].min()
    last_timestamp = raw["timestamp"].max()
    expected = int(
        (last_timestamp - first_timestamp).total_seconds()
        / (metadata["interval_minutes"] * 60)
    ) + 1
    calendar_days = (
        last_timestamp.normalize() - first_timestamp.normalize()
    ).days + 1
    result = {
        "subject_id": path.stem,
        "calendar_days": int(calendar_days),
        "whole_day_completeness": float(len(cleaned) / expected),
        **metadata,
        **summarize_glucose(cleaned["glucose_mgdl"]),
    }
    for window, hours in [("nocturnal", (0, 6)), ("dawn", (4, 8))]:
        result[f"{window}_completeness"] = _window_completeness(
            cleaned,
            first_timestamp,
            last_timestamp,
            *hours,
            metadata["interval_minutes"],
        )
        glucose = _window(cleaned, *hours)["glucose_mgdl"]
        result[f"{window}_lbgi"], result[f"{window}_hbgi"] = risk_indices(
            glucose
        )
    return result


def _virtual_temporal_metrics(
    virtual: pd.DataFrame, virtual_trace_dir: Path
) -> pd.DataFrame:
    trace_index = index_virtual_traces(virtual_trace_dir)
    rows = []
    for row in virtual.itertuples(index=False):
        trace = read_trace(
            trace_index[(str(row.virtual_id), str(row.scenario))]
        )
        values = {}
        for window, hours in [("nocturnal", (0, 6)), ("dawn", (4, 8))]:
            glucose = _window(trace, *hours)["glucose_mgdl"]
            values[f"{window}_lbgi"], values[f"{window}_hbgi"] = (
                risk_indices(glucose)
            )
        rows.append(values)
    return pd.DataFrame(rows)


def _match_external(
    external: pd.DataFrame,
    development: pd.DataFrame,
    virtual: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    scaler = MinMaxScaler().fit(
        pd.concat([development[features], virtual[features]], ignore_index=True)
    )
    external_scaled = scaler.transform(external[features])
    virtual_scaled = scaler.transform(virtual[features])
    distances = np.sqrt(
        (
            (external_scaled[:, None, :] - virtual_scaled[None, :, :]) ** 2
        ).sum(axis=2)
    )
    nearest = distances.argmin(axis=1)
    matched = pd.DataFrame(
        {
            "subject_id": external["subject_id"].to_numpy(),
            "member_key": virtual.iloc[nearest]["member_key"].to_numpy(),
            "distance": distances[np.arange(len(external)), nearest],
        }
    )
    return matched, nearest


def _agreement_rows(
    matched: pd.DataFrame,
    metrics: list[str],
    margins: dict,
    analysis: str,
    flatline_hours: float,
    replicates: int,
    seed: int,
) -> list[dict]:
    rows = []
    for metric in metrics:
        real = matched[f"real_{metric}"]
        virtual = matched[f"virtual_{metric}"]
        result = paired_tost(real, virtual, margins[metric])
        cluster_low, cluster_high = cluster_bootstrap_mean_ci(
            real - virtual,
            matched["member_key"],
            replicates=replicates,
            seed=seed,
        )
        rows.append(
            {
                "flatline_threshold_hours": flatline_hours,
                "analysis": analysis,
                "metric": metric,
                **result,
                "cluster_bootstrap_ci90_low": cluster_low,
                "cluster_bootstrap_ci90_high": cluster_high,
                "cluster_equivalent": (
                    cluster_low > -margins[metric]
                    and cluster_high < margins[metric]
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run aggregate external validation on processed CGM traces."
    )
    parser.add_argument("--external-traces", required=True)
    parser.add_argument("--development-summary", required=True)
    parser.add_argument("--virtual-summary", required=True)
    parser.add_argument("--virtual-traces", required=True)
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--output", default="outputs")
    parser.add_argument(
        "--flatline-hours",
        type=float,
        nargs="+",
        default=[2.0, 6.0, 24.0],
    )
    args = parser.parse_args()

    output = Path(args.output)
    tables = output / "tables"
    private = output / "private"
    tables.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    development = pd.read_csv(args.development_summary)
    virtual = pd.read_csv(args.virtual_summary)
    virtual["member_key"] = (
        virtual["virtual_id"].astype(str)
        + "|"
        + virtual["scenario"].astype(str)
    )
    features = config["matching"]["features"]
    virtual_temporal = _virtual_temporal_metrics(
        virtual, Path(args.virtual_traces)
    )
    files = sorted(
        path
        for path in Path(args.external_traces).glob("Train_*.csv")
        if not path.stem.endswith("_enriched")
    )

    qc_rows = []
    private_rows = []
    whole_day_rows = []
    diurnal_rows = []
    whole_day_metrics = [
        "mean_glucose_mgdl",
        "cv_percent",
        "tir_percent",
        "gmi_percent",
        "tbr_percent",
        "tar_percent",
        "lbgi",
        "hbgi",
    ]
    bootstrap = config["bootstrap"]

    for flatline_hours in args.flatline_hours:
        external = pd.DataFrame(
            [
                _external_summary(path, flatline_hours)
                for path in files
            ]
        )
        external["qualified_whole_day"] = (
            (external["calendar_days"] >= 14)
            & (external["whole_day_completeness"] >= 0.70)
        )
        external["qualified_diurnal"] = (
            external["qualified_whole_day"]
            & (external["nocturnal_completeness"] >= 0.70)
            & (external["dawn_completeness"] >= 0.70)
        )
        qualified = external[external["qualified_whole_day"]].reset_index(
            drop=True
        )
        matched, nearest = _match_external(
            qualified, development, virtual, features
        )
        for metric in whole_day_metrics:
            matched[f"real_{metric}"] = qualified[metric].to_numpy()
            matched[f"virtual_{metric}"] = virtual.iloc[nearest][
                metric
            ].to_numpy()
        whole_day_rows.extend(
            _agreement_rows(
                matched,
                whole_day_metrics,
                config["equivalence_margins"],
                "whole_day",
                flatline_hours,
                bootstrap["replicates"],
                bootstrap["seed"],
            )
        )

        diurnal_indices = np.flatnonzero(
            qualified["qualified_diurnal"].to_numpy()
        )
        diurnal_matched = matched.iloc[diurnal_indices].reset_index(drop=True)
        diurnal_qualified = qualified.iloc[diurnal_indices].reset_index(
            drop=True
        )
        diurnal_nearest = nearest[diurnal_indices]
        for metric in DIURNAL_MARGINS:
            diurnal_matched[f"real_{metric}"] = diurnal_qualified[
                metric
            ].to_numpy()
            diurnal_matched[f"virtual_{metric}"] = virtual_temporal.iloc[
                diurnal_nearest
            ][metric].to_numpy()
        diurnal_rows.extend(
            _agreement_rows(
                diurnal_matched,
                list(DIURNAL_MARGINS),
                DIURNAL_MARGINS,
                "diurnal",
                flatline_hours,
                bootstrap["replicates"],
                bootstrap["seed"],
            )
        )

        qc_rows.append(
            {
                "flatline_threshold_hours": flatline_hours,
                "source_n": len(external),
                "whole_day_qualified_n": int(
                    external["qualified_whole_day"].sum()
                ),
                "diurnal_qualified_n": int(
                    external["qualified_diurnal"].sum()
                ),
                "median_whole_day_completeness": external[
                    "whole_day_completeness"
                ].median(),
                "minimum_whole_day_completeness": external[
                    "whole_day_completeness"
                ].min(),
                "range_excluded_total": int(
                    external["excluded_range_n"].sum()
                ),
                "flatline_excluded_total": int(
                    external["excluded_flatline_n"].sum()
                ),
                "selected_members": int(matched["member_key"].nunique()),
                "mean_distance": float(matched["distance"].mean()),
            }
        )
        external["flatline_threshold_hours"] = flatline_hours
        private_rows.append(external)

    pd.DataFrame(qc_rows).to_csv(
        tables / "external_validation_qc_sensitivity.csv", index=False
    )
    pd.DataFrame(whole_day_rows).to_csv(
        tables / "external_whole_day_validation.csv", index=False
    )
    pd.DataFrame(diurnal_rows).to_csv(
        tables / "external_diurnal_validation.csv", index=False
    )
    pd.concat(private_rows, ignore_index=True).to_csv(
        private / "external_validation_subject_qc.csv", index=False
    )
    print(pd.DataFrame(qc_rows).to_string(index=False))


if __name__ == "__main__":
    main()
