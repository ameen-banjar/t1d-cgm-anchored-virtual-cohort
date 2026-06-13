from __future__ import annotations

import numpy as np
import pandas as pd

from .io import index_virtual_traces, locate_real_trace, read_trace
from .metrics import risk_indices


def _window(df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    return df[(hour >= start_hour) & (hour < end_hour)]


def _coverage(
    df: pd.DataFrame,
    start_hour: int,
    end_hour: int,
    expected_interval_minutes: int,
) -> float:
    days = df["timestamp"].dt.date.nunique()
    expected_per_day = (end_hour - start_hour) * 60 / expected_interval_minutes
    observed = _window(df, start_hour, end_hour)["timestamp"].nunique()
    return float(observed / (expected_per_day * days)) if days else np.nan


def compute_diurnal_diagnostics(
    matches: pd.DataFrame,
    real_trace_dir,
    virtual_trace_dir,
    minimum_completeness: float = 0.70,
    expected_interval_minutes: int = 15,
    nocturnal_window: tuple[int, int] = (0, 6),
    dawn_window: tuple[int, int] = (4, 8),
) -> pd.DataFrame:
    virtual_index = index_virtual_traces(virtual_trace_dir)
    virtual_cache = {}
    rows = []
    for row in matches.itertuples(index=False):
        real = read_trace(locate_real_trace(real_trace_dir, str(row.subject_id)))
        nocturnal_coverage = _coverage(
            real, *nocturnal_window, expected_interval_minutes
        )
        dawn_coverage = _coverage(real, *dawn_window, expected_interval_minutes)
        qualified = (
            nocturnal_coverage >= minimum_completeness
            and dawn_coverage >= minimum_completeness
        )
        real_nocturnal = risk_indices(
            _window(real, *nocturnal_window)["glucose_mgdl"]
        )
        real_dawn = risk_indices(_window(real, *dawn_window)["glucose_mgdl"])
        key = (str(row.virtual_id), str(row.scenario))
        if key not in virtual_cache:
            path = virtual_index.get(key)
            if path is None:
                raise FileNotFoundError(f"No virtual trace found for {key}")
            virtual = read_trace(path)
            virtual_cache[key] = (
                risk_indices(_window(virtual, *nocturnal_window)["glucose_mgdl"]),
                risk_indices(_window(virtual, *dawn_window)["glucose_mgdl"]),
            )
        virtual_nocturnal, virtual_dawn = virtual_cache[key]
        rows.append(
            {
                "subject_id": row.subject_id,
                "member_key": row.member_key,
                "qualified": qualified,
                "nocturnal_completeness": nocturnal_coverage,
                "dawn_completeness": dawn_coverage,
                "real_nocturnal_lbgi": real_nocturnal[0],
                "real_nocturnal_hbgi": real_nocturnal[1],
                "virtual_nocturnal_lbgi": virtual_nocturnal[0],
                "virtual_nocturnal_hbgi": virtual_nocturnal[1],
                "real_dawn_lbgi": real_dawn[0],
                "real_dawn_hbgi": real_dawn[1],
                "virtual_dawn_lbgi": virtual_dawn[0],
                "virtual_dawn_hbgi": virtual_dawn[1],
            }
        )
    return pd.DataFrame(rows)

