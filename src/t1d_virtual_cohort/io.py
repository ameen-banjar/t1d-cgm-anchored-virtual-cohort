from __future__ import annotations

from pathlib import Path

import pandas as pd


REAL_ALIASES = {
    "subject_id": ["subject_id", "Real_Patient_ID", "Patient_ID"],
    "mean_glucose_mgdl": ["mean_glucose_mgdl", "Real_Mean", "mean_glucose"],
    "gmi_percent": ["gmi_percent", "Real_GMI", "gmi"],
    "cv_percent": ["cv_percent", "Real_CV", "cv"],
    "tir_percent": ["tir_percent", "Real_TIR", "tir"],
    "tbr_percent": ["tbr_percent", "Real_TBR", "tbr"],
    "tar_percent": ["tar_percent", "Real_TAR", "tar"],
    "lbgi": ["lbgi", "Real_LBGI"],
    "hbgi": ["hbgi", "Real_HBGI"],
}

VIRTUAL_ALIASES = {
    "virtual_id": ["virtual_id", "Sim_ID", "Best_Digital_Twin_ID"],
    "scenario": ["scenario", "Scenario", "Best_Scenario"],
    "trace_path": ["trace_path", "File_Path", "Virtual_File_Path"],
    "mean_glucose_mgdl": ["mean_glucose_mgdl", "Sim_Mean", "mean_glucose"],
    "gmi_percent": ["gmi_percent", "Sim_GMI", "gmi"],
    "cv_percent": ["cv_percent", "Sim_CV", "cv"],
    "tir_percent": ["tir_percent", "Sim_TIR", "tir"],
    "tbr_percent": ["tbr_percent", "Sim_TBR", "tbr"],
    "tar_percent": ["tar_percent", "Sim_TAR", "tar"],
    "lbgi": ["lbgi", "Sim_LBGI"],
    "hbgi": ["hbgi", "Sim_HBGI"],
}


def _standardize(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    rename = {}
    for canonical, candidates in aliases.items():
        found = next((candidate for candidate in candidates if candidate in df), None)
        if found is not None:
            rename[found] = canonical
    return df.rename(columns=rename)


def read_real_summary(path: str | Path) -> pd.DataFrame:
    df = _standardize(pd.read_csv(path), REAL_ALIASES)
    required = {"subject_id", "mean_glucose_mgdl", "cv_percent", "tir_percent"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Real summary is missing columns: {missing}")
    return df


def read_virtual_summary(path: str | Path) -> pd.DataFrame:
    df = _standardize(pd.read_csv(path), VIRTUAL_ALIASES)
    required = {
        "virtual_id",
        "scenario",
        "mean_glucose_mgdl",
        "cv_percent",
        "tir_percent",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Virtual summary is missing columns: {missing}")
    return df


def read_trace(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = next(
        (c for c in ["timestamp", "ts", "Time", "Datetime"] if c in df), None
    )
    glucose_col = next((c for c in ["glucose_mgdl", "CGM", "Glucose"] if c in df), None)
    if time_col is None or glucose_col is None:
        raise ValueError(f"Trace {path} must contain time and glucose columns")
    out = df[[time_col, glucose_col]].rename(
        columns={time_col: "timestamp", glucose_col: "glucose_mgdl"}
    )
    out["timestamp"] = pd.to_datetime(
        out["timestamp"], errors="coerce", format="mixed"
    )
    out["glucose_mgdl"] = pd.to_numeric(out["glucose_mgdl"], errors="coerce")
    return (
        out.dropna()
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def locate_real_trace(directory: str | Path, subject_id: str) -> Path:
    directory = Path(directory)
    candidates = [
        directory / f"{subject_id}.csv",
        directory / f"{subject_id}_processed.csv",
    ]
    found = next((path for path in candidates if path.exists()), None)
    if found is None:
        raise FileNotFoundError(f"No CGM trace found for {subject_id} in {directory}")
    return found


def index_virtual_traces(directory: str | Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for path in Path(directory).rglob("*.csv"):
        if path.name.lower().startswith("summary"):
            continue
        virtual_id = path.name.split("_V0", 1)[0]
        index[(virtual_id, path.parent.name)] = path
    return index
