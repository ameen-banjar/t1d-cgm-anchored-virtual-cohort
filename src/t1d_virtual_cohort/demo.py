from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import gmi_percent


def create_demo_data(output_dir) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n_real, n_virtual = 100, 36
    real = pd.DataFrame(
        {
            "subject_id": [f"SYN-R-{i:03d}" for i in range(n_real)],
            "mean_glucose_mgdl": rng.normal(167, 30, n_real).clip(80, 320),
            "cv_percent": rng.normal(38, 7, n_real).clip(15, 70),
            "tir_percent": rng.normal(59, 16, n_real).clip(0, 100),
            "tbr_percent": rng.gamma(2, 2, n_real).clip(0, 35),
            "lbgi": rng.gamma(1.5, 0.7, n_real),
            "hbgi": rng.gamma(2.5, 3.3, n_real),
        }
    )
    real["tar_percent"] = (
        100 - real["tir_percent"] - real["tbr_percent"]
    ).clip(0)
    real["gmi_percent"] = real["mean_glucose_mgdl"].map(gmi_percent)
    virtual = pd.DataFrame(
        {
            "virtual_id": [f"SYN-V-{i:03d}" for i in range(n_virtual)],
            "scenario": [f"V{(i % 6) + 1:02d}" for i in range(n_virtual)],
            "mean_glucose_mgdl": rng.normal(160, 42, n_virtual).clip(70, 330),
            "cv_percent": rng.normal(34, 11, n_virtual).clip(12, 75),
            "tir_percent": rng.normal(62, 23, n_virtual).clip(0, 100),
            "tbr_percent": rng.gamma(2, 4, n_virtual).clip(0, 50),
            "lbgi": rng.gamma(1.8, 1.1, n_virtual),
            "hbgi": rng.gamma(2.2, 3.0, n_virtual),
        }
    )
    virtual["tar_percent"] = (
        100 - virtual["tir_percent"] - virtual["tbr_percent"]
    ).clip(0)
    virtual["gmi_percent"] = virtual["mean_glucose_mgdl"].map(gmi_percent)
    real_path = output / "real_summary_synthetic.csv"
    virtual_path = output / "virtual_summary_synthetic.csv"
    real.to_csv(real_path, index=False)
    virtual.to_csv(virtual_path, index=False)
    return real_path, virtual_path

