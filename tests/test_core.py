import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from t1d_virtual_cohort.matching import match_members
from t1d_virtual_cohort.metrics import gmi_percent, risk_indices
from t1d_virtual_cohort.pipeline import _add_robust_verdict
from t1d_virtual_cohort.statistics import paired_tost


class CoreTests(unittest.TestCase):
    def test_gmi_is_deterministic_function_of_mean(self):
        self.assertAlmostEqual(gmi_percent(100), 5.702)

    def test_risk_direction(self):
        low = risk_indices([50, 55, 60])
        high = risk_indices([220, 250, 280])
        self.assertGreater(low[0], low[1])
        self.assertGreater(high[1], high[0])

    def test_tost_handles_zero_variance(self):
        result = paired_tost([1, 2, 3], [1, 2, 3], margin=0.5)
        self.assertTrue(result["equivalent"])
        self.assertEqual(result["p_tost"], 0.0)

    def test_default_match_excludes_gmi(self):
        real = pd.DataFrame(
            {
                "subject_id": ["R1"],
                "mean_glucose_mgdl": [150],
                "cv_percent": [35],
                "tir_percent": [60],
                "gmi_percent": [gmi_percent(150)],
            }
        )
        virtual = pd.DataFrame(
            {
                "virtual_id": ["V1", "V2"],
                "scenario": ["V01", "V02"],
                "mean_glucose_mgdl": [151, 220],
                "cv_percent": [35, 50],
                "tir_percent": [61, 20],
                "gmi_percent": [gmi_percent(151), gmi_percent(220)],
            }
        )
        match = match_members(real, virtual)
        self.assertEqual(match.loc[0, "virtual_id"], "V1")

    def test_robust_verdict_requires_both_intervals(self):
        result = {
            "margin": 1.0,
            "equivalent": True,
            "cluster_bootstrap_ci90_low": -1.2,
            "cluster_bootstrap_ci90_high": 0.4,
        }
        _add_robust_verdict(result)
        self.assertFalse(result["cluster_bootstrap_equivalent"])
        self.assertEqual(result["robust_verdict"], "Not robust")

    def test_committed_virtual_cohort_is_simulator_only_and_portable(self):
        path = Path(__file__).parents[1] / "data/derived/virtual_profile_summary.csv"
        cohort = pd.read_csv(path)
        self.assertEqual(len(cohort), 180)
        self.assertEqual(cohort["virtual_id"].nunique(), 30)
        self.assertEqual(cohort["scenario"].nunique(), 6)
        self.assertNotIn("subject_id", cohort.columns)
        self.assertNotIn("trace_path", cohort.columns)
        self.assertNotIn("File_Path", cohort.columns)


if __name__ == "__main__":
    unittest.main()
