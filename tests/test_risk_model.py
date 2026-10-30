from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinsepsis.agent.risk_model import (  # noqa: E402
    DEFAULT_XGB_MODEL_PATH,
    EnsembleRisk,
    PhysioNetXGBRisk,
    StubRisk,
    load_default_risk_model,
)


def physionet_row() -> dict:
    row = {
        "HR": 118.0,
        "O2Sat": 91.0,
        "Temp": 39.1,
        "SBP": 88.0,
        "MAP": 62.0,
        "DBP": 50.0,
        "Resp": 26.0,
        "EtCO2": 30.0,
        "BaseExcess": -4.0,
        "HCO3": 20.0,
        "FiO2": 0.5,
        "pH": 7.30,
        "PaCO2": 34.0,
        "SaO2": 90.0,
        "AST": 45.0,
        "BUN": 28.0,
        "Alkalinephos": 90.0,
        "Calcium": 8.5,
        "Chloride": 101.0,
        "Creatinine": 1.6,
        "Bilirubin_direct": 0.4,
        "Glucose": 150.0,
        "Lactate": 3.4,
        "Magnesium": 2.0,
        "Phosphate": 3.5,
        "Potassium": 4.7,
        "Bilirubin_total": 1.2,
        "TroponinI": 0.05,
        "Hct": 33.0,
        "Hgb": 11.0,
        "PTT": 38.0,
        "WBC": 16.0,
        "Fibrinogen": 400.0,
        "Platelets": 180.0,
        "Age": 72.0,
        "Gender": 1.0,
        "Unit1": 1.0,
        "Unit2": 0.0,
        "HospAdmTime": -3.5,
        "ICULOS": 6,
    }
    return row


class PhysioNetXGBTests(unittest.TestCase):
    def test_artifact_loads_and_predicts(self) -> None:
        self.assertTrue(DEFAULT_XGB_MODEL_PATH.exists())
        model = PhysioNetXGBRisk()
        features = model.featurize(pd.DataFrame([physionet_row()]))
        self.assertEqual(features.shape, (1, 40))
        self.assertTrue((features.dtypes == np.float32).all())

        probability = model.predict_proba(pd.DataFrame([physionet_row()]))
        self.assertTrue(math.isfinite(probability))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertEqual(model.threshold, 0.5)

    def test_explanations_are_ranked_and_complete(self) -> None:
        model = PhysioNetXGBRisk()
        explanations = model.explain(pd.DataFrame([physionet_row()]), top_k=5)
        self.assertEqual(len(explanations), 5)
        for item in explanations:
            self.assertEqual(
                set(item), {"feature", "label", "value", "contribution", "direction"}
            )
            self.assertIn(item["direction"], {"increases", "decreases", "neutral"})
        magnitudes = [abs(item["contribution"]) for item in explanations]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_missing_columns_fail_loudly(self) -> None:
        model = PhysioNetXGBRisk()
        with self.assertRaises(ValueError):
            model.predict_proba(pd.DataFrame([{"HR": 90.0}]))

    def test_default_loader_returns_real_model(self) -> None:
        self.assertIsInstance(load_default_risk_model(), PhysioNetXGBRisk)


class EnsembleScaffoldTests(unittest.TestCase):
    def test_weights_and_threshold_load_from_config(self) -> None:
        model = EnsembleRisk()
        self.assertAlmostEqual(model.gru_weight, 0.45)
        self.assertAlmostEqual(model.xgboost_weight, 0.55)
        self.assertAlmostEqual(model.threshold, 0.81)
        self.assertAlmostEqual(model.combine(0.2, 0.6), 0.45 * 0.2 + 0.55 * 0.6)

    def test_prediction_raises_instead_of_fabricating(self) -> None:
        model = EnsembleRisk()
        with self.assertRaises(NotImplementedError):
            model.predict_proba(pd.DataFrame([{"HR": 90.0}]))


class StubTests(unittest.TestCase):
    def test_stub_is_explicitly_named(self) -> None:
        model = StubRisk()
        self.assertEqual(model.name, "StubRisk")
        self.assertTrue(model.test_only)
        self.assertIn("TEST-ONLY", (StubRisk.__doc__ or ""))


if __name__ == "__main__":
    unittest.main()
