from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinsepsis.agent import gru_model as gru_model_module  # noqa: E402
from twinsepsis.agent import risk_model as risk_model_module  # noqa: E402
from twinsepsis.agent.risk_model import (  # noqa: E402
    DEFAULT_XGB_MODEL_PATH,
    PHYSIONET_FEATURE_COLUMNS,
    EnsembleRisk,
    ModelUnavailableError,
    PhysioNetXGBRisk,
    StubRisk,
    load_default_risk_model,
    load_risk_model,
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


class FakeComponent:
    """Minimal component stand-in for blend/gating tests (no torch, no XGBoost)."""

    def __init__(self, probability: float, feature_names, artifact: str) -> None:
        self.probability = float(probability)
        # XGBoost exposes ``feature_names``; the GRU exposes ``feature_columns``.
        self.feature_names = list(feature_names)
        self.feature_columns = tuple(feature_names)
        self.artifact = artifact
        self.explain_calls = 0

    def featurize(self, window_df):
        return pd.DataFrame([{"fake": True}])

    def predict_proba(self, window_df) -> float:
        return self.probability

    def explain(self, window_df, top_k: int = 5):
        self.explain_calls += 1
        raise AssertionError("component explanations must not back the ensemble")


def ensemble_payload(**overrides) -> dict:
    payload = {
        "model": "weighted_probability_ensemble",
        "components": ["XGBoost", "GRU"],
        "xgboost_weight": 0.55,
        "gru_weight": 0.45,
        "threshold": 0.81,
    }
    payload.update(overrides)
    return payload


def write_config(directory: Path, payload: dict) -> Path:
    path = directory / "ensemble_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class EnsembleTests(unittest.TestCase):
    def make_ensemble(self, p_gru: float = 0.2, p_xgb: float = 0.6) -> EnsembleRisk:
        return EnsembleRisk(
            xgboost_model=FakeComponent(p_xgb, PHYSIONET_FEATURE_COLUMNS, "fake-xgb"),
            gru_model=FakeComponent(p_gru, PHYSIONET_FEATURE_COLUMNS, "fake-gru"),
        )

    def test_frozen_weights_and_threshold_load_from_config(self) -> None:
        model = self.make_ensemble()
        self.assertAlmostEqual(model.gru_weight, 0.45)
        self.assertAlmostEqual(model.xgboost_weight, 0.55)
        self.assertAlmostEqual(model.threshold, 0.81)
        self.assertAlmostEqual(model.combine(0.2, 0.6), 0.45 * 0.2 + 0.55 * 0.6)

    def test_threshold_cannot_be_overridden_by_callers(self) -> None:
        with self.assertRaises(TypeError):
            EnsembleRisk(threshold=0.5)  # type: ignore[call-arg]

    def test_blend_uses_gru_for_gru_and_xgboost_for_xgboost(self) -> None:
        model = self.make_ensemble(p_gru=0.9, p_xgb=0.1)
        expected = 0.45 * 0.9 + 0.55 * 0.1
        probability = model.predict_proba(pd.DataFrame([{"HR": 90.0}]))
        self.assertAlmostEqual(probability, expected)

    def test_capabilities_match_the_physionet_ensemble(self) -> None:
        model = self.make_ensemble()
        self.assertTrue(model.supports_prediction)
        self.assertFalse(model.supports_explanation)
        self.assertFalse(model.supports_simulation)
        self.assertIn("PhysioNet", model.dataset)
        self.assertNotIn("MIMIC", model.dataset)
        self.assertIsNone(model.explanation_units)

    def test_explain_is_unavailable_and_never_delegates_to_xgboost(self) -> None:
        model = self.make_ensemble()
        with self.assertRaises(NotImplementedError):
            model.explain(pd.DataFrame([{"HR": 90.0}]), top_k=5)
        self.assertEqual(model._xgboost.explain_calls, 0)

    def test_combine_rejects_invalid_probabilities(self) -> None:
        model = self.make_ensemble()
        for bad in (float("nan"), -0.1, 1.1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    model.combine(bad, 0.5)
                with self.assertRaises(ValueError):
                    model.combine(0.5, bad)

    def test_feature_order_mismatch_is_rejected(self) -> None:
        reordered = list(PHYSIONET_FEATURE_COLUMNS)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaises(ValueError):
            EnsembleRisk(
                xgboost_model=FakeComponent(
                    0.5, PHYSIONET_FEATURE_COLUMNS, "fake-xgb"
                ),
                gru_model=FakeComponent(0.5, reordered, "fake-gru"),
            )

    def test_gru_failure_raises_instead_of_falling_back(self) -> None:
        with mock.patch.object(
            gru_model_module,
            "PhysioNetGRURisk",
            side_effect=RuntimeError("synthetic GRU failure"),
        ):
            with self.assertRaises(ModelUnavailableError) as caught:
                EnsembleRisk()

        message = str(caught.exception)
        self.assertIn("GRU", message)
        self.assertIn("not fall back", message)
        self.assertIn("synthetic GRU failure", message)

    def test_xgboost_failure_raises(self) -> None:
        with mock.patch.object(
            risk_model_module,
            "PhysioNetXGBRisk",
            side_effect=RuntimeError("synthetic XGB failure"),
        ):
            with self.assertRaises(ModelUnavailableError) as caught:
                EnsembleRisk()

        self.assertIn("XGBoost", str(caught.exception))

    def test_config_validation_fails_loudly(self) -> None:
        cases = [
            (ensemble_payload(xgboost_weight=0.5, gru_weight=0.4), "sum to 1.0"),
            (ensemble_payload(components=["XGBoost"]), "components"),
            (ensemble_payload(model="something_else"), "weighted_probability_ensemble"),
            (ensemble_payload(threshold=float("nan")), "threshold"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for payload, expected in cases:
                path = write_config(root, payload)
                with self.subTest(expected=expected):
                    with self.assertRaises(ValueError) as caught:
                        EnsembleRisk(config_path=path)
                    self.assertIn(expected, str(caught.exception))

    def test_missing_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            EnsembleRisk(config_path=Path("does-not-exist.json"))

    def test_load_risk_model_dispatch(self) -> None:
        self.assertIsInstance(load_risk_model("xgboost"), PhysioNetXGBRisk)
        with self.assertRaises(ValueError):
            load_risk_model("gru")
        with self.assertRaises(TypeError):
            load_risk_model(None)  # type: ignore[arg-type]


class StubTests(unittest.TestCase):
    def test_stub_is_explicitly_named(self) -> None:
        model = StubRisk()
        self.assertEqual(model.name, "StubRisk")
        self.assertTrue(model.test_only)
        self.assertIn("TEST-ONLY", (StubRisk.__doc__ or ""))


if __name__ == "__main__":
    unittest.main()
