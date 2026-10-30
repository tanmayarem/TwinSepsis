from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gru_test_support import (  # noqa: E402
    COMMITTED_PATIENTS,
    HAS_FULL_INTEGRATION,
    HAS_GRU_INTEGRATION,
    load_committed_predictions,
    synthetic_history,
)
from twinsepsis.agent import gru_model  # noqa: E402
from twinsepsis.agent.agent import SepsisAgent  # noqa: E402
from twinsepsis.agent.gru_model import GRUCheckpointError  # noqa: E402
from twinsepsis.agent.risk_model import (  # noqa: E402
    EnsembleRisk,
    ModelUnavailableError,
)
from twinsepsis.agent.tools import get_twin_state  # noqa: E402
from twinsepsis.agent.twin import PatientTwin  # noqa: E402
from twinsepsis.patient_source import load_physionet_patient  # noqa: E402
from gru_test_support import DATASET_ROOT  # noqa: E402


def assert_trend_matches_trajectory(
    test: unittest.TestCase, trend: dict, trajectory: list[float], hours: int
) -> None:
    """The bounded trend must equal the tail of the full ensemble trajectory."""

    window = min(hours, len(trajectory) - 1)
    test.assertEqual(trend["window_hours"], window)
    test.assertEqual(trend["current_risk"], trajectory[-1])
    test.assertEqual(trend["previous_risk"], trajectory[-1 - window])
    delta = trajectory[-1] - trajectory[-1 - window]
    test.assertEqual(trend["delta"], delta)
    test.assertEqual(trend["slope_per_hour"], delta / window)


@unittest.skipUnless(HAS_GRU_INTEGRATION, "torch and the GRU checkpoint are required")
class EnsembleTrendAndCapabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ensemble = EnsembleRisk()

    def test_trend_matches_full_trajectory_slice(self) -> None:
        twin = PatientTwin(
            "synthetic-ensemble-trend", self.ensemble, synthetic_history(14)
        )
        full = twin.trajectory()

        for hours in (1, 3, 13, 14, 24):
            with self.subTest(hours=hours):
                assert_trend_matches_trajectory(
                    self, twin.trend(hours), full, hours
                )

    def test_short_history_reports_insufficient_history(self) -> None:
        twin = PatientTwin(
            "synthetic-ensemble-trend-short", self.ensemble, synthetic_history(1)
        )

        trend = twin.trend(3)

        self.assertEqual(trend["direction"], "insufficient_history")
        self.assertEqual(trend["current_risk"], twin.risk())

    def test_agent_reports_unavailable_capabilities_and_stable_identity(self) -> None:
        twin = PatientTwin(
            "synthetic-ensemble-agent", self.ensemble, synthetic_history(6)
        )
        agent = SepsisAgent(twin, None, trace_enabled=False)

        self.assertNotIn("explain_risk", agent.available_tools)
        self.assertNotIn("simulate_intervention", agent.available_tools)

        response = agent.run_tools_only(twin.patient_id, twin.hours)
        limitations = response["limitations"]
        self.assertIn("feature_attribution_unavailable", limitations)
        self.assertIn("intervention_simulation_unavailable", limitations)
        self.assertIn(
            "Intervention simulation is unavailable for this model",
            " ".join(response["caveats"]),
        )

        identity = "PhysioNetXGBGRUEnsemble"
        self.assertEqual(response["prediction_source"]["model"], identity)
        self.assertEqual(agent._risk_trajectory(max_hours=3)["risk_model"], identity)
        self.assertEqual(get_twin_state(twin)["risk_model"], identity)

        # tools_only still records the unavailable plan steps; they must be
        # marked tool_unavailable rather than executed or silently dropped.
        failed = [item for item in response["tools_used"] if not item["ok"]]
        self.assertEqual(
            sorted(item["tool"] for item in failed),
            ["explain_risk", "simulate_intervention", "simulate_intervention"],
        )
        self.assertTrue(
            all(item["error_type"] == "tool_unavailable" for item in failed)
        )


@unittest.skipUnless(
    HAS_FULL_INTEGRATION, "torch, the GRU checkpoint, PhysioNet data and predictions are required"
)
class EnsembleIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ensemble = EnsembleRisk()
        cls.committed = load_committed_predictions()

    def _history(self, patient_id: str):
        hours = int(
            self.committed[self.committed["patient_id"] == patient_id]["hour_index"].max()
        ) + 1
        twin = load_physionet_patient(
            patient_id,
            hours,
            dataset_root=DATASET_ROOT,
            model=self.ensemble,
        )
        return twin.history

    def _committed_row(self, patient_id: str, hour_index: int):
        rows = self.committed[
            (self.committed["patient_id"] == patient_id)
            & (self.committed["hour_index"] == hour_index)
        ]
        self.assertEqual(len(rows), 1, f"missing committed row for {patient_id}")
        return rows.iloc[0]

    def test_frozen_configuration_and_capabilities(self) -> None:
        self.assertAlmostEqual(self.ensemble.xgboost_weight, 0.55)
        self.assertAlmostEqual(self.ensemble.gru_weight, 0.45)
        self.assertAlmostEqual(self.ensemble.threshold, 0.81)
        self.assertTrue(self.ensemble.supports_prediction)
        self.assertFalse(self.ensemble.supports_explanation)
        self.assertFalse(self.ensemble.supports_simulation)
        self.assertIn("PhysioNet", self.ensemble.dataset)
        self.assertNotIn("MIMIC", self.ensemble.dataset)
        self.assertIn("xgboost_baseline.json", self.ensemble.artifact)
        self.assertIn("gru_baseline.pt", self.ensemble.artifact)

    def test_components_and_blend_match_committed_predictions(self) -> None:
        for patient_id in COMMITTED_PATIENTS:
            with self.subTest(patient=patient_id):
                history = self._history(patient_id)
                hour_index = len(history) - 1
                committed = self._committed_row(patient_id, hour_index)

                components = self.ensemble.component_probabilities(history)
                blended = self.ensemble.predict_proba(history)

                self.assertLess(
                    abs(components["gru"] - float(committed["gru_probability"])), 1e-4
                )
                self.assertLess(
                    abs(components["xgboost"] - float(committed["xgb_probability"])), 1e-4
                )
                self.assertLess(
                    abs(blended - float(committed["ensemble_probability"])), 1e-4
                )
                self.assertAlmostEqual(
                    blended,
                    0.45 * components["gru"] + 0.55 * components["xgboost"],
                    places=12,
                )

    def test_ensemble_probability_is_bounded_and_finite(self) -> None:
        history = self._history("p000865")
        probability = self.ensemble.predict_proba(history)
        self.assertTrue(np.isfinite(probability))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_explanation_stays_unavailable(self) -> None:
        history = self._history("p014914")
        with self.assertRaises(NotImplementedError):
            self.ensemble.explain(history, top_k=5)

    def test_gru_failure_blocks_the_ensemble(self) -> None:
        with mock.patch.object(
            gru_model,
            "PhysioNetGRURisk",
            side_effect=GRUCheckpointError("synthetic checkpoint failure"),
        ):
            with self.assertRaises(ModelUnavailableError) as caught:
                EnsembleRisk()

        message = str(caught.exception)
        self.assertIn("GRU", message)
        self.assertIn("not fall back", message)
        self.assertIn("synthetic checkpoint failure", message)


if __name__ == "__main__":
    unittest.main()
