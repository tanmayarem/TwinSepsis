from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinsepsis.agent.risk_model import StubRisk  # noqa: E402
from twinsepsis.agent.simulate import simulate  # noqa: E402
from twinsepsis.agent.twin import PatientTwin  # noqa: E402


def make_twin() -> PatientTwin:
    rows = [
        {
            "HR": 82.0 + index,
            "Temp": 37.0 + 0.1 * index,
            "Resp": 18.0,
            "SBP": 120.0,
            "MAP": 85.0,
            "WBC": 8.0,
            "Lactate": 1.0,
            "ICULOS": index + 1,
            "HospAdmTime": -2.0,
        }
        for index in range(6)
    ]
    return PatientTwin("synthetic-sim", StubRisk(), pd.DataFrame(rows))


class SimulateNoMutationTests(unittest.TestCase):
    def test_all_scenarios_leave_the_twin_unchanged(self) -> None:
        twin = make_twin()
        snapshot = twin.history.copy(deep=True)
        risks_before = twin.trajectory()

        for intervention in ("no_action", "fluid_bolus", "vasopressor", "antibiotics"):
            simulate(twin, intervention)

        pd.testing.assert_frame_equal(twin.history, snapshot)
        self.assertEqual(twin.trajectory(), risks_before)
        self.assertEqual(twin.latest.to_dict(), snapshot.iloc[-1].to_dict())

    def test_no_action_reports_baseline_risk_at_all_horizons(self) -> None:
        twin = make_twin()
        result = simulate(twin, "no_action")

        self.assertTrue(result["available"])
        self.assertEqual(result["horizons_hours"], [1, 3, 6])
        for hour in ("1", "3", "6"):
            self.assertIsInstance(result["projected_risk"][hour], float)
            self.assertEqual(result["bounds"][hour]["low"], result["projected_risk"][hour])
            self.assertEqual(result["bounds"][hour]["high"], result["projected_risk"][hour])
            self.assertEqual(
                result["baseline_no_action"][hour],
                result["projected_risk"][hour],
            )

    def test_todo_assumptions_produce_no_invented_risk(self) -> None:
        twin = make_twin()
        result = simulate(twin, "fluid_bolus")

        self.assertFalse(result["available"])
        self.assertEqual(result["effect_status"], "placeholder")
        self.assertTrue(result["unresolved"])
        self.assertTrue(any("TODO" in value for value in result["assumptions"][0].values()))
        for hour in ("1", "3", "6"):
            self.assertIsNone(result["projected_risk"][hour])
            self.assertIsNone(result["bounds"][hour]["low"])
            self.assertIsNone(result["bounds"][hour]["high"])
            self.assertIsInstance(result["baseline_no_action"][hour], float)

    def test_numeric_test_fixture_applies_effects_and_bounds(self) -> None:
        twin = make_twin()
        snapshot = twin.history.copy(deep=True)

        fixture = {
            "interventions": {
                "fluid_bolus": {
                    "label": "Test fixture only",
                    "description": "Synthetic test fixture; not a clinical assumption.",
                    "citation": "Synthetic unit-test fixture",
                    "onset_hours": 1,
                    "duration_hours": 6,
                    "effects": [
                        {
                            "target": "MAP",
                            "delta": 10.0,
                            "delta_low": 0.0,
                            "delta_high": 20.0,
                            "unit": "mmHg",
                            "citation": "Synthetic unit-test fixture",
                        }
                    ],
                }
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "effects.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            result = simulate(twin, "fluid_bolus", effects_path=path)

        self.assertTrue(result["available"])
        self.assertEqual(result["effect_status"], "numeric")
        for hour in ("1", "3", "6"):
            point = result["projected_risk"][hour]
            self.assertIsInstance(point, float)
            self.assertLessEqual(result["bounds"][hour]["low"], point)
            self.assertLessEqual(point, result["bounds"][hour]["high"])

        pd.testing.assert_frame_equal(twin.history, snapshot)


if __name__ == "__main__":
    unittest.main()
