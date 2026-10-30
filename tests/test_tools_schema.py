from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import jsonschema
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinsepsis.agent.risk_model import StubRisk  # noqa: E402
from twinsepsis.agent.tools import (  # noqa: E402
    TOOL_FUNCTIONS,
    TOOL_OUTPUT_SCHEMAS,
    clinical_scores,
    explain_risk,
    get_twin_state,
    list_interventions,
    predict_risk,
    simulate_intervention,
)
from twinsepsis.agent.twin import PatientTwin  # noqa: E402


def make_twin(rows: list[dict]) -> PatientTwin:
    return PatientTwin("synthetic-tools", StubRisk(), pd.DataFrame(rows))


class ToolSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.twin = make_twin(
            [
                {
                    "HR": 110.0,
                    "Temp": 39.0,
                    "Resp": 24.0,
                    "SBP": 90.0,
                    "MAP": 65.0,
                    "WBC": 15.0,
                    "Lactate": 3.0,
                    "PaCO2": 30.0,
                    "GCS": 14.0,
                    "ICULOS": 5,
                    "HospAdmTime": -2.0,
                }
            ]
        )

    def assert_tool_output(self, name: str, output: dict) -> None:
        schema = TOOL_OUTPUT_SCHEMAS[name]
        jsonschema.validate(instance=output, schema=schema)
        json.dumps(output, allow_nan=False)
        self.assertIn(name, TOOL_FUNCTIONS)

    def test_get_twin_state_schema(self) -> None:
        self.assert_tool_output("get_twin_state", get_twin_state(self.twin))

    def test_predict_risk_schema(self) -> None:
        self.assert_tool_output("predict_risk", predict_risk(self.twin))

    def test_explain_risk_schema(self) -> None:
        self.assert_tool_output("explain_risk", explain_risk(self.twin, top_k=3))

    def test_list_interventions_schema(self) -> None:
        output = list_interventions()
        self.assert_tool_output("list_interventions", output)
        names = {item["name"] for item in output["interventions"]}
        self.assertEqual(names, {"fluid_bolus", "vasopressor", "antibiotics", "no_action"})

    def test_simulate_intervention_schema(self) -> None:
        available = simulate_intervention(self.twin, "no_action")
        self.assert_tool_output("simulate_intervention", available)
        self.assertTrue(available["available"])

        unavailable = simulate_intervention(self.twin, "fluid_bolus")
        self.assert_tool_output("simulate_intervention", unavailable)
        self.assertFalse(unavailable["available"])

    def test_clinical_scores_schema_and_values(self) -> None:
        output = clinical_scores(self.twin)
        self.assert_tool_output("clinical_scores", output)

        self.assertEqual(output["qsofa"]["score"], 3)
        self.assertTrue(output["qsofa"]["complete"])
        self.assertTrue(output["qsofa"]["positive"])

        self.assertEqual(output["sirs"]["score"], 4)
        self.assertTrue(output["sirs"]["complete"])
        self.assertTrue(output["sirs"]["positive"])

    def test_incomplete_scores_are_reported_not_imputed(self) -> None:
        twin = make_twin([{"HR": 80.0, "Temp": 37.0, "ICULOS": 1}])
        output = clinical_scores(twin)
        self.assert_tool_output("clinical_scores", output)

        self.assertFalse(output["qsofa"]["complete"])
        self.assertEqual(output["qsofa"]["available_count"], 0)
        self.assertIsNone(output["qsofa"]["positive"])

        self.assertFalse(output["sirs"]["complete"])
        self.assertEqual(output["sirs"]["available_count"], 2)
        self.assertEqual(output["sirs"]["score"], 0)
        self.assertIsNone(output["sirs"]["positive"])


if __name__ == "__main__":
    unittest.main()
