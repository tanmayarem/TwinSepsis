from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_test_support import (  # noqa: E402
    FailingRisk,
    ScriptedLLM,
    UnavailableRisk,
    make_twin,
    tool_call,
    turn,
)
from twinsepsis.agent.agent import (  # noqa: E402
    AGENT_RESPONSE_SCHEMA,
    MAX_TOOL_CALLS,
    AgentContextError,
    SepsisAgent,
)


class AgentLoopTests(unittest.TestCase):
    def run_agent(self, turns, *, twin=None, max_tool_calls=8, trace_path=None):
        twin = twin or make_twin()
        llm = ScriptedLLM(turns)
        agent = SepsisAgent(
            twin,
            llm,
            max_tool_calls=max_tool_calls,
            trace_path=trace_path,
            trace_enabled=trace_path is not None,
        )
        response = agent.run(
            "Summarize risk, drivers and trend.",
            twin.patient_id,
            twin.hours,
        )
        return response, llm, agent, twin

    def test_selection_execution_and_structured_response(self) -> None:
        response, llm, _, twin = self.run_agent(
            [
                turn(
                    tool_call("predict_risk"),
                    tool_call("explain_risk", {"top_k": 3}),
                ),
                turn(content="done"),
            ]
        )

        self.assertIn("Model-estimated risk", response["risk_summary"])
        self.assertEqual(response["prediction_source"]["model"], "StubRisk")
        self.assertEqual(response["prediction_horizon"]["scored_icu_hour"], twin.hours)
        self.assertEqual(len(response["top_drivers"]), 3)
        self.assertEqual(
            [item["tool"] for item in response["tools_used"]],
            ["predict_risk", "explain_risk"],
        )
        self.assertTrue(all(item["ok"] for item in response["tools_used"]))
        self.assertTrue(response["needs_clinician_review"])

        exposed = {
            tool["function"]["name"] for tool in llm.calls[0]["tools"]
        }
        self.assertIn("predict_risk", exposed)
        self.assertIn("explain_risk", exposed)
        self.assertEqual(llm.calls[0]["messages"][0]["role"], "system")

    def test_multiple_tool_calls_in_one_turn(self) -> None:
        response, llm, _, _ = self.run_agent(
            [
                turn(
                    tool_call("get_twin_state", {"trend_hours": 2}),
                    tool_call("clinical_scores"),
                    tool_call("risk_trajectory", {"max_hours": 4}),
                ),
                turn(content="done"),
            ]
        )

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(len(response["tools_used"]), 4)
        self.assertIsNotNone(response["clinical_scores"])
        self.assertIsNotNone(response["trajectory"])
        self.assertEqual(len(response["trajectory"]), 4)
        self.assertEqual(response["trajectory"][-1]["hour"], 6)
        # The LLM did not ask for a prediction, so the deterministic
        # completeness check supplied one.
        self.assertIsNotNone(response["risk_summary"])
        self.assertEqual(response["tools_used"][-1]["tool"], "predict_risk")
        self.assertEqual(response["tools_used"][-1]["origin"], "fallback")

    def test_unknown_tool_and_invalid_arguments_are_rejected(self) -> None:
        response, llm, _, _ = self.run_agent(
            [
                turn(
                    tool_call("delete_patient"),
                    tool_call("predict_risk", {"unexpected": True}),
                    tool_call("explain_risk", {"top_k": 0}),
                ),
                turn(content="done"),
            ]
        )

        self.assertEqual(len(response["tools_used"]), 4)
        self.assertFalse(any(item["ok"] for item in response["tools_used"][:3]))
        self.assertTrue(response["tools_used"][3]["ok"])
        self.assertEqual(response["tools_used"][3]["tool"], "predict_risk")
        self.assertEqual(response["tools_used"][3]["origin"], "fallback")
        self.assertIsNotNone(response["risk_summary"])
        self.assertIsNotNone(response["prediction_source"])

        caveats = " ".join(response["caveats"])
        self.assertIn("not a known tool", caveats)
        self.assertIn("arguments were invalid", caveats)
        # The LLM was told about each failure rather than the run crashing.
        second_messages = llm.calls[1]["messages"]
        tool_messages = [m for m in second_messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 3)
        self.assertTrue(all(json.loads(m["content"])["ok"] is False for m in tool_messages))

    def test_prediction_failure_never_fabricates_a_risk(self) -> None:
        response, _, _, _ = self.run_agent(
            [
                turn(
                    tool_call("predict_risk"),
                    tool_call("get_twin_state"),
                    tool_call("clinical_scores"),
                ),
                turn(content="done"),
            ],
            twin=make_twin(FailingRisk()),
        )

        self.assertIsNone(response["risk_summary"])
        self.assertIsNone(response["prediction_source"])
        self.assertIsNone(response["prediction_horizon"])
        self.assertIsNone(response["top_drivers"])
        self.assertIsNotNone(response["clinical_scores"])

        caveats = " ".join(response["caveats"])
        self.assertIn("No risk value was produced", caveats)
        self.assertIn("failed", caveats)
        self.assertIn("risk_prediction_unavailable", response["limitations"])

        statuses = {item["tool"]: item["ok"] for item in response["tools_used"]}
        self.assertFalse(statuses["predict_risk"])
        self.assertFalse(statuses["get_twin_state"])
        self.assertTrue(statuses["clinical_scores"])

        predictions = [
            item for item in response["tools_used"] if item["tool"] == "predict_risk"
        ]
        self.assertEqual([item["origin"] for item in predictions], ["llm", "fallback"])
        self.assertTrue(all(not item["ok"] for item in predictions))

    def test_tool_call_limit_is_enforced(self) -> None:
        turns = [turn(tool_call("predict_risk")) for _ in range(12)]
        response, llm, _, _ = self.run_agent(turns, max_tool_calls=MAX_TOOL_CALLS)

        self.assertEqual(len(response["tools_used"]), MAX_TOOL_CALLS)
        self.assertEqual(len(llm.calls), MAX_TOOL_CALLS)
        self.assertIn(
            f"tool-call limit of {MAX_TOOL_CALLS}",
            " ".join(response["caveats"]),
        )
        self.assertIn("tool_call_limit_reached", response["limitations"])

    def test_limit_is_enforced_within_a_single_turn(self) -> None:
        response, _, _, _ = self.run_agent(
            [
                turn(
                    *[
                        tool_call("predict_risk", call_id=f"call-{index}")
                        for index in range(10)
                    ]
                )
            ],
            max_tool_calls=3,
        )
        self.assertEqual(len(response["tools_used"]), 3)

    def test_unavailable_capability_is_not_exposed(self) -> None:
        twin = make_twin(UnavailableRisk())
        response, llm, agent, _ = self.run_agent(
            [turn(tool_call("predict_risk"), tool_call("clinical_scores")), turn()],
            twin=twin,
        )

        self.assertEqual(
            agent.available_tools,
            ["clinical_scores", "list_interventions"],
        )
        exposed = {tool["function"]["name"] for tool in llm.calls[0]["tools"]}
        self.assertNotIn("predict_risk", exposed)
        self.assertIn("clinical_scores", exposed)

        self.assertIsNone(response["risk_summary"])
        self.assertIsNotNone(response["clinical_scores"])
        caveats = " ".join(response["caveats"])
        self.assertIn("does not support prediction", caveats)
        self.assertIn("not supported by risk model", caveats)

    def test_todo_intervention_does_not_invent_effects(self) -> None:
        response, _, _, _ = self.run_agent(
            [
                turn(
                    tool_call("list_interventions"),
                    tool_call("simulate_intervention", {"intervention": "fluid_bolus"}),
                ),
                turn(content="done"),
            ]
        )

        self.assertIsNotNone(response["scenarios"])
        fluid = next(
            scenario
            for scenario in response["scenarios"]
            if scenario["intervention"] == "fluid_bolus"
        )
        self.assertFalse(fluid["available"])
        self.assertEqual(fluid["effect_status"], "placeholder")
        for value in fluid["projected_risk"].values():
            self.assertIsNone(value)
        for bound in fluid["bounds"].values():
            self.assertIsNone(bound["low"])
            self.assertIsNone(bound["high"])
        self.assertIn(
            "No intervention effect was invented",
            " ".join(response["caveats"]),
        )

    def test_no_action_scenario_uses_real_model_outputs(self) -> None:
        response, _, _, _ = self.run_agent(
            [turn(tool_call("simulate_intervention", {"intervention": "no_action"})), turn()]
        )

        scenario = response["scenarios"][0]
        self.assertTrue(scenario["available"])
        self.assertEqual(scenario["intervention"], "no_action")
        self.assertEqual(set(scenario["projected_risk"]), {"1", "3", "6"})
        self.assertTrue(
            all(isinstance(value, float) for value in scenario["projected_risk"].values())
        )

    def test_response_matches_schema_and_is_json_safe(self) -> None:
        response, _, _, _ = self.run_agent(
            [turn(tool_call("predict_risk"), tool_call("risk_trajectory")), turn()]
        )
        jsonschema.validate(instance=response, schema=AGENT_RESPONSE_SCHEMA)
        json.dumps(response, allow_nan=False)

    def test_context_validation_happens_before_the_llm(self) -> None:
        twin = make_twin()
        llm = ScriptedLLM([])
        agent = SepsisAgent(twin, llm, trace_enabled=False)

        for patient_id, hour, request in (
            ("another-patient", twin.hours, "assess"),
            (twin.patient_id, twin.hours - 1, "assess"),
            (twin.patient_id, 0, "assess"),
            (twin.patient_id, twin.hours, ""),
        ):
            with self.subTest(patient_id=patient_id, hour=hour, request=request):
                with self.assertRaises(AgentContextError):
                    agent.run(request, patient_id, hour)
        self.assertEqual(llm.calls, [])

    def test_invalid_limits_are_rejected(self) -> None:
        twin = make_twin()
        for value in (0, 9, True, 2.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SepsisAgent(twin, ScriptedLLM([]), max_tool_calls=value)

    def test_no_tool_calls_still_runs_the_deterministic_prediction(self) -> None:
        response, llm, _, _ = self.run_agent([turn(content="I need no tools.")])
        self.assertEqual(len(llm.calls), 1)
        self.assertIsNotNone(response["risk_summary"])
        self.assertIsNotNone(response["prediction_source"])
        self.assertEqual(len(response["tools_used"]), 1)
        self.assertEqual(response["tools_used"][0]["tool"], "predict_risk")
        self.assertEqual(response["tools_used"][0]["origin"], "fallback")

    def test_skipped_prediction_is_recovered_by_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            response, llm, _, _ = self.run_agent(
                [
                    turn(
                        tool_call("get_twin_state"),
                        tool_call("explain_risk", {"top_k": 3}),
                    ),
                    turn(content="done"),
                ],
                trace_path=path,
            )

            self.assertEqual(len(llm.calls), 2)
            self.assertIsNotNone(response["risk_summary"])
            self.assertEqual(response["prediction_source"]["model"], "StubRisk")
            self.assertEqual(
                [item["tool"] for item in response["tools_used"]],
                ["get_twin_state", "explain_risk", "predict_risk"],
            )
            self.assertEqual(
                [item["origin"] for item in response["tools_used"]],
                ["llm", "llm", "fallback"],
            )
            self.assertTrue(response["tools_used"][-1]["ok"])

            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            fallback_events = [
                event for event in events if event["event"] == "prediction_fallback"
            ]
            self.assertEqual(len(fallback_events), 1)
            self.assertEqual(fallback_events[0]["tool"], "predict_risk")
            fallback_calls = [
                event
                for event in events
                if event["event"] == "tool_call" and event["origin"] == "fallback"
            ]
            self.assertEqual(len(fallback_calls), 1)
            self.assertEqual(fallback_calls[0]["tool"], "predict_risk")
            self.assertTrue(fallback_calls[0]["ok"])

    def test_successful_prediction_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            response, llm, _, _ = self.run_agent(
                [turn(tool_call("predict_risk")), turn(content="done")],
                trace_path=path,
            )

            self.assertEqual(len(llm.calls), 2)
            predictions = [
                item
                for item in response["tools_used"]
                if item["tool"] == "predict_risk"
            ]
            self.assertEqual(len(predictions), 1)
            self.assertEqual(predictions[0]["origin"], "llm")
            self.assertTrue(predictions[0]["ok"])
            self.assertIsNotNone(response["risk_summary"])

            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            event_names = [event["event"] for event in events]
            self.assertNotIn("prediction_fallback", event_names)
            self.assertNotIn("prediction_fallback_skipped", event_names)

    def test_prediction_failure_fallback_does_not_fabricate(self) -> None:
        response, _, _, _ = self.run_agent(
            [
                turn(tool_call("predict_risk"), tool_call("explain_risk")),
                turn(content="done"),
            ],
            twin=make_twin(FailingRisk()),
        )

        self.assertIsNone(response["risk_summary"])
        self.assertIsNone(response["prediction_source"])
        self.assertIsNone(response["prediction_horizon"])
        predictions = [
            item for item in response["tools_used"] if item["tool"] == "predict_risk"
        ]
        self.assertEqual([item["origin"] for item in predictions], ["llm", "fallback"])
        self.assertTrue(all(not item["ok"] for item in predictions))
        self.assertIn("No risk value was produced", " ".join(response["caveats"]))
        self.assertIn("risk_prediction_unavailable", response["limitations"])

    def test_fallback_respects_the_eight_call_limit(self) -> None:
        turns = [
            turn(
                *[
                    tool_call("clinical_scores", call_id=f"call-{index}")
                    for index in range(MAX_TOOL_CALLS)
                ]
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            response, llm, _, _ = self.run_agent(
                turns, max_tool_calls=MAX_TOOL_CALLS, trace_path=path
            )

            self.assertEqual(len(llm.calls), 1)
            self.assertEqual(len(response["tools_used"]), MAX_TOOL_CALLS)
            self.assertNotIn(
                "predict_risk", [item["tool"] for item in response["tools_used"]]
            )
            self.assertIsNone(response["risk_summary"])
            self.assertIsNone(response["prediction_source"])
            caveats = " ".join(response["caveats"])
            self.assertIn(
                "tool-call limit was reached before the deterministic prediction "
                "fallback could run",
                caveats,
            )
            self.assertIn(
                "prediction_fallback_blocked_by_tool_limit",
                response["limitations"],
            )
            self.assertIn("tool_call_limit_reached", response["limitations"])

            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            skipped = [
                event
                for event in events
                if event["event"] == "prediction_fallback_skipped"
            ]
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["tool"], "predict_risk")
            self.assertEqual(skipped[0]["reason"], "tool_call_limit")
            self.assertNotIn(
                "prediction_fallback", [event["event"] for event in events]
            )

    def test_trace_omits_patient_identifiers_and_tool_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            response, _, _, twin = self.run_agent(
                [turn(tool_call("get_twin_state"), tool_call("predict_risk")), turn()],
                trace_path=path,
            )
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")

            self.assertNotIn(twin.patient_id, text)
            self.assertNotIn("latest_values", text)
            self.assertNotIn("test-key", text)
            self.assertNotIn("Authorization", text)
            self.assertNotIn(str(response["risk_summary"]), text)

            lines = [json.loads(line) for line in text.splitlines()]
            events = [line["event"] for line in lines]
            self.assertIn("run_start", events)
            self.assertIn("tool_call", events)
            self.assertIn("run_end", events)

            tool_events = [line for line in lines if line["event"] == "tool_call"]
            self.assertEqual([event["tool"] for event in tool_events], ["get_twin_state", "predict_risk"])
            for event in tool_events:
                self.assertIn("arguments", event)
                self.assertIn("duration_ms", event)
                self.assertEqual(event["llm_model"], "scripted-test-model")
                self.assertEqual(event["llm_provider"], "test-provider")
                self.assertEqual(event["risk_model"], "StubRisk")
                self.assertTrue(event["ok"])

    def test_llm_exception_is_traced_and_reraised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            twin = make_twin()
            agent = SepsisAgent(
                twin,
                ScriptedLLM([RuntimeError("synthetic LLM failure")]),
                trace_path=path,
            )
            with self.assertRaises(RuntimeError):
                agent.run("assess", twin.patient_id, twin.hours)

            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(lines[-1]["event"], "run_end")
            self.assertEqual(lines[-1]["status"], "error")
            self.assertEqual(lines[-1]["error_type"], "RuntimeError")


class ToolsOnlyTests(unittest.TestCase):
    def test_tools_only_mode_is_labelled_and_uses_real_tool_outputs(self) -> None:
        twin = make_twin()
        agent = SepsisAgent(twin, None, trace_enabled=False)
        response = agent.run_tools_only(twin.patient_id, twin.hours)

        self.assertEqual(response["mode"], "tools_only")
        self.assertIn("Model-estimated risk", response["risk_summary"])
        self.assertEqual(len(response["tools_used"]), 8)
        self.assertIn("No LLM was used", " ".join(response["caveats"]))

        scenarios = {item["intervention"]: item for item in response["scenarios"]}
        self.assertFalse(scenarios["fluid_bolus"]["available"])
        self.assertTrue(scenarios["no_action"]["available"])

    def test_tools_only_reports_unavailable_capabilities(self) -> None:
        twin = make_twin(UnavailableRisk())
        agent = SepsisAgent(twin, None, trace_enabled=False)
        response = agent.run_tools_only(twin.patient_id, twin.hours)

        self.assertEqual(response["mode"], "tools_only")
        self.assertIsNone(response["risk_summary"])
        self.assertEqual(len(response["tools_used"]), 8)
        self.assertTrue(any(not item["ok"] for item in response["tools_used"]))
        self.assertIn("does not support prediction", " ".join(response["caveats"]))


if __name__ == "__main__":
    unittest.main()
