from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_test_support import make_twin  # noqa: E402
from twinsepsis.agent import cli  # noqa: E402
from twinsepsis.agent.llm import (  # noqa: E402
    LLMConfigurationError,
    LLMRateLimitError,
)
from twinsepsis.agent.risk_model import ModelUnavailableError  # noqa: E402
from twinsepsis.patient_source import PatientSourceError  # noqa: E402


class CliTests(unittest.TestCase):
    def test_tools_only_prints_labelled_json(self) -> None:
        twin = make_twin()
        stdout = io.StringIO()
        with mock.patch.object(cli, "load_physionet_patient", return_value=twin):
            with contextlib.redirect_stdout(stdout):
                code = cli.main(
                    [
                        "--patient-id",
                        twin.patient_id,
                        "--hour",
                        str(twin.hours),
                        "--tools-only",
                        "--no-trace",
                    ]
                )

        self.assertEqual(code, 0)
        response = json.loads(stdout.getvalue())
        self.assertEqual(response["mode"], "tools_only")
        self.assertIn("Model-estimated risk", response["risk_summary"])
        self.assertTrue(response["needs_clinician_review"])

    def test_output_file_is_written(self) -> None:
        twin = make_twin()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.json"
            stdout = io.StringIO()
            with mock.patch.object(cli, "load_physionet_patient", return_value=twin):
                with contextlib.redirect_stdout(stdout):
                    code = cli.main(
                        [
                            "--patient-id",
                            twin.patient_id,
                            "--hour",
                            str(twin.hours),
                            "--tools-only",
                            "--no-trace",
                            "--output",
                            str(path),
                        ]
                    )
            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue(), "")
            response = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(response["patient_id"], twin.patient_id)

    def test_patient_load_failure_returns_2(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(
            cli,
            "load_physionet_patient",
            side_effect=PatientSourceError("synthetic missing patient"),
        ):
            with contextlib.redirect_stderr(stderr):
                code = cli.main(
                    ["--patient-id", "p999999", "--hour", "1", "--tools-only", "--no-trace"]
                )

        self.assertEqual(code, 2)
        payload = json.loads(stderr.getvalue())
        self.assertEqual(payload["error"]["type"], "PatientSourceError")

    def test_default_risk_model_is_xgboost_and_is_forwarded(self) -> None:
        twin = make_twin()
        sentinel = object()
        stdout = io.StringIO()
        with mock.patch.object(cli, "load_risk_model", return_value=sentinel) as load_model:
            with mock.patch.object(
                cli, "load_physionet_patient", return_value=twin
            ) as load_patient:
                with contextlib.redirect_stdout(stdout):
                    code = cli.main(
                        [
                            "--patient-id",
                            twin.patient_id,
                            "--hour",
                            str(twin.hours),
                            "--tools-only",
                            "--no-trace",
                        ]
                    )

        self.assertEqual(code, 0)
        load_model.assert_called_once_with("xgboost")
        self.assertIs(load_patient.call_args.kwargs["model"], sentinel)
        self.assertEqual(load_patient.call_args.kwargs["dataset_root"], None)

    def test_ensemble_flag_is_forwarded(self) -> None:
        twin = make_twin()
        sentinel = object()
        stdout = io.StringIO()
        with mock.patch.object(cli, "load_risk_model", return_value=sentinel) as load_model:
            with mock.patch.object(cli, "load_physionet_patient", return_value=twin):
                with contextlib.redirect_stdout(stdout):
                    code = cli.main(
                        [
                            "--patient-id",
                            twin.patient_id,
                            "--hour",
                            str(twin.hours),
                            "--risk-model",
                            "ensemble",
                            "--tools-only",
                            "--no-trace",
                        ]
                    )

        self.assertEqual(code, 0)
        load_model.assert_called_once_with("ensemble")

    def test_model_unavailable_returns_2(self) -> None:
        stderr = io.StringIO()
        stdout = io.StringIO()
        with mock.patch.object(
            cli,
            "load_risk_model",
            side_effect=ModelUnavailableError(
                "The GRU component of the PhysioNet ensemble could not be loaded."
            ),
        ):
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                code = cli.main(
                    [
                        "--patient-id",
                        "p000001",
                        "--hour",
                        "1",
                        "--risk-model",
                        "ensemble",
                        "--tools-only",
                        "--no-trace",
                    ]
                )

        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        payload = json.loads(stderr.getvalue())
        self.assertEqual(payload["error"]["type"], "ModelUnavailableError")
        self.assertIn("GRU", payload["error"]["message"])

    def test_invalid_risk_model_choice_is_rejected(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                cli.main(
                    [
                        "--patient-id",
                        "p000001",
                        "--hour",
                        "1",
                        "--risk-model",
                        "gru",
                    ]
                )
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())

    def test_rate_limit_error_payload_includes_retry_metadata(self) -> None:
        payload = cli._error_payload(
            LLMRateLimitError(
                "The LLM provider rate limit was reached (HTTP 429).",
                status_code=429,
                retry_after_s=12.5,
            )
        )

        self.assertEqual(payload["error"]["type"], "LLMRateLimitError")
        self.assertEqual(payload["error"]["status_code"], 429)
        self.assertEqual(payload["error"]["retry_after_s"], 12.5)

    def test_missing_llm_config_returns_2(self) -> None:
        twin = make_twin()
        stderr = io.StringIO()
        with mock.patch.object(cli, "load_physionet_patient", return_value=twin):
            with mock.patch.object(
                cli.LLMConfig,
                "from_env",
                side_effect=LLMConfigurationError("missing test configuration"),
            ):
                with contextlib.redirect_stderr(stderr):
                    code = cli.main(
                        [
                            "--patient-id",
                            twin.patient_id,
                            "--hour",
                            str(twin.hours),
                            "--no-trace",
                        ]
                    )

        self.assertEqual(code, 2)
        payload = json.loads(stderr.getvalue())
        self.assertEqual(payload["error"]["type"], "LLMConfigurationError")


if __name__ == "__main__":
    unittest.main()
