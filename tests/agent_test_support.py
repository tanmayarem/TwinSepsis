"""Shared offline fixtures for agent tests.

Nothing here makes a network call.  StubRisk is explicitly test-only and its
numbers are never presented as research results.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinsepsis.agent.llm import AssistantTurn, ToolCallRequest  # noqa: E402
from twinsepsis.agent.risk_model import RiskModel, StubRisk  # noqa: E402
from twinsepsis.agent.twin import PatientTwin  # noqa: E402


class ScriptedLLM:
    """Deterministic ChatClient that replays prepared turns."""

    provider = "test-provider"
    model = "scripted-test-model"

    def __init__(self, turns: list[AssistantTurn | BaseException]) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, tools) -> AssistantTurn:
        self.calls.append(
            {
                "messages": copy.deepcopy(list(messages)),
                "tools": copy.deepcopy(list(tools)),
            }
        )
        if not self.turns:
            raise AssertionError("ScriptedLLM ran out of prepared turns")
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn


def tool_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    call_id: str | None = None,
) -> ToolCallRequest:
    arguments = {} if arguments is None else arguments
    return ToolCallRequest(
        id=call_id or f"call-{name}-{abs(hash(json.dumps(arguments, sort_keys=True))) % 100000}",
        name=name,
        arguments=json.dumps(arguments),
    )


def turn(*calls: ToolCallRequest, content: str | None = None) -> AssistantTurn:
    return AssistantTurn(content=content, tool_calls=list(calls))


def synthetic_rows(n: int = 6) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(n):
        rows.append(
            {
                "HR": 92.0 + 3.0 * index,
                "Temp": 37.2 + 0.25 * index,
                "Resp": 18.0 + index,
                "SBP": 118.0 - 2.0 * index,
                "MAP": 82.0 - 2.0 * index,
                "DBP": 62.0,
                "WBC": 9.0 + 0.7 * index,
                "Lactate": 1.2 + 0.3 * index,
                "PaCO2": 38.0,
                "ICULOS": index + 1,
                "HospAdmTime": -2.0,
            }
        )
    return rows


def make_twin(
    model: RiskModel | None = None,
    *,
    rows: list[dict[str, Any]] | None = None,
    patient_id: str = "synthetic-agent-1",
) -> PatientTwin:
    return PatientTwin(
        patient_id,
        model or StubRisk(),
        pd.DataFrame(rows if rows is not None else synthetic_rows()),
    )


class FailingRisk(RiskModel):
    """Raises on every prediction/explanation, for failure-path tests."""

    test_only = True
    dataset = "synthetic failure fixture"

    def __init__(self) -> None:
        super().__init__(threshold=0.5, name="FailingRisk")

    def featurize(self, window_df: pd.DataFrame) -> pd.DataFrame:
        return window_df.copy()

    def predict_proba(self, window_df: pd.DataFrame) -> float:
        raise RuntimeError("synthetic prediction failure")

    def explain(self, window_df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        raise RuntimeError("synthetic explanation failure")


class UnavailableRisk(RiskModel):
    """Models the capability state of EnsembleRisk without needing torch."""

    supports_prediction = False
    supports_explanation = False
    supports_simulation = False
    test_only = True
    dataset = "synthetic unavailable fixture"
    prediction_horizon = "Unknown (fixture)."

    def __init__(self) -> None:
        super().__init__(threshold=0.81, name="UnavailableRisk")

    def featurize(self, window_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("unavailable")

    def predict_proba(self, window_df: pd.DataFrame) -> float:
        raise NotImplementedError("unavailable")

    def explain(self, window_df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        raise NotImplementedError("unavailable")
