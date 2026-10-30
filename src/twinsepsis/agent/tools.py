"""JSON-returning agent tools.

Every public tool returns a plain ``dict`` that is safe to serialise with
``json.dumps(..., allow_nan=False)``.  Output schemas live in
``TOOL_OUTPUT_SCHEMAS`` and are exercised by ``tests/test_tools_schema.py``.

There are no LLM calls here.  The tools are the stable seam an agent (or a
future MCP server) can call.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .simulate import list_intervention_specs, simulate
from .twin import PatientTwin

CLINICAL_SCORE_DISCLAIMER = (
    "Rule-based screening score computed from the variables present at the "
    "latest hour. Missing criteria are reported, not imputed. This is not a "
    "diagnosis and not a substitute for clinician assessment."
)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def _require_twin(twin: PatientTwin) -> PatientTwin:
    if not isinstance(twin, PatientTwin):
        raise TypeError("twin must be a PatientTwin")
    if twin.hours == 0:
        raise ValueError("twin has no history")
    return twin


# ---------------------------------------------------------------------------
# Clinical scores (fresh implementations; no Partial_SOFA / Modified_SIRS)
# ---------------------------------------------------------------------------


def _first_number(row: pd.Series, aliases: tuple[str, ...]) -> tuple[float | None, str | None]:
    for name in aliases:
        if name not in row.index:
            continue
        value = row[name]
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number, name
    return None, None


def _temperature_c(row: pd.Series) -> tuple[float | None, str | None, list[dict[str, Any]]]:
    value, source = _first_number(row, ("Temp", "Temperature", "temperature", "Temp_C", "Temp_F"))
    if value is None:
        return None, source, []
    if source in {"Temp_F", "temp_f"} or value > 50.0:
        converted = (value - 32.0) * 5.0 / 9.0
        return converted, source, [
            {
                "variable": "Temp",
                "input": value,
                "output": converted,
                "reason": "value > 50 or Fahrenheit column; converted to degC",
            }
        ]
    return value, source, []


def _gcs_total(row: pd.Series) -> tuple[float | None, str | None]:
    total, source = _first_number(row, ("GCS", "GCS_Total", "gcs_total", "GCS_total"))
    if total is not None:
        return total, source

    eye, eye_source = _first_number(row, ("GCS_Eye", "GCS_eye", "gcs_eye", "GcsEye"))
    verbal, verbal_source = _first_number(row, ("GCS_Verbal", "GCS_verbal", "gcs_verbal", "GcsVerbal"))
    motor, motor_source = _first_number(row, ("GCS_Motor", "GCS_motor", "gcs_motor", "GcsMotor"))
    if eye is not None and verbal is not None and motor is not None:
        return eye + verbal + motor, f"{eye_source}+{verbal_source}+{motor_source}"
    return None, None


def _mental_status(row: pd.Series) -> tuple[bool | None, str | None]:
    for name in ("AlteredMentalStatus", "MentalStatus", "AMS", "altered_mental_status"):
        if name not in row.index:
            continue
        value = row[name]
        if isinstance(value, (bool, np.bool_)):
            return bool(value), name
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"altered", "yes", "true", "1", "confused", "unresponsive"}:
                return True, name
            if normalized in {"normal", "no", "false", "0", "alert"}:
                return False, name
    return None, None


def _criterion(
    name: str,
    value: float | None,
    source: str | None,
    rule: str,
    met: bool | None,
    *,
    missing_inputs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "variable": source,
        "value": value,
        "rule": rule,
        "available": value is not None and met is not None,
        "met": met if value is not None else None,
        "missing_inputs": missing_inputs or ([] if value is not None else [name]),
    }


def _qsofa(row: pd.Series) -> dict[str, Any]:
    criteria: list[dict[str, Any]] = []

    resp, resp_source = _first_number(row, ("Resp", "RR", "resp_rate", "RespiratoryRate"))
    criteria.append(
        _criterion(
            "Respiratory rate",
            resp,
            resp_source,
            ">= 22 breaths/min",
            None if resp is None else resp >= 22.0,
        )
    )

    sbp, sbp_source = _first_number(row, ("SBP", "sbp", "SystolicBP", "systolic_bp"))
    criteria.append(
        _criterion(
            "Systolic blood pressure",
            sbp,
            sbp_source,
            "<= 100 mmHg",
            None if sbp is None else sbp <= 100.0,
        )
    )

    gcs, gcs_source = _gcs_total(row)
    altered, altered_source = _mental_status(row)
    cns_met: bool | None
    cns_value: float | None
    cns_source: str | None
    if gcs is not None:
        cns_met = gcs < 15.0
        cns_value = gcs
        cns_source = gcs_source
    elif altered is not None:
        cns_met = altered
        cns_value = 1.0 if altered else 0.0
        cns_source = altered_source
    else:
        cns_met = None
        cns_value = None
        cns_source = None
    criteria.append(
        _criterion(
            "Mental status",
            cns_value,
            cns_source,
            "GCS < 15 or altered mental status",
            cns_met,
            missing_inputs=[] if cns_met is not None else ["GCS or mental status"],
        )
    )

    available = [item for item in criteria if item["available"]]
    score = sum(1 for item in available if item["met"])
    complete = len(available) == 3
    if score >= 2:
        positive: bool | None = True
    elif complete:
        positive = False
    else:
        positive = None

    if positive is True:
        interpretation = "Two or more qSOFA criteria are met."
    elif positive is False:
        interpretation = "Fewer than two qSOFA criteria are met."
    else:
        interpretation = "qSOFA is incomplete; fewer than two criteria are met on available data."

    return {
        "score": score,
        "max_score": 3,
        "available_count": len(available),
        "complete": complete,
        "positive": positive,
        "criteria": criteria,
        "interpretation": interpretation,
        "definition": "qSOFA: respiratory rate >= 22, SBP <= 100, GCS < 15 or altered mentation.",
    }


def _sirs(row: pd.Series) -> dict[str, Any]:
    criteria: list[dict[str, Any]] = []
    normalizations: list[dict[str, Any]] = []

    temp, temp_source, temp_conversions = _temperature_c(row)
    normalizations.extend(temp_conversions)
    temp_met = None if temp is None else (temp > 38.0 or temp < 36.0)
    criteria.append(
        _criterion(
            "Temperature",
            temp,
            temp_source,
            "> 38 or < 36 degC",
            temp_met,
        )
    )

    hr, hr_source = _first_number(row, ("HR", "heart_rate", "HeartRate"))
    criteria.append(
        _criterion(
            "Heart rate",
            hr,
            hr_source,
            "> 90 beats/min",
            None if hr is None else hr > 90.0,
        )
    )

    resp, resp_source = _first_number(row, ("Resp", "RR", "resp_rate", "RespiratoryRate"))
    paco2, paco2_source = _first_number(row, ("PaCO2", "paco2", "PaCO2_mmHg"))
    if resp is None and paco2 is None:
        resp_met = None
        resp_value = None
        resp_display = None
        missing = ["Resp or PaCO2"]
    else:
        resp_met = bool((resp is not None and resp > 20.0) or (paco2 is not None and paco2 < 32.0))
        resp_value = resp if resp is not None else paco2
        resp_display = "+".join(part for part in (resp_source, paco2_source) if part)
        missing = []
    criteria.append(
        _criterion(
            "Respiratory",
            resp_value,
            resp_display,
            "Resp > 20 breaths/min or PaCO2 < 32 mmHg",
            resp_met,
            missing_inputs=missing,
        )
    )

    wbc, wbc_source = _first_number(row, ("WBC", "wbc", "WhiteBloodCells", "WBC_count"))
    criteria.append(
        _criterion(
            "White blood cell count",
            wbc,
            wbc_source,
            "> 12 or < 4 x10^9/L",
            None if wbc is None else (wbc > 12.0 or wbc < 4.0),
        )
    )

    available = [item for item in criteria if item["available"]]
    score = sum(1 for item in available if item["met"])
    complete = len(available) == 4
    if score >= 2:
        positive: bool | None = True
    elif complete:
        positive = False
    else:
        positive = None

    if positive is True:
        interpretation = "Two or more SIRS criteria are met."
    elif positive is False:
        interpretation = "Fewer than two SIRS criteria are met."
    else:
        interpretation = "SIRS is incomplete; fewer than two criteria are met on available data."

    return {
        "score": score,
        "max_score": 4,
        "available_count": len(available),
        "complete": complete,
        "positive": positive,
        "criteria": criteria,
        "normalizations": normalizations,
        "interpretation": interpretation,
        "definition": (
            "SIRS (4 available criteria): temperature, heart rate, respiratory "
            "rate/PaCO2, and WBC. Band forms are not available; missing criteria "
            "are not treated as normal."
        ),
    }


def clinical_scores(twin: PatientTwin) -> dict[str, Any]:
    """Compute qSOFA and SIRS from the latest hour's available variables."""

    twin = _require_twin(twin)
    row = twin.latest
    return _jsonable(
        {
            "patient_id": twin.patient_id,
            "hour": twin.latest_hour,
            "qsofa": _qsofa(row),
            "sirs": _sirs(row),
            "disclaimer": CLINICAL_SCORE_DISCLAIMER,
        }
    )


# ---------------------------------------------------------------------------
# Twin / risk tools
# ---------------------------------------------------------------------------


def get_twin_state(twin: PatientTwin, trend_hours: int = 3) -> dict[str, Any]:
    """Return a compact, JSON-safe snapshot of the twin at its latest hour."""

    twin = _require_twin(twin)
    if trend_hours < 1:
        raise ValueError("trend_hours must be >= 1")

    risk = twin.risk()
    return _jsonable(
        {
            "patient_id": twin.patient_id,
            "hours": twin.hours,
            "latest_hour": twin.latest_hour,
            "current_risk": risk,
            "threshold": twin.model.threshold,
            "exceeds_threshold": risk >= twin.model.threshold,
            "trend": twin.trend(trend_hours),
            "risk_model": type(twin.model).__name__,
            "latest_values": twin.latest.to_dict(),
            "disclaimer": (
                "Risk is produced by the configured model for the latest hour. "
                "It is not a diagnosis."
            ),
        }
    )


def predict_risk(twin: PatientTwin) -> dict[str, Any]:
    """Return the latest-hour risk and threshold decision."""

    twin = _require_twin(twin)
    risk = twin.risk()
    return _jsonable(
        {
            "patient_id": twin.patient_id,
            "hour": twin.latest_hour,
            "risk": risk,
            "threshold": twin.model.threshold,
            "exceeds_threshold": risk >= twin.model.threshold,
            "risk_model": type(twin.model).__name__,
        }
    )


def explain_risk(twin: PatientTwin, top_k: int = 5) -> dict[str, Any]:
    """Return the model's top feature contributions at the latest hour."""

    twin = _require_twin(twin)
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    risk = twin.risk()
    explanations = twin.model.explain(twin.history, top_k=top_k)
    return _jsonable(
        {
            "patient_id": twin.patient_id,
            "hour": twin.latest_hour,
            "risk": risk,
            "threshold": twin.model.threshold,
            "top_k": top_k,
            "explanations": explanations,
            "risk_model": type(twin.model).__name__,
            "disclaimer": (
                "Contributions are model attributions for this hour, not causal "
                "effects. Positive values push the model output up."
            ),
        }
    )


def list_interventions() -> dict[str, Any]:
    """List scenario interventions and whether their assumptions are usable."""

    return _jsonable(list_intervention_specs())


def simulate_intervention(
    twin: PatientTwin,
    intervention: str,
) -> dict[str, Any]:
    """Run one model-based scenario. Never mutates the twin."""

    twin = _require_twin(twin)
    return _jsonable(simulate(twin, intervention))


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

_NULLABLE_NUMBER = {"type": ["number", "null"]}

_CRITERION_SCHEMA = {
    "type": "object",
    "required": ["name", "variable", "value", "rule", "available", "met", "missing_inputs"],
    "properties": {
        "name": {"type": "string"},
        "variable": {"type": ["string", "null"]},
        "value": _NULLABLE_NUMBER,
        "rule": {"type": "string"},
        "available": {"type": "boolean"},
        "met": {"type": ["boolean", "null"]},
        "missing_inputs": {"type": "array", "items": {"type": "string"}},
    },
}

_SCORE_SCHEMA = {
    "type": "object",
    "required": [
        "score", "max_score", "available_count", "complete", "positive",
        "criteria", "interpretation", "definition",
    ],
    "properties": {
        "score": {"type": "integer", "minimum": 0},
        "max_score": {"type": "integer", "minimum": 1},
        "available_count": {"type": "integer", "minimum": 0},
        "complete": {"type": "boolean"},
        "positive": {"type": ["boolean", "null"]},
        "criteria": {"type": "array", "items": _CRITERION_SCHEMA},
        "interpretation": {"type": "string"},
        "definition": {"type": "string"},
    },
}

TOOL_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_twin_state": {
        "type": "object",
        "required": [
            "patient_id", "hours", "latest_hour", "current_risk", "threshold",
            "exceeds_threshold", "trend", "risk_model", "latest_values", "disclaimer",
        ],
        "properties": {
            "patient_id": {"type": "string"},
            "hours": {"type": "integer", "minimum": 1},
            "latest_hour": {"type": "integer", "minimum": 0},
            "current_risk": {"type": "number", "minimum": 0, "maximum": 1},
            "threshold": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
            "exceeds_threshold": {"type": "boolean"},
            "trend": {
                "type": "object",
                "required": ["direction", "delta", "slope_per_hour", "window_hours", "previous_risk", "current_risk"],
                "properties": {
                    "direction": {"enum": ["rising", "falling", "stable", "insufficient_history"]},
                    "delta": _NULLABLE_NUMBER,
                    "slope_per_hour": _NULLABLE_NUMBER,
                    "window_hours": {"type": "integer", "minimum": 0},
                    "previous_risk": _NULLABLE_NUMBER,
                    "current_risk": {"type": "number"},
                },
            },
            "risk_model": {"type": "string"},
            "latest_values": {"type": "object"},
            "disclaimer": {"type": "string"},
        },
    },
    "predict_risk": {
        "type": "object",
        "required": ["patient_id", "hour", "risk", "threshold", "exceeds_threshold", "risk_model"],
        "properties": {
            "patient_id": {"type": "string"},
            "hour": {"type": "integer", "minimum": 0},
            "risk": {"type": "number", "minimum": 0, "maximum": 1},
            "threshold": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
            "exceeds_threshold": {"type": "boolean"},
            "risk_model": {"type": "string"},
        },
    },
    "explain_risk": {
        "type": "object",
        "required": ["patient_id", "hour", "risk", "threshold", "top_k", "explanations", "risk_model", "disclaimer"],
        "properties": {
            "patient_id": {"type": "string"},
            "hour": {"type": "integer", "minimum": 0},
            "risk": {"type": "number", "minimum": 0, "maximum": 1},
            "threshold": {"type": "number"},
            "top_k": {"type": "integer", "minimum": 1},
            "explanations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["feature", "label", "value", "contribution", "direction"],
                    "properties": {
                        "feature": {"type": "string"},
                        "label": {"type": "string"},
                        "value": _NULLABLE_NUMBER,
                        "contribution": {"type": "number"},
                        "direction": {"enum": ["increases", "decreases", "neutral"]},
                    },
                },
            },
            "risk_model": {"type": "string"},
            "disclaimer": {"type": "string"},
        },
    },
    "list_interventions": {
        "type": "object",
        "required": ["interventions", "source", "source_kind", "any_available", "disclaimer"],
        "properties": {
            "interventions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "label", "description", "available", "effect_status", "citation", "effects", "unresolved"],
                    "properties": {
                        "name": {"enum": ["fluid_bolus", "vasopressor", "antibiotics", "no_action"]},
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                        "available": {"type": "boolean"},
                        "effect_status": {"enum": ["numeric", "placeholder", "not_applicable"]},
                        "citation": {"type": "string"},
                        "effects": {"type": "array", "items": {"type": "object"}},
                        "unresolved": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "source": {"type": "string"},
            "source_kind": {"type": "string"},
            "any_available": {"type": "boolean"},
            "disclaimer": {"type": "string"},
        },
    },
    "simulate_intervention": {
        "type": "object",
        "required": [
            "intervention", "label", "available", "reason", "source", "source_kind",
            "effect_status", "horizons_hours", "projected_risk", "bounds",
            "bound_type", "baseline_no_action", "assumptions", "unresolved",
            "disclaimer", "generated_by",
        ],
        "properties": {
            "intervention": {"enum": ["fluid_bolus", "vasopressor", "antibiotics", "no_action"]},
            "label": {"type": "string"},
            "available": {"type": "boolean"},
            "reason": {"type": ["string", "null"]},
            "source": {"type": "string"},
            "source_kind": {"type": "string"},
            "effect_status": {"enum": ["numeric", "placeholder", "invalid", "not_applicable"]},
            "horizons_hours": {"type": "array", "items": {"type": "integer"}},
            "projected_risk": {"type": "object", "additionalProperties": _NULLABLE_NUMBER},
            "bounds": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "required": ["low", "high"],
                    "properties": {"low": _NULLABLE_NUMBER, "high": _NULLABLE_NUMBER},
                },
            },
            "bound_type": {"type": ["string", "null"]},
            "baseline_no_action": {"type": "object", "additionalProperties": {"type": "number"}},
            "assumptions": {"type": "array", "items": {"type": "object"}},
            "unresolved": {"type": "array", "items": {"type": "string"}},
            "disclaimer": {"type": "string"},
            "generated_by": {"type": "string"},
        },
    },
    "clinical_scores": {
        "type": "object",
        "required": ["patient_id", "hour", "qsofa", "sirs", "disclaimer"],
        "properties": {
            "patient_id": {"type": "string"},
            "hour": {"type": "integer", "minimum": 0},
            "qsofa": _SCORE_SCHEMA,
            "sirs": _SCORE_SCHEMA,
            "disclaimer": {"type": "string"},
        },
    },
}

TOOL_FUNCTIONS = {
    "get_twin_state": get_twin_state,
    "predict_risk": predict_risk,
    "explain_risk": explain_risk,
    "list_interventions": list_interventions,
    "simulate_intervention": simulate_intervention,
    "clinical_scores": clinical_scores,
}


__all__ = [
    "get_twin_state",
    "predict_risk",
    "explain_risk",
    "list_interventions",
    "simulate_intervention",
    "clinical_scores",
    "TOOL_OUTPUT_SCHEMAS",
    "TOOL_FUNCTIONS",
]
