"""Model-based scenario simulation for a patient twin.

These are **model-based scenarios, not causal treatment effects**.  A scenario
carries observed values forward for six hours, applies a configured effect to
one or more variables, and asks the risk model what it would predict.  It does
not identify a treatment effect, does not adjust for confounding, and does not
represent clinical guidance.

Effect sizes are *never* hard-coded here.  They come from, in order:

1. ``results/mimic/effect_profiles.json`` if that file exists;
2. otherwise ``config/assumptions.yaml``.

The shipped YAML deliberately contains ``TODO`` placeholders.  If they have
not been replaced with sourced numeric values, the intervention is reported as
unavailable and no risk is projected -- rather than inventing a number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .twin import PatientTwin

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EFFECT_PROFILES_PATH = PROJECT_ROOT / "results" / "mimic" / "effect_profiles.json"
ASSUMPTIONS_PATH = PROJECT_ROOT / "config" / "assumptions.yaml"

INTERVENTIONS = ("fluid_bolus", "vasopressor", "antibiotics", "no_action")
HORIZONS = (1, 3, 6)
MAX_HORIZON = max(HORIZONS)

DISCLAIMER = (
    "Model-based scenario, not a causal treatment effect. Projections carry "
    "observed values forward, apply configured assumptions, and re-score with "
    "the same risk model. They are not clinical advice and must not be used "
    "for treatment decisions."
)

_TIME_COLUMNS = (
    "ICULOS",
    "Hour",
    "hour",
    "hour_index",
    "hours_since_admission",
    "HoursSinceAdmission",
)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_todo(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper().startswith("TODO")


def _load_effect_config(
    effects_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path, str]:
    if effects_path is not None:
        path = Path(effects_path)
        kind = "custom_json" if path.suffix.lower() == ".json" else "custom_yaml"
    elif EFFECT_PROFILES_PATH.exists():
        path = EFFECT_PROFILES_PATH
        kind = "effect_profiles_json"
    else:
        path = ASSUMPTIONS_PATH
        kind = "assumptions_yaml"

    if not path.exists():
        raise FileNotFoundError(f"effect configuration not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle)

    if not isinstance(data, Mapping) or not isinstance(data.get("interventions"), Mapping):
        raise ValueError(f"{path} must define an 'interventions' mapping")
    return dict(data), path, kind


def _assess_intervention(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    unresolved: list[str] = []

    if not isinstance(spec, Mapping):
        return {"available": False, "unresolved": ["intervention definition missing"], "effects": []}

    effects = spec.get("effects")
    if not isinstance(effects, list) or not effects:
        unresolved.append("effects")

    usable_effects: list[dict[str, Any]] = []
    if isinstance(effects, list):
        for index, effect in enumerate(effects):
            if not isinstance(effect, Mapping):
                unresolved.append(f"effects[{index}]")
                continue
            target = effect.get("target")
            if not isinstance(target, str) or not target.strip():
                unresolved.append(f"effects[{index}].target")
            for field in ("delta", "delta_low", "delta_high"):
                if not _is_number(effect.get(field)):
                    unresolved.append(f"effects[{index}].{field}")
            if not isinstance(effect.get("unit"), str) or not effect.get("unit").strip():
                unresolved.append(f"effects[{index}].unit")
            citation = effect.get("citation")
            if not isinstance(citation, str) or not citation.strip() or _is_todo(citation):
                unresolved.append(f"effects[{index}].citation")
            usable_effects.append(dict(effect))

    for field in ("onset_hours", "duration_hours"):
        if not _is_number(spec.get(field)):
            unresolved.append(field)

    if _is_number(spec.get("onset_hours")) and float(spec["onset_hours"]) < 0:
        unresolved.append("onset_hours (must be >= 0)")
    if _is_number(spec.get("duration_hours")) and float(spec["duration_hours"]) <= 0:
        unresolved.append("duration_hours (must be > 0)")

    citation = spec.get("citation")
    if not isinstance(citation, str) or not citation.strip() or _is_todo(citation):
        unresolved.append("citation")

    return {
        "available": not unresolved,
        "unresolved": unresolved,
        "effects": usable_effects,
    }


def _project_history(
    history: pd.DataFrame,
    spec: Mapping[str, Any] | None,
    scenario: str,
    max_hour: int = MAX_HORIZON,
) -> pd.DataFrame:
    """Carry the last observed row forward, then apply an effect if supplied."""

    if history.empty:
        raise ValueError("twin has no history")

    base = history.iloc[-1].copy()
    rows: list[pd.Series] = []

    for hour in range(1, max_hour + 1):
        row = base.copy()
        for column in _TIME_COLUMNS:
            if column in row.index and _is_number(row[column]):
                row[column] = float(row[column]) + hour

        if spec is not None:
            onset = float(spec["onset_hours"])
            duration = float(spec["duration_hours"])
            if onset <= hour < onset + duration:
                for effect in spec.get("effects", []):
                    target = str(effect["target"])
                    if target not in row.index:
                        continue
                    current = row[target]
                    try:
                        current_value = float(current)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(current_value):
                        continue
                    key = "delta" if scenario == "point" else f"delta_{scenario}"
                    row[target] = current_value + float(effect[key])
        rows.append(row)

    return pd.DataFrame(rows, columns=history.columns)


def _projected_risks(
    history: pd.DataFrame,
    projected: pd.DataFrame,
    model: Any,
) -> dict[int, float]:
    risks: dict[int, float] = {}
    for horizon in HORIZONS:
        prefix = pd.concat([history, projected.iloc[:horizon]], ignore_index=True)
        value = float(model.predict_proba(prefix))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"model returned invalid probability at +{horizon}h: {value}")
        risks[horizon] = value
    return risks


def _empty_bounds() -> dict[str, dict[str, None]]:
    return {str(hour): {"low": None, "high": None} for hour in HORIZONS}


def _result(
    *,
    intervention: str,
    label: str,
    available: bool,
    reason: str | None,
    source: Path | str,
    source_kind: str,
    effect_status: str,
    projected_risk: dict[str, float | None],
    bounds: dict[str, dict[str, float | None]],
    baseline_no_action: dict[str, float],
    assumptions: list[dict[str, Any]],
    unresolved: list[str],
) -> dict[str, Any]:
    return {
        "intervention": intervention,
        "label": label,
        "available": available,
        "reason": reason,
        "source": str(source),
        "source_kind": source_kind,
        "effect_status": effect_status,
        "horizons_hours": list(HORIZONS),
        "projected_risk": projected_risk,
        "bounds": bounds,
        "bound_type": "assumption_envelope" if available and intervention != "no_action" else None,
        "baseline_no_action": baseline_no_action,
        "assumptions": assumptions,
        "unresolved": unresolved,
        "disclaimer": DISCLAIMER,
        "generated_by": "twinsepsis.agent.simulate",
    }


def simulate(
    twin: PatientTwin,
    intervention: str,
    effects_path: str | Path | None = None,
) -> dict[str, Any]:
    """Project risk at +1/+3/+6 h under one scenario.

    Parameters
    ----------
    twin:
        Patient whose latest hour is the starting point.  The twin is never
        mutated.
    intervention:
        One of ``fluid_bolus``, ``vasopressor``, ``antibiotics``, ``no_action``.
    effects_path:
        Optional override for tests.  By default the resolution order is
        ``results/mimic/effect_profiles.json`` then ``config/assumptions.yaml``.

    Returns
    -------
    dict
        A JSON-serialisable result.  ``projected_risk`` is ``None`` when the
        configured effects are still TODO placeholders.

    Notes
    -----
    These are model-based scenarios, not causal treatment effects.
    """

    if intervention not in INTERVENTIONS:
        raise ValueError(
            f"unknown intervention {intervention!r}; expected one of {INTERVENTIONS}"
        )
    if twin.hours == 0:
        raise ValueError("twin has no history")

    history = twin.history

    baseline_projected = _project_history(history, None, "point")
    baseline = _projected_risks(history, baseline_projected, twin.model)
    baseline_strings = {str(hour): value for hour, value in baseline.items()}

    if intervention == "no_action":
        return _result(
            intervention=intervention,
            label="No action",
            available=True,
            reason=None,
            source="not_applicable",
            source_kind="none",
            effect_status="not_applicable",
            projected_risk=baseline_strings,
            bounds={key: {"low": value, "high": value} for key, value in baseline_strings.items()},
            baseline_no_action=baseline_strings,
            assumptions=[],
            unresolved=[],
        )

    config, source, source_kind = _load_effect_config(effects_path)

    spec = config["interventions"].get(intervention)
    assessment = _assess_intervention(spec)
    unresolved = list(assessment["unresolved"])

    if isinstance(spec, Mapping):
        for effect in spec.get("effects", []) or []:
            if isinstance(effect, Mapping):
                target = effect.get("target")
                if isinstance(target, str) and target not in twin.columns:
                    unresolved.append(f"target column not in twin: {target}")

    label = str(spec.get("label", intervention)) if isinstance(spec, Mapping) else intervention
    assumptions = [
        dict(effect)
        for effect in (assessment["effects"] if isinstance(spec, Mapping) else [])
    ]

    if unresolved:
        status = "placeholder" if any(
            _is_todo(value)
            for effect in assumptions
            for value in effect.values()
        ) else "invalid"
        reason = (
            "Effect assumptions are not usable yet. No risk was projected. "
            "Fill the TODO values and citations in the configured effect source, "
            "or provide results/mimic/effect_profiles.json."
        )
        return _result(
            intervention=intervention,
            label=label,
            available=False,
            reason=reason,
            source=source,
            source_kind=source_kind,
            effect_status=status,
            projected_risk={str(hour): None for hour in HORIZONS},
            bounds=_empty_bounds(),
            baseline_no_action=baseline_strings,
            assumptions=assumptions,
            unresolved=unresolved,
        )

    scenario_risks: dict[str, dict[int, float]] = {}
    for scenario in ("point", "low", "high"):
        projected = _project_history(history, spec, scenario)
        scenario_risks[scenario] = _projected_risks(history, projected, twin.model)

    point = {str(hour): scenario_risks["point"][hour] for hour in HORIZONS}
    bounds: dict[str, dict[str, float]] = {}
    for hour in HORIZONS:
        values = [scenario_risks[name][hour] for name in ("point", "low", "high")]
        bounds[str(hour)] = {"low": min(values), "high": max(values)}

    return _result(
        intervention=intervention,
        label=label,
        available=True,
        reason=None,
        source=source,
        source_kind=source_kind,
        effect_status="numeric",
        projected_risk=point,
        bounds=bounds,
        baseline_no_action=baseline_strings,
        assumptions=assumptions,
        unresolved=[],
    )


def list_intervention_specs(
    effects_path: str | Path | None = None,
) -> dict[str, Any]:
    """Describe all interventions and whether their assumptions are usable."""

    config, source, source_kind = _load_effect_config(effects_path)
    items: list[dict[str, Any]] = []

    for name in INTERVENTIONS:
        if name == "no_action":
            items.append(
                {
                    "name": name,
                    "label": "No action",
                    "description": "Carry the latest observed values forward without an intervention.",
                    "available": True,
                    "effect_status": "not_applicable",
                    "citation": "Not applicable",
                    "effects": [],
                    "unresolved": [],
                }
            )
            continue

        spec = config["interventions"].get(name)
        assessment = _assess_intervention(spec)
        effects = assessment["effects"] if isinstance(spec, Mapping) else []
        status = "numeric" if assessment["available"] else "placeholder"
        items.append(
            {
                "name": name,
                "label": str(spec.get("label", name)) if isinstance(spec, Mapping) else name,
                "description": str(spec.get("description", "")) if isinstance(spec, Mapping) else "",
                "available": assessment["available"],
                "effect_status": status,
                "citation": str(spec.get("citation", "")) if isinstance(spec, Mapping) else "",
                "effects": effects,
                "unresolved": assessment["unresolved"],
            }
        )

    return {
        "interventions": items,
        "source": str(source),
        "source_kind": source_kind,
        "any_available": any(item["available"] for item in items),
        "disclaimer": DISCLAIMER,
    }


__all__ = [
    "INTERVENTIONS",
    "HORIZONS",
    "DISCLAIMER",
    "simulate",
    "list_intervention_specs",
]
