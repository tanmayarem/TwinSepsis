"""Backend adapter for the TwinSepsis Streamlit dashboard.

This module contains **no Streamlit imports**.  It is the seam the UI calls and
it is unit-tested directly, so the dashboard never grows a second copy of the
model, patient-loading, trajectory or agent logic.

Everything clinical comes from the existing backend:

* patients: :func:`twinsepsis.patient_source.load_physionet_patient`;
* risk, trend, drivers, trajectory, scores and scenarios: the same
  :mod:`twinsepsis.agent.tools` / :class:`twinsepsis.agent.SepsisAgent`
  functions the CLI and LLM agent use;
* thresholds and model identity: read from the loaded model, never hard-coded.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from ..agent import tools as agent_tools
from ..agent.agent import SepsisAgent
from ..agent.gru_model import DEFAULT_GRU_MODEL_PATH
from ..agent.llm import LLMConfig, LLMConfigurationError
from ..agent.risk_model import (
    DEFAULT_ENSEMBLE_CONFIG_PATH,
    DEFAULT_XGB_MODEL_PATH,
    RiskModel,
    load_risk_model,
    model_identity,
)
from ..agent.twin import PatientTwin
from ..patient_source import (
    PatientSourceError,
    find_patient_file,
    load_physionet_patient,
    resolve_dataset_root,
)

MODEL_CHOICES = ("xgboost", "ensemble")

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "ensemble": "XGBoost + GRU ensemble",
}

MODEL_SUMMARIES = {
    "xgboost": (
        "PhysioNet XGBoost baseline. Per-hour 40-feature model with TreeSHAP "
        "attribution and model-based scenarios."
    ),
    "ensemble": (
        "Frozen 0.55 XGBoost / 0.45 GRU probability blend. No ensemble-level "
        "feature attribution and no intervention simulation."
    ),
}

#: Quick picks offered in the sidebar.  They are only shortcuts for patients
#: that exist in the local dataset (plus the two used by the committed
#: regressions); any valid ``pNNNNNN`` identifier can be typed instead.
REFERENCE_PATIENTS = ("p000001", "p000002", "p000003", "p000010", "p018469", "p114889")

#: Candidate trajectory windows, filtered against the patient's history.
TRAJECTORY_WINDOWS = (6, 12, 24, 48, 72, 168)

ENSEMBLE_EXPLANATION_NOTE = (
    "Ensemble-level feature attribution is unavailable. XGBoost-only TreeSHAP "
    "values describe one component of the weighted blend, not the ensemble, so "
    "they are deliberately not shown as ensemble explanations."
)
GENERIC_EXPLANATION_NOTE = (
    "Feature attribution is unavailable for the configured model, so no "
    "contribution ranking is shown."
)
ENSEMBLE_SIMULATION_NOTE = (
    "Intervention simulation is disabled for the XGBoost + GRU ensemble: "
    "counterfactual re-scoring of the blend has not been validated. No "
    "scenario is projected and none is implied."
)
GENERIC_SIMULATION_NOTE = (
    "Intervention simulation is unavailable for the configured model, so no "
    "scenario is projected."
)


class DashboardError(RuntimeError):
    """User-facing dashboard failure carrying a safe, readable message."""


@dataclass(frozen=True)
class MeasurementSpec:
    """One patient column the dashboard may display."""

    column: str
    label: str
    unit: str | None = None
    group: str = "vital"


MEASUREMENTS: tuple[MeasurementSpec, ...] = (
    MeasurementSpec("HR", "Heart rate", "bpm"),
    MeasurementSpec("O2Sat", "Oxygen saturation", "%"),
    MeasurementSpec("Temp", "Temperature", "°C"),
    MeasurementSpec("SBP", "Systolic blood pressure", "mmHg"),
    MeasurementSpec("MAP", "Mean arterial pressure", "mmHg"),
    MeasurementSpec("DBP", "Diastolic blood pressure", "mmHg"),
    MeasurementSpec("Resp", "Respiratory rate", "breaths/min"),
    MeasurementSpec("EtCO2", "End-tidal CO2", "mmHg"),
    MeasurementSpec("SaO2", "Arterial oxygen saturation", "%"),
    MeasurementSpec("FiO2", "Fraction of inspired oxygen", None),
    MeasurementSpec("Lactate", "Lactate", "mmol/L", "lab"),
    MeasurementSpec("WBC", "White blood cell count", "10³/µL", "lab"),
    MeasurementSpec("Creatinine", "Creatinine", "mg/dL", "lab"),
    MeasurementSpec("BUN", "Blood urea nitrogen", "mg/dL", "lab"),
    MeasurementSpec("Glucose", "Glucose", "mg/dL", "lab"),
    MeasurementSpec("pH", "Blood pH", None, "lab"),
    MeasurementSpec("Hct", "Hematocrit", "%", "lab"),
    MeasurementSpec("Hgb", "Hemoglobin", "g/dL", "lab"),
    MeasurementSpec("Platelets", "Platelet count", "10³/µL", "lab"),
    MeasurementSpec("Calcium", "Calcium", "mg/dL", "lab"),
)

DEMOGRAPHICS: tuple[MeasurementSpec, ...] = (
    MeasurementSpec("Age", "Age", "years", "demographic"),
    MeasurementSpec("Gender", "Sex code (as recorded)", None, "demographic"),
    MeasurementSpec("HospAdmTime", "Hospital admission to ICU", "hours", "demographic"),
    MeasurementSpec("ICULOS", "ICU length of stay", "hours", "demographic"),
    MeasurementSpec("Unit1", "ICU unit 1 indicator", None, "demographic"),
    MeasurementSpec("Unit2", "ICU unit 2 indicator", None, "demographic"),
)


def normalize_model_name(name: str) -> str:
    """Return the canonical model key or raise a readable error."""

    if not isinstance(name, str):
        raise DashboardError("model name must be a string")
    key = name.strip().lower()
    if key not in MODEL_CHOICES:
        raise DashboardError(
            f"unknown model {name!r}; expected one of {', '.join(MODEL_CHOICES)}"
        )
    return key


def model_label(name_or_model: str | RiskModel) -> str:
    """Human label for a model key or a loaded model instance."""

    if isinstance(name_or_model, str):
        key = name_or_model.strip().lower()
        return MODEL_LABELS.get(key, name_or_model)

    identity = model_identity(name_or_model)
    lowered = identity.lower()
    if "ensemble" in lowered:
        return MODEL_LABELS["ensemble"]
    if "xgb" in lowered:
        return MODEL_LABELS["xgboost"]
    return identity


def explanation_note(model: Any) -> str:
    """Why feature attribution is not shown, for this model."""

    if type(model).__name__ == "EnsembleRisk":
        return ENSEMBLE_EXPLANATION_NOTE
    return GENERIC_EXPLANATION_NOTE


def simulation_note(model: Any) -> str:
    """Why intervention simulation is not shown, for this model."""

    if type(model).__name__ == "EnsembleRisk":
        return ENSEMBLE_SIMULATION_NOTE
    return GENERIC_SIMULATION_NOTE


class ModelCache:
    """Loads each risk model at most once per process.

    Streamlit re-executes the page script on every interaction, so a model built
    inside the script would rebuild the XGBoost booster (and, for the ensemble,
    re-read the GRU checkpoint and its frozen preprocessing statistics) on every
    click.  One instance of this cache is held in a Streamlit ``cache_resource``
    and shared by all sessions in the process.
    """

    def __init__(self, loader: Callable[[str], RiskModel] = load_risk_model) -> None:
        self._loader = loader
        self._models: dict[str, RiskModel] = {}
        self._loads: dict[str, int] = {}

    def get(self, name: str) -> RiskModel:
        key = normalize_model_name(name)
        if key not in self._models:
            self._models[key] = self._loader(key)
            self._loads[key] = self._loads.get(key, 0) + 1
        return self._models[key]

    def load_count(self, name: str) -> int:
        """How many times this model has actually been constructed."""

        return self._loads.get(normalize_model_name(name), 0)

    @property
    def loaded(self) -> tuple[str, ...]:
        return tuple(self._models)


def llm_status(env: Any = None) -> dict[str, Any]:
    """Describe LLM configuration without ever exposing the credential.

    Returns only the provider and model id (both non-secret) plus a reason
    string naming the missing variables.  The API key is never read into this
    structure and never logged.
    """

    try:
        config = LLMConfig.from_env(env)
    except LLMConfigurationError as exc:
        return {
            "available": False,
            "provider": None,
            "model": None,
            "reason": str(exc),
        }
    return {
        "available": True,
        "provider": config.provider,
        "model": config.model,
        "reason": None,
    }


def trajectory_window_options(total: int) -> list[int]:
    """Selectable history windows for a patient with ``total`` hours."""

    if total < 1:
        return []
    options = [window for window in TRAJECTORY_WINDOWS if window < total]
    options.append(total)
    return options


class DashboardService:
    """Adapter the dashboard renders; holds the process-wide model cache."""

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        *,
        model_cache: ModelCache | None = None,
        trace_enabled: bool = False,
    ) -> None:
        self._dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.models = model_cache or ModelCache()
        self.trace_enabled = trace_enabled

    # -- dataset -------------------------------------------------------

    @property
    def dataset_root(self) -> Path:
        if self._dataset_root is None:
            try:
                self._dataset_root = resolve_dataset_root(None)
            except PatientSourceError as exc:
                raise DashboardError(str(exc)) from exc
        return self._dataset_root

    def _patient_file(self, patient_id: str) -> Path:
        try:
            return find_patient_file(
                patient_id,
                self._dataset_root if self._dataset_root is not None else None,
            )
        except PatientSourceError as exc:
            raise DashboardError(str(exc)) from exc

    def patient_row_count(self, patient_id: str) -> int:
        """Number of hourly rows available for one patient."""

        path = self._patient_file(patient_id)
        try:
            with path.open("r", encoding="utf-8") as handle:
                lines = sum(1 for line in handle if line.strip())
        except OSError as exc:
            raise DashboardError(f"Could not read {path}: {exc}") from exc

        rows = lines - 1  # header
        if rows < 1:
            raise DashboardError(f"Patient file has no hourly rows: {path}")
        return rows

    def reference_patients(self) -> list[str]:
        """Quick-pick identifiers that exist in the configured dataset."""

        existing: list[str] = []
        for patient_id in REFERENCE_PATIENTS:
            try:
                self._patient_file(patient_id)
            except DashboardError:
                continue
            existing.append(patient_id)
        return existing

    def load_twin(
        self,
        patient_id: str,
        hour: int,
        model_name: str = "xgboost",
    ) -> PatientTwin:
        """Load history rows ``1..hour`` with the selected model attached."""

        model = self.model(model_name)
        root = self._dataset_root
        if root is None:
            try:
                root = resolve_dataset_root(None)
            except PatientSourceError as exc:
                raise DashboardError(str(exc)) from exc
            self._dataset_root = root

        try:
            return load_physionet_patient(
                patient_id,
                hour,
                dataset_root=root,
                model=model,
            )
        except PatientSourceError as exc:
            raise DashboardError(str(exc)) from exc

    # -- models --------------------------------------------------------

    def model(self, name: str = "xgboost") -> RiskModel:
        """The cached model instance for ``name`` (loaded at most once)."""

        try:
            return self.models.get(name)
        except (ValueError, TypeError) as exc:
            raise DashboardError(str(exc)) from exc
        except Exception as exc:  # ModelUnavailableError and loader failures
            raise DashboardError(
                f"The {model_label(name)} model could not be loaded: {exc}"
            ) from exc

    def model_availability(self, name: str) -> tuple[bool, str | None]:
        """Cheap pre-flight check used to explain a missing backend."""

        key = normalize_model_name(name)
        if key == "xgboost":
            if not Path(DEFAULT_XGB_MODEL_PATH).is_file():
                return False, f"XGBoost artifact is missing: {DEFAULT_XGB_MODEL_PATH}"
            return True, None

        missing: list[str] = []
        if importlib.util.find_spec("torch") is None:
            missing.append("PyTorch is not installed in this interpreter")
        if not Path(DEFAULT_GRU_MODEL_PATH).is_file():
            missing.append(f"GRU checkpoint is missing: {DEFAULT_GRU_MODEL_PATH}")
        if not Path(DEFAULT_ENSEMBLE_CONFIG_PATH).is_file():
            missing.append(
                f"ensemble config is missing: {DEFAULT_ENSEMBLE_CONFIG_PATH}"
            )
        if missing:
            return False, (
                "The ensemble requires CPython 3.12 with CPU PyTorch "
                "(requirements-gru.txt): " + "; ".join(missing)
            )
        return True, None

    def capabilities(self, model: RiskModel) -> dict[str, Any]:
        """What the selected backend actually supports, with honest notes."""

        prediction = bool(getattr(model, "supports_prediction", True))
        explanation = bool(getattr(model, "supports_explanation", True))
        simulation = bool(getattr(model, "supports_simulation", True))

        limitations: list[str] = []
        if not explanation:
            limitations.append("feature_attribution_unavailable")
        if not simulation:
            limitations.append("intervention_simulation_unavailable")

        return {
            "model": model_identity(model),
            "model_label": model_label(model),
            "dataset": getattr(model, "dataset", "unknown"),
            "threshold": model.threshold,
            "prediction": prediction,
            "explanation": explanation,
            "simulation": simulation,
            "explanation_note": None if explanation else explanation_note(model),
            "simulation_note": None if simulation else simulation_note(model),
            "prediction_horizon": getattr(model, "prediction_horizon", None),
            "limitations": limitations,
        }

    # -- patient overview ----------------------------------------------

    @staticmethod
    def _column_summary(
        history: pd.DataFrame,
        specs: Sequence[MeasurementSpec],
    ) -> list[dict[str, Any]]:
        rows = len(history)
        latest = history.iloc[-1] if rows else None
        summaries: list[dict[str, Any]] = []

        for spec in specs:
            if spec.column not in history.columns:
                continue  # not part of this patient's schema at all

            series = pd.to_numeric(history[spec.column], errors="coerce")
            observed = int(series.notna().sum())
            latest_value = None
            if latest is not None:
                raw = latest[spec.column]
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    if pd.notna(raw):
                        latest_value = float(raw)

            summaries.append(
                {
                    "column": spec.column,
                    "label": spec.label,
                    "unit": spec.unit,
                    "group": spec.group,
                    "available": observed > 0,
                    "observed": observed,
                    "total": rows,
                    "missing_share": float(1.0 - observed / rows) if rows else 1.0,
                    "latest_value": latest_value,
                    "latest_available": latest_value is not None,
                }
            )
        return summaries

    def overview(self, twin: PatientTwin) -> dict[str, Any]:
        """Patient identity, demographics, measurements and missingness."""

        history = twin.history
        rows = len(history)
        missing = history.isna()
        per_column = missing.sum()
        never_recorded = [
            column for column in history.columns if int(per_column[column]) >= rows
        ]

        return {
            "patient_id": twin.patient_id,
            "hour": twin.hours,
            "hours": rows,
            "latest_hour": twin.latest_hour if rows else None,
            "dataset": getattr(twin.model, "dataset", "unknown"),
            "model": model_identity(twin.model),
            "model_label": model_label(twin.model),
            "threshold": twin.model.threshold,
            "demographics": self._column_summary(history, DEMOGRAPHICS),
            "measurements": self._column_summary(history, MEASUREMENTS),
            "missing": {
                "feature_columns": len(history.columns),
                "observed_columns": len(history.columns) - len(never_recorded),
                "never_recorded": never_recorded,
                "cells_missing": int(missing.to_numpy().sum()),
                "cells_total": int(rows * len(history.columns)),
                "share_missing": (
                    float(missing.to_numpy().mean()) if rows and len(history.columns) else 1.0
                ),
            },
        }

    # -- risk ----------------------------------------------------------

    def risk_assessment(
        self,
        twin: PatientTwin,
        *,
        trend_hours: int = 3,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Current-hour risk, frozen threshold decision, trend and drivers."""

        model = twin.model
        prediction = agent_tools.predict_risk(twin)
        state = agent_tools.get_twin_state(twin, trend_hours=trend_hours)

        explanation_available = bool(getattr(model, "supports_explanation", True))
        drivers: list[dict[str, Any]] | None = None
        note: str | None = None

        if explanation_available:
            explanation = agent_tools.explain_risk(twin, top_k=top_k)
            drivers = list(explanation.get("explanations") or [])
        else:
            note = explanation_note(model)

        components: dict[str, Any] | None = None
        if callable(getattr(model, "component_probabilities", None)):
            probabilities = model.component_probabilities(twin.history)
            components = {
                "xgboost": float(probabilities["xgboost"]),
                "gru": float(probabilities["gru"]),
                "xgboost_weight": float(getattr(model, "xgboost_weight", 0.0)),
                "gru_weight": float(getattr(model, "gru_weight", 0.0)),
            }

        return {
            "patient_id": prediction["patient_id"],
            "risk": float(prediction["risk"]),
            "threshold": float(prediction["threshold"]),
            "exceeds_threshold": bool(prediction["exceeds_threshold"]),
            "risk_model": prediction["risk_model"],
            "history_index": prediction["hour"],
            "trend": state["trend"],
            "current_risk": float(state["current_risk"]),
            "drivers": drivers,
            "explanation_available": explanation_available,
            "explanation_note": note,
            "explanation_units": getattr(model, "explanation_units", None),
            "prediction_horizon": getattr(model, "prediction_horizon", None),
            "dataset": getattr(model, "dataset", "unknown"),
            "artifact": getattr(model, "artifact", None),
            "components": components,
        }

    # -- historical trajectory -----------------------------------------

    def risk_trajectory(self, twin: PatientTwin, *, max_hours: int = 24) -> dict[str, Any]:
        """Prefix-scored historical risk, via the agent's own implementation."""

        agent = SepsisAgent(twin, None, trace_enabled=self.trace_enabled)
        return agent.risk_trajectory(max_hours=max_hours)

    @staticmethod
    def trajectory_frame(result: dict[str, Any]) -> pd.DataFrame:
        """Chart-ready frame of the historical scores (1-based ICU hours)."""

        points = result.get("risks") or []
        return pd.DataFrame(
            {
                "icu_hour": [int(point["hour"]) for point in points],
                "risk": [float(point["risk"]) for point in points],
            }
        )

    # -- measurements ---------------------------------------------------

    def vitals_frame(
        self,
        twin: PatientTwin,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Wide frame of selected measurements, missing values preserved."""

        history = twin.history
        if columns is None:
            selected = [
                spec.column for spec in MEASUREMENTS if spec.column in history.columns
            ]
        else:
            selected = [str(column) for column in columns]

        unknown = [column for column in selected if column not in history.columns]
        if unknown:
            raise DashboardError(
                "These measurements are not part of the patient's data: "
                + ", ".join(unknown)
            )

        frame = pd.DataFrame({"icu_hour": range(1, len(history) + 1)})
        for column in selected:
            frame[column] = pd.to_numeric(history[column], errors="coerce")
        return frame

    def vitals_long(
        self,
        twin: PatientTwin,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        """Long frame for faceted charting; ``NaN`` rows are kept as gaps."""

        wide = self.vitals_frame(twin, columns)
        labels = {
            spec.column: (
                f"{spec.label} ({spec.unit})" if spec.unit else spec.label
            )
            for spec in MEASUREMENTS
        }

        long = wide.melt(
            id_vars="icu_hour",
            var_name="column",
            value_name="value",
        )
        long["measurement"] = long["column"].map(lambda key: labels.get(key, key))
        return long.sort_values(["column", "icu_hour"]).reset_index(drop=True)

    def available_measurements(self, twin: PatientTwin) -> list[dict[str, Any]]:
        """Measurements that have at least one recorded value."""

        history = twin.history
        rows = len(history)
        available: list[dict[str, Any]] = []

        for spec in MEASUREMENTS:
            if spec.column not in history.columns:
                continue
            series = pd.to_numeric(history[spec.column], errors="coerce")
            observed = int(series.notna().sum())
            if observed == 0:
                continue
            available.append(
                {
                    "column": spec.column,
                    "label": spec.label,
                    "unit": spec.unit,
                    "group": spec.group,
                    "observed": observed,
                    "total": rows,
                    "missing_share": float(1.0 - observed / rows) if rows else 1.0,
                }
            )
        return available

    # -- assistant -------------------------------------------------------

    def assistant(
        self,
        twin: PatientTwin,
        request: str,
        *,
        llm: Any = None,
    ) -> dict[str, Any]:
        """Run the existing agent, or its deterministic tools-only workflow."""

        agent = SepsisAgent(twin, llm, trace_enabled=self.trace_enabled)
        if llm is None:
            return agent.run_tools_only(twin.patient_id, twin.hours)
        return agent.run(request, twin.patient_id, twin.hours)

    # -- scenarios -------------------------------------------------------

    def scenario_catalogue(self) -> dict[str, Any]:
        """Available interventions and whether their assumptions resolve."""

        return agent_tools.list_interventions()

    def scenario(self, twin: PatientTwin, intervention: str) -> dict[str, Any]:
        """Run one model-based scenario for the selected patient."""

        if not getattr(twin.model, "supports_simulation", True):
            raise DashboardError(simulation_note(twin.model))
        return agent_tools.simulate_intervention(twin, intervention)


__all__ = [
    "MODEL_CHOICES",
    "MODEL_LABELS",
    "MODEL_SUMMARIES",
    "MEASUREMENTS",
    "DEMOGRAPHICS",
    "REFERENCE_PATIENTS",
    "TRAJECTORY_WINDOWS",
    "DashboardError",
    "DashboardService",
    "MeasurementSpec",
    "ModelCache",
    "explanation_note",
    "llm_status",
    "model_label",
    "normalize_model_name",
    "simulation_note",
    "trajectory_window_options",
]
