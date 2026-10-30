"""Model-agnostic risk-model interface for the TwinSepsis agent.

The agent should not care whether risk comes from the PhysioNet XGBoost
artifact, the PhysioNet XGBoost + GRU ensemble, or a test double.  Every
implementation of :class:`RiskModel` therefore exposes the same four things:

* ``featurize(window_df)`` -- turn a patient's hourly window into model input;
* ``predict_proba(window_df)`` -- risk at the *latest* hour in the window;
* ``explain(window_df, top_k)`` -- per-feature contributions for that hour;
* ``threshold`` -- the frozen operating threshold for this model.

``window_df`` is always a patient's history **up to and including the hour
being scored**.  Implementations must never read rows after the last row.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XGB_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_baseline.json"
DEFAULT_ENSEMBLE_CONFIG_PATH = PROJECT_ROOT / "results" / "ensemble_config.json"


def model_identity(model: Any) -> str:
    """Stable, human-readable model identifier for responses and traces.

    Uses the model's declared ``name`` when present and falls back to the
    class name for third-party implementations that do not set one.
    """

    name = getattr(model, "name", None)
    if isinstance(name, str) and name.strip():
        return name
    return type(model).__name__


class ModelUnavailableError(RuntimeError):
    """Raised when a required model component cannot be loaded.

    The agent never substitutes a different model (for example XGBoost-only
    risk when the GRU is missing), so this error always aborts the run.
    """


class RiskModel(ABC):
    """Common interface for every risk model used by the agent."""

    # Capability flags and provenance.  The agent uses these to decide which
    # tools it may expose.  Defaults are intentionally permissive so that
    # existing models keep working; unusable models must override them.
    supports_prediction = True
    supports_explanation = True
    supports_simulation = True
    test_only = False
    dataset = "unknown"
    artifact: str | None = None
    explanation_units: str | None = None
    prediction_horizon = "Scored hour only; no multi-hour forecast."

    def __init__(self, threshold: float = 0.5, name: str | None = None) -> None:
        self._threshold = self._validate_threshold(threshold)
        self.name = name or type(self).__name__

    @staticmethod
    def _validate_threshold(threshold: float) -> float:
        value = float(threshold)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold!r}")
        return value

    @property
    def threshold(self) -> float:
        """Frozen decision threshold.  Never tune this at inference time."""

        return self._threshold

    @abstractmethod
    def featurize(self, window_df: pd.DataFrame) -> Any:
        """Convert an hourly history into this model's input representation."""

    @abstractmethod
    def predict_proba(self, window_df: pd.DataFrame) -> float:
        """Return P(sepsis at/after the latest hour) in [0, 1]."""

    @abstractmethod
    def explain(self, window_df: pd.DataFrame, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k feature contributions for the latest hour."""


# ---------------------------------------------------------------------------
# PhysioNet XGBoost
# ---------------------------------------------------------------------------

# The 40 columns the saved model was trained on, in training order.  The
# loader prefers the feature names stored in the artifact and only uses this
# list as a fallback / documentation aid.
PHYSIONET_FEATURE_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS",
]

FEATURE_LABELS = {
    "HR": "Heart rate",
    "O2Sat": "Oxygen saturation",
    "Temp": "Temperature",
    "SBP": "Systolic blood pressure",
    "MAP": "Mean arterial pressure",
    "DBP": "Diastolic blood pressure",
    "Resp": "Respiratory rate",
    "EtCO2": "End-tidal CO2",
    "BaseExcess": "Base excess",
    "HCO3": "Bicarbonate",
    "FiO2": "Fraction of inspired oxygen",
    "pH": "Blood pH",
    "PaCO2": "Partial pressure of CO2",
    "SaO2": "Arterial oxygen saturation",
    "AST": "Aspartate aminotransferase",
    "BUN": "Blood urea nitrogen",
    "Alkalinephos": "Alkaline phosphatase",
    "Calcium": "Calcium",
    "Chloride": "Chloride",
    "Creatinine": "Creatinine",
    "Bilirubin_direct": "Direct bilirubin",
    "Glucose": "Glucose",
    "Lactate": "Lactate",
    "Magnesium": "Magnesium",
    "Phosphate": "Phosphate",
    "Potassium": "Potassium",
    "Bilirubin_total": "Total bilirubin",
    "TroponinI": "Troponin I",
    "Hct": "Hematocrit",
    "Hgb": "Hemoglobin",
    "PTT": "Partial thromboplastin time",
    "WBC": "White blood cell count",
    "Fibrinogen": "Fibrinogen",
    "Platelets": "Platelet count",
    "Age": "Age",
    "Gender": "Sex",
    "Unit1": "ICU unit 1",
    "Unit2": "ICU unit 2",
    "HospAdmTime": "Hours from hospital admission to ICU",
    "ICULOS": "ICU length of stay (hours)",
}


def _as_frame(window_df: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
    if isinstance(window_df, Mapping):
        frame = pd.DataFrame([window_df])
    elif isinstance(window_df, pd.DataFrame):
        frame = window_df
    else:
        frame = pd.DataFrame(window_df)
    if frame.empty:
        raise ValueError("window_df must contain at least one hourly row")
    return frame


class PhysioNetXGBRisk(RiskModel):
    """Risk model backed by ``models/xgboost_baseline.json``.

    The audit confirmed that this artifact is a tracked, 40-feature
    ``XGBClassifier`` JSON and that it loads with the installed
    ``xgboost==3.4.1``.  It is a *PhysioNet* model: its features are the 40
    PhysioNet columns, not MIMIC-IV concepts.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(threshold=threshold, name="PhysioNetXGBRisk")
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_XGB_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(f"XGBoost artifact not found: {self.model_path}")

        model = XGBClassifier()
        model.load_model(str(self.model_path))
        booster = model.get_booster()
        feature_names = list(booster.feature_names or PHYSIONET_FEATURE_COLUMNS)

        if len(feature_names) != int(model.n_features_in_):
            raise ValueError(
                "XGBoost artifact feature names do not match n_features_in_: "
                f"{len(feature_names)} vs {model.n_features_in_}"
            )
        if len(feature_names) == 0:
            raise ValueError("XGBoost artifact exposes no feature names")

        self._model = model
        self._booster = booster
        self.feature_names = feature_names
        self.best_iteration = getattr(model, "best_iteration", None)

        self.dataset = "PhysioNet Challenge 2019"
        self.artifact = str(self.model_path)
        self.explanation_units = "log-odds (XGBoost TreeSHAP margin)"
        self.prediction_horizon = (
            "Scored ICU hour (PhysioNet hourly SepsisLabel); not a multi-hour forecast."
        )

    def featurize(self, window_df: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
        frame = _as_frame(window_df)
        row = frame.iloc[[-1]].copy()

        missing = [column for column in self.feature_names if column not in row.columns]
        if missing:
            raise ValueError(
                "window_df is missing PhysioNet XGBoost feature columns: "
                + ", ".join(missing)
            )

        numeric = pd.DataFrame(index=row.index)
        for column in self.feature_names:
            original = row[column]
            converted = pd.to_numeric(original, errors="coerce")
            if original.notna().sum() != converted.notna().sum():
                raise ValueError(f"feature {column!r} contains non-numeric values")
            numeric[column] = converted

        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        return numeric.astype(np.float32).reset_index(drop=True)

    def predict_proba(self, window_df: pd.DataFrame | Mapping[str, Any]) -> float:
        features = self.featurize(window_df)
        probability = float(self._model.predict_proba(features)[0, 1])
        if not math.isfinite(probability):
            raise ValueError("XGBoost returned a non-finite probability")
        return min(1.0, max(0.0, probability))

    def explain(
        self,
        window_df: pd.DataFrame | Mapping[str, Any],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        features = self.featurize(window_df)
        dmatrix = DMatrix(features, feature_names=self.feature_names)
        # TreeSHAP contributions in margin (log-odds) space.  The final
        # column is the model bias and is intentionally excluded here.
        contributions = self._booster.predict(dmatrix, pred_contribs=True)[0, :-1]
        values = features.iloc[0]

        records: list[dict[str, Any]] = []
        for column, contribution in zip(self.feature_names, contributions):
            contribution_value = float(contribution)
            value = values[column]
            records.append(
                {
                    "feature": column,
                    "label": FEATURE_LABELS.get(column, column),
                    "value": float(value) if pd.notna(value) else None,
                    "contribution": contribution_value,
                    "direction": (
                        "increases"
                        if contribution_value > 0
                        else "decreases"
                        if contribution_value < 0
                        else "neutral"
                    ),
                }
            )

        records.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return records[:top_k]


# ---------------------------------------------------------------------------
# Weighted XGBoost + GRU ensemble
# ---------------------------------------------------------------------------


def _required_config_float(config: Mapping[str, Any], key: str) -> float:
    if key not in config:
        raise ValueError(f"ensemble config must define {key!r}")
    try:
        value = float(config[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ensemble config {key!r} is not numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"ensemble config {key!r} must be finite")
    return value


class EnsembleRisk(RiskModel):
    """Weighted probability ensemble ``0.55 * p_xgb + 0.45 * p_gru``.

    Weights and the decision threshold are read from
    ``results/ensemble_config.json`` and are never passed in by callers, so the
    frozen validation-selected values (0.55 / 0.45, threshold 0.81) cannot
    drift at inference time.

    Both components are constructed eagerly.  If either fails to load, the
    constructor raises :class:`ModelUnavailableError`; it never falls back to
    XGBoost-only risk.

    ``supports_explanation`` is ``False``: the repository defines no
    ensemble-level attribution, and XGBoost-only TreeSHAP values would
    describe one component rather than the blend.  ``supports_simulation`` is
    ``False`` because counterfactual re-scoring of the ensemble has not been
    validated.
    """

    supports_prediction = True
    supports_explanation = False
    supports_simulation = False

    dataset = "PhysioNet Challenge 2019"
    explanation_units = None
    prediction_horizon = (
        "Scored ICU hour; XGBoost uses the scored hour's 40 features and the "
        "GRU uses the causal hourly prefix. Not a multi-hour forecast."
    )

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        xgboost_model: Any | None = None,
        gru_model: Any | None = None,
        gru_model_path: str | Path | None = None,
    ) -> None:
        path = (
            Path(config_path)
            if config_path is not None
            else DEFAULT_ENSEMBLE_CONFIG_PATH
        )
        if not path.exists():
            raise FileNotFoundError(f"ensemble config not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise ValueError(f"ensemble config must contain a JSON object: {path}")

        if config.get("model") != "weighted_probability_ensemble":
            raise ValueError(
                "ensemble config model must be 'weighted_probability_ensemble', "
                f"got {config.get('model')!r}"
            )

        declared_components = [str(item) for item in config.get("components", ())]
        if sorted(declared_components) != ["GRU", "XGBoost"]:
            raise ValueError(
                "ensemble config components must be ['XGBoost', 'GRU'], got "
                f"{declared_components}"
            )

        xgboost_weight = _required_config_float(config, "xgboost_weight")
        gru_weight = _required_config_float(config, "gru_weight")
        if not math.isclose(xgboost_weight + gru_weight, 1.0, abs_tol=1e-6):
            raise ValueError(
                "ensemble weights must sum to 1.0, got "
                f"{xgboost_weight} + {gru_weight}"
            )

        threshold = _required_config_float(config, "threshold")

        super().__init__(threshold=threshold, name="PhysioNetXGBGRUEnsemble")

        self.config_path = path
        self.config = dict(config)
        self.xgboost_weight = xgboost_weight
        self.gru_weight = gru_weight
        self.components = tuple(declared_components)

        if xgboost_model is None:
            xgboost_component = self._construct("XGBoost", PhysioNetXGBRisk)
        else:
            xgboost_component = xgboost_model

        if gru_model is None:
            def _build_gru() -> RiskModel:
                from .gru_model import PhysioNetGRURisk

                return PhysioNetGRURisk(model_path=gru_model_path)

            gru_component = self._construct("GRU", _build_gru)
        else:
            gru_component = gru_model

        for component_name, component in (
            ("XGBoost", xgboost_component),
            ("GRU", gru_component),
        ):
            if not callable(getattr(component, "predict_proba", None)):
                raise TypeError(
                    f"{component_name} component does not implement predict_proba()"
                )

        xgboost_features = getattr(xgboost_component, "feature_names", None)
        gru_features = getattr(gru_component, "feature_columns", None)
        if xgboost_features is not None and gru_features is not None:
            if [str(column) for column in xgboost_features] != [
                str(column) for column in gru_features
            ]:
                raise ValueError(
                    "XGBoost and GRU feature names/order differ; refusing to "
                    "blend components trained on different inputs"
                )

        self._xgboost = xgboost_component
        self._gru = gru_component

        artifacts = [
            getattr(xgboost_component, "artifact", None),
            getattr(gru_component, "artifact", None),
        ]
        self.artifact = " + ".join(
            item for item in artifacts if isinstance(item, str)
        ) or None

    @staticmethod
    def _construct(component_name: str, factory: Any) -> RiskModel:
        try:
            return factory()
        except Exception as exc:
            raise ModelUnavailableError(
                f"The {component_name} component of the PhysioNet ensemble could "
                "not be loaded. The ensemble requires both components and will "
                "not fall back to XGBoost-only risk. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc

    def combine(self, p_gru: float, p_xgb: float) -> float:
        """Apply the frozen linear combination to two probabilities."""

        probabilities = {"GRU": float(p_gru), "XGBoost": float(p_xgb)}
        for name, value in probabilities.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} probability must be finite and in [0, 1], got {value!r}"
                )
        return self.gru_weight * probabilities["GRU"] + self.xgboost_weight * probabilities["XGBoost"]

    def featurize(self, window_df: pd.DataFrame | Mapping[str, Any]) -> dict[str, Any]:
        return {
            "xgboost": self._xgboost.featurize(window_df),
            "gru": self._gru.featurize(window_df),
        }

    def component_probabilities(
        self, window_df: pd.DataFrame | Mapping[str, Any]
    ) -> dict[str, float]:
        """Return both component probabilities used by the frozen blend."""

        return {
            "gru": float(self._gru.predict_proba(window_df)),
            "xgboost": float(self._xgboost.predict_proba(window_df)),
        }

    def predict_proba(self, window_df: pd.DataFrame | Mapping[str, Any]) -> float:
        components = self.component_probabilities(window_df)
        probability = self.combine(components["gru"], components["xgboost"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("ensemble produced an invalid probability")
        return probability

    def explain(
        self,
        window_df: pd.DataFrame | Mapping[str, Any],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "Ensemble-level feature attribution is not defined for this model. "
            "XGBoost TreeSHAP values describe only the XGBoost component and "
            "would misrepresent the weighted blend, so they are not presented "
            "as ensemble explanations."
        )


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


class StubRisk(RiskModel):
    """**TEST-ONLY** deterministic risk model.

    This exists so the twin, simulation, and tool layers can be tested without
    a real model or patient data.  Its numbers are arbitrary fixtures and must
    never be reported as experimental results.  Production code should use
    :class:`PhysioNetXGBRisk` or a future MIMIC model.
    """

    test_only = True
    dataset = "synthetic test fixture (not clinical data)"
    explanation_units = "arbitrary test heuristic (not clinical)"
    prediction_horizon = "Synthetic scored hour (test fixture)."

    _WEIGHTS = {
        "HR": 0.01,
        "Temp": 0.20,
        "Resp": 0.02,
        "SBP": -0.005,
        "MAP": -0.01,
        "WBC": 0.01,
        "Lactate": 0.05,
    }
    _BASELINES = {
        "HR": 80.0,
        "Temp": 37.0,
        "Resp": 18.0,
        "SBP": 120.0,
        "MAP": 85.0,
        "WBC": 8.0,
        "Lactate": 1.0,
    }

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(threshold=threshold, name="StubRisk")

    def featurize(self, window_df: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
        frame = _as_frame(window_df)
        row = frame.iloc[[-1]].copy()
        numeric = {}
        for column, value in row.iloc[0].items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            numeric[str(column)] = number if math.isfinite(number) else float("nan")
        return pd.DataFrame([numeric])

    def _score(self, features: pd.DataFrame) -> float:
        z = 0.0
        for column, weight in self._WEIGHTS.items():
            if column not in features.columns:
                continue
            value = features.iloc[0][column]
            if pd.isna(value):
                continue
            z += weight * (float(value) - self._BASELINES[column])
        return 1.0 / (1.0 + math.exp(-z))

    def predict_proba(self, window_df: pd.DataFrame | Mapping[str, Any]) -> float:
        return float(self._score(self.featurize(window_df)))

    def explain(
        self,
        window_df: pd.DataFrame | Mapping[str, Any],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        features = self.featurize(window_df)
        records = []
        for column, weight in self._WEIGHTS.items():
            if column not in features.columns:
                continue
            value = features.iloc[0][column]
            if pd.isna(value):
                continue
            contribution = weight * (float(value) - self._BASELINES[column])
            records.append(
                {
                    "feature": column,
                    "label": FEATURE_LABELS.get(column, column),
                    "value": float(value),
                    "contribution": float(contribution),
                    "direction": (
                        "increases"
                        if contribution > 0
                        else "decreases"
                        if contribution < 0
                        else "neutral"
                    ),
                }
            )
        records.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return records[:top_k]


def load_default_risk_model() -> RiskModel:
    """Load the only real model currently available (PhysioNet XGBoost).

    No silent stub fallback is provided: if the real artifact cannot be
    loaded, callers get an exception rather than fabricated risk.
    """

    try:
        return PhysioNetXGBRisk()
    except Exception as exc:  # pragma: no cover - message is the important part
        raise RuntimeError(
            "Could not load a real risk model. The PhysioNet XGBoost artifact "
            f"at {DEFAULT_XGB_MODEL_PATH} failed to load: {exc}"
        ) from exc


def load_risk_model(name: str = "xgboost") -> RiskModel:
    """Load one of the supported production risk models by name.

    ``xgboost`` keeps the original XGBoost-only behaviour and works in the
    project's Python 3.14 environment.  ``ensemble`` is the PhysioNet
    XGBoost + GRU blend and requires the isolated CPython 3.12 environment
    with CPU PyTorch.  Failures are reported as :class:`ModelUnavailableError`;
    there is no fallback from ``ensemble`` to ``xgboost``.
    """

    if not isinstance(name, str):
        raise TypeError("risk model name must be a string")
    normalized = name.strip().lower()

    if normalized == "xgboost":
        try:
            return PhysioNetXGBRisk()
        except Exception as exc:
            raise ModelUnavailableError(
                "The PhysioNet XGBoost artifact could not be loaded from "
                f"{DEFAULT_XGB_MODEL_PATH}: {type(exc).__name__}: {exc}"
            ) from exc

    if normalized == "ensemble":
        try:
            return EnsembleRisk()
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                "The PhysioNet XGBoost + GRU ensemble could not be loaded using "
                f"{DEFAULT_ENSEMBLE_CONFIG_PATH}: {type(exc).__name__}: {exc}"
            ) from exc

    raise ValueError(
        f"unknown risk model {name!r}; expected 'xgboost' or 'ensemble'"
    )


__all__ = [
    "RiskModel",
    "PhysioNetXGBRisk",
    "EnsembleRisk",
    "ModelUnavailableError",
    "model_identity",
    "StubRisk",
    "load_default_risk_model",
    "load_risk_model",
    "FEATURE_LABELS",
    "PHYSIONET_FEATURE_COLUMNS",
]
