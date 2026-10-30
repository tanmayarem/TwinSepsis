"""Model-agnostic risk-model interface for the TwinSepsis agent.

The agent should not care whether risk comes from the existing PhysioNet
XGBoost artifact, the (future) MIMIC ensemble, or a test double.  Every
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
# Weighted XGBoost + GRU ensemble (scaffold)
# ---------------------------------------------------------------------------


class EnsembleRisk(RiskModel):
    """Weighted probability ensemble: ``w * p_gru + (1 - w) * p_xgb``.

    Weights and threshold are loaded from ``results/ensemble_config.json``.
    The class is deliberately a scaffold: the repository has no MIMIC-IV
    component models, and the existing PhysioNet GRU checkpoint cannot be
    loaded in the committed environment (``torch`` is absent).  Calling
    :meth:`predict_proba` or :meth:`explain` raises ``NotImplementedError``
    rather than returning a fabricated probability.
    """

    supports_prediction = False
    supports_explanation = False
    supports_simulation = False
    dataset = "MIMIC-IV (model not available)"
    explanation_units = None
    prediction_horizon = "Unknown (no MIMIC-IV model is available)."

    def __init__(
        self,
        config_path: str | Path | None = None,
        threshold: float | None = None,
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

        xgboost_weight = config.get("xgboost_weight")
        gru_weight = config.get("gru_weight")
        if xgboost_weight is None or gru_weight is None:
            raise ValueError("ensemble config must define xgboost_weight and gru_weight")

        xgboost_weight = float(xgboost_weight)
        gru_weight = float(gru_weight)
        if not math.isclose(xgboost_weight + gru_weight, 1.0, abs_tol=1e-6):
            raise ValueError(
                "ensemble weights must sum to 1.0, got "
                f"{xgboost_weight} + {gru_weight}"
            )

        configured_threshold = threshold if threshold is not None else config.get("threshold", 0.5)
        super().__init__(threshold=float(configured_threshold), name="EnsembleRisk")

        self.config_path = path
        self.config = config
        self.xgboost_weight = xgboost_weight
        self.gru_weight = gru_weight
        self.components = tuple(config.get("components", ()))

    def combine(self, p_gru: float, p_xgb: float) -> float:
        """Apply the frozen linear combination to two probabilities."""

        p_gru = float(p_gru)
        p_xgb = float(p_xgb)
        if not (0.0 <= p_gru <= 1.0 and 0.0 <= p_xgb <= 1.0):
            raise ValueError("component probabilities must be in [0, 1]")
        return self.gru_weight * p_gru + self.xgboost_weight * p_xgb

    def _unavailable(self) -> NotImplementedError:
        return NotImplementedError(
            "EnsembleRisk is scaffolded but not usable: no MIMIC-IV component "
            "models exist in this repository, and the PhysioNet GRU checkpoint "
            "cannot be loaded because torch is not installed. Weights and "
            "threshold were loaded from the config; no probability was produced."
        )

    def featurize(self, window_df: pd.DataFrame) -> Any:
        raise self._unavailable()

    def predict_proba(self, window_df: pd.DataFrame) -> float:
        raise self._unavailable()

    def explain(self, window_df: pd.DataFrame, top_k: int = 5) -> list[dict[str, Any]]:
        raise self._unavailable()


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


__all__ = [
    "RiskModel",
    "PhysioNetXGBRisk",
    "EnsembleRisk",
    "StubRisk",
    "load_default_risk_model",
    "FEATURE_LABELS",
    "PHYSIONET_FEATURE_COLUMNS",
]
