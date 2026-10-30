"""PhysioNet GRU risk component for the TwinSepsis agent.

This module is the production port of the GRU branch that was verified against
``results/ensemble_test_predictions.csv`` (worst deviation 1.8e-7).  It
reproduces the notebook pipeline exactly:

1. select the checkpoint's 40 raw PhysioNet columns, ``inf`` -> ``NaN``;
2. record which of the 34 measured columns were observed *before* imputation;
3. forward-fill each patient's history (causal, prefix-safe);
4. fill remaining gaps with the *training* medians;
5. z-score with the *training* means/stds and clip to +/- ``clip``;
6. append the 34 observation masks -> 74 features per hour;
7. ``sigmoid(GRU(x))`` and take the latest valid hour.

Only the checkpoint's frozen preprocessing statistics are used.  Nothing is
computed from the patient being scored, so scoring a prefix cannot leak future
information.

``torch`` is imported lazily.  Importing this module, and importing the agent
package, therefore still works in interpreters where PyTorch is absent; the
error only appears when a GRU-backed model is actually constructed.

The checkpoint pickles NumPy arrays, so it must be loaded with
``weights_only=False``.  That is only safe because the loader is hard-wired to
the repository's trusted local artifact and never accepts a caller-supplied
path from the CLI.

The model is a **PhysioNet Challenge 2019** component.  It is not a MIMIC-IV
model, not a multi-hour forecast, and carries no treatment-effect claim.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .risk_model import PHYSIONET_FEATURE_COLUMNS, RiskModel, _as_frame

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRU_MODEL_PATH = PROJECT_ROOT / "models" / "gru_baseline.pt"

_REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "clip",
        "feature_columns",
        "means",
        "measured_columns",
        "medians",
        "model_config",
        "model_state_dict",
        "stds",
    }
)


class GRUUnavailableError(RuntimeError):
    """Raised when the GRU component cannot be used in this interpreter."""


class GRUCheckpointError(ValueError):
    """Raised when the local checkpoint does not match the verified contract."""


def _import_torch():
    """Import PyTorch lazily and convert a missing install into a clear error."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in the 3.14 venv
        raise GRUUnavailableError(
            "PyTorch is not installed in this interpreter, so the PhysioNet GRU "
            "cannot be loaded. The GRU/ensemble requires CPython 3.12 with "
            "torch==2.15.0+cpu (see requirements-gru.txt and the isolated "
            "environment C:\\TwinSepsis-venvs\\gru312). "
            f"Current interpreter: {sys.executable}"
        ) from exc
    return torch


_SEPSIS_GRU_CLASS: type | None = None


def _sepsis_gru_class() -> type:
    """Return the notebook's ``SepsisGRU`` class, creating it once on demand.

    The class must subclass ``torch.nn.Module``, which is impossible without
    importing torch, so it is defined lazily rather than at module import time.
    """

    global _SEPSIS_GRU_CLASS
    if _SEPSIS_GRU_CLASS is not None:
        return _SEPSIS_GRU_CLASS

    torch = _import_torch()
    nn = torch.nn

    class SepsisGRU(nn.Module):
        """Exact architecture from notebooks/08_gru.ipynb cell 11."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            out, _ = self.gru(x)
            return self.head(out).squeeze(-1)

    SepsisGRU.__module__ = __name__
    SepsisGRU.__qualname__ = "SepsisGRU"
    _SEPSIS_GRU_CLASS = SepsisGRU
    return SepsisGRU


def __getattr__(name: str) -> Any:
    """Expose ``SepsisGRU`` without importing torch at module import time."""

    if name == "SepsisGRU":
        return _sepsis_gru_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class GRUArtifact:
    """Everything needed to reproduce the verified GRU preprocessing."""

    model: Any
    path: Path
    feature_columns: tuple[str, ...]
    measured_columns: tuple[str, ...]
    measured_indices: tuple[int, ...]
    medians: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    clip: float
    model_config: dict[str, Any]


def _require_finite_vector(checkpoint: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    if key not in checkpoint:
        raise GRUCheckpointError(f"GRU checkpoint is missing {key!r}")
    array = np.asarray(checkpoint[key], dtype=np.float32)
    if array.shape != (size,):
        raise GRUCheckpointError(
            f"GRU checkpoint {key!r} has shape {array.shape}, expected ({size},)"
        )
    if not np.isfinite(array).all():
        raise GRUCheckpointError(f"GRU checkpoint {key!r} contains non-finite values")
    return array


def load_gru_artifact(model_path: str | Path | None = None) -> GRUArtifact:
    """Load and validate the frozen GRU checkpoint.

    Every deviation from the verified contract raises :class:`GRUCheckpointError`
    rather than silently scoring with a different preprocessing or architecture.
    """

    torch = _import_torch()
    path = Path(model_path) if model_path is not None else DEFAULT_GRU_MODEL_PATH
    if not path.is_file():
        raise GRUUnavailableError(f"GRU checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - corrupt files are not shipped
        raise GRUCheckpointError(f"Could not load GRU checkpoint {path}: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise GRUCheckpointError(
            f"GRU checkpoint {path} did not contain a mapping, got {type(checkpoint).__name__}"
        )

    missing = _REQUIRED_CHECKPOINT_KEYS - set(checkpoint)
    if missing:
        raise GRUCheckpointError(
            "GRU checkpoint is missing required keys: " + ", ".join(sorted(missing))
        )

    try:
        model_config = dict(checkpoint["model_config"])
    except (TypeError, ValueError) as exc:
        raise GRUCheckpointError("GRU checkpoint model_config is not a mapping") from exc

    feature_columns = tuple(str(column) for column in checkpoint["feature_columns"])
    measured_columns = tuple(str(column) for column in checkpoint["measured_columns"])

    # The XGBoost artifact and the GRU checkpoint were trained on the same
    # order.  A mismatch means the two components cannot be blended honestly.
    if feature_columns != tuple(PHYSIONET_FEATURE_COLUMNS):
        raise GRUCheckpointError(
            "GRU checkpoint feature order does not match the verified PhysioNet "
            "feature order; refusing to score with mismatched inputs"
        )
    if len(set(measured_columns)) != len(measured_columns):
        raise GRUCheckpointError("GRU checkpoint measured_columns contains duplicates")
    unknown = [column for column in measured_columns if column not in feature_columns]
    if unknown:
        raise GRUCheckpointError(
            "GRU checkpoint measured_columns are not a subset of feature_columns: "
            + ", ".join(unknown)
        )

    expected_input_size = len(feature_columns) + len(measured_columns)
    if int(model_config.get("input_size", -1)) != expected_input_size:
        raise GRUCheckpointError(
            "GRU checkpoint input_size does not match feature_columns + "
            f"measured_columns: {model_config.get('input_size')} vs {expected_input_size}"
        )

    medians = _require_finite_vector(checkpoint, "medians", len(feature_columns))
    means = _require_finite_vector(checkpoint, "means", len(feature_columns))
    stds = _require_finite_vector(checkpoint, "stds", len(feature_columns))
    if not (stds > 0).all():
        raise GRUCheckpointError("GRU checkpoint stds must all be strictly positive")

    try:
        clip = float(checkpoint["clip"])
    except (TypeError, ValueError) as exc:
        raise GRUCheckpointError("GRU checkpoint clip is not numeric") from exc
    if not math.isfinite(clip) or clip <= 0:
        raise GRUCheckpointError(f"GRU checkpoint clip must be finite and > 0, got {clip!r}")

    gru_class = _sepsis_gru_class()
    try:
        model = gru_class(**model_config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as exc:
        raise GRUCheckpointError(
            "GRU weights do not match the documented SepsisGRU architecture: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    model.eval()
    model.to("cpu")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    measured_indices = tuple(feature_columns.index(column) for column in measured_columns)
    return GRUArtifact(
        model=model,
        path=path,
        feature_columns=feature_columns,
        measured_columns=measured_columns,
        measured_indices=measured_indices,
        medians=medians,
        means=means,
        stds=stds,
        clip=clip,
        model_config=model_config,
    )


class PhysioNetGRURisk(RiskModel):
    """GRU component of the PhysioNet XGBoost + GRU ensemble.

    ``supports_explanation`` is ``False``: no gradient/attention attribution
    was validated for this checkpoint, so the agent does not expose an
    explanation tool for it.  ``supports_simulation`` is ``False`` as well
    because counterfactual re-scoring has not been validated for the ensemble.
    """

    supports_prediction = True
    supports_explanation = False
    supports_simulation = False

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(threshold=threshold, name="PhysioNetGRURisk")
        artifact = load_gru_artifact(model_path)
        self._artifact = artifact
        self._model = artifact.model
        self._torch = _import_torch()

        self.model_path = artifact.path
        self.feature_columns = artifact.feature_columns
        self.measured_columns = artifact.measured_columns
        self.model_config = dict(artifact.model_config)

        self.dataset = "PhysioNet Challenge 2019"
        self.artifact = str(artifact.path)
        self.explanation_units = None
        # This is a component model; the validated operating point belongs to
        # the ensemble (0.81), not to this standalone score.
        self.prediction_horizon = (
            "Scored ICU hour; the GRU consumes the causal hourly prefix. "
            "Not a multi-hour forecast."
        )

    # -- preprocessing -------------------------------------------------

    def _numeric_frame(self, window_df: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
        frame = _as_frame(window_df)
        missing = [
            column for column in self.feature_columns if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                "window_df is missing PhysioNet GRU feature columns: "
                + ", ".join(missing)
            )

        numeric = pd.DataFrame(index=frame.index)
        for column in self.feature_columns:
            original = frame[column]
            converted = pd.to_numeric(original, errors="coerce")
            if original.notna().sum() != converted.notna().sum():
                raise ValueError(f"feature {column!r} contains non-numeric values")
            numeric[column] = converted

        return numeric.replace([np.inf, -np.inf], np.nan)

    def featurize(self, window_df: pd.DataFrame | Mapping[str, Any]) -> np.ndarray:
        """Return the ``[hours, 74]`` float32 GRU input for this whole prefix."""

        numeric = self._numeric_frame(window_df)
        values = numeric.to_numpy(dtype=np.float32)

        # Observation masks come from the raw values, before any imputation.
        observed = (
            ~np.isnan(values[:, list(self._artifact.measured_indices)])
        ).astype(np.float32)

        # Forward fill is causal: ffill(prefix) == ffill(full)[:len(prefix)].
        filled = pd.DataFrame(values).ffill().to_numpy(dtype=np.float32, copy=True)
        nan_mask = np.isnan(filled)
        if nan_mask.any():
            filled[nan_mask] = np.take(self._artifact.medians, np.where(nan_mask)[1])

        scaled = np.clip(
            (filled - self._artifact.means) / self._artifact.stds,
            -self._artifact.clip,
            self._artifact.clip,
        )
        features = np.concatenate([scaled, observed], axis=1).astype(np.float32)

        if features.shape != (len(values), len(self.feature_columns) + len(self.measured_columns)):
            raise ValueError(f"GRU featurization produced an unexpected shape: {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("GRU featurization produced non-finite values")
        return features

    # -- inference -----------------------------------------------------

    def predict_hourly(self, window_df: pd.DataFrame | Mapping[str, Any]) -> np.ndarray:
        """Score every hour in the prefix with one causal forward pass.

        Entry ``t`` is the score for hour ``t`` given only rows ``0..t``.  This
        exists for verified regression tests and mirrors the notebook; the
        agent itself only uses :meth:`predict_proba`.
        """

        features = self.featurize(window_df)
        tensor = self._torch.from_numpy(np.ascontiguousarray(features)).unsqueeze(0)
        with self._torch.no_grad():
            probabilities = (
                self._torch.sigmoid(self._model(tensor)).squeeze(0).cpu().numpy()
            )

        result = probabilities.astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError("GRU returned a non-finite probability")
        if ((result < 0.0) | (result > 1.0)).any():
            raise ValueError("GRU returned a probability outside [0, 1]")
        return result

    def predict_proba(self, window_df: pd.DataFrame | Mapping[str, Any]) -> float:
        return float(self.predict_hourly(window_df)[-1])

    def explain(
        self,
        window_df: pd.DataFrame | Mapping[str, Any],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "No validated GRU feature-attribution method exists for this "
            "checkpoint, so no explanation is produced."
        )


__all__ = [
    "DEFAULT_GRU_MODEL_PATH",
    "GRUArtifact",
    "GRUCheckpointError",
    "GRUUnavailableError",
    "PhysioNetGRURisk",
    "SepsisGRU",
    "load_gru_artifact",
]
