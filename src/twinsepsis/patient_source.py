"""Load a real PhysioNet 2019 patient into a :class:`PatientTwin`.

Only the 40 model features are copied into the twin.  ``SepsisLabel`` (and any
other extra column) is deliberately dropped so the outcome never reaches the
model, the tools, the LLM or a trace.

This module reads local files only.  It does not download data.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from .agent.risk_model import (
    PHYSIONET_FEATURE_COLUMNS,
    PhysioNetXGBRisk,
    RiskModel,
)
from .agent.twin import PatientTwin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATIENT_ID_PATTERN = re.compile(r"^p\d{6}$")
SPLITS = ("training_setA", "training_setB")
PATIENT_FILE_SUFFIX = ".psv"


class PatientSourceError(Exception):
    """Raised when a local PhysioNet patient cannot be loaded safely."""


def _resolve_root(dataset_root: str | Path | None) -> Path:
    if dataset_root is not None:
        root = Path(dataset_root)
    else:
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env", override=False)
        except ImportError:  # pragma: no cover - python-dotenv is pinned
            pass
        raw = os.getenv("PHYSIONET_ROOT", "").strip()
        if not raw:
            raise PatientSourceError(
                "PHYSIONET_ROOT is not set and --dataset-root was not supplied"
            )
        root = Path(raw)

    if not root.exists():
        raise PatientSourceError(f"PhysioNet dataset root does not exist: {root}")
    if not root.is_dir():
        raise PatientSourceError(f"PhysioNet dataset root is not a directory: {root}")
    return root


def resolve_dataset_root(dataset_root: str | Path | None = None) -> Path:
    """Return the validated PhysioNet dataset root.

    Public wrapper over the internal resolution order (explicit argument, then
    ``PHYSIONET_ROOT`` from the environment/``.env``).  Read-only callers such
    as the dashboard need the resolved directory for display and directory
    listing without re-implementing the lookup.
    """

    return _resolve_root(dataset_root)


def find_patient_file(
    patient_id: str,
    dataset_root: str | Path | None = None,
) -> Path:
    """Find one ``.psv`` file, refusing unsafe identifiers."""

    if not isinstance(patient_id, str) or not PATIENT_ID_PATTERN.fullmatch(patient_id):
        raise PatientSourceError(
            "patient_id must look like a PhysioNet identifier (for example p000001)"
        )

    root = _resolve_root(dataset_root)
    candidates = [root / f"{patient_id}{PATIENT_FILE_SUFFIX}"]
    candidates.extend(
        root / split / f"{patient_id}{PATIENT_FILE_SUFFIX}" for split in SPLITS
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise PatientSourceError(
        f"Patient {patient_id!r} was not found. Searched: {searched}"
    )


def load_physionet_patient(
    patient_id: str,
    hour: int,
    dataset_root: str | Path | None = None,
    model: RiskModel | None = None,
) -> PatientTwin:
    """Load history rows ``1..hour`` for one PhysioNet patient.

    Parameters
    ----------
    patient_id:
        PhysioNet identifier such as ``p000001``.  Path separators and other
        unsafe identifiers are rejected.
    hour:
        1-based ICU hour.  The twin contains exactly this many rows, so the
        latest row is the scored hour and no future data is present.
    dataset_root:
        Directory containing ``training_setA``/``training_setB``.  Defaults to
        ``PHYSIONET_ROOT`` from the environment (``.env`` is loaded if present).
    model:
        Risk model to attach to the twin.  Defaults to
        :class:`PhysioNetXGBRisk`, preserving the original XGBoost-only
        behaviour.  Pass an :class:`EnsembleRisk` (or any other
        :class:`RiskModel`) to score with a different backend; the model is
        used as given, never replaced on failure.
    """

    if isinstance(hour, bool) or not isinstance(hour, int):
        raise PatientSourceError("hour must be an integer")
    if hour < 1:
        raise PatientSourceError("hour must be >= 1")

    path = find_patient_file(patient_id, dataset_root)

    try:
        frame = pd.read_csv(path, sep="|")
    except Exception as exc:
        raise PatientSourceError(f"Could not read {path}: {exc}") from exc

    if frame.empty:
        raise PatientSourceError(f"Patient file is empty: {path}")

    missing = [
        column for column in PHYSIONET_FEATURE_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise PatientSourceError(
            f"Patient file {path} is missing required feature columns: "
            + ", ".join(missing)
        )

    if hour > len(frame):
        raise PatientSourceError(
            f"hour {hour} is out of range for {patient_id!r}; "
            f"the file has {len(frame)} rows"
        )

    # Deliberately select only model features: SepsisLabel never enters the
    # twin, tool output, LLM context or trace.
    history = frame.loc[: hour - 1, list(PHYSIONET_FEATURE_COLUMNS)].copy()
    history = history.reset_index(drop=True)

    model = model if model is not None else PhysioNetXGBRisk()
    return PatientTwin(patient_id, model, history)


__all__ = [
    "PatientSourceError",
    "PATIENT_ID_PATTERN",
    "find_patient_file",
    "load_physionet_patient",
    "resolve_dataset_root",
]
