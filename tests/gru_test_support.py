"""Shared helpers for the GRU and ensemble tests.

Importing this module never imports torch.  Tests use the ``HAS_*`` flags to
skip cleanly in the project's Python 3.14 environment, where the GRU cannot
run, and to skip dataset-dependent regressions when the local PhysioNet data is
absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRU_MODEL_PATH = ROOT / "models" / "gru_baseline.pt"
COMMITTED_PREDICTIONS = ROOT / "results" / "ensemble_test_predictions.csv"

# Same five patients used during GRU verification: shortest, longest, a
# positive-heavy case and a no-positive case.
COMMITTED_PATIENTS = ("p014914", "p000865", "p018469", "p000185", "p000575")

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_CHECKPOINT = GRU_MODEL_PATH.is_file()


def _dataset_root() -> Path | None:
    try:
        from twinsepsis.patient_source import _resolve_root

        return Path(_resolve_root(None))
    except Exception:
        candidate = (
            ROOT
            / "dataset"
            / "physionet.org"
            / "files"
            / "challenge-2019"
            / "1.0.0"
            / "training"
        )
        return candidate if candidate.is_dir() else None


DATASET_ROOT = _dataset_root()
HAS_DATASET = (
    DATASET_ROOT is not None
    and (DATASET_ROOT / "training_setA").is_dir()
    and COMMITTED_PREDICTIONS.is_file()
)

HAS_GRU_INTEGRATION = HAS_TORCH and HAS_CHECKPOINT
HAS_FULL_INTEGRATION = HAS_GRU_INTEGRATION and HAS_DATASET


def synthetic_history(hours: int = 6) -> pd.DataFrame:
    """A deterministic 40-column PhysioNet-shaped frame with missing values."""

    from twinsepsis.agent.risk_model import PHYSIONET_FEATURE_COLUMNS

    rng = np.random.default_rng(7)
    data = {
        column: rng.normal(80.0, 5.0, size=hours) for column in PHYSIONET_FEATURE_COLUMNS
    }
    # Overwriting keeps the canonical column order from the list above.
    data["Age"] = np.full(hours, 65.0)
    data["Gender"] = np.full(hours, 1.0)
    data["Unit1"] = np.full(hours, 1.0)
    data["Unit2"] = np.zeros(hours)
    data["HospAdmTime"] = np.full(hours, -3.0)
    data["ICULOS"] = np.arange(1, hours + 1, dtype=float)

    frame = pd.DataFrame(data)
    frame.loc[0, ["HR", "Lactate", "Creatinine"]] = np.nan
    if hours > 2:
        frame.loc[2, "Temp"] = np.inf
    return frame


def load_full_patient_frame(patient_id: str) -> pd.DataFrame:
    """Read a whole patient file and keep only the 40 model features."""

    from twinsepsis.agent.risk_model import PHYSIONET_FEATURE_COLUMNS
    from twinsepsis.patient_source import find_patient_file

    path = find_patient_file(patient_id, DATASET_ROOT)
    frame = pd.read_csv(path, sep="|")
    return frame[list(PHYSIONET_FEATURE_COLUMNS)].copy()


def load_committed_predictions() -> pd.DataFrame:
    return pd.read_csv(
        COMMITTED_PREDICTIONS,
        usecols=[
            "patient_id",
            "hour_index",
            "SepsisLabel",
            "xgb_probability",
            "gru_probability",
            "ensemble_probability",
        ],
    )
