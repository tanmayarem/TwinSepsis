"""A single patient's hourly twin.

The twin is deliberately dumb: it stores rows in order and always asks the
risk model to score a *prefix* of that history.  It never backfills, smooths,
or looks ahead.  This is what makes ``trajectory()`` honest: the risk shown at
hour *t* is exactly the risk that would have been computed at hour *t*, even
after future rows are appended.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .risk_model import RiskModel


class PatientTwin:
    """One patient's hourly history plus a pluggable risk model."""

    def __init__(
        self,
        patient_id: str,
        model: RiskModel,
        history: pd.DataFrame | None = None,
    ) -> None:
        if model is None:
            raise ValueError("model is required")
        self.patient_id = str(patient_id)
        self.model = model
        self._columns: list[str] = []
        self._history = pd.DataFrame()

        if history is not None:
            if not isinstance(history, pd.DataFrame):
                history = pd.DataFrame(history)
            if not history.empty:
                self._columns = [str(column) for column in history.columns]
                frame = history.copy()
                frame.columns = self._columns
                self._history = frame.reset_index(drop=True)

    # -- state ---------------------------------------------------------

    @property
    def history(self) -> pd.DataFrame:
        """A copy, so callers cannot mutate the twin's stored history."""

        return self._history.copy()

    @property
    def latest(self) -> pd.Series:
        if self._history.empty:
            raise ValueError("twin has no history")
        return self._history.iloc[-1].copy()

    @property
    def hours(self) -> int:
        return int(len(self._history))

    @property
    def latest_hour(self) -> int:
        if self._history.empty:
            raise ValueError("twin has no history")
        return int(len(self._history) - 1)

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    def __len__(self) -> int:
        return self.hours

    # -- updates -------------------------------------------------------

    def update(self, new_row: pd.Series | Mapping[str, Any] | pd.DataFrame) -> "PatientTwin":
        """Append one hourly row.  Existing rows are never modified.

        The first row fixes the column schema.  Later rows may omit columns
        (they become NaN), but may not introduce new ones -- silently adding a
        column would change the meaning of earlier feature windows.
        """

        row = self._coerce_row(new_row)

        if not self._columns:
            self._columns = [str(column) for column in row.index]
        else:
            extra = [str(column) for column in row.index if str(column) not in self._columns]
            if extra:
                raise ValueError(
                    "new_row introduces columns not present in the twin schema: "
                    + ", ".join(extra)
                )
            row.index = [str(column) for column in row.index]
            row = row.reindex(self._columns)

        frame_row = row.to_frame().T
        frame_row.columns = self._columns
        self._history = pd.concat([self._history, frame_row], ignore_index=True)
        return self

    @staticmethod
    def _coerce_row(new_row: pd.Series | Mapping[str, Any] | pd.DataFrame) -> pd.Series:
        if isinstance(new_row, pd.DataFrame):
            if len(new_row) != 1:
                raise ValueError("update() accepts exactly one row")
            return new_row.iloc[0].copy()
        if isinstance(new_row, pd.Series):
            return new_row.copy()
        if isinstance(new_row, Mapping):
            return pd.Series(dict(new_row))
        raise TypeError("new_row must be a mapping, pandas Series, or one-row DataFrame")

    # -- scoring -------------------------------------------------------

    def risk(self) -> float:
        """Risk at the latest hour, using only history up to that hour."""

        if self._history.empty:
            raise ValueError("twin has no history")
        value = float(self.model.predict_proba(self._history.copy()))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"model returned probability outside [0, 1]: {value}")
        return value

    def trajectory(self) -> list[float]:
        """Risk at every hour, each computed from that hour's prefix only.

        Complexity is O(n^2) in the number of hours because each prefix is
        passed separately.  That is intentional: it makes leakage impossible
        to hide behind a single batch call.
        """

        risks: list[float] = []
        for end in range(1, len(self._history) + 1):
            window = self._history.iloc[:end].copy()
            value = float(self.model.predict_proba(window))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"model returned probability outside [0, 1]: {value}")
            risks.append(value)
        return risks

    def trend(self, hours: int = 3) -> dict[str, Any]:
        """Change in risk over the last ``hours`` hours."""

        if hours < 1:
            raise ValueError("hours must be >= 1")
        if self._history.empty:
            raise ValueError("twin has no history")

        trajectory = self.trajectory()
        window = min(int(hours), len(trajectory) - 1)
        current = trajectory[-1]

        if window == 0:
            return {
                "direction": "insufficient_history",
                "delta": None,
                "slope_per_hour": None,
                "window_hours": 0,
                "previous_risk": None,
                "current_risk": current,
            }

        previous = trajectory[-1 - window]
        delta = current - previous
        if abs(delta) < 1e-9:
            direction = "stable"
        elif delta > 0:
            direction = "rising"
        else:
            direction = "falling"

        return {
            "direction": direction,
            "delta": float(delta),
            "slope_per_hour": float(delta / window),
            "window_hours": window,
            "previous_risk": float(previous),
            "current_risk": float(current),
        }


__all__ = ["PatientTwin"]
