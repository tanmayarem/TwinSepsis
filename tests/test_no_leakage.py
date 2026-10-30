from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinsepsis.agent.risk_model import RiskModel  # noqa: E402
from twinsepsis.agent.twin import PatientTwin  # noqa: E402


class WindowMeanRisk(RiskModel):
    """Deliberately non-causal if given a whole future-containing window."""

    def __init__(self) -> None:
        super().__init__(threshold=0.5, name="WindowMeanRisk")

    def featurize(self, window_df: pd.DataFrame) -> pd.DataFrame:
        return window_df.copy()

    def predict_proba(self, window_df: pd.DataFrame) -> float:
        values = pd.to_numeric(window_df["HR"], errors="coerce").dropna()
        if values.empty:
            return 0.5
        mean = float(values.mean())
        return 1.0 / (1.0 + math.exp(-(mean - 80.0) / 10.0))

    def explain(self, window_df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        return []


def history(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "HR": values,
            "Temp": [37.0] * len(values),
            "ICULOS": list(range(1, len(values) + 1)),
        }
    )


class NoLeakageTests(unittest.TestCase):
    def test_appending_future_rows_does_not_change_past_risks(self) -> None:
        model = WindowMeanRisk()
        twin = PatientTwin("synthetic-1", model, history([70, 72, 74, 76, 78, 80]))

        before = twin.trajectory()
        self.assertEqual(len(before), 6)

        twin.update({"HR": 140, "Temp": 39.0, "ICULOS": 7})
        twin.update({"HR": 160, "Temp": 40.0, "ICULOS": 8})
        after = twin.trajectory()

        self.assertEqual(len(after), 8)
        self.assertEqual(before, after[:6])

    def test_each_trajectory_point_uses_only_its_prefix(self) -> None:
        model = WindowMeanRisk()
        twin = PatientTwin("synthetic-2", model, history([60, 90, 120, 70]))

        for index, risk in enumerate(twin.trajectory()):
            prefix = twin.history.iloc[: index + 1].copy()
            self.assertEqual(risk, model.predict_proba(prefix))

        # The last row lowers the mean; if a future row leaked backwards the
        # earlier point would change.  The assertEqual loop above is the real
        # check, but this guards the fixture itself.
        self.assertLess(twin.trajectory()[-1], twin.trajectory()[-2])

    def test_history_property_is_a_copy(self) -> None:
        twin = PatientTwin("synthetic-3", WindowMeanRisk(), history([70, 80]))
        snapshot = twin.history
        snapshot.loc[0, "HR"] = 999
        self.assertEqual(twin.history.loc[0, "HR"], 70)


if __name__ == "__main__":
    unittest.main()
