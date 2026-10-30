from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinsepsis.agent.risk_model import PhysioNetXGBRisk, RiskModel  # noqa: E402
from twinsepsis.agent.twin import PatientTwin  # noqa: E402
from twinsepsis.agent.gru_model import PhysioNetGRURisk  # noqa: E402
from gru_test_support import HAS_GRU_INTEGRATION, synthetic_history  # noqa: E402


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


class CountingPrefixRisk(WindowMeanRisk):
    """WindowMeanRisk that records the length of every prefix it scores."""

    def __init__(self) -> None:
        super().__init__()
        self.scored_prefix_lengths: list[int] = []

    def predict_proba(self, window_df: pd.DataFrame) -> float:
        self.scored_prefix_lengths.append(len(window_df))
        return super().predict_proba(window_df)


def assert_trend_matches_trajectory(
    test: unittest.TestCase, trend: dict, trajectory: list[float], hours: int
) -> None:
    """The bounded trend must equal the tail of the full trajectory exactly."""

    window = min(hours, len(trajectory) - 1)
    test.assertEqual(trend["window_hours"], window)
    test.assertEqual(trend["current_risk"], trajectory[-1])
    test.assertEqual(trend["previous_risk"], trajectory[-1 - window])
    delta = trajectory[-1] - trajectory[-1 - window]
    test.assertEqual(trend["delta"], delta)
    test.assertEqual(trend["slope_per_hour"], delta / window)
    if abs(delta) < 1e-9:
        expected = "stable"
    elif delta > 0:
        expected = "rising"
    else:
        expected = "falling"
    test.assertEqual(trend["direction"], expected)


class TrendBoundedScoringTests(unittest.TestCase):
    def test_trend_matches_the_full_trajectory_for_every_window(self) -> None:
        twin = PatientTwin(
            "synthetic-trend-1", WindowMeanRisk(), history([60, 64, 70, 76, 82, 90])
        )
        full = twin.trajectory()

        for hours in (1, 2, 3, 5, 6, 24):
            with self.subTest(hours=hours):
                assert_trend_matches_trajectory(self, twin.trend(hours), full, hours)

    def test_trend_scores_only_the_recent_prefixes(self) -> None:
        model = CountingPrefixRisk()
        twin = PatientTwin(
            "synthetic-trend-bounded", model, history(list(range(60, 80)))
        )
        model.scored_prefix_lengths.clear()

        twin.trend(3)

        # 20 rows, hours=3 -> prefixes 17, 18, 19, 20 only.
        self.assertEqual(model.scored_prefix_lengths, [17, 18, 19, 20])

    def test_short_history_is_clamped_and_reported(self) -> None:
        twin = PatientTwin("synthetic-trend-2", WindowMeanRisk(), history([70, 95]))
        full = twin.trajectory()

        trend = twin.trend(5)

        assert_trend_matches_trajectory(self, trend, full, 5)
        self.assertEqual(trend["window_hours"], 1)

    def test_single_row_reports_insufficient_history(self) -> None:
        twin = PatientTwin("synthetic-trend-3", WindowMeanRisk(), history([100]))

        trend = twin.trend(3)

        self.assertEqual(trend["direction"], "insufficient_history")
        self.assertEqual(trend["window_hours"], 0)
        self.assertIsNone(trend["delta"])
        self.assertIsNone(trend["slope_per_hour"])
        self.assertIsNone(trend["previous_risk"])
        self.assertEqual(trend["current_risk"], twin.risk())

    def test_invalid_window_and_empty_history_are_rejected(self) -> None:
        twin = PatientTwin("synthetic-trend-4", WindowMeanRisk(), history([70, 80]))
        with self.assertRaises(ValueError):
            twin.trend(0)
        with self.assertRaises(ValueError):
            PatientTwin("synthetic-trend-5", WindowMeanRisk()).trend(3)


class PhysioNetXGBTrendTests(unittest.TestCase):
    """Regression coverage for the real default XGBoost backend."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = PhysioNetXGBRisk()

    def test_trend_matches_full_trajectory_slice(self) -> None:
        twin = PatientTwin(
            "synthetic-xgb-trend", self.model, synthetic_history(12)
        )
        full = twin.trajectory()

        for hours in (1, 3, 11, 12, 24):
            with self.subTest(hours=hours):
                assert_trend_matches_trajectory(self, twin.trend(hours), full, hours)

    def test_single_hour_history(self) -> None:
        twin = PatientTwin(
            "synthetic-xgb-trend-short", self.model, synthetic_history(1)
        )

        trend = twin.trend(3)

        self.assertEqual(trend["direction"], "insufficient_history")
        self.assertEqual(trend["current_risk"], twin.risk())


@unittest.skipUnless(HAS_GRU_INTEGRATION, "torch and the GRU checkpoint are required")
class GRUNoLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = PhysioNetGRURisk()

    def test_appending_future_rows_does_not_change_past_risks(self) -> None:
        frame = synthetic_history(6)
        twin = PatientTwin("synthetic-gru-1", self.model, frame)

        before = twin.trajectory()
        self.assertEqual(len(before), 6)

        twin.update({"HR": 240.0, "Temp": 41.0, "Lactate": 12.0, "ICULOS": 7})
        twin.update({"HR": 250.0, "Temp": 42.0, "Lactate": 15.0, "ICULOS": 8})
        after = twin.trajectory()

        self.assertEqual(len(after), 8)
        self.assertEqual(before, after[:6])

    def test_prefix_scores_match_the_causal_full_pass(self) -> None:
        frame = synthetic_history(10)
        full = self.model.predict_hourly(frame)

        for end in range(1, len(frame) + 1):
            prefix_score = self.model.predict_proba(frame.iloc[:end])
            self.assertAlmostEqual(prefix_score, float(full[end - 1]), places=9)


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
