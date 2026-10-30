from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twinsepsis import patient_source  # noqa: E402
from twinsepsis.agent.risk_model import PHYSIONET_FEATURE_COLUMNS, PhysioNetXGBRisk  # noqa: E402
from twinsepsis.patient_source import (  # noqa: E402
    PatientSourceError,
    load_physionet_patient,
)


def write_patient(
    root: Path,
    patient_id: str = "p123456",
    *,
    rows: int = 3,
    split: str = "training_setA",
    columns: list[str] | None = None,
) -> Path:
    columns = columns or (list(PHYSIONET_FEATURE_COLUMNS) + ["SepsisLabel"])
    data = {column: [float("nan")] * rows for column in columns}
    if "HR" in data:
        data["HR"] = [90.0 + 5 * index for index in range(rows)]
    if "ICULOS" in data:
        data["ICULOS"] = [float(index + 1) for index in range(rows)]
    if "Age" in data:
        data["Age"] = [70.0] * rows
    if "Gender" in data:
        data["Gender"] = [1.0] * rows
    if "SepsisLabel" in data:
        data["SepsisLabel"] = [0] * rows

    path = root / split / f"{patient_id}.psv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, columns=columns).to_csv(
        path,
        sep="|",
        index=False,
        na_rep="NaN",
    )
    return path


class PatientSourceTests(unittest.TestCase):
    def test_loads_rows_and_drops_the_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_patient(root)

            twin = load_physionet_patient("p123456", 2, dataset_root=root)

            self.assertEqual(twin.patient_id, "p123456")
            self.assertEqual(twin.hours, 2)
            self.assertEqual(twin.columns, list(PHYSIONET_FEATURE_COLUMNS))
            self.assertNotIn("SepsisLabel", twin.columns)
            self.assertIsInstance(twin.model, PhysioNetXGBRisk)
            self.assertEqual(float(twin.latest["ICULOS"]), 2.0)
            self.assertGreaterEqual(twin.risk(), 0.0)
            self.assertLessEqual(twin.risk(), 1.0)

    def test_finds_patient_in_training_set_b(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_patient(root, split="training_setB")
            twin = load_physionet_patient("p123456", 1, dataset_root=root)
            self.assertEqual(twin.hours, 1)

    def test_rejects_unsafe_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for patient_id in ("../p123456", "p12345", "p1234567", "p12345a", ""):
                with self.subTest(patient_id=patient_id):
                    with self.assertRaises(PatientSourceError):
                        load_physionet_patient(patient_id, 1, dataset_root=root)

    def test_rejects_invalid_hours(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_patient(root, rows=3)
            for hour in (0, 4, True, "2"):
                with self.subTest(hour=hour):
                    with self.assertRaises(PatientSourceError):
                        load_physionet_patient("p123456", hour, dataset_root=root)  # type: ignore[arg-type]

    def test_missing_patient_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training_setA").mkdir()
            with self.assertRaises(PatientSourceError):
                load_physionet_patient("p999999", 1, dataset_root=root)

    def test_missing_columns_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_patient(root, columns=["HR", "ICULOS", "SepsisLabel"])
            with self.assertRaises(PatientSourceError) as context:
                load_physionet_patient("p123456", 1, dataset_root=root)
            self.assertIn("missing required feature columns", str(context.exception))

    def test_missing_root_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "does-not-exist"
            with self.assertRaises(PatientSourceError):
                load_physionet_patient("p123456", 1, dataset_root=missing)

    def test_project_root_is_the_repository_root(self) -> None:
        # Regression: PROJECT_ROOT must not point above the repository, or
        # .env is read from the wrong directory and PHYSIONET_ROOT is missed.
        self.assertEqual(patient_source.PROJECT_ROOT, ROOT)
        self.assertTrue((patient_source.PROJECT_ROOT / "src").is_dir())

    def test_physionet_root_env_is_used_when_no_dataset_root_given(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_patient(root)
            with mock.patch.dict(os.environ, {"PHYSIONET_ROOT": str(root)}):
                twin = load_physionet_patient("p123456", 1)
            self.assertEqual(twin.hours, 1)


if __name__ == "__main__":
    unittest.main()
