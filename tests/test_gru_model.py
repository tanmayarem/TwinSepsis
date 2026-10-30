from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gru_test_support import (  # noqa: E402
    COMMITTED_PATIENTS,
    HAS_CHECKPOINT,
    HAS_DATASET,
    HAS_GRU_INTEGRATION,
    HAS_TORCH,
    load_committed_predictions,
    load_full_patient_frame,
    synthetic_history,
)
from twinsepsis.agent import gru_model  # noqa: E402
from twinsepsis.agent.gru_model import (  # noqa: E402
    GRUCheckpointError,
    GRUUnavailableError,
    PhysioNetGRURisk,
    load_gru_artifact,
)
from twinsepsis.agent.risk_model import (  # noqa: E402
    PHYSIONET_FEATURE_COLUMNS,
    EnsembleRisk,
    ModelUnavailableError,
)


class GRUAvailabilityTests(unittest.TestCase):
    def test_module_import_does_not_import_torch_eagerly(self) -> None:
        # ``_import_torch`` keeps torch in a local variable, so even after the
        # GRU has been used the module itself has no torch global.
        self.assertNotIn("torch", vars(gru_model))

    @unittest.skipIf(HAS_TORCH, "torch is installed in this interpreter")
    def test_missing_torch_is_reported_clearly(self) -> None:
        with self.assertRaises(GRUUnavailableError) as caught:
            PhysioNetGRURisk()
        message = str(caught.exception).lower()
        self.assertIn("torch", message)
        self.assertIn("3.12", message)

    @unittest.skipIf(HAS_TORCH, "torch is installed in this interpreter")
    def test_ensemble_fails_loudly_without_torch(self) -> None:
        with self.assertRaises(ModelUnavailableError) as caught:
            EnsembleRisk()
        message = str(caught.exception)
        self.assertIn("GRU", message)
        self.assertIn("not fall back", message)


@unittest.skipUnless(HAS_GRU_INTEGRATION, "torch and the GRU checkpoint are required")
class GRUArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = load_gru_artifact()
        cls.model = PhysioNetGRURisk()

    def test_checkpoint_contract(self) -> None:
        artifact = self.artifact
        self.assertEqual(
            artifact.model_config,
            {
                "input_size": 74,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
            },
        )
        self.assertEqual(artifact.feature_columns, tuple(PHYSIONET_FEATURE_COLUMNS))
        self.assertEqual(len(artifact.feature_columns), 40)
        self.assertEqual(len(artifact.measured_columns), 34)
        self.assertEqual(artifact.medians.shape, (40,))
        self.assertEqual(artifact.means.shape, (40,))
        self.assertEqual(artifact.stds.shape, (40,))
        self.assertTrue((artifact.stds > 0).all())
        self.assertAlmostEqual(artifact.clip, 10.0)
        self.assertEqual(type(artifact.model).__name__, "SepsisGRU")
        self.assertFalse(artifact.model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in artifact.model.parameters()))
        self.assertEqual(
            sum(parameter.numel() for parameter in artifact.model.parameters()), 56_065
        )

    def test_missing_checkpoint_is_reported(self) -> None:
        with self.assertRaises(GRUUnavailableError):
            load_gru_artifact(Path("does-not-exist.pt"))

    def test_corrupt_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.pt"
            path.write_bytes(b"not a torch checkpoint")
            with self.assertRaises(GRUCheckpointError):
                load_gru_artifact(path)

    def test_featurize_shape_masks_and_clip(self) -> None:
        frame = synthetic_history(6)
        features = self.model.featurize(frame)

        self.assertEqual(features.shape, (6, 74))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue((np.abs(features[:, :40]) <= 10.0 + 1e-6).all())
        self.assertTrue((features[:, 40:] >= 0.0).all())
        self.assertTrue((features[:, 40:] <= 1.0).all())

        measured_indices = [
            PHYSIONET_FEATURE_COLUMNS.index(column)
            for column in self.model.measured_columns
        ]
        raw = frame.to_numpy(dtype=np.float32)
        # The pipeline replaces +/-inf with NaN before recording observations,
        # exactly like the notebook.
        raw = np.where(np.isfinite(raw), raw, np.nan)
        expected_masks = (~np.isnan(raw[:, measured_indices])).astype(np.float32)
        np.testing.assert_array_equal(features[:, 40:], expected_masks)

    def test_missing_and_non_numeric_columns_fail_loudly(self) -> None:
        frame = synthetic_history(3)
        with self.assertRaises(ValueError):
            self.model.predict_proba(frame.drop(columns=["HR"]))

        broken = frame.copy()
        broken["HR"] = broken["HR"].astype(object)
        broken.loc[0, "HR"] = "not-a-number"
        with self.assertRaises(ValueError):
            self.model.predict_proba(broken)

    def test_scores_are_deterministic_and_bounded(self) -> None:
        frame = synthetic_history(6)
        first = self.model.predict_hourly(frame)
        second = self.model.predict_hourly(frame)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (6,))
        self.assertTrue(((first > 0.0) & (first < 1.0)).all())
        self.assertAlmostEqual(self.model.predict_proba(frame), float(first[-1]))

    def test_prefix_scores_are_causal(self) -> None:
        frame = synthetic_history(8)
        full = self.model.predict_hourly(frame)
        for end in (1, 3, 5, 8):
            with self.subTest(end=end):
                prefix = self.model.predict_hourly(frame.iloc[:end])
                np.testing.assert_allclose(prefix, full[:end], atol=1e-6, rtol=0)

    def test_single_row_mapping_is_accepted(self) -> None:
        row = synthetic_history(1).iloc[0].to_dict()
        probability = self.model.predict_proba(row)
        self.assertTrue(math.isfinite(probability))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_explanation_and_simulation_are_unavailable(self) -> None:
        self.assertFalse(self.model.supports_explanation)
        self.assertFalse(self.model.supports_simulation)
        with self.assertRaises(NotImplementedError):
            self.model.explain(synthetic_history(2))

    def test_threshold_validation(self) -> None:
        with self.assertRaises(ValueError):
            PhysioNetGRURisk(threshold=0.0)


@unittest.skipUnless(HAS_DATASET, "PhysioNet data and committed predictions are required")
class GRURegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not HAS_GRU_INTEGRATION:
            raise unittest.SkipTest("torch and the GRU checkpoint are required")
        cls.model = PhysioNetGRURisk()
        cls.committed = load_committed_predictions()

    def test_reproduces_committed_gru_probabilities(self) -> None:
        for patient_id in COMMITTED_PATIENTS:
            with self.subTest(patient=patient_id):
                frame = load_full_patient_frame(patient_id)
                rows = (
                    self.committed[self.committed["patient_id"] == patient_id]
                    .sort_values("hour_index")
                )
                self.assertEqual(len(rows), len(frame))

                expected = rows["gru_probability"].to_numpy(dtype=np.float64)
                actual = self.model.predict_hourly(frame)

                self.assertEqual(len(actual), len(expected))
                worst = float(np.abs(actual - expected).max())
                self.assertLess(
                    worst,
                    1e-4,
                    f"{patient_id}: GRU probabilities diverge from committed values "
                    f"by {worst:.3e}",
                )

    def test_checkpoint_is_the_repository_artifact(self) -> None:
        self.assertTrue(HAS_CHECKPOINT)
        self.assertTrue(Path(self.model.artifact).is_file())


if __name__ == "__main__":
    unittest.main()
