"""
tests/test_ml_trainer.py
-------------------------
Unit tests for ml/trainer.py.

WHY these tests exist:
  The ML trainer persists four scikit-learn models to disk so they can be
  reused across runs without retraining from scratch.  Bugs in the training
  logic (wrong feature columns, wrong label column, data-leakage split) would
  produce silently incorrect predictions.  These tests verify:
    - When there are fewer rows than ML_MIN_TRAIN_ROWS, training is skipped
      and the stub dict with status='insufficient_data' is returned.
    - When enough data is provided, all four models are trained and the
      returned dict contains them under the expected keys.
    - All four .pkl model files are written to the models directory.
    - The metrics dict contains the expected accuracy / MAE keys.

A temporary directory is used for all model file I/O so the real models/
directory is never modified by the test run.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.trainer import load_or_train, FEATURE_COLS
from config import ML_MIN_TRAIN_ROWS


# ---------------------------------------------------------------------------
# Helper — synthetic training CSV
# ---------------------------------------------------------------------------

def _make_training_csv(n: int, tmp_dir: str, seed: int = 55) -> str:
    """Write a synthetic features.csv to *tmp_dir* and return its path.

    WHY synthetic data:
      We need a file that exactly mirrors what csv_exporter produces:
        - One row per trading day.
        - All FEATURE_COLS columns (used as model inputs).
        - direction_5d, direction_21d (0/1 classification labels).
        - return_5d, return_21d (float regression labels).
        - The last 21 rows have NaN labels (simulating 'today' — future unknown).

    The values are seeded random numbers which have no real financial meaning;
    they just need to be numeric so sklearn can fit the models.
    """
    rng  = np.random.default_rng(seed)
    rows = n

    data: dict = {}

    # Generate a value for every feature column.
    for col in FEATURE_COLS:
        if col == "RSI":
            # RSI is bounded [0, 100] so we generate accordingly.
            data[col] = rng.uniform(20, 80, size=rows)
        elif col == "BB_PctB":
            data[col] = rng.uniform(0, 1, size=rows)
        elif col == "ADX":
            data[col] = rng.uniform(10, 60, size=rows)
        elif col in ("Vol_5d", "Vol_21d"):
            data[col] = rng.uniform(0.005, 0.04, size=rows)
        else:
            data[col] = rng.normal(0, 1, size=rows)

    # Add supervised labels: last 21 rows get NaN (future unknown — inference rows).
    direction_5d  = rng.integers(0, 2, size=rows).astype(float)
    direction_21d = rng.integers(0, 2, size=rows).astype(float)
    return_5d     = rng.normal(0.002, 0.01, size=rows)
    return_21d    = rng.normal(0.004, 0.02, size=rows)

    # Simulate the real exporter behaviour: last horizon rows have no label.
    direction_5d[-5:]   = np.nan
    direction_21d[-21:] = np.nan
    return_5d[-5:]      = np.nan
    return_21d[-21:]    = np.nan

    data["direction_5d"]  = direction_5d
    data["direction_21d"] = direction_21d
    data["return_5d"]     = return_5d
    data["return_21d"]    = return_21d

    df   = pd.DataFrame(data)
    path = os.path.join(tmp_dir, "features.csv")
    df.to_csv(path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadOrTrain(unittest.TestCase):
    """Tests for load_or_train() — the main public entry point in trainer.py."""

    def test_insufficient_data_returns_stub(self):
        # When the CSV has fewer rows than ML_MIN_TRAIN_ROWS, the trainer
        # must skip training and return a stub dict — not raise an exception.
        with tempfile.TemporaryDirectory() as tmp:
            # Write a CSV with only 10 rows (well below the minimum).
            csv_path = _make_training_csv(n=10, tmp_dir=tmp)
            result   = load_or_train("TEST", csv_path, force_retrain=True)
            self.assertEqual(
                result["status"], "insufficient_data",
                msg=f"Expected 'insufficient_data' status, got '{result['status']}'",
            )

    def test_trains_four_models(self):
        # With enough data, all four models (clf_5d, clf_21d, reg_5d, reg_21d)
        # must be present in the returned dict.
        n = ML_MIN_TRAIN_ROWS + 50   # comfortably above the minimum
        with tempfile.TemporaryDirectory() as tmp:
            # Patch ML_MODELS_DIR to point at the temp dir so no real model files
            # are written to the project's models/ directory.
            import ml.trainer as trainer_module
            original_dir = trainer_module.ML_MODELS_DIR

            try:
                trainer_module.ML_MODELS_DIR = tmp
                csv_path = _make_training_csv(n=n, tmp_dir=tmp)
                result   = load_or_train("TEST", csv_path, force_retrain=True)

                self.assertIn(result["status"], ("trained", "loaded"))
                for key in ("clf_5d", "clf_21d", "reg_5d", "reg_21d"):
                    self.assertIn(key, result.get("models", {}),
                                  msg=f"Model key '{key}' missing after training")
            finally:
                # Always restore the original directory to avoid test pollution.
                trainer_module.ML_MODELS_DIR = original_dir

    def test_model_pkl_files_written(self):
        # The four .pkl files must exist on disk after a successful training run.
        # Verifying file existence ensures models can be loaded on the next run.
        n = ML_MIN_TRAIN_ROWS + 50
        with tempfile.TemporaryDirectory() as tmp:
            import ml.trainer as trainer_module
            original_dir = trainer_module.ML_MODELS_DIR
            try:
                trainer_module.ML_MODELS_DIR = tmp
                csv_path = _make_training_csv(n=n, tmp_dir=tmp)
                load_or_train("TEST", csv_path, force_retrain=True)

                model_dir = os.path.join(tmp, "TEST")
                # _save_models() writes files named after the dict keys:
                # clf_5d.pkl, clf_21d.pkl, reg_5d.pkl, reg_21d.pkl.
                for fname in ("clf_5d.pkl", "clf_21d.pkl",
                               "reg_5d.pkl",  "reg_21d.pkl"):
                    self.assertTrue(
                        os.path.isfile(os.path.join(model_dir, fname)),
                        msg=f"Expected model file '{fname}' not found in {model_dir}",
                    )
            finally:
                trainer_module.ML_MODELS_DIR = original_dir

    def test_metrics_keys_present(self):
        # After training, the metrics dict must contain accuracy and MAE entries
        # for both horizons so the terminal display can render them.
        n = ML_MIN_TRAIN_ROWS + 50
        with tempfile.TemporaryDirectory() as tmp:
            import ml.trainer as trainer_module
            original_dir = trainer_module.ML_MODELS_DIR
            try:
                trainer_module.ML_MODELS_DIR = tmp
                csv_path = _make_training_csv(n=n, tmp_dir=tmp)
                result   = load_or_train("TEST", csv_path, force_retrain=True)
                metrics  = result.get("metrics", {})

                for key in ("clf_5d_accuracy", "clf_21d_accuracy",
                            "reg_5d_mae", "reg_21d_mae"):
                    self.assertIn(key, metrics,
                                  msg=f"Metrics key '{key}' missing")
            finally:
                trainer_module.ML_MODELS_DIR = original_dir


if __name__ == "__main__":
    unittest.main()
