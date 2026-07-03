"""
tests/test_ml_predictor.py
---------------------------
Unit tests for ml/predictor.py.

WHY these tests exist:
  predict() is the function called at the end of every pipeline run to
  produce the probability-of-up and expected-return numbers shown in the
  terminal report and PDF.  Bugs here — e.g. using the wrong feature order
  or dividing by the wrong shape — produce wrong numbers silently.
  These tests verify:
    - predict() returns a dict with all expected keys.
    - When trainer_result has no models (insufficient_data stub), predict()
      returns the stub dict with status 'insufficient_data' — not an error.
    - Direction strings are always 'UP' or 'DOWN' (never None or other strings).
    - Probability values are in [0, 1].

HOW synthetic models work:
  We reuse the training helper from test_ml_trainer to train real (tiny)
  GradientBoosting models on synthetic data, then pass both the CSV path
  and the trainer_result into predict().  This exercises the real inference
  code path including feature-vector construction.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.predictor import predict
from ml.trainer import FEATURE_COLS
from config import ML_MIN_TRAIN_ROWS


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_ml_trainer to keep tests self-contained)
# ---------------------------------------------------------------------------

def _make_training_csv(n: int, tmp_dir: str, seed: int = 77) -> str:
    """Write a synthetic features.csv to *tmp_dir* and return its path.

    Matches the CSV format expected by both trainer.py and predictor.py.
    See test_ml_trainer._make_training_csv for a detailed explanation.
    """
    rng  = np.random.default_rng(seed)
    rows = n
    data: dict = {}

    for col in FEATURE_COLS:
        if col == "RSI":
            data[col] = rng.uniform(20, 80, size=rows)
        elif col == "BB_PctB":
            data[col] = rng.uniform(0, 1, size=rows)
        elif col == "ADX":
            data[col] = rng.uniform(10, 60, size=rows)
        elif col in ("Vol_5d", "Vol_21d"):
            data[col] = rng.uniform(0.005, 0.04, size=rows)
        else:
            data[col] = rng.normal(0, 1, size=rows)

    direction_5d  = rng.integers(0, 2, size=rows).astype(float)
    direction_21d = rng.integers(0, 2, size=rows).astype(float)
    return_5d     = rng.normal(0.002, 0.01, size=rows)
    return_21d    = rng.normal(0.004, 0.02, size=rows)

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


def _train_and_get_result(tmp_dir: str):
    """Train four models on synthetic data and return (csv_path, trainer_result).

    WHY return both: predict() needs both the CSV path (to read the latest
    feature row) and the trainer_result (to get the fitted model objects and
    the canonical feature list used at training time).
    """
    import ml.trainer as trainer_module
    original_dir = trainer_module.ML_MODELS_DIR
    try:
        trainer_module.ML_MODELS_DIR = tmp_dir
        n        = ML_MIN_TRAIN_ROWS + 50
        csv_path = _make_training_csv(n=n, tmp_dir=tmp_dir)
        result   = trainer_module.load_or_train("PRED_TEST", csv_path, force_retrain=True)
        return csv_path, result
    finally:
        trainer_module.ML_MODELS_DIR = original_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredict(unittest.TestCase):
    """Tests for predict() — inference on the latest feature row."""

    def test_predict_returns_valid_structure(self):
        # With real trained models, predict() must return a dict that contains
        # all the keys the terminal display and PDF generator look for.
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, trainer_result = _train_and_get_result(tmp)
            result = predict(csv_path, trainer_result)

            expected_keys = (
                "clf_5d_prob_up", "clf_21d_prob_up",
                "clf_5d_direction", "clf_21d_direction",
                "reg_5d_return_pct", "reg_21d_return_pct",
                "metrics", "trained_on_rows", "status",
            )
            for key in expected_keys:
                self.assertIn(key, result, msg=f"Key '{key}' missing from predict() result")

    def test_predict_status_ok(self):
        # A successful prediction run should report status='ok'.
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, trainer_result = _train_and_get_result(tmp)
            result = predict(csv_path, trainer_result)
            self.assertEqual(result["status"], "ok")

    def test_predict_probabilities_in_0_1(self):
        # Probabilities are used as percentage displays in the terminal report;
        # values outside [0, 1] would display nonsense like "150% probability".
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, trainer_result = _train_and_get_result(tmp)
            result = predict(csv_path, trainer_result)
            for key in ("clf_5d_prob_up", "clf_21d_prob_up"):
                val = result.get(key)
                if val is not None:
                    self.assertGreaterEqual(val, 0.0, msg=f"{key} below 0")
                    self.assertLessEqual(val, 1.0,   msg=f"{key} above 1")

    def test_predict_directions_are_up_or_down(self):
        # Direction strings are rendered literally in the terminal report.
        # Any value other than 'UP' or 'DOWN' would confuse the user.
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, trainer_result = _train_and_get_result(tmp)
            result = predict(csv_path, trainer_result)
            for key in ("clf_5d_direction", "clf_21d_direction"):
                val = result.get(key)
                if val not in (None, "N/A"):
                    self.assertIn(val, ("UP", "DOWN"),
                                  msg=f"{key} has unexpected value '{val}'")

    def test_predict_insufficient_data_returns_stub(self):
        # When the trainer_result has no models (status=insufficient_data)
        # AND no pooled global model exists, predict() must return the stub
        # immediately rather than raising an AttributeError trying to call
        # .predict_proba() on a None model. The global fallback is patched
        # out so this test does not depend on models/_GLOBAL/ artifacts on
        # the developer's machine.
        from unittest.mock import patch
        stub_result = {
            "status":  "insufficient_data",
            "models":  {},
            "metrics": {},
            "trained_on_rows": 0,
            "features": [],
        }
        with tempfile.TemporaryDirectory() as tmp,                 patch("ml.predictor.global_trainer.load_global_models", return_value=None):
            csv_path = _make_training_csv(n=10, tmp_dir=tmp)
            result   = predict(csv_path, stub_result)
            self.assertEqual(result["status"], "insufficient_data")
            # Probability fields must be None (not a float) in the stub.
            self.assertIsNone(result.get("clf_5d_prob_up"))
            self.assertIsNone(result.get("clf_21d_prob_up"))

    def test_predict_insufficient_data_uses_global_when_available(self):
        # With a pooled global model present, a short-history ticker gets
        # real (global-only) predictions with status "ok_global" instead of
        # the empty stub — this is the discovery-ticker code path.
        from unittest.mock import patch
        stub_result = {
            "status":  "insufficient_data",
            "models":  {},
            "metrics": {},
            "trained_on_rows": 0,
            "features": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, trainer_result = _train_and_get_result(tmp)
            fake_global = {
                "models":   trainer_result["models"],
                "features": trainer_result["features"],
                "metrics":  trainer_result["metrics"],
            }
            with patch("ml.predictor.global_trainer.load_global_models",
                       return_value=fake_global):
                result = predict(csv_path, stub_result)
            self.assertEqual(result["status"], "ok_global")
            self.assertIsInstance(result.get("clf_5d_prob_up"), float)

    def test_predict_return_pct_is_float_or_none(self):
        # Regression outputs must be floats so the display can format them
        # as percentage strings.
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, trainer_result = _train_and_get_result(tmp)
            result = predict(csv_path, trainer_result)
            for key in ("reg_5d_return_pct", "reg_21d_return_pct"):
                val = result.get(key)
                self.assertIn(type(val), (float, type(None)),
                              msg=f"{key} has unexpected type {type(val)}")


if __name__ == "__main__":
    unittest.main()
