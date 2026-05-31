"""
tests/test_csv_exporter.py
---------------------------
Unit tests for data/csv_exporter.py.

WHY these tests exist:
  The CSV produced by export_features_csv() is the sole input to the ML
  trainer and predictor.  If a feature column is missing or the row count
  is wrong the models will either fail to train or produce misleading output.
  These tests verify the file is created, contains the expected ML feature
  columns, and has one row per trading day in the source price DataFrame.

WHAT is tested:
  - export_features_csv() creates a file at the expected path.
  - The resulting CSV contains every column listed in ml.trainer.FEATURE_COLS.
  - The row count of the exported CSV matches the row count of the input
    price DataFrame (one feature row per trading day).

A temporary directory is used so no real files are left on disk after the tests.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from data.csv_exporter import export_features_csv
from ml.trainer import FEATURE_COLS

# We use compute_all_technicals to build a realistic technical dict —
# this is intentional: csv_exporter consumes the technical dict produced
# by the analysis pipeline, so using the real function is more robust
# than hand-crafting a stub.
from analysis.technical import compute_all_technicals


# ---------------------------------------------------------------------------
# Helper — synthetic price DataFrame
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 260, seed: int = 11) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with n trading-day bars.

    WHY 260 rows: matches approximately one year of daily data, which is
    enough for all rolling-window features (SMA-200, 21-day momentum, etc.)
    to produce non-NaN values in the final rows used by the ML predictor.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    log_returns = rng.normal(loc=0.0003, scale=0.012, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    noise = rng.uniform(0.005, 0.015, size=n)
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = close * (1 + rng.uniform(-0.005, 0.005, size=n))
    vol   = rng.integers(1_000_000, 5_000_000, size=n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def _make_inputs():
    """Build all inputs required by export_features_csv() in one call.

    Returns (price_df, technical, info, sentiment) so every test can call
    this helper rather than repeating setup boilerplate.
    """
    price_df  = _make_price_df()
    technical = compute_all_technicals(price_df)
    info      = {
        "currentPrice": float(price_df["Close"].iloc[-1]),
        "beta": 1.0,
        "fiftyTwoWeekHigh": float(price_df["High"].max()),
        "fiftyTwoWeekLow":  float(price_df["Low"].min()),
    }
    sentiment = {"overall_score": 0.1, "label": "POSITIVE", "summary": "Bullish tone."}
    return price_df, technical, info, sentiment


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExportFeaturesCSV(unittest.TestCase):
    """Tests for export_features_csv() — feature matrix file creation."""

    def test_file_is_created(self):
        # The function must write a file to the requested directory.
        # We use a temp dir so the test is isolated from the project's reports/.
        price_df, technical, info, sentiment = _make_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = export_features_csv("TEST", price_df, technical, info, sentiment, tmp)
            self.assertTrue(
                os.path.isfile(path),
                msg=f"Expected file at {path} but it was not created.",
            )

    def test_file_is_named_features_csv(self):
        # The path returned should end with 'features.csv' so ml/trainer.py can
        # locate it using the same naming convention.
        price_df, technical, info, sentiment = _make_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = export_features_csv("TEST", price_df, technical, info, sentiment, tmp)
            self.assertTrue(
                path.endswith("features.csv"),
                msg=f"Unexpected filename: {path}",
            )

    def test_all_feature_columns_present(self):
        # Every column in FEATURE_COLS must be present in the exported CSV.
        # Missing columns mean the ML predictor will receive a zero-filled row,
        # silently degrading prediction accuracy.
        price_df, technical, info, sentiment = _make_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = export_features_csv("TEST", price_df, technical, info, sentiment, tmp)
            df   = pd.read_csv(path, index_col=0)
            for col in FEATURE_COLS:
                self.assertIn(col, df.columns, msg=f"Feature column '{col}' missing from CSV")

    def test_row_count_matches_price_df(self):
        # One feature row must be generated for every trading day in price_df.
        # A lower row count would mean data was dropped; a higher count would
        # indicate accidental duplication.
        price_df, technical, info, sentiment = _make_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = export_features_csv("TEST", price_df, technical, info, sentiment, tmp)
            df   = pd.read_csv(path, index_col=0)
            self.assertEqual(
                len(df), len(price_df),
                msg=(
                    f"CSV has {len(df)} rows but price_df has {len(price_df)} rows. "
                    "Row count mismatch detected."
                ),
            )

    def test_sentiment_score_in_last_row(self):
        # The sentiment score is only attached to the last (most recent) row;
        # all historical rows default to 0.0.  Verify the last row carries
        # the provided score.
        price_df, technical, info, sentiment = _make_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path = export_features_csv("TEST", price_df, technical, info, sentiment, tmp)
            df   = pd.read_csv(path, index_col=0)
            if "sentiment_score" in df.columns:
                last_score = df["sentiment_score"].iloc[-1]
                self.assertAlmostEqual(
                    last_score, sentiment["overall_score"], places=4,
                    msg="sentiment_score in last row does not match provided score",
                )


if __name__ == "__main__":
    unittest.main()
