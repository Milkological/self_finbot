"""
tests/test_feedback_tracker.py
-------------------------------
Unit tests for feedback/tracker.py.

WHY these tests exist:
  save_prediction() is the sole entry point that writes data to the
  feedback CSV.  If the column list is wrong, a column is missing from the
  row dict, or the file-append logic is broken, the resolver and accuracy
  modules will silently read garbage or fail entirely.  These tests verify:
    - The CSV is created with the correct header on the first call.
    - All COLUMNS are present as headers in the written file.
    - The 'provider' field is recorded correctly.
    - A second call appends a new row rather than overwriting.
    - The function returns a valid file path that exists on disk.
    - Minimal inputs (empty llm_result, empty ml_result) do not raise.

HOW isolation works:
  A temporary directory is injected by patching FEEDBACK_DIR so no real
  feedback files are created or modified during the test run.
"""

import csv
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper — minimal synthetic inputs
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1,
         "Close": close, "Volume": 1_000_000},
        index=dates,
    )


def _make_llm_result() -> dict:
    return {
        "recommendation": "BUY",
        "confidence":     "HIGH",
        "target_prices": {
            "1_week":   {"price": 105.0},
            "2_weeks":  {"price": 107.0},
            "3_weeks":  {"price": 109.0},
            "1_month":  {"price": 112.0},
            "3_months": {"price": 120.0},
            "6_months": {"price": 130.0},
            "9_months": {"price": 140.0},
            "12_months":{"price": 150.0},
        },
        "technical_verdict":   "BULLISH",
        "fundamental_verdict": "STRONG",
        "valuation_verdict":   "FAIR",
        "sentiment_verdict":   "POSITIVE",
        "catalysts":           ["Strong earnings"],
        "key_bull_case":       ["Market leader"],
        "key_bear_case":       ["Valuation stretched"],
    }


def _make_info() -> dict:
    return {
        "currentPrice":          100.0,
        "shortRatio":            3.5,
        "shortPercentOfFloat":   0.04,
    }


def _make_analyst_data() -> dict:
    return {
        "news": [
            {"title": "FAKE beats earnings"},
            {"title": "FAKE raises guidance"},
            {"title": "Analyst upgrades FAKE"},
        ]
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSavePrediction(unittest.TestCase):
    """Tests for feedback.tracker.save_prediction()."""

    def setUp(self):
        """Create a fresh temporary directory for each test."""
        self._tmpdir = tempfile.mkdtemp()

    def _call_save(self, provider: str = "Azure OpenAI"):
        """Import and call save_prediction with the temp FEEDBACK_DIR patched."""
        # NOTE: do NOT importlib.reload() the module here. _csv_path() reads the
        # module-level FEEDBACK_DIR at call time, so patching the attribute is
        # sufficient. Reloading re-runs `from config import FEEDBACK_DIR`, which
        # rebinds the name back to the real path and defeats the patch — that
        # caused tests to write to the real feedback/ dir and accumulate rows.
        with patch("feedback.tracker.FEEDBACK_DIR", self._tmpdir):
            import feedback.tracker as mod
            return mod.save_prediction(
                ticker       = "FAKE",
                mode         = "llm",
                provider     = provider,
                price_df     = _make_price_df(),
                info         = _make_info(),
                llm_result   = _make_llm_result(),
                ml_result    = {},
                analyst_data = _make_analyst_data(),
                options_data = {"put_call_ratio": 0.9, "options_iv_avg": 0.25},
            )

    def test_returns_valid_path(self):
        path = self._call_save()
        self.assertTrue(os.path.isfile(path),
                        msg=f"Expected CSV file to exist at: {path}")

    def test_header_contains_all_columns(self):
        from feedback.tracker import COLUMNS
        path = self._call_save()
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
        for col in COLUMNS:
            self.assertIn(col, fieldnames,
                          msg=f"Column '{col}' missing from CSV header")

    def test_provider_field_written(self):
        path = self._call_save(provider="DeepSeek (deepseek-chat)")
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["provider"], "DeepSeek (deepseek-chat)")

    def test_recommendation_field_written(self):
        path = self._call_save()
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        self.assertEqual(row["recommendation"], "BUY")

    def test_second_call_appends_row(self):
        self._call_save(provider="Azure OpenAI")
        path = self._call_save(provider="Google Gemini (gemini-2.0-flash)")
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        self.assertEqual(len(rows), 2,
                         msg="Second call should append, giving 2 rows total")

    def test_empty_ml_result_does_not_raise(self):
        with patch("feedback.tracker.FEEDBACK_DIR", self._tmpdir):
            import feedback.tracker as mod
            # Should not raise even with minimal inputs
            mod.save_prediction(
                ticker       = "FAKE",
                mode         = "no_llm",
                provider     = "rule_based",
                price_df     = _make_price_df(),
                info         = _make_info(),
                llm_result   = {},
                ml_result    = None,
                analyst_data = {},
                options_data = {},
            )

    def test_outcome_columns_are_blank_at_creation(self):
        """Outcome columns must be empty when first written — resolver fills them later."""
        path = self._call_save()
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = next(reader)
        for horizon in ["1w", "2w", "3w", "1m", "3m", "6m", "9m", "12m"]:
            self.assertEqual(row[f"actual_{horizon}"], "",
                             msg=f"actual_{horizon} should be blank at prediction time")
            self.assertEqual(row[f"dir_correct_{horizon}"], "",
                             msg=f"dir_correct_{horizon} should be blank at prediction time")


if __name__ == "__main__":
    unittest.main()
