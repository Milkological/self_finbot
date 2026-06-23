"""
tests/test_accuracy.py
----------------------
Unit tests for feedback/accuracy.py.

WHY these tests exist:
  accuracy.py turns resolved feedback rows into the signals that actually
  improve predictions: per-horizon bias correction, calibration tables, and
  the historical-accuracy block injected into the LLM Judge. Bugs here mean
  the model never learns from its own track record. These tests verify:
    - Bias correction uses the MEDIAN signed error (robust to outliers).
    - Correction is SHRUNK toward 1.0 on thin samples (no overcorrection at N=5).
    - Directional accuracy aggregation is correct.
    - build_calibration_table survives malformed CSV rows (on_bad_lines skip).
    - format_llm_injection emits grounding + caution instructions when the
      live track record is poor.

HOW isolation works:
  Synthetic CSVs are written to a temp dir and FEEDBACK_DIR / ML_MODELS_DIR
  are patched, so no real feedback or model files are touched. No
  importlib.reload (it would re-bind the patched constants).
"""

import csv
import datetime
import os
import json
import tempfile
import unittest
from unittest.mock import patch

from feedback.tracker import COLUMNS
from config import FEEDBACK_MIN_SAMPLES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_row() -> dict:
    return {c: "" for c in COLUMNS}


def _write_csv(path: str, rows: list[dict], extra_bad_line: bool = False) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        if extra_bad_line:
            # A row with one extra field (mirrors the real corrupted CSVs:
            # "Expected 72 fields, saw 73"). Plain read_csv throws on this.
            fh.write(",".join(["x"] * (len(COLUMNS) + 1)) + "\n")


def _resolved_rows(pct_errors: list[float], dir_correct: list[int],
                   provider: str = "Azure OpenAI") -> list[dict]:
    """Build resolved 1w rows with distinct run_dates (so dedup keeps them all)."""
    rows = []
    base = datetime.date(2024, 1, 1)
    for i, pe in enumerate(pct_errors):
        r = _blank_row()
        r.update({
            "run_id": f"FAKE_{i}", "ticker": "FAKE",
            "run_date": (base + datetime.timedelta(days=i)).isoformat(),
            "provider": provider, "current_price": "100.0",
            "recommendation": "BUY",
            "actual_1w": "105.0",
            "pct_error_1w": str(pe),
            "dir_correct_1w": str(dir_correct[i]) if i < len(dir_correct) else "",
        })
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetContext(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._path = os.path.join(self._tmpdir, "FAKE_feedback.csv")

    def _get_context(self, **kwargs):
        with patch("feedback.accuracy.FEEDBACK_DIR", self._tmpdir):
            import feedback.accuracy as mod
            return mod.get_context("FAKE", **kwargs)

    def test_bias_uses_median_not_mean(self):
        # Errors with an outlier: mean=18, median=10. We expect MEDIAN.
        rows = _resolved_rows([10, 12, 8, 50, 10], [1, 1, 1, 1, 1])
        _write_csv(self._path, rows)
        ctx = self._get_context()
        h = ctx["horizon_stats"]["1w"]
        self.assertEqual(h["n"], 5)
        self.assertAlmostEqual(h["mean_bias"], 10.0, places=2)

    def test_correction_shrinks_on_thin_samples(self):
        # N=5 == FEEDBACK_MIN_SAMPLES → shrink factor = 5/(2*5) = 0.5.
        # median bias 10% → 0.10 clamped → *0.5 → cf = 1.05.
        rows = _resolved_rows([10, 10, 10, 10, 10], [1, 1, 1, 1, 1])
        _write_csv(self._path, rows)
        ctx = self._get_context()
        self.assertAlmostEqual(
            ctx["horizon_stats"]["1w"]["correction_factor"], 1.05, places=3)

    def test_correction_full_strength_with_enough_samples(self):
        # N=12 → shrink factor = min(1, 12/10) = 1.0 → cf = 1.10.
        rows = _resolved_rows([10] * 12, [1] * 12)
        _write_csv(self._path, rows)
        ctx = self._get_context()
        self.assertAlmostEqual(
            ctx["horizon_stats"]["1w"]["correction_factor"], 1.10, places=3)

    def test_correction_clamped(self):
        # median bias 50% → clamp to ±0.20 → cf = 1.20 at full strength.
        rows = _resolved_rows([50] * 12, [1] * 12)
        _write_csv(self._path, rows)
        ctx = self._get_context()
        self.assertAlmostEqual(
            ctx["horizon_stats"]["1w"]["correction_factor"], 1.20, places=3)

    def test_directional_accuracy_aggregation(self):
        rows = _resolved_rows([1, 1, 1, 1, 1], [1, 1, 1, 0, 0])  # 3/5 correct
        _write_csv(self._path, rows)
        ctx = self._get_context()
        self.assertAlmostEqual(ctx["horizon_stats"]["1w"]["dir_accuracy"], 60.0, places=1)


class TestCalibrationRobustness(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._models = tempfile.mkdtemp()
        self._path = os.path.join(self._tmpdir, "FAKE_feedback.csv")

    def test_builds_despite_malformed_row(self):
        rows = []
        base = datetime.date(2024, 1, 1)
        for i in range(12):
            r = _blank_row()
            r.update({
                "run_id": f"FAKE_{i}", "ticker": "FAKE",
                "run_date": (base + datetime.timedelta(days=i)).isoformat(),
                "provider": "Azure OpenAI",
                "ml_5d_prob_up": "0.75",
                "ml_5d_correct": str(i % 2),  # alternating 1/0
            })
            rows.append(r)
        _write_csv(self._path, rows, extra_bad_line=True)  # corrupt trailing row

        with patch("feedback.accuracy.FEEDBACK_DIR", self._tmpdir), \
             patch("feedback.accuracy.ML_MODELS_DIR", self._models):
            import feedback.accuracy as mod
            result = mod.build_calibration_table("FAKE")

        self.assertIn("5d", result, msg="Calibration must build even with a malformed row")
        self.assertTrue(os.path.isfile(os.path.join(self._models, "FAKE", "calibration.json")))


class TestLLMInjection(unittest.TestCase):

    def _ctx(self, rec_hit_rate, dir_acc_1w=70.0, n=8):
        base = {
            "horizon_stats": {h: {"n": 0, "dir_accuracy": None, "mean_bias": None,
                                  "correction_factor": 1.0}
                              for h in ["1w", "2w", "3w", "1m", "3m", "6m", "9m", "12m"]},
            "ml_accuracy": {"5d": 55.0, "21d": 52.0},
            "rec_hit_rate": rec_hit_rate,
            "adjusted_weights": {}, "weights_adjusted": False,
            "total_resolved": n, "has_data": True,
        }
        base["horizon_stats"]["1w"] = {"n": n, "dir_accuracy": dir_acc_1w,
                                       "mean_bias": 2.0, "correction_factor": 1.0}
        return base

    def test_grounding_instruction_present(self):
        from feedback.accuracy import format_llm_injection
        txt = format_llm_injection("FAKE", self._ctx(rec_hit_rate=0.7))
        # The Judge must be told to use realized accuracy as the BASE.
        self.assertIn("BASE", txt)

    def test_low_hit_rate_triggers_caution(self):
        from feedback.accuracy import format_llm_injection
        txt = format_llm_injection("FAKE", self._ctx(rec_hit_rate=0.3))
        low = txt.lower()
        self.assertTrue("caution" in low or "cap confidence" in low,
                        msg="Poor track record must inject an explicit caution instruction")

    def test_good_hit_rate_no_caution(self):
        from feedback.accuracy import format_llm_injection
        txt = format_llm_injection("FAKE", self._ctx(rec_hit_rate=0.8)).lower()
        self.assertNotIn("cap confidence", txt)


if __name__ == "__main__":
    unittest.main()
