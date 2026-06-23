"""
tests/test_resolver.py
----------------------
Unit tests for feedback/resolver.py.

WHY these tests exist:
  resolve_pending() is what turns a saved prediction into a *measured*
  outcome — it is the engine of the whole feedback/accuracy/calibration loop.
  If it silently fails (as it did with the removed `show_errors` yfinance
  kwarg) nothing ever resolves and every accuracy feature stays dormant.
  These tests verify, with yfinance fully mocked:
    - A run old enough gets its outcome columns filled correctly.
    - Directional correctness is computed from predicted-vs-actual move.
    - Price history is fetched ONCE per call (batched), not once per
      horizon per row.
    - The deprecated `show_errors` kwarg is never passed to yf.download.
    - Horizons whose window hasn't elapsed are left blank.

HOW mocking works here:
  We patch 'feedback.resolver.yf.download' with a MagicMock returning a
  synthetic daily-close DataFrame, and patch FEEDBACK_DIR to a temp dir so
  no real feedback files are touched.
"""

import csv
import datetime
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from feedback.tracker import COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_row() -> dict:
    """A feedback row with every column present and blank."""
    return {c: "" for c in COLUMNS}


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fake_history(step_date: str = "2024-01-08",
                  before: float = 100.0, after: float = 105.0) -> pd.DataFrame:
    """Daily-close history: `before` until step_date, then `after` onward.

    Spans a wide range so any past resolve date lands inside it. Mimics
    yf.download output (DatetimeIndex + 'Close' column).
    """
    idx = pd.date_range("2023-12-01", datetime.date.today().isoformat(), freq="D")
    close = pd.Series(before, index=idx, dtype=float)
    close.loc[pd.Timestamp(step_date):] = after
    return pd.DataFrame({"Close": close})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResolvePending(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._path = os.path.join(self._tmpdir, "FAKE_feedback.csv")

    def _run_resolver(self, mock_download):
        with patch("feedback.resolver.FEEDBACK_DIR", self._tmpdir), \
             patch("feedback.resolver.yf.download", mock_download):
            import feedback.resolver as mod
            return mod.resolve_pending("FAKE")

    def test_fills_outcomes_for_elapsed_horizon(self):
        row = _blank_row()
        row.update({
            "run_id": "FAKE_1", "ticker": "FAKE",
            "run_date": "2024-01-02", "provider": "Azure OpenAI",
            "current_price": "100.0", "recommendation": "BUY",
            "target_1w": "110.0", "ml_5d_direction": "UP",
        })
        _write_csv(self._path, [row])

        mock_dl = MagicMock(return_value=_fake_history())
        updated = self._run_resolver(mock_dl)

        self.assertGreaterEqual(updated, 1)
        out = _read_rows(self._path)[0]
        self.assertAlmostEqual(float(out["actual_1w"]), 105.0, places=2)
        # predicted up (110>100) and actual up (105>100) → correct
        self.assertEqual(out["dir_correct_1w"], "1")
        # pct_error = (105 - 110) / 110 * 100 = -4.55
        self.assertAlmostEqual(float(out["pct_error_1w"]), -4.55, places=1)

    def test_batched_single_download(self):
        """Price history must be fetched once per call, not per-horizon."""
        row = _blank_row()
        row.update({
            "run_id": "FAKE_1", "ticker": "FAKE",
            "run_date": "2024-01-02", "provider": "Azure OpenAI",
            "current_price": "100.0", "recommendation": "BUY",
            "target_1w": "110.0", "target_1m": "120.0",
        })
        _write_csv(self._path, [row])

        mock_dl = MagicMock(return_value=_fake_history())
        self._run_resolver(mock_dl)

        self.assertEqual(mock_dl.call_count, 1,
                         msg="Expected exactly one batched download for the ticker")

    def test_no_show_errors_kwarg(self):
        row = _blank_row()
        row.update({
            "run_id": "FAKE_1", "ticker": "FAKE",
            "run_date": "2024-01-02", "provider": "Azure OpenAI",
            "current_price": "100.0", "target_1w": "110.0",
        })
        _write_csv(self._path, [row])

        mock_dl = MagicMock(return_value=_fake_history())
        self._run_resolver(mock_dl)

        self.assertTrue(mock_dl.called)
        _, kwargs = mock_dl.call_args
        self.assertNotIn("show_errors", kwargs,
                         msg="show_errors was removed from yfinance and must not be passed")

    def test_wrong_direction_marked_incorrect(self):
        row = _blank_row()
        row.update({
            "run_id": "FAKE_1", "ticker": "FAKE",
            "run_date": "2024-01-02", "provider": "Azure OpenAI",
            "current_price": "100.0", "target_1w": "90.0",  # predicts DOWN
        })
        _write_csv(self._path, [row])

        # actual goes UP to 105 → prediction was wrong
        mock_dl = MagicMock(return_value=_fake_history(after=105.0))
        self._run_resolver(mock_dl)

        out = _read_rows(self._path)[0]
        self.assertEqual(out["dir_correct_1w"], "0")

    def test_unelapsed_horizon_left_blank(self):
        recent = (datetime.date.today() - datetime.timedelta(days=3)).isoformat()
        row = _blank_row()
        row.update({
            "run_id": "FAKE_1", "ticker": "FAKE",
            "run_date": recent, "provider": "Azure OpenAI",
            "current_price": "100.0", "target_1w": "110.0",
        })
        _write_csv(self._path, [row])

        mock_dl = MagicMock(return_value=_fake_history())
        self._run_resolver(mock_dl)

        out = _read_rows(self._path)[0]
        self.assertEqual(out["actual_1w"].strip(), "",
                         msg="1-week horizon only 3 days old must remain unresolved")

    def test_missing_csv_returns_zero(self):
        mock_dl = MagicMock(return_value=_fake_history())
        with patch("feedback.resolver.FEEDBACK_DIR", self._tmpdir), \
             patch("feedback.resolver.yf.download", mock_dl):
            import feedback.resolver as mod
            self.assertEqual(mod.resolve_pending("NOEXIST"), 0)


if __name__ == "__main__":
    unittest.main()
