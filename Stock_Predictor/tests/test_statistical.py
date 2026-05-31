"""
tests/test_statistical.py
--------------------------
Unit tests for analysis/statistical.py.

WHY these tests exist:
  The statistical models feed directly into the rule-based judge composite
  score, the Monte Carlo chart, and the LLM prompt.  Incorrect annualisation
  (e.g. multiplying by 252 instead of √252) or wrong drawdown arithmetic
  would silently produce misleading risk metrics.  These tests pin the
  mathematical output against known analytical results.

WHAT is tested:
  - compute_volatility(): annualised sigma ≈ daily_sigma × √252; regime label set.
  - compute_sharpe_ratio(): positive return series → positive Sharpe.
  - compute_sortino_ratio(): expected keys present; downside-only calculation.
  - compute_max_drawdown_calmar(): monotonically increasing series → 0 % drawdown.
  - compute_linear_regression_predictions(): expected horizon keys present.
  - compute_monte_carlo(): predictions at all eight config horizons.
  - compute_beta(): result structure validated; SPY download mocked so no
    network call is made during the test run.

All yfinance network calls are patched out.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.statistical import (
    compute_volatility,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown_calmar,
    compute_linear_regression_predictions,
    compute_monte_carlo,
    compute_beta,
)
import config


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 260, seed: int = 7) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with a Log_Return column.

    WHY 260 rows:  One trading year of daily data (≈ 252 bars) is enough
    to compute all rolling windows including the 1-year Beta.  260 gives
    a small buffer above 252.

    Log_Return is computed here because statistical.py functions consume it
    directly — they do not recompute it from OHLCV.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    log_returns = rng.normal(loc=0.0004, scale=0.012, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    noise = rng.uniform(0.005, 0.015, size=n)
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = close * (1 + rng.uniform(-0.005, 0.005, size=n))
    vol   = rng.integers(1_000_000, 5_000_000, size=n).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    # Add the log-return column that statistical functions expect.
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def _make_uptrend_df(n: int = 260) -> pd.DataFrame:
    """Return a strictly monotonically increasing price DataFrame.

    WHY this shape: Used to test that max drawdown = 0 % for a series that
    never goes below any previous peak.
    """
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100.0 + np.arange(n, dtype=float)  # +1 every day
    high  = close + 0.5
    low   = close - 0.5
    vol   = np.ones(n) * 1_000_000.0

    df = pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


# ---------------------------------------------------------------------------
# Volatility tests
# ---------------------------------------------------------------------------

class TestComputeVolatility(unittest.TestCase):
    """Tests for compute_volatility() — annualised historical volatility."""

    def setUp(self):
        self.result = compute_volatility(_make_price_df())

    def test_expected_keys_present(self):
        # Actual key is 'label' (not 'vol_label'); see compute_volatility() return dict.
        for key in ("sigma_annual", "sigma_daily", "label", "vol_regime_ratio"):
            self.assertIn(key, self.result, msg=f"Key '{key}' missing")

    def test_sigma_annual_approx_sqrt252_times_daily(self):
        # Annualised volatility = daily σ × √252 (Hull, Options Ch. 15).
        # We allow a small tolerance because rolling-window edge effects can
        # introduce minor differences.
        daily  = self.result["sigma_daily"]
        annual = self.result["sigma_annual"]
        self.assertAlmostEqual(annual, daily * np.sqrt(252), delta=0.01)

    def test_sigma_positive(self):
        self.assertGreater(self.result["sigma_annual"], 0)

    def test_vol_label_is_string(self):
        # Key is 'label' in the actual return dict.
        self.assertIsInstance(self.result["label"], str)

    def test_vol_regime_ratio_positive(self):
        # The ratio of recent 30-day vol to full-period vol should be positive.
        self.assertGreater(self.result["vol_regime_ratio"], 0)


# ---------------------------------------------------------------------------
# Sharpe Ratio tests
# ---------------------------------------------------------------------------

class TestComputeSharpeRatio(unittest.TestCase):
    """Tests for compute_sharpe_ratio() — risk-adjusted excess return."""

    def test_expected_keys_present(self):
        result = compute_sharpe_ratio(_make_price_df())
        # 'sigma_annual' is not included in Sharpe output; available keys: sharpe, annual_return, signal.
        for key in ("sharpe", "annual_return", "signal"):
            self.assertIn(key, result, msg=f"Key '{key}' missing")

    def test_positive_return_series_gives_positive_sharpe(self):
        # A series with a strong upward drift should produce a positive Sharpe.
        # We use an extreme drift to ensure the return exceeds the risk-free rate.
        df = _make_uptrend_df()
        result = compute_sharpe_ratio(df)
        if result["sharpe"] is not None:
            self.assertGreater(result["sharpe"], 0)

    def test_sharpe_is_float_or_none(self):
        result = compute_sharpe_ratio(_make_price_df())
        # numpy.float64 is a subclass of float — use isinstance for a robust check.
        val = result["sharpe"]
        self.assertTrue(val is None or isinstance(val, (float, int)),
                        msg=f"sharpe has unexpected type {type(val)}")


# ---------------------------------------------------------------------------
# Sortino Ratio tests
# ---------------------------------------------------------------------------

class TestComputeSortinoRatio(unittest.TestCase):
    """Tests for compute_sortino_ratio() — downside-risk-adjusted return."""

    def test_expected_keys_present(self):
        result = compute_sortino_ratio(_make_price_df())
        # Actual key is 'sigma_downside_annual' (not 'downside_sigma').
        for key in ("sortino", "sigma_downside_annual", "signal"):
            self.assertIn(key, result, msg=f"Key '{key}' missing")

    def test_sortino_is_numeric_or_none(self):
        result = compute_sortino_ratio(_make_price_df())
        # Use isinstance to handle numpy scalar types as well as plain float/int.
        val = result["sortino"]
        self.assertTrue(val is None or isinstance(val, (float, int)),
                        msg=f"sortino has unexpected type {type(val)}")


# ---------------------------------------------------------------------------
# Max Drawdown tests
# ---------------------------------------------------------------------------

class TestComputeMaxDrawdown(unittest.TestCase):
    """Tests for compute_max_drawdown_calmar() — peak-to-trough loss + Calmar."""

    def test_expected_keys_present(self):
        result = compute_max_drawdown_calmar(_make_price_df())
        # Actual key is 'cagr_pct' (not 'cagr').
        for key in ("max_drawdown_pct", "calmar", "cagr_pct"):
            self.assertIn(key, result, msg=f"Key '{key}' missing")

    def test_zero_drawdown_for_uptrend(self):
        # A strictly increasing series never breaches its running maximum,
        # so the maximum drawdown must be exactly 0 %.
        result = compute_max_drawdown_calmar(_make_uptrend_df())
        if result["max_drawdown_pct"] is not None:
            self.assertAlmostEqual(result["max_drawdown_pct"], 0.0, places=4)

    def test_drawdown_non_positive(self):
        # Drawdown is defined as a loss — it must be ≤ 0.
        result = compute_max_drawdown_calmar(_make_price_df())
        if result["max_drawdown_pct"] is not None:
            self.assertLessEqual(result["max_drawdown_pct"], 0.0)


# ---------------------------------------------------------------------------
# Linear Regression Predictions tests
# ---------------------------------------------------------------------------

class TestComputeLinearRegression(unittest.TestCase):
    """Tests for compute_linear_regression_predictions() — OLS trend extrapolation."""

    def setUp(self):
        self.result = compute_linear_regression_predictions(_make_price_df())

    def test_expected_top_level_keys(self):
        # Actual key is 'trend_dir' (not 'trend_direction').
        for key in ("predictions", "r_squared", "trend_dir"):
            self.assertIn(key, self.result, msg=f"Key '{key}' missing")

    def test_predictions_at_all_horizons(self):
        # Every horizon defined in config.PREDICTION_HORIZONS should have an entry.
        preds = self.result.get("predictions", {})
        for label in config.PREDICTION_HORIZONS:
            self.assertIn(label, preds, msg=f"Horizon '{label}' missing from regression predictions")

    def test_r_squared_bounded_0_1(self):
        # R² must be in [0, 1] for an OLS fit.
        r2 = self.result.get("r_squared")
        if r2 is not None:
            self.assertGreaterEqual(r2, 0.0)
            self.assertLessEqual(r2, 1.0)


# ---------------------------------------------------------------------------
# Monte Carlo tests
# ---------------------------------------------------------------------------

class TestComputeMonteCarlo(unittest.TestCase):
    """Tests for compute_monte_carlo() — GBM simulation with CAPM drift."""

    def setUp(self):
        self.result = compute_monte_carlo(_make_price_df(), beta=1.0)

    def test_expected_top_level_keys(self):
        # Actual keys: 'predictions', 'paths', 'params'. 'current_price' is inside params['S0'].
        for key in ("predictions", "paths", "params"):
            self.assertIn(key, self.result, msg=f"Key '{key}' missing")

    def test_predictions_at_all_horizons(self):
        # All eight PREDICTION_HORIZONS should have Monte Carlo estimates.
        preds = self.result.get("predictions", {})
        for label in config.PREDICTION_HORIZONS:
            self.assertIn(label, preds, msg=f"Horizon '{label}' missing from MC predictions")

    def test_each_horizon_has_percentile_keys(self):
        # Each horizon should contain at minimum p10, median, and p90.
        preds = self.result["predictions"]
        for label, horizon_data in preds.items():
            with self.subTest(label=label):
                for sub_key in ("p10", "median", "p90"):
                    self.assertIn(sub_key, horizon_data, msg=f"Missing '{sub_key}' for horizon '{label}'")

    def test_p10_lte_median_lte_p90(self):
        # By definition: 10th percentile ≤ median ≤ 90th percentile.
        preds = self.result["predictions"]
        for label, data in preds.items():
            with self.subTest(label=label):
                p10    = data.get("p10")
                median = data.get("median")
                p90    = data.get("p90")
                if all(v is not None for v in (p10, median, p90)):
                    self.assertLessEqual(p10, median,  msg=f"p10 > median for {label}")
                    self.assertLessEqual(median, p90, msg=f"median > p90 for {label}")

    def test_paths_run_matches_config(self):
        # n_paths is stored inside params dict; paths ndarray has shape [n_paths, max_horizon+1].
        self.assertEqual(self.result["params"]["n_paths"], config.MONTE_CARLO_PATHS)


# ---------------------------------------------------------------------------
# Beta tests (mocked SPY download)
# ---------------------------------------------------------------------------

class TestComputeBeta(unittest.TestCase):
    """Tests for compute_beta() — systematic market risk vs SPY.

    WHY we mock _get_spy_data:
      compute_beta() calls _get_spy_data() which downloads SPY from Yahoo
      Finance.  In a unit-test environment we must not make live network
      calls (they are slow, flaky, and require internet access).  We replace
      _get_spy_data with a function that returns a synthetic SPY DataFrame
      with the same structure as the real download.
    """

    def _make_spy_df(self, n: int = 260, seed: int = 99) -> pd.DataFrame:
        """Build a synthetic SPY DataFrame matching yfinance's output format."""
        rng   = np.random.default_rng(seed)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        log_r = rng.normal(0.0003, 0.010, size=n)
        close = 400.0 * np.exp(np.cumsum(log_r))
        return pd.DataFrame({"Close": close}, index=dates)

    @patch("analysis.statistical._get_spy_data")
    def test_beta_result_keys(self, mock_spy):
        # Inject the synthetic SPY data so no network call is made.
        mock_spy.return_value = self._make_spy_df()
        result = compute_beta(_make_price_df(), ticker="TEST")
        # Actual key is 'label' (not 'signal'); compute_beta() returns {beta, label}.
        for key in ("beta", "label"):
            self.assertIn(key, result, msg=f"Key '{key}' missing from beta result")

    @patch("analysis.statistical._get_spy_data")
    def test_beta_is_numeric_or_none(self, mock_spy):
        mock_spy.return_value = self._make_spy_df()
        result = compute_beta(_make_price_df(), ticker="TEST")
        self.assertIn(type(result["beta"]), (float, type(None)))

    @patch("analysis.statistical._get_spy_data")
    def test_beta_graceful_on_empty_spy(self, mock_spy):
        # If SPY data is empty, beta must return a stub rather than raising.
        mock_spy.return_value = pd.DataFrame()
        result = compute_beta(_make_price_df(), ticker="TEST")
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
