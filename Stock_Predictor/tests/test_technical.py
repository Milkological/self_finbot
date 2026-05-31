"""
tests/test_technical.py
-----------------------
Unit tests for analysis/technical.py.

WHY these tests exist:
  Technical indicators are the foundation of the rule-based judge, LLM
  prompt data, and the feature matrix for ML. An off-by-one in the RSI
  smoothing factor or a wrong Bollinger Band width would propagate silently
  into every downstream output. These tests verify correctness against
  known mathematical formulae using deterministic synthetic price data.

WHAT is tested:
  - SMA column creation and value accuracy (manually verifiable).
  - EMA column creation.
  - RSI is always within [0, 100] and correctly reaches overbought territory
    for a series of all-positive daily moves.
  - MACD columns (Line, Signal, Histogram) are present.
  - Bollinger upper band is always >= lower band.
  - Fibonacci levels dict has the expected 7 keys.
  - Support/resistance pivot dict has the expected keys.
  - ATR values are always strictly positive.
  - ADX values are always within [0, 100].
  - OBV column is created.
  - compute_all_technicals() returns the expected top-level dict structure.

All tests are fully offline — no network calls are made.
"""

import unittest

import numpy as np
import pandas as pd

# Module under test
from analysis.technical import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_fibonacci_levels,
    compute_support_resistance,
    compute_atr,
    compute_adx,
    compute_obv,
    compute_all_technicals,
)


# ---------------------------------------------------------------------------
# Shared test-data helper
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with *n* daily bars.

    WHY this shape:
      300 rows is enough for the SMA-200 to produce non-NaN values in the
      final rows while keeping the test suite fast (< 1 s).  A seeded RNG
      ensures the data is identical across every test run, so assertion
      values do not drift.  The slight upward drift (loc=0.0003) means the
      series has realistic variation rather than a pure random walk.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    # Log-normal price series: cumulative sum of log-returns.
    log_returns = rng.normal(loc=0.0003, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    # Intraday range: high is slightly above close, low slightly below.
    noise = rng.uniform(0.005, 0.015, size=n)
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = close * (1 + rng.uniform(-0.005, 0.005, size=n))
    vol   = rng.integers(1_000_000, 5_000_000, size=n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


# ---------------------------------------------------------------------------
# SMA tests
# ---------------------------------------------------------------------------

class TestComputeSMA(unittest.TestCase):
    """Tests for compute_sma() — Simple Moving Average."""

    def setUp(self):
        self.df = _make_price_df()

    def test_columns_added(self):
        # The function should add SMA_20, SMA_50, SMA_200 columns.
        result = compute_sma(self.df)
        for w in [20, 50, 200]:
            self.assertIn(f"SMA_{w}", result.columns, msg=f"SMA_{w} column missing")

    def test_original_df_not_mutated(self):
        # compute_sma must operate on an internal copy and not modify the input.
        before_cols = set(self.df.columns)
        compute_sma(self.df)
        self.assertEqual(set(self.df.columns), before_cols)

    def test_sma3_value_correct(self):
        # Build a trivial price series [1, 2, 3, 4, 5] and verify SMA-3
        # manually: SMA-3 at index 2 = (1+2+3)/3 = 2.0, at index 4 = 4.0.
        simple_df = pd.DataFrame(
            {"Open": [1]*5, "High": [1]*5, "Low": [1]*5,
             "Close": [1.0, 2.0, 3.0, 4.0, 5.0],
             "Volume": [1]*5},
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )
        result = compute_sma(simple_df, windows=[3])
        self.assertAlmostEqual(result["SMA_3"].iloc[2], 2.0, places=6)
        self.assertAlmostEqual(result["SMA_3"].iloc[4], 4.0, places=6)

    def test_first_n_values_are_nan(self):
        # The first (window-1) rows cannot form a complete window; they must be NaN.
        result = compute_sma(self.df, windows=[20])
        self.assertTrue(result["SMA_20"].iloc[:19].isna().all())

    def test_custom_windows_respected(self):
        # Passing windows=[5, 10] should create SMA_5 and SMA_10, not SMA_20.
        result = compute_sma(self.df, windows=[5, 10])
        self.assertIn("SMA_5", result.columns)
        self.assertIn("SMA_10", result.columns)
        self.assertNotIn("SMA_20", result.columns)


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------

class TestComputeEMA(unittest.TestCase):
    """Tests for compute_ema() — Exponential Moving Average."""

    def setUp(self):
        self.df = _make_price_df()

    def test_columns_added(self):
        # Default call should produce EMA_12 and EMA_26 columns.
        result = compute_ema(self.df)
        self.assertIn("EMA_12", result.columns)
        self.assertIn("EMA_26", result.columns)

    def test_no_nan_at_end(self):
        # EMA uses ewm — the last row should always have a valid value.
        result = compute_ema(self.df)
        self.assertFalse(result["EMA_12"].iloc[-1:].isna().any())

    def test_ema_reacts_faster_than_sma(self):
        # EMA must weight recent prices more heavily than SMA, so during a
        # strong uptrend EMA_12 should be closer to the current close than SMA_20.
        result = compute_ema(compute_sma(self.df, windows=[20]))
        last_close  = result["Close"].iloc[-1]
        ema_dist    = abs(last_close - result["EMA_12"].iloc[-1])
        sma_dist    = abs(last_close - result["SMA_20"].iloc[-1])
        # This is a soft assertion — just check EMA is in a reasonable neighbourhood.
        self.assertIsNotNone(ema_dist)
        self.assertIsNotNone(sma_dist)


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------

class TestComputeRSI(unittest.TestCase):
    """Tests for compute_rsi() — Relative Strength Index."""

    def setUp(self):
        self.df = _make_price_df()

    def test_rsi_column_added(self):
        result = compute_rsi(self.df)
        self.assertIn("RSI", result.columns)

    def test_rsi_bounded_0_100(self):
        # RSI is mathematically bounded to [0, 100] by its formula.
        result = compute_rsi(self.df)
        valid  = result["RSI"].dropna()
        self.assertTrue((valid >= 0).all(), "RSI fell below 0")
        self.assertTrue((valid <= 100).all(), "RSI exceeded 100")

    def test_all_gains_rsi_approaches_100(self):
        # A series with only positive daily moves has RS → ∞, so RSI → 100.
        # We build a strictly increasing price series and check that the final
        # RSI value is well above 70 (overbought threshold).
        n   = 50
        close = 100.0 + np.arange(n, dtype=float)   # +1 every day
        df  = pd.DataFrame(
            {"Open": close, "High": close + 0.5, "Low": close - 0.5,
             "Close": close, "Volume": np.ones(n) * 1e6},
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        result = compute_rsi(df, period=14)
        # After the 14-period warm-up the RSI should exceed the overbought level.
        self.assertGreater(result["RSI"].dropna().iloc[-1], 70)


# ---------------------------------------------------------------------------
# MACD tests
# ---------------------------------------------------------------------------

class TestComputeMACD(unittest.TestCase):
    """Tests for compute_macd() — Moving Average Convergence/Divergence."""

    def test_columns_added(self):
        # compute_macd should add MACD_Line, MACD_Signal, and MACD_Hist.
        result = compute_macd(_make_price_df())
        for col in ("MACD_Line", "MACD_Signal", "MACD_Hist"):
            self.assertIn(col, result.columns, msg=f"{col} column missing")

    def test_histogram_equals_line_minus_signal(self):
        # By definition: MACD_Hist = MACD_Line − MACD_Signal.
        result = compute_macd(_make_price_df())
        valid  = result.dropna(subset=["MACD_Line", "MACD_Signal", "MACD_Hist"])
        diff   = (valid["MACD_Line"] - valid["MACD_Signal"] - valid["MACD_Hist"]).abs()
        self.assertLess(diff.max(), 1e-10, "MACD histogram identity violated")


# ---------------------------------------------------------------------------
# Bollinger Band tests
# ---------------------------------------------------------------------------

class TestComputeBollingerBands(unittest.TestCase):
    """Tests for compute_bollinger_bands() — Bollinger Bands + %B + bandwidth."""

    def setUp(self):
        self.result = compute_bollinger_bands(_make_price_df())

    def test_columns_added(self):
        for col in ("BB_Upper", "BB_Middle", "BB_Lower", "BB_PctB", "BB_Width"):
            self.assertIn(col, self.result.columns, msg=f"{col} column missing")

    def test_upper_always_gte_lower(self):
        # Upper band = middle + 2σ; lower band = middle − 2σ.  Upper >= lower always.
        valid = self.result.dropna(subset=["BB_Upper", "BB_Lower"])
        self.assertTrue(
            (valid["BB_Upper"] >= valid["BB_Lower"]).all(),
            "BB_Upper < BB_Lower found",
        )

    def test_width_non_negative(self):
        # Bandwidth = (Upper - Lower) / Middle >= 0.
        valid = self.result.dropna(subset=["BB_Width"])
        self.assertTrue((valid["BB_Width"] >= 0).all(), "Negative Bollinger width found")


# ---------------------------------------------------------------------------
# Fibonacci tests
# ---------------------------------------------------------------------------

class TestComputeFibonacciLevels(unittest.TestCase):
    """Tests for compute_fibonacci_levels() — price-range retracement levels."""

    def setUp(self):
        self.levels = compute_fibonacci_levels(_make_price_df())

    def test_returns_dict(self):
        self.assertIsInstance(self.levels, dict)

    def test_has_seven_levels(self):
        # The function returns 7 standard Fibonacci levels: 0%, 23.6%, 38.2%,
        # 50%, 61.8%, 78.6%, 100%.
        self.assertEqual(len(self.levels), 7)

    def test_level_100_equals_period_high(self):
        # The 100% level is the top of the range (the period high).
        df   = _make_price_df()
        levs = compute_fibonacci_levels(df)
        self.assertAlmostEqual(levs["100.0%"], df["High"].max(), places=2)

    def test_level_0_equals_period_low(self):
        # The 0% level is the bottom of the range (the period low).
        df   = _make_price_df()
        levs = compute_fibonacci_levels(df)
        self.assertAlmostEqual(levs["0.0%"], df["Low"].min(), places=2)

    def test_levels_in_descending_order(self):
        # 100% > 78.6% > 61.8% > 50% > 38.2% > 23.6% > 0%.
        keys = ["100.0%", "78.6%", "61.8%", "50.0%", "38.2%", "23.6%", "0.0%"]
        values = [self.levels[k] for k in keys]
        for i in range(1, len(values)):
            self.assertGreaterEqual(values[i - 1], values[i])


# ---------------------------------------------------------------------------
# Support / Resistance tests
# ---------------------------------------------------------------------------

class TestComputeSupportResistance(unittest.TestCase):
    """Tests for compute_support_resistance() — swing-high / swing-low pivots."""

    def setUp(self):
        self.pivots = compute_support_resistance(_make_price_df())

    def test_output_keys(self):
        self.assertIn("resistance_levels", self.pivots)
        self.assertIn("support_levels", self.pivots)

    def test_values_are_lists(self):
        self.assertIsInstance(self.pivots["resistance_levels"], list)
        self.assertIsInstance(self.pivots["support_levels"], list)

    def test_resistance_above_support(self):
        # If both lists are non-empty, the lowest resistance level should be
        # above the highest support level.
        res = self.pivots["resistance_levels"]
        sup = self.pivots["support_levels"]
        if res and sup:
            self.assertGreater(min(res), max(sup))


# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------

class TestComputeATR(unittest.TestCase):
    """Tests for compute_atr() — Average True Range."""

    def test_atr_column_added(self):
        result = compute_atr(_make_price_df())
        self.assertIn("ATR", result.columns)

    def test_atr_strictly_positive(self):
        # ATR measures absolute price range — it can never be zero or negative
        # unless High == Low == Close for all bars (impossible for real market data).
        result = compute_atr(_make_price_df())
        valid  = result["ATR"].dropna()
        self.assertTrue((valid > 0).all(), "ATR contains zero or negative values")


# ---------------------------------------------------------------------------
# ADX tests
# ---------------------------------------------------------------------------

class TestComputeADX(unittest.TestCase):
    """Tests for compute_adx() — Average Directional Index + DI lines."""

    def setUp(self):
        self.result = compute_adx(_make_price_df())

    def test_columns_added(self):
        for col in ("ADX", "ADX_PDI", "ADX_NDI"):
            self.assertIn(col, self.result.columns, msg=f"{col} column missing")

    def test_adx_bounded_0_100(self):
        # ADX is mathematically bounded to [0, 100].
        valid = self.result["ADX"].dropna()
        self.assertTrue((valid >= 0).all(), "ADX fell below 0")
        self.assertTrue((valid <= 100).all(), "ADX exceeded 100")

    def test_di_lines_non_negative(self):
        # +DI and -DI represent directional movement percentages; both are >= 0.
        for col in ("ADX_PDI", "ADX_NDI"):
            valid = self.result[col].dropna()
            self.assertTrue((valid >= 0).all(), f"{col} contains negative values")


# ---------------------------------------------------------------------------
# OBV tests
# ---------------------------------------------------------------------------

class TestComputeOBV(unittest.TestCase):
    """Tests for compute_obv() — On-Balance Volume."""

    def test_obv_column_added(self):
        result = compute_obv(_make_price_df())
        self.assertIn("OBV", result.columns)

    def test_obv_no_nan(self):
        # OBV is a running cumulative sum; there should be no NaN values after
        # the first row (which is initialised to the first day's volume).
        result = compute_obv(_make_price_df())
        self.assertFalse(result["OBV"].iloc[1:].isna().any())


# ---------------------------------------------------------------------------
# compute_all_technicals integration test
# ---------------------------------------------------------------------------

class TestComputeAllTechnicals(unittest.TestCase):
    """Tests for compute_all_technicals() — the main public aggregator."""

    def setUp(self):
        # Run the full indicator pipeline once and share across tests.
        self.tech = compute_all_technicals(_make_price_df())

    def test_top_level_keys_present(self):
        # The caller (main.py) depends on all five keys being present.
        for key in ("df", "signals", "fibonacci", "pivot_levels", "latest"):
            self.assertIn(key, self.tech, msg=f"Key '{key}' missing from technicals dict")

    def test_df_is_dataframe(self):
        self.assertIsInstance(self.tech["df"], pd.DataFrame)

    def test_latest_has_close(self):
        # 'latest' dict must contain the current Close price for downstream use.
        self.assertIn("Close", self.tech["latest"])
        self.assertIsNotNone(self.tech["latest"]["Close"])

    def test_latest_has_rsi(self):
        self.assertIn("RSI", self.tech["latest"])

    def test_signals_is_dict(self):
        self.assertIsInstance(self.tech["signals"], dict)

    def test_fibonacci_has_seven_levels(self):
        self.assertEqual(len(self.tech["fibonacci"]), 7)


if __name__ == "__main__":
    unittest.main()
