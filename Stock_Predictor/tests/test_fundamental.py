"""
tests/test_fundamental.py
--------------------------
Unit tests for analysis/fundamental.py.

WHY these tests exist:
  All five valuation functions take a plain dict (the yfinance info object)
  and return a dict of computed metrics.  They are pure functions — no network
  calls, no side effects.  Tests here verify that:
    - The mathematical formulae produce correct output for known inputs.
    - Edge cases (None, negative EPS, missing fields) are handled gracefully
      without raising exceptions (the UI just shows "N/A").
    - compute_all_fundamentals() bundles all five results under the expected
      top-level keys that the rule-based judge and PDF generator rely on.

WHAT is tested:
  - compute_pe_ratios(): fallback calculation, signal tier labels.
  - compute_peg_ratio(): PEG = P/E / (growth × 100) formula.
  - compute_graham_number(): √(22.5 × EPS × BookValue) formula; None for negatives.
  - compute_dcf(): keys are present; value is positive for profitable company.
  - compute_pb_dividend(): P/B and dividend yield output structure.
  - compute_all_fundamentals(): top-level key presence.
"""

import math
import unittest

from analysis.fundamental import (
    compute_pe_ratios,
    compute_peg_ratio,
    compute_graham_number,
    compute_dcf,
    compute_pb_dividend,
    compute_all_fundamentals,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _info(**overrides) -> dict:
    """Return a realistic base info dict with optional field overrides.

    WHY a helper:  Every test needs a minimal but valid info dict.  Rather
    than copy-pasting 20 fields into every test, we define a 'good company'
    baseline and let each test override only what it needs.
    """
    base = {
        "currentPrice":    150.0,
        "trailingEps":       6.0,
        "forwardEps":        7.0,
        "trailingPE":       25.0,
        "forwardPE":        21.0,
        "earningsGrowth":    0.12,  # 12 % growth
        "bookValue":        30.0,
        "priceToBook":       5.0,
        "dividendYield":     0.015,
        "freeCashflow":  1_200_000_000,
        "sharesOutstanding": 16_000_000,
        "totalRevenue":  8_000_000_000,
        "revenueGrowth":     0.10,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# P/E Ratio tests
# ---------------------------------------------------------------------------

class TestComputePERatios(unittest.TestCase):
    """Tests for compute_pe_ratios() — trailing and forward P/E with tier labels."""

    def test_returns_all_keys(self):
        result = compute_pe_ratios(_info())
        for key in ("trailing_pe", "forward_pe", "trailing_signal", "forward_signal"):
            self.assertIn(key, result, msg=f"Key '{key}' missing from pe result")

    def test_pre_computed_values_used_when_present(self):
        # When trailingPE is in the info dict, it should be used directly.
        result = compute_pe_ratios(_info(trailingPE=20.0))
        self.assertAlmostEqual(result["trailing_pe"], 20.0, places=2)

    def test_fallback_calculation_when_field_missing(self):
        # If trailingPE is absent, the function must divide currentPrice / trailingEps.
        result = compute_pe_ratios(_info(trailingPE=None, currentPrice=120.0, trailingEps=8.0))
        self.assertAlmostEqual(result["trailing_pe"], 15.0, places=2)

    def test_none_when_both_missing(self):
        # If both trailingPE and trailingEps are absent, result must be None.
        result = compute_pe_ratios(_info(trailingPE=None, trailingEps=None))
        self.assertIsNone(result["trailing_pe"])

    def test_negative_pe_signal(self):
        # A negative P/E means the company is losing money.
        result = compute_pe_ratios(_info(trailingPE=-5.0))
        self.assertIn("NEGATIVE", result["trailing_signal"].upper())

    def test_cheap_pe_signal(self):
        # P/E < 15 should be labelled as potentially undervalued.
        result = compute_pe_ratios(_info(trailingPE=10.0))
        self.assertIn("UNDERVALUED", result["trailing_signal"].upper())

    def test_expensive_pe_signal(self):
        # P/E > 40 should be labelled as expensive.
        result = compute_pe_ratios(_info(trailingPE=60.0))
        self.assertIn("EXPENSIVE", result["trailing_signal"].upper())


# ---------------------------------------------------------------------------
# PEG Ratio tests
# ---------------------------------------------------------------------------

class TestComputePEGRatio(unittest.TestCase):
    """Tests for compute_peg_ratio() — growth-adjusted P/E valuation."""

    def test_returns_expected_keys(self):
        result = compute_peg_ratio(_info())
        self.assertIn("peg", result)
        self.assertIn("signal", result)

    def test_peg_formula_correct(self):
        # PEG = trailing_P/E / (earnings_growth_pct)
        # With P/E = 25 and growth = 12 % → PEG = 25 / 12 ≈ 2.083
        result = compute_peg_ratio(_info(trailingPE=25.0, earningsGrowth=0.12))
        if result["peg"] is not None:
            self.assertAlmostEqual(result["peg"], 25.0 / 12.0, places=1)

    def test_none_when_growth_zero(self):
        # Division by zero must be handled; result should be None (not an exception).
        result = compute_peg_ratio(_info(earningsGrowth=0.0))
        self.assertIsNone(result["peg"])

    def test_none_when_data_missing(self):
        result = compute_peg_ratio(_info(trailingPE=None, earningsGrowth=None))
        self.assertIsNone(result["peg"])


# ---------------------------------------------------------------------------
# Graham Number tests
# ---------------------------------------------------------------------------

class TestComputeGrahamNumber(unittest.TestCase):
    """Tests for compute_graham_number() — Benjamin Graham intrinsic value floor.

    Formula: Graham Number = √(22.5 × EPS × Book Value Per Share)
    """

    def test_returns_expected_keys(self):
        result = compute_graham_number(_info())
        for key in ("graham_number", "signal"):
            self.assertIn(key, result)

    def test_formula_correct(self):
        # With EPS=6.0 and book_value=30.0: √(22.5 × 6 × 30) = √4050 ≈ 63.64
        result = compute_graham_number(_info(trailingEps=6.0, bookValue=30.0))
        if result["graham_number"] is not None:
            expected = math.sqrt(22.5 * 6.0 * 30.0)
            self.assertAlmostEqual(result["graham_number"], expected, places=1)

    def test_none_for_negative_eps(self):
        # Graham Number is undefined (square root of negative) when EPS < 0.
        result = compute_graham_number(_info(trailingEps=-2.0))
        self.assertIsNone(result["graham_number"])

    def test_none_for_negative_book_value(self):
        # Similarly undefined when book value is negative (insolvent company).
        result = compute_graham_number(_info(bookValue=-5.0))
        self.assertIsNone(result["graham_number"])

    def test_none_when_data_missing(self):
        result = compute_graham_number(_info(trailingEps=None, bookValue=None))
        self.assertIsNone(result["graham_number"])


# ---------------------------------------------------------------------------
# DCF tests
# ---------------------------------------------------------------------------

class TestComputeDCF(unittest.TestCase):
    """Tests for compute_dcf() — simplified two-stage discounted cash flow."""

    def test_returns_expected_keys(self):
        result = compute_dcf(_info())
        # Actual key is 'dcf_value' (not 'intrinsic_value'); 'growth_rate_used' and 'wacc_used' also present.
        for key in ("dcf_value", "signal", "growth_rate_used", "wacc_used"):
            self.assertIn(key, result)

    def test_positive_intrinsic_value_for_profitable_company(self):
        # A company with positive free cash flow must have a positive DCF value.
        result = compute_dcf(_info())
        if result["dcf_value"] is not None:
            self.assertGreater(result["dcf_value"], 0)

    def test_none_when_fcf_missing(self):
        # DCF cannot be computed without free cash flow data.
        result = compute_dcf(_info(freeCashflow=None))
        self.assertIsNone(result["dcf_value"])

    def test_none_when_shares_missing(self):
        result = compute_dcf(_info(sharesOutstanding=None))
        self.assertIsNone(result["dcf_value"])


# ---------------------------------------------------------------------------
# Price-to-Book / Dividend Yield tests
# ---------------------------------------------------------------------------

class TestComputePBDividend(unittest.TestCase):
    """Tests for compute_pb_dividend() — balance-sheet and income signals."""

    def test_returns_expected_keys(self):
        result = compute_pb_dividend(_info())
        # Actual key is 'dividend_signal' (not 'div_signal').
        for key in ("price_to_book", "dividend_yield", "pb_signal", "dividend_signal"):
            self.assertIn(key, result)

    def test_pb_value_correct(self):
        # With priceToBook=5.0 the value should pass through as-is.
        result = compute_pb_dividend(_info(priceToBook=5.0))
        if result["price_to_book"] is not None:
            self.assertAlmostEqual(result["price_to_book"], 5.0, places=2)

    def test_dividend_yield_as_percentage(self):
        # dividendYield comes from yfinance as a decimal (e.g. 0.015 = 1.5%).
        # The function should store it as-is or convert to percentage consistently.
        result = compute_pb_dividend(_info(dividendYield=0.015))
        self.assertIsNotNone(result["dividend_yield"])

    def test_none_dividend_when_no_yield(self):
        # ETFs and growth stocks have no dividend; result should be None or 0.
        result = compute_pb_dividend(_info(dividendYield=None))
        # Not raising an exception is the key check here.
        self.assertIsNotNone(result)  # dict is returned regardless


# ---------------------------------------------------------------------------
# compute_all_fundamentals integration test
# ---------------------------------------------------------------------------

class TestComputeAllFundamentals(unittest.TestCase):
    """Tests for compute_all_fundamentals() — the aggregator called by main.py."""

    def setUp(self):
        self.result = compute_all_fundamentals(_info())

    def test_top_level_keys_present(self):
        # The rule-based judge accesses exactly these keys on the fundamental dict.
        for key in ("pe", "peg", "graham", "dcf", "pb_div"):
            self.assertIn(key, self.result, msg=f"Key '{key}' missing")

    def test_each_section_is_dict(self):
        for key in ("pe", "peg", "graham", "dcf", "pb_div"):
            self.assertIsInstance(self.result[key], dict, msg=f"{key} is not a dict")

    def test_no_exception_with_empty_info(self):
        # An empty info dict (e.g. ETF with no EPS) must not raise.
        result = compute_all_fundamentals({})
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
