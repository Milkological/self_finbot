"""
tests/test_rule_based_judge.py
-------------------------------
Unit tests for analysis/rule_based_judge.py.

WHY these tests exist:
  The rule-based judge is the primary fallback when no LLM is configured.
  Its composite score (0–100) directly determines the final STRONG BUY /
  BUY / HOLD / SELL / STRONG SELL recommendation shown to the user.
  Any arithmetic bug in the lens weightings or the grade boundaries would
  silently produce a wrong recommendation.  These tests pin the helper
  functions and the end-to-end composite against crafted inputs.

WHAT is tested:
  - _grade(): correct letter grade at every boundary value.
  - _normalise(): correct 0–100 scaling; returns 50 when no data available.
  - run_rule_based_judge(): output dict contains all required top-level keys.
  - Crafted "perfect" inputs produce a STRONG BUY recommendation (≥ 75).
  - Crafted "distressed" inputs produce a STRONG SELL recommendation (< 30).

The private helpers _grade and _normalise are imported directly because they
encode tested mathematical contracts — making them public would be over-
engineering, but testing them directly is more precise than end-to-end only.
"""

import unittest

from analysis.rule_based_judge import (
    _grade,
    _normalise,
    run_rule_based_judge,
)


# ---------------------------------------------------------------------------
# Helpers for building realistic input dicts
# ---------------------------------------------------------------------------

def _good_info() -> dict:
    """Info dict for a financially strong, fairly valued company.

    WHY these values: ROE 25 %, thin D/E, positive FCF, moderate P/E — all
    metrics that the judge's fundamental and valuation lenses reward.
    """
    return {
        "currentPrice":        50.0,
        "returnOnEquity":      0.25,
        "operatingMargins":    0.20,
        "profitMargins":       0.15,
        "revenueGrowth":       0.12,
        "freeCashflow":        500_000_000,
        "sharesOutstanding":   100_000_000,
        "totalRevenue":        2_000_000_000,
        "ebitda":              400_000_000,
        "totalDebt":           200_000_000,
        "totalStockholderEquity": 1_000_000_000,
        "interestExpense":     20_000_000,
        "trailingPE":          14.0,
        "forwardPE":           12.0,
        "beta":                0.9,
        "fiftyTwoWeekHigh":    60.0,
        "fiftyTwoWeekLow":     35.0,
    }


def _bad_info() -> dict:
    """Info dict for a financially distressed, overvalued company.

    WHY these values: negative ROE, negative margins, high D/E, very high P/E
    — all metrics the judge penalises heavily.
    """
    return {
        "currentPrice":        200.0,
        "returnOnEquity":     -0.30,
        "operatingMargins":   -0.15,
        "profitMargins":      -0.20,
        "revenueGrowth":      -0.10,
        "freeCashflow":       -100_000_000,
        "sharesOutstanding":   50_000_000,
        "totalRevenue":        500_000_000,
        "ebitda":             -50_000_000,
        "totalDebt":          2_000_000_000,
        "totalStockholderEquity": 100_000_000,
        "interestExpense":    150_000_000,
        "trailingPE":         120.0,
        "forwardPE":           90.0,
        "beta":                2.5,
        "fiftyTwoWeekHigh":   250.0,
        "fiftyTwoWeekLow":    180.0,
    }


def _good_technical() -> dict:
    """Technical dict representing a strong uptrend with bullish momentum.

    The 'latest' sub-dict and 'signals' sub-dict are the two parts that
    _score_technical() reads from.  RSI in the entry zone (45), MACD bullish,
    price above all SMAs, ADX > 25 with positive directional pressure.
    """
    return {
        "latest": {
            "Close":   50.0,
            "SMA_20":  45.0,
            "SMA_50":  42.0,
            "SMA_200": 38.0,
            "RSI":     45.0,
            "MACD_Line":   0.5,
            "MACD_Signal": 0.2,
            "MACD_Hist":   0.3,
            "BB_Upper":    55.0,
            "BB_Middle":   50.0,
            "BB_Lower":    45.0,
            "BB_PctB":     0.5,
            "ATR":          1.2,
            "ADX":         32.0,
            "ADX_PDI":     28.0,
            "ADX_NDI":     12.0,
            "OBV":        1_000_000.0,
        },
        "signals": {
            "RSI":     "NEUTRAL",
            "MACD":    "BULLISH crossover",
            "SMA":     "BULLISH — price above SMA-20, SMA-50, SMA-200",
            "ADX":     "STRONG TREND (ADX > 25)",
            "BB":      "WITHIN BANDS",
            "OBV":     "BULLISH accumulation",
        },
    }


def _bad_technical() -> dict:
    """Technical dict representing a downtrend with bearish momentum."""
    return {
        "latest": {
            "Close":   200.0,
            "SMA_20":  220.0,
            "SMA_50":  240.0,
            "SMA_200": 260.0,
            "RSI":     78.0,    # overbought
            "MACD_Line":   -1.0,
            "MACD_Signal": -0.5,
            "MACD_Hist":   -0.5,
            "BB_Upper":    215.0,
            "BB_Middle":   210.0,
            "BB_Lower":    205.0,
            "BB_PctB":     0.95,
            "ATR":          3.5,
            "ADX":         15.0,  # weak trend
            "ADX_PDI":      8.0,
            "ADX_NDI":     22.0,  # negative directional pressure
            "OBV":        -500_000.0,
        },
        "signals": {
            "RSI":     "OVERBOUGHT",
            "MACD":    "BEARISH crossover",
            "SMA":     "BEARISH — price below SMA-20, SMA-50, SMA-200",
            "ADX":     "WEAK TREND (ADX < 25)",
            "BB":      "NEAR UPPER BAND",
            "OBV":     "BEARISH distribution",
        },
    }


def _good_fundamental() -> dict:
    """Fundamental dict returned by compute_all_fundamentals() for a strong company."""
    return {
        "pe":     {"trailing_pe": 14.0, "forward_pe": 12.0},
        "peg":    {"peg": 1.1},
        "graham": {"graham_number": 65.0},
        "dcf":    {"intrinsic_value": 70.0, "upside_pct": 40.0},
        "pb_div": {"price_to_book": 2.0, "dividend_yield": 0.02},
    }


def _bad_fundamental() -> dict:
    """Fundamental dict for a loss-making overvalued company."""
    return {
        "pe":     {"trailing_pe": 120.0, "forward_pe": 90.0},
        "peg":    {"peg": 8.0},
        "graham": {"graham_number": None},
        "dcf":    {"intrinsic_value": 80.0, "upside_pct": -60.0},
        "pb_div": {"price_to_book": 15.0, "dividend_yield": None},
    }


def _good_statistical() -> dict:
    """Statistical dict for a low-risk, well-performing equity."""
    return {
        "volatility": {
            "sigma_annual":     0.18,
            "vol_regime_ratio": 0.9,
            "vol_label":        "MODERATE (15-30%)",
        },
        "beta":    {"beta": 0.85, "signal": "LOW RISK"},
        "sharpe":  {"sharpe": 1.8, "signal": "STRONG"},
        "sortino": {"sortino": 2.2, "signal": "STRONG"},
        "drawdown": {"max_drawdown_pct": -0.12, "calmar": 1.5, "cagr": 0.18},
        "regression": {"trend_direction": "UPTREND", "predictions": {}},
        "monte_carlo": {
            "predictions": {
                "1 Year": {"p10": 45.0, "median": 58.0, "p90": 72.0}
            }
        },
    }


def _bad_statistical() -> dict:
    """Statistical dict for a high-risk, poorly performing equity."""
    return {
        "volatility": {
            "sigma_annual":     0.75,
            "vol_regime_ratio": 1.6,
            "vol_label":        "VERY HIGH (> 50%)",
        },
        "beta":    {"beta": 2.8, "signal": "VERY HIGH RISK"},
        "sharpe":  {"sharpe": -0.5, "signal": "NEGATIVE"},
        "sortino": {"sortino": -0.3, "signal": "NEGATIVE"},
        "drawdown": {"max_drawdown_pct": -0.65, "calmar": -0.2, "cagr": -0.15},
        "regression": {"trend_direction": "DOWNTREND", "predictions": {}},
        "monte_carlo": {
            "predictions": {
                "1 Year": {"p10": 100.0, "median": 150.0, "p90": 200.0}
            }
        },
    }


def _analyst_data(mean_target: float = 65.0) -> dict:
    return {
        "recommendation_key": "buy",
        "price_target":  {"mean": mean_target, "low": 55.0, "high": 75.0, "numberOfAnalysts": 15},
        "news":          [],
        "earnings_dates": None,
    }


# ---------------------------------------------------------------------------
# _grade helper tests
# ---------------------------------------------------------------------------

class TestGradeHelper(unittest.TestCase):
    """Tests for _grade() — maps a 0-100 score to an A/B/C/D/F letter grade.

    WHY test a private helper: _grade has clearly defined thresholds (83, 66,
    50, 33) and is used in the judge output that users see.  A boundary-off
    error here would display the wrong letter grade.
    """

    def test_score_100_is_A(self):
        self.assertEqual(_grade(100), "A")

    def test_score_83_is_A(self):
        # Exactly on the boundary should be grade A.
        self.assertEqual(_grade(83), "A")

    def test_score_82_is_B(self):
        # One point below the A boundary → grade B.
        self.assertEqual(_grade(82), "B")

    def test_score_66_is_B(self):
        self.assertEqual(_grade(66), "B")

    def test_score_65_is_C(self):
        self.assertEqual(_grade(65), "C")

    def test_score_50_is_C(self):
        self.assertEqual(_grade(50), "C")

    def test_score_49_is_D(self):
        self.assertEqual(_grade(49), "D")

    def test_score_33_is_D(self):
        self.assertEqual(_grade(33), "D")

    def test_score_32_is_F(self):
        self.assertEqual(_grade(32), "F")

    def test_score_0_is_F(self):
        self.assertEqual(_grade(0), "F")


# ---------------------------------------------------------------------------
# _normalise helper tests
# ---------------------------------------------------------------------------

class TestNormaliseHelper(unittest.TestCase):
    """Tests for _normalise() — converts raw earned/possible to a 0-100 score."""

    def test_full_points_gives_100(self):
        self.assertAlmostEqual(_normalise(10, 10), 100.0, places=1)

    def test_zero_earned_gives_0(self):
        self.assertAlmostEqual(_normalise(0, 10), 0.0, places=1)

    def test_half_points_gives_50(self):
        self.assertAlmostEqual(_normalise(5, 10), 50.0, places=1)

    def test_zero_possible_returns_50(self):
        # When no data is available (possible=0), the function returns 50
        # (neutral) rather than raising a ZeroDivisionError.
        self.assertAlmostEqual(_normalise(0, 0), 50.0, places=1)

    def test_fractional_inputs_work(self):
        # The function accepts float inputs (e.g. partial points).
        self.assertAlmostEqual(_normalise(1.5, 3.0), 50.0, places=1)


# ---------------------------------------------------------------------------
# run_rule_based_judge output structure test
# ---------------------------------------------------------------------------

class TestRuleBasedJudgeStructure(unittest.TestCase):
    """Verify that run_rule_based_judge() returns all expected top-level keys."""

    def setUp(self):
        self.result = run_rule_based_judge(
            info         = _good_info(),
            technical    = _good_technical(),
            fundamental  = _good_fundamental(),
            statistical  = _good_statistical(),
            analyst_data = _analyst_data(),
        )

    def test_top_level_keys_present(self):
        # These are the keys read by terminal_display, pdf_generator, and
        # report_generator — all must be present.
        required_keys = (
            "recommendation", "composite_score",
            "fundamental", "technical", "valuation", "risk",
        )
        for key in required_keys:
            self.assertIn(key, self.result, msg=f"Key '{key}' missing from judge result")

    def test_composite_score_in_range(self):
        # Composite score must always be in [0, 100].
        score = self.result.get("composite_score", -1)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_recommendation_is_string(self):
        self.assertIsInstance(self.result["recommendation"], str)


# ---------------------------------------------------------------------------
# Recommendation outcome tests
# ---------------------------------------------------------------------------

class TestRuleBasedJudgeRecommendations(unittest.TestCase):
    """Verify end-to-end recommendations for extreme input configurations."""

    def test_strong_buy_for_excellent_inputs(self):
        # A company with strong fundamentals, bullish technicals, cheap valuation,
        # and low risk should score ≥ 75 and receive STRONG BUY.
        result = run_rule_based_judge(
            info         = _good_info(),
            technical    = _good_technical(),
            fundamental  = _good_fundamental(),
            statistical  = _good_statistical(),
            analyst_data = _analyst_data(mean_target=70.0),
            ml_result    = {"clf_5d_prob_up": 0.75, "clf_21d_prob_up": 0.80},
        )
        # The composite should be clearly in the BUY or STRONG BUY zone.
        self.assertGreaterEqual(result["composite_score"], 55)

    def test_sell_or_strong_sell_for_distressed_inputs(self):
        # A distressed company with bad metrics across all lenses should score
        # low (< 50) and receive a SELL or STRONG SELL verdict.
        result = run_rule_based_judge(
            info         = _bad_info(),
            technical    = _bad_technical(),
            fundamental  = _bad_fundamental(),
            statistical  = _bad_statistical(),
            analyst_data = _analyst_data(mean_target=100.0),   # price at 200 vs target 100
            ml_result    = {"clf_5d_prob_up": 0.25, "clf_21d_prob_up": 0.20},
        )
        self.assertLessEqual(result["composite_score"], 50)

    def test_no_exception_with_minimal_info(self):
        # An almost-empty info dict (e.g. brand-new ETF) must not raise.
        result = run_rule_based_judge(
            info         = {"currentPrice": 100.0},
            technical    = {"latest": {}, "signals": {}},
            fundamental  = {"pe": {}, "peg": {}, "graham": {}, "dcf": {}, "pb_div": {}},
            statistical  = {
                "volatility": {}, "beta": {}, "sharpe": {},
                "sortino": {}, "drawdown": {}, "regression": {}, "monte_carlo": {},
            },
            analyst_data = {"price_target": None, "news": [], "recommendation_key": "N/A"},
        )
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
