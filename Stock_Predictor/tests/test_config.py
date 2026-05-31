"""
tests/test_config.py
--------------------
Unit tests for config.py.

WHY these tests exist:
  config.py is imported by every other module at startup. If a constant
  has the wrong type (e.g. RISK_FREE_RATE accidentally becomes a string
  because os.getenv was not wrapped in float()), downstream arithmetic
  silently breaks. These tests act as a cheap safety net that runs in
  milliseconds and requires no network access.

WHAT is tested:
  - Every exported constant has the expected Python type.
  - Numeric constants fall within a plausible real-world range so a
    mis-configured .env (e.g. RISK_FREE_RATE="52.5" instead of "0.0525")
    is caught early.
  - PREDICTION_HORIZONS contains exactly 8 entries, all with positive
    integer trading-day values, and the values increase monotonically
    (a later horizon must have more days than an earlier one).
"""

import unittest

import config


class TestConfigTypes(unittest.TestCase):
    """Verify that all config constants are the correct Python types.

    A type mismatch here would surface as a cryptic TypeError deep inside
    a statistical calculation rather than at startup, so we check types
    explicitly rather than relying on downstream tests to catch them.
    """

    def test_azure_openai_key_is_str(self):
        # Even when unset the constant must be a str (empty string), not None.
        self.assertIsInstance(config.AZURE_OPENAI_KEY, str)

    def test_azure_enabled_is_bool(self):
        # AZURE_ENABLED is derived from two string checks — confirm bool coercion.
        self.assertIsInstance(config.AZURE_ENABLED, bool)

    def test_google_enabled_is_bool(self):
        self.assertIsInstance(config.GOOGLE_ENABLED, bool)

    def test_llm_enabled_is_bool(self):
        # LLM_ENABLED must be True iff at least one provider is configured.
        self.assertIsInstance(config.LLM_ENABLED, bool)
        self.assertEqual(
            config.LLM_ENABLED,
            config.AZURE_ENABLED or config.GOOGLE_ENABLED or config.DEEPSEEK_ENABLED,
        )

    def test_deepseek_enabled_is_bool(self):
        self.assertIsInstance(config.DEEPSEEK_ENABLED, bool)

    def test_deepseek_api_key_is_str(self):
        self.assertIsInstance(config.DEEPSEEK_API_KEY, str)

    def test_deepseek_model_is_str(self):
        self.assertIsInstance(config.DEEPSEEK_MODEL, str)
        self.assertTrue(len(config.DEEPSEEK_MODEL) > 0)

    def test_deepseek_base_url_is_str(self):
        self.assertIsInstance(config.DEEPSEEK_BASE_URL, str)
        self.assertTrue(len(config.DEEPSEEK_BASE_URL) > 0)

    def test_deepseek_max_tokens_is_int(self):
        self.assertIsInstance(config.LLM_MAX_TOKENS_DEEPSEEK, int)
        self.assertGreater(config.LLM_MAX_TOKENS_DEEPSEEK, 0)

    def test_risk_free_rate_is_float(self):
        # Must be a float so it can be used directly in arithmetic (e.g.
        # excess_return = annual_return - RISK_FREE_RATE).
        self.assertIsInstance(config.RISK_FREE_RATE, float)

    def test_default_wacc_is_float(self):
        self.assertIsInstance(config.DEFAULT_WACC, float)

    def test_terminal_growth_rate_is_float(self):
        self.assertIsInstance(config.TERMINAL_GROWTH_RATE, float)

    def test_equity_risk_premium_is_float(self):
        self.assertIsInstance(config.EQUITY_RISK_PREMIUM, float)

    def test_monte_carlo_paths_is_int(self):
        self.assertIsInstance(config.MONTE_CARLO_PATHS, int)

    def test_ml_min_train_rows_is_int(self):
        self.assertIsInstance(config.ML_MIN_TRAIN_ROWS, int)

    def test_ml_retrain_days_is_int(self):
        self.assertIsInstance(config.ML_RETRAIN_DAYS, int)

    def test_prediction_horizons_is_dict(self):
        self.assertIsInstance(config.PREDICTION_HORIZONS, dict)

    def test_reports_dir_is_str(self):
        self.assertIsInstance(config.REPORTS_DIR, str)


class TestConfigValues(unittest.TestCase):
    """Verify that numeric constants are within plausible real-world bounds.

    These tests do NOT enforce a single correct value — they just guard
    against obviously wrong configurations (e.g. risk-free rate of 52.5 %
    because someone forgot to write 0.0525).
    """

    def test_risk_free_rate_plausible(self):
        # Risk-free rate should be between 0 % and 20 % annualised.
        self.assertGreater(config.RISK_FREE_RATE, 0.0)
        self.assertLess(config.RISK_FREE_RATE, 0.20)

    def test_default_wacc_plausible(self):
        # WACC proxy is typically 8–15 % for most equities.
        self.assertGreater(config.DEFAULT_WACC, 0.05)
        self.assertLess(config.DEFAULT_WACC, 0.25)

    def test_terminal_growth_rate_plausible(self):
        # Terminal growth must be less than WACC; typical range 1–5 %.
        self.assertGreater(config.TERMINAL_GROWTH_RATE, 0.0)
        self.assertLess(config.TERMINAL_GROWTH_RATE, config.DEFAULT_WACC)

    def test_equity_risk_premium_plausible(self):
        # Damodaran ERP estimates are typically 4–7 % for the US market.
        self.assertGreater(config.EQUITY_RISK_PREMIUM, 0.02)
        self.assertLess(config.EQUITY_RISK_PREMIUM, 0.15)

    def test_monte_carlo_paths_positive(self):
        # At least 100 paths are needed for P10/P90 to be meaningful.
        self.assertGreater(config.MONTE_CARLO_PATHS, 100)

    def test_ml_min_train_rows_positive(self):
        self.assertGreater(config.ML_MIN_TRAIN_ROWS, 0)

    def test_ml_retrain_days_positive(self):
        self.assertGreater(config.ML_RETRAIN_DAYS, 0)


class TestPredictionHorizons(unittest.TestCase):
    """Verify the structure and internal consistency of PREDICTION_HORIZONS.

    The pipeline uses this dict to drive all multi-horizon outputs (Monte
    Carlo predictions, LLM target prices, rule-based judge forecasts).  An
    incorrect entry here would silently produce wrong date-range labels.
    """

    def test_has_eight_entries(self):
        # Eight horizons are documented: 1W, 2W, 3W, 1M, 3M, 6M, 9M, 1Y.
        self.assertEqual(len(config.PREDICTION_HORIZONS), 8)

    def test_all_values_are_positive_ints(self):
        # Each value is a trading-day count used in array indexing — must be int > 0.
        for label, days in config.PREDICTION_HORIZONS.items():
            with self.subTest(label=label):
                self.assertIsInstance(days, int)
                self.assertGreater(days, 0)

    def test_values_are_monotonically_increasing(self):
        # A later horizon must always represent more trading days than an
        # earlier one. Violation would mean e.g. "6 Months" < "3 Months".
        values = list(config.PREDICTION_HORIZONS.values())
        for i in range(1, len(values)):
            self.assertGreater(
                values[i], values[i - 1],
                msg=f"Horizon values not monotonically increasing at index {i}",
            )

    def test_one_year_is_252_days(self):
        # 252 trading days per year is the standard convention in finance.
        self.assertEqual(config.PREDICTION_HORIZONS.get("1 Year"), 252)


if __name__ == "__main__":
    unittest.main()
