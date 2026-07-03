"""
tests/test_analyst_fetcher.py
------------------------------
Unit tests for data/analyst_fetcher.py.

WHY these tests exist:
  fetch_analyst_data() makes several separate yfinance API calls (info,
  recommendations, analyst_price_targets, news, earnings_dates).  Each call
  can fail independently on a live network.  The function is designed to
  return partial data gracefully rather than raising.  These tests verify:
    - All expected top-level keys are always present in the returned dict.
    - When yfinance raises an exception (simulating a network error or an
      ETF ticker with no analyst coverage), the function still returns a
      safe default dict rather than propagating the exception.

HOW mocking works here:
  We patch 'data.market_data.yf' (the process-wide cache layer every
  fetcher routes through) to control what yfinance returns.
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from data.analyst_fetcher import fetch_analyst_data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFetchAnalystData(unittest.TestCase):
    """Tests for fetch_analyst_data() with yfinance network calls mocked."""

    def setUp(self):
        # market_data memoises per-symbol results for the whole process;
        # clear it so each test's mock configuration takes effect.
        from data import market_data
        market_data.clear_cache()

    # Required keys that every caller (display, report, PDF) reads from the dict.
    REQUIRED_KEYS = (
        "recommendation_key",
        "price_target",
        "news",
        "earnings_dates",
    )

    def _make_mock_ticker(self, raise_on_info: bool = False) -> MagicMock:
        """Build a minimal yfinance Ticker mock.

        WHY a helper: avoids copy-pasting the MagicMock configuration across
        every test method.  Pass raise_on_info=True to simulate a failure
        in the analyst data endpoint (e.g. the ticker is an ETF).
        """
        mock = MagicMock()
        if raise_on_info:
            # Simulate yfinance raising a 404-equivalent for an ETF.
            type(mock).info = property(lambda self: (_ for _ in ()).throw(Exception("No data")))
        else:
            mock.info = {
                "recommendationKey": "buy",
                "currentPrice": 150.0,
            }
        # Return a minimal recommendations DataFrame (columns must match yfinance format).
        mock.recommendations = pd.DataFrame(
            {"period": ["0m", "-1m"], "strongBuy": [5, 3], "buy": [10, 8],
             "hold": [4, 5], "sell": [1, 2], "strongSell": [0, 0]},
        )
        mock.analyst_price_targets = {"mean": 175.0, "low": 150.0, "high": 200.0,
                                       "numberOfAnalysts": 15}
        mock.news = [{"title": "Company beats earnings", "link": "http://example.com"}]
        mock.earnings_dates = None
        return mock

    @patch("data.market_data.yf.download")
    @patch("data.market_data.yf.Ticker")
    def test_all_required_keys_present(self, mock_ticker_cls, _mock_download):
        # Normal happy-path: all fields returned by yfinance.
        mock_ticker_cls.return_value = self._make_mock_ticker()
        result = fetch_analyst_data("FAKE")
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, result, msg=f"Key '{key}' missing from analyst_data")

    @patch("data.market_data.yf.download")
    @patch("data.market_data.yf.Ticker")
    def test_graceful_failure_still_returns_dict(self, mock_ticker_cls, _mock_download):
        # Even when the .info property raises (simulating ETF / no coverage),
        # the function must return a dict — never propagate the exception.
        mock_ticker_cls.return_value = self._make_mock_ticker(raise_on_info=True)
        try:
            result = fetch_analyst_data("ETFLIKE")
            self.assertIsInstance(result, dict)
        except Exception as exc:
            self.fail(
                f"fetch_analyst_data raised an unexpected exception: {exc!r}"
            )

    @patch("data.market_data.yf.download")
    @patch("data.market_data.yf.Ticker")
    def test_news_is_list(self, mock_ticker_cls, _mock_download):
        # Downstream code iterates over news with a for-loop;
        # it must always be a list (never None).
        mock_ticker_cls.return_value = self._make_mock_ticker()
        result = fetch_analyst_data("FAKE")
        self.assertIsInstance(result.get("news"), list)

    @patch("data.market_data.yf.download")
    @patch("data.market_data.yf.Ticker")
    def test_recommendation_key_is_string(self, mock_ticker_cls, _mock_download):
        # The recommendation_key is used as a display string and for mapping
        # to colour codes — it must always be a str.
        mock_ticker_cls.return_value = self._make_mock_ticker()
        result = fetch_analyst_data("FAKE")
        self.assertIsInstance(result.get("recommendation_key"), str)


if __name__ == "__main__":
    unittest.main()
