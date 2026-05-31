"""
tests/test_stock_fetcher.py
----------------------------
Unit tests for data/stock_fetcher.py.

WHY these tests exist:
  fetch_stock_data() is the entry point for all market data in the pipeline.
  It makes a live yfinance network call which we must NOT make during tests
  (slow, flaky, needs internet).  By mocking yf.Ticker we can verify:
    - The function returns (DataFrame, dict) for a valid ticker.
    - It raises ValueError when yfinance returns an empty DataFrame
      (how yfinance signals an unknown ticker).
    - The returned DataFrame has a 'Log_Return' column added by the function.

HOW mocking works here:
  We use unittest.mock.patch to replace 'data.stock_fetcher.yf.Ticker' with
  a MagicMock whose .history() method returns a pre-built synthetic DataFrame.
  The real yfinance library is never contacted.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from data.stock_fetcher import fetch_stock_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_history(n: int = 260) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame that mimics yfinance's output.

    WHY mimic yfinance: yfinance returns a DatetimeIndex DataFrame with
    columns Open, High, Low, Close, Volume.  Our fake must have the same
    structure so the function's downstream operations (dropna, log-return
    computation) behave identically to the real case.
    """
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 150.0 + np.cumsum(rng.normal(0, 1.5, size=n))
    noise = rng.uniform(0.005, 0.015, size=n)
    return pd.DataFrame(
        {
            "Open":   close * (1 + rng.uniform(-0.005, 0.005, size=n)),
            "High":   close * (1 + noise),
            "Low":    close * (1 - noise),
            "Close":  close,
            "Volume": rng.integers(1_000_000, 5_000_000, size=n).astype(float),
        },
        index=dates,
    )


def _make_fake_info() -> dict:
    """Return a minimal yfinance info dict for mocking purposes."""
    return {
        "shortName":    "Fake Corp",
        "sector":       "Technology",
        "currentPrice": 150.0,
        "trailingPE":   25.0,
        "beta":          1.1,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFetchStockData(unittest.TestCase):
    """Tests for fetch_stock_data() with yfinance mocked out."""

    @patch("data.stock_fetcher.yf.Ticker")
    def test_returns_dataframe_and_dict(self, mock_ticker_cls):
        # Configure the mock: .history() returns a valid synthetic DataFrame,
        # .info returns a minimal metadata dict.
        mock_instance = MagicMock()
        mock_instance.history.return_value = _make_fake_history()
        mock_instance.info = _make_fake_info()
        mock_ticker_cls.return_value = mock_instance

        price_df, info = fetch_stock_data("FAKE")

        self.assertIsInstance(price_df, pd.DataFrame, "price_df should be a DataFrame")
        self.assertIsInstance(info, dict, "info should be a dict")

    @patch("data.stock_fetcher.yf.Ticker")
    def test_log_return_column_added(self, mock_ticker_cls):
        # fetch_stock_data() must append a 'Log_Return' column before returning.
        # This column is consumed by every statistical model.
        mock_instance = MagicMock()
        mock_instance.history.return_value = _make_fake_history()
        mock_instance.info = _make_fake_info()
        mock_ticker_cls.return_value = mock_instance

        price_df, _ = fetch_stock_data("FAKE")
        self.assertIn("Log_Return", price_df.columns)

    @patch("data.stock_fetcher.yf.Ticker")
    def test_invalid_ticker_raises_value_error(self, mock_ticker_cls):
        # yfinance signals an invalid ticker by returning an empty DataFrame.
        # fetch_stock_data() must translate this into a ValueError with a
        # descriptive message so main.py can print it and exit cleanly.
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()  # empty = bad ticker
        mock_instance.info = {}
        mock_ticker_cls.return_value = mock_instance

        with self.assertRaises(ValueError):
            fetch_stock_data("INVALID_XYZ_TICKER")

    @patch("data.stock_fetcher.yf.Ticker")
    def test_required_ohlcv_columns_present(self, mock_ticker_cls):
        # The returned DataFrame must have the standard OHLCV columns that
        # the entire downstream pipeline depends on.
        mock_instance = MagicMock()
        mock_instance.history.return_value = _make_fake_history()
        mock_instance.info = _make_fake_info()
        mock_ticker_cls.return_value = mock_instance

        price_df, _ = fetch_stock_data("FAKE")
        for col in ("Open", "High", "Low", "Close", "Volume"):
            self.assertIn(col, price_df.columns, msg=f"Column '{col}' missing from price_df")

    @patch("data.stock_fetcher.yf.Ticker")
    def test_ticker_normalised_to_uppercase(self, mock_ticker_cls):
        # Tickers should be normalised so 'aapl' and 'AAPL' both work.
        mock_instance = MagicMock()
        mock_instance.history.return_value = _make_fake_history()
        mock_instance.info = _make_fake_info()
        mock_ticker_cls.return_value = mock_instance

        # Should not raise for a lowercase ticker.
        price_df, _ = fetch_stock_data("aapl")
        self.assertIsInstance(price_df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
