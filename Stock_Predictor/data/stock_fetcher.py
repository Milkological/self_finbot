"""
data/stock_fetcher.py — Fetches price history and fundamental data for a ticker.

Data source: Yahoo Finance via the `yfinance` library (free, no API key needed).

Returns two artefacts:
  • price_df  — pandas DataFrame with OHLCV columns + daily log-returns
  • info_dict — flat dict of fundamental / company metadata fields
"""

import yfinance as yf
import pandas as pd
import numpy as np
from config import HISTORY_PERIOD


def fetch_stock_data(ticker: str) -> tuple[pd.DataFrame, dict]:
    """
    Download historical OHLCV prices and fundamental info for *ticker*.

    Parameters
    ----------
    ticker : str
        Valid Yahoo Finance ticker symbol (e.g. "AAPL", "MSFT", "TSLA").

    Returns
    -------
    price_df : pd.DataFrame
        Daily OHLCV with an additional 'Log_Return' column.
        Index is DatetimeIndex (UTC-normalised).
    info : dict
        Flat dictionary of ~100+ fundamental fields from yf.Ticker.info.
        Keys of particular interest include:
          shortName, sector, industry, marketCap,
          trailingPE, forwardPE, trailingEps, forwardEps,
          bookValue, priceToBook, dividendYield,
          totalRevenue, grossProfits, freeCashflow,
          revenueGrowth, earningsGrowth, returnOnEquity, beta,
          fiftyTwoWeekHigh, fiftyTwoWeekLow, currentPrice / regularMarketPrice

    Raises
    ------
    ValueError
        If the ticker is not found or history is empty (invalid symbol).
    """

    stock = yf.Ticker(ticker)

    # ------------------------------------------------------------------ #
    # 1. Price history
    # ------------------------------------------------------------------ #
    # HISTORY_PERIOD is set in config.py (default "2y").
    # We use 'auto_adjust=True' so dividends / splits are baked in,
    # which is the standard in quantitative work to avoid spurious drops.
    price_df: pd.DataFrame = stock.history(period=HISTORY_PERIOD, auto_adjust=True)

    if price_df.empty:
        raise ValueError(
            f"No price data returned for '{ticker}'. "
            "Check that the ticker symbol is valid on Yahoo Finance."
        )

    # Normalise column names to title-case so every downstream module
    # can safely reference price_df['Close'], price_df['Volume'], etc.
    idx = pd.to_datetime(price_df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    price_df.index = idx
    price_df = price_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    price_df.dropna(inplace=True)

    # Log-return: ln(P_t / P_{t-1})
    # Used in volatility, Sharpe Ratio, and GBM drift calculations.
    # Formula source: standard continuous compounding return definition.
    price_df["Log_Return"] = np.log(price_df["Close"] / price_df["Close"].shift(1))

    # ------------------------------------------------------------------ #
    # 2. Fundamental / company info
    # ------------------------------------------------------------------ #
    # ETFs (e.g. VT, SPY, QQQ) don't have a quoteSummary on Yahoo Finance,
    # so stock.info raises an HTTP 404. Fall back to an empty dict so the
    # rest of the pipeline degrades gracefully with N/A fundamental metrics.
    try:
        info: dict = stock.info  # dict with 100+ fields; missing fields → None
    except Exception:
        info = {}

    # Provide safe fallbacks for critical fields so downstream code can
    # always access them without KeyError / None-check boilerplate.
    defaults = {
        "shortName": ticker.upper(),
        "sector": "N/A",
        "industry": "N/A",
        "marketCap": None,
        "trailingPE": None,
        "forwardPE": None,
        "trailingEps": None,
        "forwardEps": None,
        "bookValue": None,
        "priceToBook": None,
        "dividendYield": None,
        "totalRevenue": None,
        "grossProfits": None,
        "freeCashflow": None,
        "revenueGrowth": None,
        "earningsGrowth": None,
        "returnOnEquity": None,
        "beta": None,
        "fiftyTwoWeekHigh": None,
        "fiftyTwoWeekLow": None,
        "currentPrice": None,
        "regularMarketPrice": None,
    }
    for key, default in defaults.items():
        info.setdefault(key, default)

    # Resolve current price: prefer currentPrice, fall back to last Close
    if not info["currentPrice"] and not info["regularMarketPrice"]:
        info["currentPrice"] = float(price_df["Close"].iloc[-1])
    elif not info["currentPrice"]:
        info["currentPrice"] = info["regularMarketPrice"]

    return price_df, info


def fetch_options_data(ticker: str) -> dict:
    """
    Fetch near-term options data for *ticker* via yfinance.

    Returns a dict with:
        put_call_ratio : float | None  — total put volume / total call volume
                         on the nearest available expiry.
                         >1 = net bearish flow, <1 = net bullish flow.
        options_iv_avg : float | None  — average implied volatility across
                         all near-term option contracts (calls + puts).

    Returns empty values on any error (e.g. ETFs with no options chain).
    """
    result = {"put_call_ratio": None, "options_iv_avg": None}
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return result

        # Use the nearest expiry with available data
        chain = stock.option_chain(expirations[0])
        calls = chain.calls
        puts  = chain.puts

        # Put/Call ratio by volume
        call_vol = calls["volume"].fillna(0).sum()
        put_vol  = puts["volume"].fillna(0).sum()
        if call_vol > 0:
            result["put_call_ratio"] = round(float(put_vol / call_vol), 3)

        # Average implied volatility across both sides
        iv_values = pd.concat([
            calls["impliedVolatility"].dropna(),
            puts["impliedVolatility"].dropna(),
        ])
        if len(iv_values) > 0:
            result["options_iv_avg"] = round(float(iv_values.mean()), 4)

    except Exception:
        pass  # Graceful degradation — options data is supplementary

    return result
