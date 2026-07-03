"""
data/market_data.py — Process-wide, thread-safe cache for yfinance data.

Every network-touching accessor in the pipeline routes through this
module so each fact is fetched from Yahoo Finance at most once per
process, no matter how many modules ask for it:

  • get_ticker()          — one yf.Ticker object per symbol
  • get_history()         — OHLCV history per (symbol, period)
  • get_info()            — .info dict (returns a copy; {} on failure)
  • get_news()            — raw news list
  • get_earnings_dates()  — earnings calendar DataFrame
  • get_vix()             — ^VIX close series (single wide fetch, sliced)
  • get_spy()             — SPY benchmark history (for Beta)
  • prewarm_symbol()      — batch-mode helper: pre-fetch a symbol's data
                            from a worker thread so the sequential
                            pipeline afterwards runs from cache.

Before this module existed a single run fetched .info up to 3×, VIX 2×,
and earnings dates 2×; batch runs repeated VIX for every ticker.
"""

import datetime
import logging
import threading

import pandas as pd
import yfinance as yf

from config import BENCHMARK_TICKER

logger = logging.getLogger(__name__)

# One lock for the per-symbol memo dicts (cheap dict ops only — the
# prewarm executor works on distinct symbols so fetches don't collide),
# plus dedicated locks held across the shared VIX / SPY downloads so
# concurrent workers can't trigger duplicate fetches of those.
_lock      = threading.Lock()
_vix_lock  = threading.Lock()
_spy_lock  = threading.Lock()

_tickers:  dict[str, yf.Ticker]        = {}
_history:  dict[tuple, pd.DataFrame]   = {}
_info:     dict[str, dict]             = {}
_news:     dict[str, list]             = {}
_earnings: dict[str, "pd.DataFrame | None"] = {}

_vix_series: "pd.Series | None"   = None
_spy_df:     "pd.DataFrame | None" = None


def clear_cache() -> None:
    """Empty every memo (used by tests and long-lived callers)."""
    global _vix_series, _spy_df
    with _lock:
        _tickers.clear()
        _history.clear()
        _info.clear()
        _news.clear()
        _earnings.clear()
    with _vix_lock:
        _vix_series = None
    with _spy_lock:
        _spy_df = None


# ------------------------------------------------------------------ #
# Per-symbol accessors
# ------------------------------------------------------------------ #

def get_ticker(symbol: str) -> yf.Ticker:
    """Return the process-wide yf.Ticker instance for *symbol*."""
    symbol = symbol.upper().strip()
    with _lock:
        if symbol not in _tickers:
            _tickers[symbol] = yf.Ticker(symbol)
        return _tickers[symbol]


def get_history(symbol: str, period: str) -> pd.DataFrame:
    """
    Return raw OHLCV history for *symbol* (auto-adjusted).
    Cached per (symbol, period). Callers must .copy() before mutating.
    """
    symbol = symbol.upper().strip()
    key = (symbol, period)
    with _lock:
        cached = _history.get(key)
    if cached is not None:
        return cached
    df = get_ticker(symbol).history(period=period, auto_adjust=True)
    with _lock:
        _history[key] = df
    return df


def get_info(symbol: str) -> dict:
    """
    Return a shallow copy of yf .info for *symbol* ({} when unavailable,
    e.g. ETFs whose quoteSummary 404s). The copy keeps the cached
    original pristine when callers add defaults.
    """
    symbol = symbol.upper().strip()
    with _lock:
        if symbol in _info:
            return dict(_info[symbol])
    try:
        info = get_ticker(symbol).info or {}
    except Exception:
        info = {}
    with _lock:
        _info[symbol] = info
    return dict(info)


def get_news(symbol: str) -> list:
    """Return the raw yfinance news list for *symbol* ([] on failure)."""
    symbol = symbol.upper().strip()
    with _lock:
        if symbol in _news:
            return _news[symbol]
    try:
        news = get_ticker(symbol).news or []
    except Exception:
        news = []
    with _lock:
        _news[symbol] = news
    return news


def get_earnings_dates(symbol: str) -> "pd.DataFrame | None":
    """
    Return the earnings calendar for *symbol* (limit=20 — a superset of
    every caller's need). None when unavailable. Callers must .copy()
    before mutating.
    """
    symbol = symbol.upper().strip()
    with _lock:
        if symbol in _earnings:
            return _earnings[symbol]
    try:
        cal = get_ticker(symbol).get_earnings_dates(limit=20)
    except Exception:
        cal = None
    with _lock:
        _earnings[symbol] = cal
    return cal


# ------------------------------------------------------------------ #
# Shared market series (VIX / SPY)
# ------------------------------------------------------------------ #

def get_vix(start: str, end: str) -> pd.Series:
    """
    Return daily ^VIX closes for [start, end) as a tz-naive Series.

    The first call downloads one generous range (≥6 years back through
    tomorrow); every later call — any ticker, any range — is a slice of
    that cached series. Returns an empty Series on failure.
    """
    global _vix_series
    with _vix_lock:
        if _vix_series is None:
            today      = datetime.date.today()
            wide_start = min(
                pd.Timestamp(start).date() if start else today,
                today - datetime.timedelta(days=6 * 365),
            )
            wide_end = today + datetime.timedelta(days=2)
            try:
                df = yf.download("^VIX", start=wide_start.isoformat(),
                                 end=wide_end.isoformat(),
                                 auto_adjust=True, progress=False)
                if df is None or df.empty:
                    return pd.Series(dtype=float)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                series = df["Close"].copy()
                series.index = pd.to_datetime(series.index)
                if series.index.tz is not None:
                    series.index = series.index.tz_convert(None)
                _vix_series = series
            except Exception:
                return pd.Series(dtype=float)
        series = _vix_series
    try:
        return series.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()
    except Exception:
        return series.copy()


def get_vix_latest() -> "float | None":
    """Return the most recent VIX close, or None when unavailable."""
    today  = datetime.date.today()
    series = get_vix((today - datetime.timedelta(days=10)).isoformat(),
                     (today + datetime.timedelta(days=1)).isoformat())
    series = series.dropna()
    return float(series.iloc[-1]) if not series.empty else None


def get_spy() -> pd.DataFrame:
    """Return cached benchmark (SPY) history, downloading at most once."""
    global _spy_df
    with _spy_lock:
        if _spy_df is None:
            spy = yf.download(BENCHMARK_TICKER, period="1y",
                              auto_adjust=True, progress=False)
            # Recent yfinance returns MultiIndex columns (field, ticker)
            # even for one symbol. Flatten so spy["Close"] is a Series.
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            _spy_df = spy
        return _spy_df


# ------------------------------------------------------------------ #
# Batch-mode prewarm
# ------------------------------------------------------------------ #

def prewarm_symbol(symbol: str, period: str) -> None:
    """
    Fetch everything the pipeline will need for *symbol* into the cache.
    Designed to run from a worker thread; failures are non-fatal (the
    pipeline's own error handling deals with truly missing data).
    """
    for fn, args in (
        (get_history,        (symbol, period)),
        (get_info,           (symbol,)),
        (get_news,           (symbol,)),
        (get_earnings_dates, (symbol,)),
    ):
        try:
            fn(*args)
        except Exception as exc:
            logger.debug("prewarm %s %s failed: %s", symbol, fn.__name__, exc)
