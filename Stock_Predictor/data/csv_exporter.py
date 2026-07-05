"""
data/csv_exporter.py — Export a feature-rich CSV for ML training.

Each row represents one trading day.  The final N rows also carry
forward-return labels (direction_5d / direction_21d and return_5d /
return_21d) which are used as supervised-learning targets.  The very
last row (today) receives a NaN for the labels because the future is
unknown; that row is used at inference time only.

Sentiment score (from the LLM headline scorer) is attached to the
last row; all historical rows default to 0.0 / "NEUTRAL".
"""

import os
import numpy as np
import pandas as pd

from data import market_data
from data.stock_fetcher import fetch_vix_data


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def export_features_csv(
    ticker: str,
    price_df: pd.DataFrame,
    technical: dict,
    info: dict,
    sentiment: dict,
    dir_path: str,
) -> str:
    """
    Build a feature matrix from price, technical, fundamental, and
    sentiment data then save it to ``dir_path/features.csv``.

    Parameters
    ----------
    ticker      : Ticker symbol (used only for the filename title).
    price_df    : Raw OHLCV DataFrame from yfinance (DatetimeIndex).
    technical   : Dict returned by compute_all_technicals(); must
                  contain a "df" key with indicator columns.
    info        : yfinance info dict (fundamental scalars).
    sentiment   : Dict returned by score_headlines(); keys:
                  overall_score (float), label (str).
    dir_path    : Directory where features.csv will be written.

    Returns
    -------
    Absolute path to the saved CSV file.
    """
    df = _build_feature_df(ticker, price_df, technical, info, sentiment)
    path = os.path.join(dir_path, "features.csv")
    df.to_csv(path)
    return path


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _build_feature_df(
    ticker: str,
    price_df: pd.DataFrame,
    technical: dict,
    info: dict,
    sentiment: dict,
) -> pd.DataFrame:
    """Assemble all features into a single time-indexed DataFrame."""

    tech_df = technical.get("df", pd.DataFrame())

    # ── 1. Price-derived features ──────────────────────────────── #
    feat = pd.DataFrame(index=price_df.index)

    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in price_df.columns:
            feat[col] = price_df[col]

    feat["Log_Return"]      = np.log(feat["Close"] / feat["Close"].shift(1))
    feat["Close_pct_change"] = feat["Close"].pct_change()
    feat["High_Low_Range"]  = (feat["High"] - feat["Low"]) / feat["Close"]

    rolling_252 = feat["Close"].rolling(252, min_periods=126)
    feat["dist_52w_high"] = feat["Close"] / rolling_252.max() - 1
    feat["dist_52w_low"]  = feat["Close"] / rolling_252.min() - 1

    # ── 2. Technical indicator columns ─────────────────────────── #
    # Raw values (kept for human readability in the CSV)
    for col in ("SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
                "RSI", "MACD_Line", "MACD_Signal", "MACD_Hist",
                "BB_PctB", "BB_Width",
                "ADX"):          # ADX for market-regime feature
        if col in tech_df.columns:
            feat[col] = tech_df[col]

    # Normalised MA ratios: (Close / MA) - 1  → price-scale independent
    for ma_col, ratio_col in [
        ("SMA_20",  "SMA_20_ratio"),  ("SMA_50",  "SMA_50_ratio"),
        ("SMA_200", "SMA_200_ratio"), ("EMA_12",  "EMA_12_ratio"),
        ("EMA_26",  "EMA_26_ratio"),
    ]:
        if ma_col in tech_df.columns:
            feat[ratio_col] = feat["Close"] / tech_df[ma_col] - 1

    # Normalised MACD: divide by Close to remove price scale
    for macd_col, norm_col in [
        ("MACD_Line",   "MACD_Line_norm"),
        ("MACD_Signal", "MACD_Signal_norm"),
        ("MACD_Hist",   "MACD_Hist_norm"),
    ]:
        if macd_col in tech_df.columns:
            feat[norm_col] = tech_df[macd_col] / feat["Close"]

    # ── 3. Rolling engineered features ─────────────────────────── #
    if "RSI" in feat.columns:
        feat["RSI_5d_mean"]  = feat["RSI"].rolling(5,  min_periods=1).mean()
        feat["RSI_21d_mean"] = feat["RSI"].rolling(21, min_periods=1).mean()

    feat["Vol_5d"]  = feat["Log_Return"].rolling(5,  min_periods=1).std()
    feat["Vol_21d"] = feat["Log_Return"].rolling(21, min_periods=1).std()

    feat["Momentum_5d"]  = feat["Close"].pct_change(5)
    feat["Momentum_21d"] = feat["Close"].pct_change(21)

    # Lagged return features (gives model a sense of recent direction)
    for lag in (1, 2, 3):
        feat[f"Log_Return_lag{lag}"] = feat["Log_Return"].shift(lag)

    # Volume ratio: today's volume relative to its 20-day average
    if "Volume" in feat.columns:
        feat["Volume_ratio"] = (
            feat["Volume"] / feat["Volume"].rolling(20, min_periods=5).mean()
        )

    # Vol regime ratio: 30-day realized vol / full-period realized vol.
    # ratio > 1.3 → vol expanding (risk-off); < 0.77 → compressing (calm).
    feat["Vol_30d"] = feat["Log_Return"].rolling(30, min_periods=15).std() * np.sqrt(252)
    full_vol = feat["Log_Return"].std() * np.sqrt(252)
    feat["vol_regime_ratio"] = (feat["Vol_30d"] / full_vol).clip(0.1, 5.0) if full_vol > 0 else 1.0

    # ── 4. Sentiment (latest row only; 0.0 for history) ─────────── #
    feat["sentiment_score"] = 0.0
    feat["sentiment_label"] = "NEUTRAL"
    if len(feat) > 0:
        feat.at[feat.index[-1], "sentiment_score"] = float(
            sentiment.get("overall_score", 0.0)
        )
        feat.at[feat.index[-1], "sentiment_label"] = str(
            sentiment.get("label", "NEUTRAL")
        )

    # ── 5. VIX macro features ────────────────────────────────────── #
    # Market fear context: vix_level captures current fear regime;
    # vix_5d_change captures whether fear is rising or falling.
    try:
        start_str = feat.index[0].strftime("%Y-%m-%d")
        end_str   = (feat.index[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        vix_series = fetch_vix_data(start_str, end_str)
        if not vix_series.empty:
            vix_series.index = pd.to_datetime(vix_series.index).normalize()
            feat.index = pd.to_datetime(feat.index).normalize()
            feat["vix_level"] = vix_series.reindex(feat.index, method="ffill")
            feat["vix_5d_change"] = feat["vix_level"].pct_change(5)
        else:
            feat["vix_level"]    = np.nan
            feat["vix_5d_change"] = np.nan
    except Exception:
        feat["vix_level"]    = np.nan
        feat["vix_5d_change"] = np.nan

    # ── 6. Earnings proximity flag ───────────────────────────────── #
    # Binary: 1 if this trading day is within 14 calendar days of an
    # earnings announcement, 0 otherwise.  Earnings dates are a major
    # disruptor of technical patterns.
    feat["earnings_within_14d"] = 0.0
    try:
        earnings_df = market_data.get_earnings_dates(ticker)
        if earnings_df is not None and not earnings_df.empty:
            earn_dates = pd.to_datetime(earnings_df.index).normalize()
            earn_dates_tz_naive = earn_dates.tz_localize(None) if earn_dates.tz is not None else earn_dates
            feat_dates = pd.to_datetime(feat.index).normalize()
            if feat_dates.tz is not None:
                feat_dates = feat_dates.tz_localize(None)
            for d in feat_dates:
                diffs = abs((earn_dates_tz_naive - d).days)
                if diffs.min() <= 14:
                    feat.at[d, "earnings_within_14d"] = 1.0
    except Exception:
        pass  # Non-fatal — defaults to 0

    # ── 6b. Additional leak-free engineered features ─────────────── #
    # All derived from data already in hand (past-only windows).

    # Relative strength vs the benchmark: stock return minus SPY return
    # over the same window. Cross-sectional momentum — a stock rising 5%
    # while the market rises 10% is lagging, not leading.
    try:
        from config import BENCHMARK_TICKER, HISTORY_PERIOD
        spy_hist = market_data.get_history(BENCHMARK_TICKER, HISTORY_PERIOD)
        spy_close = spy_hist["Close"].copy()
        spy_close.index = pd.to_datetime(spy_close.index)
        if spy_close.index.tz is not None:
            spy_close.index = spy_close.index.tz_convert(None)
        spy_close = spy_close.reindex(
            pd.to_datetime(feat.index).normalize(), method="ffill"
        )
        spy_close.index = feat.index
        for w in (21, 63):
            feat[f"rel_strength_{w}d"] = (
                feat["Close"].pct_change(w) - spy_close.pct_change(w)
            )
    except Exception:
        feat["rel_strength_21d"] = np.nan
        feat["rel_strength_63d"] = np.nan

    # Overnight gap: open vs previous close, plus its 5-day mean.
    # Persistent gapping signals institutional/news-driven repricing.
    if "Open" in feat.columns:
        feat["overnight_gap"] = feat["Open"] / feat["Close"].shift(1) - 1
        feat["gap_5d_mean"]   = feat["overnight_gap"].rolling(5, min_periods=1).mean()

    # ATR normalised by price (scale-independent daily-range regime).
    prev_close = feat["Close"].shift(1)
    tr = pd.concat([
        feat["High"] - feat["Low"],
        (feat["High"] - prev_close).abs(),
        (feat["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    feat["ATR_norm"] = (
        tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean() / feat["Close"]
    )

    # Drawdown from running peak (0 = at high; -0.3 = 30% below peak).
    feat["drawdown_from_peak"] = feat["Close"] / feat["Close"].cummax() - 1

    # Volume z-score: how unusual is today's volume vs the last 20 days.
    if "Volume" in feat.columns:
        vol_mean = feat["Volume"].rolling(20, min_periods=5).mean()
        vol_std  = feat["Volume"].rolling(20, min_periods=5).std()
        feat["volume_zscore_20d"] = (
            (feat["Volume"] - vol_mean) / vol_std.replace(0, np.nan)
        )

    # ── 7. Price z-score (mean-reversion signal) ─────────────────── #
    # Captures how stretched price is relative to its recent mean,
    # independent of Bollinger %B which is range-normalised differently.
    rolling_std_20 = feat["Close"].rolling(20, min_periods=5).std()
    feat["price_zscore_20d"] = (
        (feat["Close"] - feat["Close"].rolling(20, min_periods=5).mean())
        / rolling_std_20.replace(0, np.nan)
    )

    # ── 7b. SEC EDGAR point-in-time fundamentals ─────────────────── #
    # Leak-free quarterly fundamentals joined on each figure's SEC FILING
    # date (not its period end), so a trading day only ever sees numbers
    # already public by then. Left as NaN when unavailable (foreign/ADR or
    # no lxml-free JSON facts) — the trainer drops all-NaN columns per
    # ticker, so this never becomes a degenerate constant.
    try:
        from data import edgar_data
        edgar_feats = edgar_data.get_fundamental_features(ticker, feat.index)
        if edgar_feats is not None:
            for col in edgar_data.FEATURE_NAMES:
                feat[col] = edgar_feats[col].values if col in edgar_feats.columns else np.nan
        else:
            for col in edgar_data.FEATURE_NAMES:
                feat[col] = np.nan
    except Exception:
        from data import edgar_data
        for col in edgar_data.FEATURE_NAMES:
            feat[col] = np.nan

    # ── 7c. Earnings-surprise features (point-in-time) ───────────── #
    # yfinance's earnings calendar carries a "Surprise(%)" column (actual
    # vs estimate). A surprise is only known from the earnings date onward,
    # so we forward-fill from each earnings date — never before it. Depends
    # on lxml (yfinance scrapes HTML); degrades to NaN when unavailable.
    feat["earnings_surprise_last"] = np.nan
    feat["earnings_surprise_avg4"] = np.nan
    try:
        cal = market_data.get_earnings_dates(ticker)
        if cal is not None and not cal.empty:
            surprise_col = next((c for c in cal.columns if "surprise" in str(c).lower()), None)
            if surprise_col is not None:
                sdf = cal[[surprise_col]].copy()
                sdf.index = pd.to_datetime(sdf.index).tz_localize(None) \
                    if getattr(sdf.index, "tz", None) is not None else pd.to_datetime(sdf.index)
                sdf = sdf[pd.to_numeric(sdf[surprise_col], errors="coerce").notna()]
                sdf["val"] = pd.to_numeric(sdf[surprise_col], errors="coerce")
                sdf = sdf.sort_index()
                sdf["avg4"] = sdf["val"].rolling(4, min_periods=1).mean()
                fidx = pd.to_datetime(feat.index).tz_localize(None) \
                    if getattr(feat.index, "tz", None) is not None else pd.to_datetime(feat.index)
                # As-of backward join: only surprises already announced.
                left = pd.DataFrame({"date": fidx}).reset_index(names="pos").sort_values("date")
                right = sdf.reset_index().rename(columns={sdf.index.name or "index": "date"})
                right["date"] = pd.to_datetime(right["date"]).astype("datetime64[ns]")
                left["date"]  = left["date"].astype("datetime64[ns]")
                right = right.sort_values("date")
                m = pd.merge_asof(left, right[["date", "val", "avg4"]],
                                  on="date", direction="backward").sort_values("pos")
                feat["earnings_surprise_last"] = m["val"].values
                feat["earnings_surprise_avg4"] = m["avg4"].values
    except Exception:
        pass

    # ── 8. Forward-return labels (supervised targets) ───────────── #
    # direction: 1 = price up, 0 = price down or flat
    # return:    raw percentage change
    for h in (5, 21):
        future_close      = feat["Close"].shift(-h)
        feat[f"return_{h}d"]    = (future_close - feat["Close"]) / feat["Close"]
        feat[f"direction_{h}d"] = (feat[f"return_{h}d"] > 0).astype(float)
        # Last h rows have no valid future — set to NaN
        feat.loc[feat.index[-h:], [f"return_{h}d", f"direction_{h}d"]] = np.nan

    return feat
