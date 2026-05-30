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
    df = _build_feature_df(price_df, technical, info, sentiment)
    path = os.path.join(dir_path, "features.csv")
    df.to_csv(path)
    return path


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _build_feature_df(
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

    # ── 5. Forward-return labels (supervised targets) ───────────── #
    # direction: 1 = price up, 0 = price down or flat
    # return:    raw percentage change
    for h in (5, 21):
        future_close      = feat["Close"].shift(-h)
        feat[f"return_{h}d"]    = (future_close - feat["Close"]) / feat["Close"]
        feat[f"direction_{h}d"] = (feat[f"return_{h}d"] > 0).astype(float)
        # Last h rows have no valid future — set to NaN
        feat.loc[feat.index[-h:], [f"return_{h}d", f"direction_{h}d"]] = np.nan

    return feat
