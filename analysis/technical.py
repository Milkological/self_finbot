"""
analysis/technical.py — Technical indicator calculations for FinBot.

All seven indicators are computed purely from the OHLCV price DataFrame
(no network calls). Each function is annotated with its full formula,
the academic/industry source, and the trading rationale for including it.

Indicators implemented
----------------------
 1. Simple Moving Average (SMA)         — trend direction
 2. Exponential Moving Average (EMA)    — weighted trend signal
 3. Relative Strength Index (RSI-14)    — momentum overbought/oversold
 4. MACD + Signal Line                  — momentum crossover
 5. Bollinger Bands                     — volatility expansion/contraction
 6. Fibonacci Retracement Levels        — key support/resistance zones
 7. Local Support & Resistance Pivots   — rolling-window swing highs/lows
"""

import pandas as pd
import numpy as np


# ------------------------------------------------------------------ #
# 1. Simple Moving Average (SMA)
# ------------------------------------------------------------------ #
# Formula: SMA_n = (1/n) * Σ P_t  for t = [today - n + 1 … today]
# Source:  Dow Theory / Charles Dow (late 1800s); universally taught in
#          every technical analysis textbook (e.g. Murphy, "Technical
#          Analysis of the Financial Markets", 1999, Ch. 9).
# Rationale: Price above SMA → uptrend; price below SMA → downtrend.
#            SMA-20 ≈ monthly trend, SMA-50 ≈ mid-term, SMA-200 ≈ long-term.
#            Golden Cross (SMA-50 crosses above SMA-200) and Death Cross
#            (SMA-50 crosses below SMA-200) are widely-watched signals.
def compute_sma(price_df: pd.DataFrame, windows: list[int] = [20, 50, 200]) -> pd.DataFrame:
    """Add SMA columns to a copy of *price_df*."""
    df = price_df.copy()
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


# ------------------------------------------------------------------ #
# 2. Exponential Moving Average (EMA)
# ------------------------------------------------------------------ #
# Formula: EMA_t = α * P_t + (1 - α) * EMA_{t-1},  where α = 2 / (n + 1)
# Source:  Appel, G. "Technical Analysis: Power Tools for Active Investors"
#          (2005). Used as the basis for MACD.
# Rationale: EMA weights recent prices more heavily than SMA, making it
#             more responsive to short-term price changes.
def compute_ema(price_df: pd.DataFrame, windows: list[int] = [12, 26]) -> pd.DataFrame:
    df = price_df.copy()
    for w in windows:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df


# ------------------------------------------------------------------ #
# 3. Relative Strength Index (RSI-14)
# ------------------------------------------------------------------ #
# Formula:
#   RS     = Average Gain over n days / Average Loss over n days
#   RSI    = 100 - (100 / (1 + RS))
# Source:  Wilder, J.W. "New Concepts in Technical Trading Systems" (1978).
# Rationale: RSI > 70 → overbought (potential reversal / sell signal).
#             RSI < 30 → oversold  (potential bounce   / buy signal).
#             RSI = 50 acts as a trend confirmation midline.
def compute_rsi(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = price_df.copy()
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)          # Keep only positive changes
    loss = (-delta).clip(lower=0)       # Keep only negative changes (as positive)

    # Wilder's smoothing = EMA with α = 1/period
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


# ------------------------------------------------------------------ #
# 4. MACD (Moving Average Convergence Divergence) + Signal Line
# ------------------------------------------------------------------ #
# Formula:
#   MACD_line   = EMA_12 - EMA_26
#   Signal_line = EMA_9 of MACD_line
#   Histogram   = MACD_line - Signal_line
# Source:  Appel, G. "System and Forecasts" (1979); refined in
#          "Technical Analysis: Power Tools for Active Investors" (2005).
# Rationale: Bullish signal when MACD crosses above Signal line (momentum
#             turning positive). Bearish signal on the opposite crossover.
#             Histogram shows the magnitude of divergence.
def compute_macd(price_df: pd.DataFrame,
                 fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = price_df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    # Store the underlying EMAs as columns so downstream code (latest dict,
    # csv_exporter) can reference EMA_12 / EMA_26 directly.
    df[f"EMA_{fast}"] = ema_fast
    df[f"EMA_{slow}"] = ema_slow

    df["MACD_Line"]   = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD_Line"] - df["MACD_Signal"]
    return df


# ------------------------------------------------------------------ #
# 5. Bollinger Bands
# ------------------------------------------------------------------ #
# Formula:
#   Middle Band = SMA_20
#   Upper Band  = SMA_20 + (k * σ_20)    where k = 2 (default)
#   Lower Band  = SMA_20 - (k * σ_20)
#   %B          = (Price - Lower) / (Upper - Lower)  [0 = lower touch, 1 = upper]
#   Bandwidth   = (Upper - Lower) / Middle            [squeeze indicator]
# Source:  Bollinger, J. "Bollinger on Bollinger Bands" (2001).
# Rationale: Price touching/exceeding upper band → overextended/overbought.
#             Price touching/falling below lower band → oversold.
#             Bandwidth squeeze (BW < historical low) often precedes a
#             large directional move.
def compute_bollinger_bands(price_df: pd.DataFrame,
                             window: int = 20, k: float = 2.0) -> pd.DataFrame:
    df = price_df.copy()
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()

    df["BB_Upper"]  = sma + k * std
    df["BB_Middle"] = sma
    df["BB_Lower"]  = sma - k * std
    df["BB_PctB"]   = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    return df


# ------------------------------------------------------------------ #
# 6. Fibonacci Retracement Levels
# ------------------------------------------------------------------ #
# Formula:
#   Retracement level = High - (High - Low) * ratio
#   where ratio ∈ {0.236, 0.382, 0.500, 0.618, 0.786}
# Source:  Based on Fibonacci sequence ratios documented by Leonardo
#          Bonacci (c. 1202). Applied to markets in Prechter & Frost,
#          "Elliott Wave Principle" (1978).
# Rationale: These levels (23.6%, 38.2%, 50%, 61.8%, 78.6%) are widely
#             watched by traders as potential support on pullbacks in an
#             uptrend and resistance on bounces in a downtrend.
def compute_fibonacci_levels(price_df: pd.DataFrame) -> dict:
    """
    Compute Fibonacci retracement levels over the full price history.

    Returns a dict mapping each label → price level, e.g.:
      {"100%": 190.5, "78.6%": 185.2, ..., "0%": 120.0}
    """
    high = price_df["High"].max()
    low  = price_df["Low"].min()
    diff = high - low

    ratios = {
        "100.0%": 0.000,
        "78.6%":  0.214,   # 1 - 0.786
        "61.8%":  0.382,   # 1 - 0.618
        "50.0%":  0.500,
        "38.2%":  0.618,   # 1 - 0.382
        "23.6%":  0.764,   # 1 - 0.236
        "0.0%":   1.000,
    }

    levels = {}
    for label, pull in ratios.items():
        levels[label] = round(high - diff * pull, 4)
    return levels


# ------------------------------------------------------------------ #
# 7. Local Support & Resistance Pivots
# ------------------------------------------------------------------ #
# Formula:
#   A bar at index i is a swing HIGH if High[i] = max(High[i-w : i+w+1])
#   A bar at index i is a swing LOW  if Low[i]  = min(Low[i-w  : i+w+1])
# Source:  Standard pivot-point methodology; see Williams, L. "Long-Term
#          Secrets to Short-Term Trading" (2nd ed., 2011).
# Rationale: Swing highs/lows are price levels where supply/demand
#             previously reversed direction and are therefore likely to
#             act as future resistance/support.
def compute_support_resistance(price_df: pd.DataFrame, window: int = 10) -> dict:
    """
    Identify the most significant recent support and resistance price levels.

    Parameters
    ----------
    window : int
        Number of bars to look left and right of each candidate pivot.

    Returns
    -------
    dict with keys:
      "resistance_levels" — list[float], up to 5 highest swing highs
      "support_levels"    — list[float], up to 5 lowest  swing lows
    """
    highs = price_df["High"].values
    lows  = price_df["Low"].values
    n = len(highs)

    swing_highs = []
    swing_lows  = []

    for i in range(window, n - window):
        local_max = highs[i - window: i + window + 1].max()
        local_min = lows[i  - window: i + window + 1].min()

        if highs[i] == local_max:
            swing_highs.append(highs[i])
        if lows[i] == local_min:
            swing_lows.append(lows[i])

    # De-duplicate levels that are within 0.5 % of each other
    def deduplicate(levels: list[float], tol: float = 0.005) -> list[float]:
        if not levels:
            return []
        sorted_levels = sorted(set(levels))
        deduped = [sorted_levels[0]]
        for lvl in sorted_levels[1:]:
            if (lvl - deduped[-1]) / deduped[-1] > tol:
                deduped.append(lvl)
        return deduped

    resistance = sorted(deduplicate(swing_highs), reverse=True)[:5]
    support    = sorted(deduplicate(swing_lows))[:5]

    return {
        "resistance_levels": [round(x, 4) for x in resistance],
        "support_levels":    [round(x, 4) for x in support],
    }


# ------------------------------------------------------------------ #
# 8. Average True Range (ATR-14)
# ------------------------------------------------------------------ #
# Formula:
#   True Range (TR) = max( High - Low,
#                          |High - Close_prev|,
#                          |Low  - Close_prev| )
#   ATR_14 = Wilder's smoothed average of TR over 14 periods
#           = (ATR_{t-1} × 13 + TR_t) / 14
# Source:  Wilder, J.W. "New Concepts in Technical Trading Systems" (1978).
# Rationale: ATR measures absolute price volatility and is the standard
#             input for stop-loss sizing (e.g. 2×ATR trailing stop).
#             Unlike percentage-based volatility it accounts for gaps.
def compute_atr(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df   = price_df.copy()
    high = df["High"]
    low  = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder smoothing: equivalent to EMA with α = 1/period
    df["ATR"] = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return df


# ------------------------------------------------------------------ #
# 9. Average Directional Index (ADX-14)
# ------------------------------------------------------------------ #
# Formula:
#   +DM = High_t − High_{t-1}  if positive and > |Low_t − Low_{t-1}|, else 0
#   −DM = Low_{t-1} − Low_t   if positive and > |High_t − High_{t-1}|, else 0
#   Smoothed +DM14, −DM14 via Wilder's method
#   +DI14 = 100 × (Smoothed +DM14 / ATR14)
#   −DI14 = 100 × (Smoothed −DM14 / ATR14)
#   DX    = 100 × |+DI14 − −DI14| / (+DI14 + −DI14)
#   ADX   = Wilder-smoothed DX over 14 periods
# Source:  Wilder, J.W. "New Concepts in Technical Trading Systems" (1978).
# Rationale: ADX measures trend *strength* independent of direction.
#             ADX > 25 → trend is strong (trade with it).
#             ADX < 20 → no trend (mean-reversion strategies preferred).
#             Also used to validate RSI/MACD signals.
def compute_adx(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df   = price_df.copy()
    high = df["High"]
    low  = df["Low"]
    prev_close = df["Close"].shift(1)
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)

    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move)   & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move)    & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / period
    tr_s   = pd.Series(tr.values,       index=df.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    pdm_s  = pd.Series(plus_dm,         index=df.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    ndm_s  = pd.Series(minus_dm,        index=df.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    pdi = 100 * pdm_s / tr_s.replace(0, np.nan)
    ndi = 100 * ndm_s / tr_s.replace(0, np.nan)
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)

    df["ADX"]     = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    df["ADX_PDI"] = pdi
    df["ADX_NDI"] = ndi
    return df


# ------------------------------------------------------------------ #
# 10. On-Balance Volume (OBV)
# ------------------------------------------------------------------ #
# Formula:
#   OBV_t = OBV_{t-1} + Volume_t   if Close_t > Close_{t-1}
#   OBV_t = OBV_{t-1} − Volume_t   if Close_t < Close_{t-1}
#   OBV_t = OBV_{t-1}              if Close_t = Close_{t-1}
# Source:  Granville, J. "Granville's New Strategy of Daily Stock Market
#          Timing for Maximum Profit" (1960).
# Rationale: OBV is the cumulative volume-flow proxy for institutional
#             participation. Rising price + rising OBV = healthy trend.
#             Rising price + falling OBV = divergence (suspect rally,
#             potential reversal). Also shows accumulation/distribution.
def compute_obv(price_df: pd.DataFrame) -> pd.DataFrame:
    df     = price_df.copy()
    closes = df["Close"]
    volume = df["Volume"]

    direction = np.sign(closes.diff().fillna(0))
    df["OBV"] = (direction * volume).cumsum()
    return df


# ------------------------------------------------------------------ #
# Master function: run all technical indicators at once
# ------------------------------------------------------------------ #
def compute_all_technicals(price_df: pd.DataFrame) -> dict:
    """
    Run every technical indicator and return a summary dict.

    Parameters
    ----------
    price_df : pd.DataFrame
        OHLCV DataFrame from stock_fetcher.fetch_stock_data().

    Returns
    -------
    dict with keys:
      "df"            — enriched DataFrame (all indicator columns appended)
      "signals"       — dict of actionable signal strings (BUY/SELL/NEUTRAL)
      "fibonacci"     — dict of retracement price levels
      "pivot_levels"  — dict with "support_levels" and "resistance_levels"
      "latest"        — dict of the most recent indicator values (floats)
    """

    df = compute_sma(price_df)
    df = compute_rsi(df)
    df = compute_macd(df)   # internally computes EMA_12 and EMA_26
    df = compute_bollinger_bands(df)
    df = compute_atr(df)
    df = compute_adx(df)
    df = compute_obv(df)

    fib    = compute_fibonacci_levels(df)
    pivots = compute_support_resistance(df)

    # Extract the most recent valid value for each indicator independently.
    # Per-column last-valid avoids an empty result when a long-window indicator
    # (e.g. SMA-200 requires 200 bars) has all-NaN values for recently-listed
    # or low-history tickers — df.dropna() would return an empty DataFrame
    # in that case, causing an IndexError on .iloc[-1].
    def _lv(col: str):
        """Return the last non-NaN float in column *col*, or None."""
        s = df[col].dropna()
        return float(s.iloc[-1]) if not s.empty else None

    def _rv(val, decimals: int = 4):
        """Round *val* to *decimals* places; returns None if val is None."""
        return round(val, decimals) if val is not None else None

    price = _lv("Close")
    if price is None:
        raise ValueError("Price history is empty after indicator computation.")

    latest = {
        "Close":       _rv(price),
        "SMA_20":      _rv(_lv("SMA_20")),
        "SMA_50":      _rv(_lv("SMA_50")),
        "SMA_200":     _rv(_lv("SMA_200")),
        "EMA_12":      _rv(_lv("EMA_12")),
        "EMA_26":      _rv(_lv("EMA_26")),
        "RSI":         _rv(_lv("RSI"), 2),
        "MACD_Line":   _rv(_lv("MACD_Line")),
        "MACD_Signal": _rv(_lv("MACD_Signal")),
        "MACD_Hist":   _rv(_lv("MACD_Hist")),
        "BB_Upper":    _rv(_lv("BB_Upper")),
        "BB_Middle":   _rv(_lv("BB_Middle")),
        "BB_Lower":    _rv(_lv("BB_Lower")),
        "BB_PctB":     _rv(_lv("BB_PctB")),
        "ATR":         _rv(_lv("ATR")),
        "ADX":         _rv(_lv("ADX"), 2),
        "ADX_PDI":     _rv(_lv("ADX_PDI"), 2),
        "ADX_NDI":     _rv(_lv("ADX_NDI"), 2),
        "OBV":         _rv(_lv("OBV"), 0),
    }

    # ---- Build human-readable signals -------------------------------- #
    # Each block guards against None so short-history tickers degrade
    # gracefully rather than raising a TypeError on comparison.
    signals = {}

    # RSI signal
    rsi = latest["RSI"]
    if rsi is None:
        signals["RSI"] = "N/A — insufficient history"
    elif rsi > 70:
        signals["RSI"] = "OVERBOUGHT (>70) — potential pullback"
    elif rsi < 30:
        signals["RSI"] = "OVERSOLD (<30) — potential bounce"
    else:
        signals["RSI"] = f"NEUTRAL ({rsi:.1f})"

    # MACD signal
    macd_hist = latest["MACD_Hist"]
    macd_line = latest["MACD_Line"]
    macd_sig  = latest["MACD_Signal"]
    if macd_hist is None or macd_line is None or macd_sig is None:
        signals["MACD"] = "N/A — insufficient history"
    elif macd_hist > 0 and macd_line > macd_sig:
        signals["MACD"] = "BULLISH — MACD above Signal line"
    elif macd_hist < 0 and macd_line < macd_sig:
        signals["MACD"] = "BEARISH — MACD below Signal line"
    else:
        signals["MACD"] = "NEUTRAL / CROSSOVER"

    # Price vs SMAs
    sma200 = latest["SMA_200"]
    sma50  = latest["SMA_50"]
    if sma200 is None:
        signals["MA_Trend"] = "N/A — SMA-200 requires more history"
    else:
        ma_pos = "ABOVE SMA-200 (long-term uptrend)" if price > sma200 else "BELOW SMA-200 (long-term downtrend)"
        if sma50 is None:
            signals["MA_Trend"] = f"{ma_pos} | SMA-50 unavailable"
        else:
            golden_cross = sma50 > sma200
            signals["MA_Trend"] = f"{ma_pos} | {'Golden Cross' if golden_cross else 'Death Cross'} (SMA-50 vs SMA-200)"

    # Bollinger Band position
    pb = latest["BB_PctB"]
    if pb is None:
        signals["Bollinger"] = "N/A — insufficient history"
    elif pb > 1.0:
        signals["Bollinger"] = "PRICE ABOVE UPPER BAND — overextended"
    elif pb < 0.0:
        signals["Bollinger"] = "PRICE BELOW LOWER BAND — oversold squeeze"
    elif pb > 0.8:
        signals["Bollinger"] = "APPROACHING UPPER BAND"
    elif pb < 0.2:
        signals["Bollinger"] = "APPROACHING LOWER BAND"
    else:
        signals["Bollinger"] = f"MID-BAND (BB%B = {pb:.2f})"

    # ATR signal (dollar-value volatility)
    atr = latest["ATR"]
    if atr is None:
        signals["ATR"] = "N/A — insufficient history"
    elif price and price > 0:
        atr_pct = atr / price * 100
        signals["ATR"] = f"${atr:.4f} ({atr_pct:.1f}% of price) — daily range proxy"
    else:
        signals["ATR"] = f"${atr:.4f}"

    # ADX trend-strength signal
    adx = latest["ADX"]
    pdi = latest["ADX_PDI"]
    ndi = latest["ADX_NDI"]
    if adx is None:
        signals["ADX"] = "N/A — insufficient history"
    elif adx >= 40:
        bias = "BULLISH TREND" if (pdi or 0) > (ndi or 0) else "BEARISH TREND"
        signals["ADX"] = f"VERY STRONG TREND (ADX={adx:.1f}) — {bias}"
    elif adx >= 25:
        bias = "BULLISH TREND" if (pdi or 0) > (ndi or 0) else "BEARISH TREND"
        signals["ADX"] = f"STRONG TREND (ADX={adx:.1f}) — {bias}"
    elif adx >= 20:
        signals["ADX"] = f"DEVELOPING TREND (ADX={adx:.1f}) — watch for breakout"
    else:
        signals["ADX"] = f"NO TREND / RANGING (ADX={adx:.1f}) — mean-reversion favoured"

    # OBV divergence signal (simple: compare OBV direction with price direction)
    obv = latest["OBV"]
    if obv is None:
        signals["OBV"] = "N/A — insufficient history"
    else:
        # Use last 20 bars to detect divergence
        obv_series   = df["OBV"].dropna()
        close_series = df["Close"].dropna()
        if len(obv_series) >= 20 and len(close_series) >= 20:
            price_up = float(close_series.iloc[-1]) > float(close_series.iloc[-20])
            obv_up   = float(obv_series.iloc[-1])   > float(obv_series.iloc[-20])
            if price_up and obv_up:
                signals["OBV"] = "CONFIRMING UPTREND — volume supports price rise"
            elif not price_up and not obv_up:
                signals["OBV"] = "CONFIRMING DOWNTREND — volume supports price decline"
            elif price_up and not obv_up:
                signals["OBV"] = "BEARISH DIVERGENCE — price rising but volume retreating"
            else:
                signals["OBV"] = "BULLISH DIVERGENCE — price falling but volume accumulating"
        else:
            signals["OBV"] = f"OBV={obv:.0f} (insufficient history for divergence check)"

    return {
        "df":           df,
        "signals":      signals,
        "fibonacci":    fib,
        "pivot_levels": pivots,
        "latest":       latest,
    }
