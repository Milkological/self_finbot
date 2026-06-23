"""
analysis/statistical.py — Statistical models and Monte Carlo simulation for FinBot.

Seven models are implemented:
 1. Historical (Realised) Volatility   — annualised daily σ
 2. Beta                               — systematic market risk vs S&P 500
 3. Sharpe Ratio                       — risk-adjusted return
 4. Sortino Ratio                      — downside-only risk-adjusted return
 5. Maximum Drawdown & Calmar Ratio    — peak-to-trough loss + CAGR/MDD
 6. Linear Regression on log-prices    — trend-based price extrapolation
 7. Monte Carlo GBM (1 000 paths)      — CAPM-drift probabilistic price-range forecasts

All price predictions are computed at five horizons defined in config.py:
  1 Week (5d), 1 Month (21d), 3 Months (63d), 6 Months (126d), 1 Year (252d)
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import yfinance as yf
from config import (
    RISK_FREE_RATE,
    EQUITY_RISK_PREMIUM,
    BENCHMARK_TICKER,
    MONTE_CARLO_PATHS,
    PREDICTION_HORIZONS,
)

# Module-level SPY cache — downloaded once per process to avoid a
# redundant network round-trip on every compute_beta() call.
_spy_cache: pd.DataFrame | None = None


def _get_spy_data() -> pd.DataFrame:
    """Return cached SPY price history, downloading it at most once."""
    global _spy_cache
    if _spy_cache is None:
        spy = yf.download(BENCHMARK_TICKER, period="1y",
                          auto_adjust=True, progress=False)
        # Recent yfinance returns MultiIndex columns (field, ticker) even for a
        # single symbol. Flatten to the field level so `spy["Close"]` is a
        # Series, not a 1-column DataFrame (which breaks the Beta covariance).
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        _spy_cache = spy
    return _spy_cache


# ------------------------------------------------------------------ #
# 1. Historical (Realised) Volatility
# ------------------------------------------------------------------ #
# Formula:
#   σ_daily  = std(log returns)
#   σ_annual = σ_daily × √252
# Source:  Hull, J. "Options, Futures, and Other Derivatives" (10th ed.,
#          2018), Chapter 15 — Volatility estimation.
# Rationale: Realised volatility is the foundational risk metric in
#             modern portfolio theory (Markowitz, 1952). It is used as
#             the σ parameter in Black-Scholes and GBM models.
def compute_volatility(price_df: pd.DataFrame) -> dict:
    log_returns = price_df["Log_Return"].dropna()

    sigma_daily  = log_returns.std()
    sigma_annual = sigma_daily * np.sqrt(252)

    if sigma_annual < 0.15:
        vol_label = "LOW (< 15%)"
    elif sigma_annual < 0.30:
        vol_label = "MODERATE (15–30%)"
    elif sigma_annual < 0.50:
        vol_label = "HIGH (30–50%)"
    else:
        vol_label = "VERY HIGH (> 50%)"

    # 30-day vol regime: compare recent volatility to full-period volatility.
    # ratio > 1.3  → vol is expanding (risk-off signal)
    # ratio < 0.77 → vol is compressing (calm / low-conviction)
    # 0.77–1.30   → stable regime
    sigma_30d_raw = log_returns.rolling(30, min_periods=15).std().iloc[-1] if len(log_returns) >= 15 else np.nan
    if not np.isnan(sigma_30d_raw) and sigma_annual > 0:
        sigma_30d_annual = float(sigma_30d_raw) * np.sqrt(252)
        vol_regime_ratio = round(sigma_30d_annual / sigma_annual, 4)
    else:
        sigma_30d_annual = sigma_annual
        vol_regime_ratio = 1.0

    if vol_regime_ratio > 1.3:
        vol_regime = "EXPANDING"
    elif vol_regime_ratio < 0.77:
        vol_regime = "COMPRESSING"
    else:
        vol_regime = "STABLE"

    return {
        "sigma_daily":       round(sigma_daily, 6),
        "sigma_annual":      round(sigma_annual, 4),
        "label":             vol_label,
        "vol_regime_ratio":  vol_regime_ratio,
        "vol_regime":        vol_regime,
    }


# ------------------------------------------------------------------ #
# 2. Beta
# ------------------------------------------------------------------ #
# Formula:
#   β = Cov(R_stock, R_market) / Var(R_market)
# Source:  Sharpe, W. "Capital Asset Prices: A Theory of Market
#          Equilibrium under Conditions of Risk" (1964); Lintner, J.
#          "The Valuation of Risk Assets" (1965). (CAPM).
# Rationale: Beta measures sensitivity to broad market moves.
#             β = 1 → moves in line with the market.
#             β > 1 → amplified swings (higher risk/return).
#             β < 1 → dampened swings (defensive).
#             β < 0 → inversely correlated with market (rare).
def compute_beta(price_df: pd.DataFrame, ticker: str) -> dict:
    """
    Compute Beta against SPY (S&P 500 ETF) using 1 year of daily returns.

    We download SPY data independently to ensure the same date range is
    used for both the stock and the benchmark.
    """
    try:
        spy = _get_spy_data()
        if spy.empty:
            raise ValueError("SPY download returned empty DataFrame")

        # Align on common dates
        spy_ret   = np.log(spy["Close"] / spy["Close"].shift(1)).dropna()
        stock_ret = price_df["Log_Return"].dropna()

        common_idx = spy_ret.index.intersection(stock_ret.index)
        if len(common_idx) < 30:
            raise ValueError("Fewer than 30 overlapping trading days with SPY")

        s = stock_ret.loc[common_idx].values
        m = spy_ret.loc[common_idx].values

        # Flatten to 1-D in case yfinance returns a multi-index Series
        s = s.flatten()
        m = m.flatten()

        cov_matrix = np.cov(s, m)
        beta       = cov_matrix[0, 1] / cov_matrix[1, 1]

        if beta < 0:
            label = "INVERSE — moves opposite to market"
        elif beta < 0.5:
            label = "DEFENSIVE (β < 0.5)"
        elif beta < 1.0:
            label = "BELOW-MARKET (0.5–1.0)"
        elif beta < 1.5:
            label = "MARKET-LIKE (1.0–1.5)"
        else:
            label = "AGGRESSIVE (β > 1.5)"

        return {"beta": round(float(beta), 4), "label": label}

    except Exception as exc:
        # Use Yahoo Finance's pre-computed beta as fallback
        return {"beta": None, "label": f"Could not compute (reason: {exc})"}


# ------------------------------------------------------------------ #
# 3. Sharpe Ratio
# ------------------------------------------------------------------ #
# Formula:
#   Sharpe = (R_annual - R_f) / σ_annual
#   where R_annual = mean(log_return) × 252
#         R_f      = RISK_FREE_RATE  (config.py, default 5.25%)
#         σ_annual = σ_daily × √252
# Source:  Sharpe, W. "Mutual Fund Performance" (1966); refined in
#          "The Sharpe Ratio" (Journal of Portfolio Management, 1994).
# Rationale: The Sharpe Ratio is the industry-standard measure of
#             risk-adjusted return. A ratio > 1.0 is generally considered
#             good; > 2.0 is excellent; < 0 means the stock underperformed
#             the risk-free rate on a risk-adjusted basis.
def compute_sharpe_ratio(price_df: pd.DataFrame) -> dict:
    log_returns  = price_df["Log_Return"].dropna()
    mu_annual    = log_returns.mean() * 252
    sigma_annual = log_returns.std()  * np.sqrt(252)

    if sigma_annual == 0:
        return {"sharpe": None, "signal": "N/A — zero volatility"}

    sharpe = (mu_annual - RISK_FREE_RATE) / sigma_annual

    if sharpe > 2.0:
        signal = "EXCELLENT risk-adjusted return (> 2.0)"
    elif sharpe > 1.0:
        signal = "GOOD risk-adjusted return (1.0–2.0)"
    elif sharpe > 0:
        signal = "MODEST risk-adjusted return (0–1.0)"
    else:
        signal = "UNDERPERFORMING risk-free rate (< 0)"

    return {
        "sharpe":        round(sharpe, 4),
        "annual_return": f"{(float(np.exp(mu_annual)) - 1) * 100:.2f}%",
        "signal":        signal,
    }


# ------------------------------------------------------------------ #
# 4. Sortino Ratio
# ------------------------------------------------------------------ #
# Formula:
#   Sortino = (R_annual - R_f) / σ_downside
#   where σ_downside = std(negative log returns only) × √252
# Source:  Sortino, F. & van der Meer, R. "Downside Risk" (1991)
#          Journal of Portfolio Management.
# Rationale: Sharpe penalises both upside and downside volatility equally.
#             Sortino only penalises harmful volatility (losses), giving a
#             more realistic risk-adjusted view for asymmetric return profiles.
#             Sortino > 2.0 is excellent; < 0 underperforms risk-free rate.
def compute_sortino_ratio(price_df: pd.DataFrame) -> dict:
    log_returns  = price_df["Log_Return"].dropna()
    mu_annual    = log_returns.mean() * 252

    # Downside deviation: use only negative daily returns
    downside     = log_returns[log_returns < 0]
    if len(downside) < 5:
        return {"sortino": None, "signal": "N/A — insufficient negative returns"}

    sigma_down_annual = downside.std() * np.sqrt(252)
    if sigma_down_annual == 0:
        return {"sortino": None, "signal": "N/A — zero downside volatility"}

    sortino = (mu_annual - RISK_FREE_RATE) / sigma_down_annual

    if sortino > 3.0:
        signal = "EXCELLENT downside-adjusted return (> 3.0)"
    elif sortino > 2.0:
        signal = "VERY GOOD downside-adjusted return (2.0–3.0)"
    elif sortino > 1.0:
        signal = "GOOD downside-adjusted return (1.0–2.0)"
    elif sortino > 0:
        signal = "MODEST downside-adjusted return (0–1.0)"
    else:
        signal = "UNDERPERFORMING risk-free rate (< 0)"

    return {
        "sortino": round(sortino, 4),
        "sigma_downside_annual": round(sigma_down_annual, 4),
        "signal": signal,
    }


# ------------------------------------------------------------------ #
# 5. Maximum Drawdown & Calmar Ratio
# ------------------------------------------------------------------ #
# Max Drawdown formula:
#   MDD = min over all t of [ (P_t - peak_t) / peak_t ]
#   where peak_t = max(P_0 … P_t)
# Calmar Ratio formula:
#   Calmar = CAGR / |MDD|
#   where CAGR = (P_T/P_0)^(252/T) - 1
# Source:  Young, T.W. "Calmar Ratio: A Smoother Tool" (1991)
#          Futures Magazine. Standard prop-desk risk screening metric.
# Rationale: Max Drawdown is the worst-case loss a buy-and-hold investor
#             suffered over the period. Calmar tells you how much annual
#             return you earn per unit of max drawdown risk — prop desks
#             typically require Calmar > 0.5; > 1.0 is strong.
def compute_max_drawdown_calmar(price_df: pd.DataFrame) -> dict:
    closes   = price_df["Close"].dropna()
    if len(closes) < 10:
        return {"max_drawdown_pct": None, "calmar": None, "signal": "N/A — insufficient data"}

    cummax   = closes.cummax()
    drawdown = (closes - cummax) / cummax
    mdd      = float(drawdown.min())        # most negative value

    # CAGR over the available period
    n_days   = len(closes)
    p0, pt   = float(closes.iloc[0]), float(closes.iloc[-1])
    cagr     = (pt / p0) ** (252 / n_days) - 1 if p0 > 0 else 0.0

    calmar   = cagr / abs(mdd) if mdd != 0 else None

    if calmar is None:
        signal = "N/A — zero drawdown (not enough history)"
    elif calmar > 1.5:
        signal = "STRONG (Calmar > 1.5)"
    elif calmar > 0.5:
        signal = "ACCEPTABLE (0.5–1.5)"
    elif calmar > 0:
        signal = "WEAK (< 0.5) — poor return per unit of drawdown"
    else:
        signal = "NEGATIVE — CAGR below zero"

    return {
        "max_drawdown_pct": round(mdd * 100, 2),
        "calmar":           round(calmar, 4) if calmar is not None else None,
        "cagr_pct":         round(cagr * 100, 2),
        "signal":           signal,
    }


# ------------------------------------------------------------------ #
# 6. Linear Regression on Log-Prices (Trend Extrapolation)
# ------------------------------------------------------------------ #
# Formula:
#   Fit: ln(P_t) = α + β_trend × t + ε
#   Predicted price at horizon h:
#     P̂_{t+h} = exp( α + β_trend × (T + h) )
# Source:  Least-Squares Ordinary Linear Regression — Gauss, C.F.
#          "Theoria Motus Corporum Coelestium" (1809); applied to
#          financial price modelling by Lo, A. & MacKinlay, A.C.
#          "A Non-Random Walk Down Wall Street" (1999).
# Rationale: Log-price regression captures the *trend* component of price
#             movement. It is deterministic and does not model randomness —
#             it should be read as the trend-implied price assuming the
#             historical drift continues uninterrupted.
def compute_linear_regression_predictions(price_df: pd.DataFrame) -> dict:
    closes       = price_df["Close"].values
    log_prices   = np.log(closes)

    t = np.arange(len(log_prices))
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(t, log_prices)

    T      = len(log_prices)  # last observed index
    last   = float(closes[-1])

    predictions = {}
    for label, h in PREDICTION_HORIZONS.items():
        predicted_log  = intercept + slope * (T + h)
        predicted_price = np.exp(predicted_log)
        pct_change      = (predicted_price - last) / last * 100
        predictions[label] = {
            "price":   round(float(predicted_price), 4),
            "change%": round(float(pct_change), 2),
        }

    return {
        "predictions":  predictions,
        "slope_daily":  round(float(slope), 8),
        "r_squared":    round(float(r_value ** 2), 4),
        "trend_dir":    "UPWARD" if slope > 0 else "DOWNWARD",
    }


# ------------------------------------------------------------------ #
# 7. Monte Carlo Simulation — Geometric Brownian Motion (GBM)
# ------------------------------------------------------------------ #
# Formula (GBM exact discrete solution):
#   S_{t+1} = S_t × exp( (μ_capm - σ²/2)·Δt + σ·√Δt·Z )
#   where:
#     μ_capm = Rf + β × ERP           (CAPM-implied annualised drift)
#     σ      = std(log-returns) × √252 (annualised realised volatility)
#     Δt     = 1/252 (one trading day)
#     Z      ~ N(0, 1) independent standard normal draw
# Source:  Black, F. & Scholes, M. "The Pricing of Options and Corporate
#          Liabilities" (1973). GBM was formalised by Samuelson, P.
#          "Rational Theory of Warrant Pricing" (1965).
#          Simulation methodology: Glasserman, P. "Monte Carlo Methods
#          in Financial Engineering" (2004), Chapter 3.
#          CAPM drift: Sharpe (1964), Lintner (1965).
# Rationale: Using CAPM-implied drift anchors the simulation to
#             systematic risk (β × ERP) rather than the noisy historical
#             realised return, which can be boosted by recent luck.
#             Beta is passed in so both functions share the same estimate.
def compute_monte_carlo(price_df: pd.DataFrame, beta: float | None = None) -> dict:
    """
    Simulate MONTE_CARLO_PATHS price paths via GBM and return
    percentile forecasts at each horizon in PREDICTION_HORIZONS.

    Returns
    -------
    dict with keys:
      "paths"       — np.ndarray [n_paths × max_horizon+1] for chart plotting
      "predictions" — dict of horizon → {p10, median, p90, current}
      "params"      — dict of drift/vol used
    """
    log_returns  = price_df["Log_Return"].dropna().values
    S0           = float(price_df["Close"].iloc[-1])

    sigma_daily_long = log_returns.std()               # full-period daily vol
    # Recent 30-day vol reflects current regime better than 2-year average.
    # Blend 70% long-term + 30% recent to balance stability vs responsiveness.
    sigma_daily_30d = log_returns[-30:].std() if len(log_returns) >= 30 else sigma_daily_long
    sigma_daily  = 0.70 * sigma_daily_long + 0.30 * sigma_daily_30d
    sigma_annual = sigma_daily * np.sqrt(252)

    # Blended drift: 60% CAPM-implied + 40% historical mean.
    # Pure CAPM is theoretically clean but ignores persistent momentum;
    # blending adds responsiveness to actual stock performance while
    # keeping CAPM as the dominant anchor.
    beta_val = beta if (beta is not None and np.isfinite(beta)) else 1.0
    mu_annual_capm = RISK_FREE_RATE + beta_val * EQUITY_RISK_PREMIUM
    mu_annual_hist = float(log_returns.mean()) * 252
    mu_annual_blended = 0.60 * mu_annual_capm + 0.40 * mu_annual_hist
    mu_daily       = mu_annual_blended / 252

    max_horizon = max(PREDICTION_HORIZONS.values())    # longest: 252 days

    # Use an isolated, reproducible RNG instance (thread-safe; does not
    # mutate global numpy random state).
    rng = np.random.default_rng(42)

    # GBM simulation: [n_paths × (max_horizon+1)]
    # Time step Δt = 1 day
    Z        = rng.standard_normal((MONTE_CARLO_PATHS, max_horizon))
    # Exact GBM log-return at each step: (μ - σ²/2)·Δt + σ·√Δt·Z
    daily_log_rets = (mu_daily - 0.5 * sigma_daily ** 2) + sigma_daily * Z
    log_price_paths = np.cumsum(daily_log_rets, axis=1)  # cumulative sum

    # Prepend t=0 column so index i maps to day i
    log_price_paths = np.hstack([
        np.zeros((MONTE_CARLO_PATHS, 1)),
        log_price_paths
    ])
    price_paths = S0 * np.exp(log_price_paths)           # shape: [n_paths, max_horizon+1]

    predictions = {}
    for label, h in PREDICTION_HORIZONS.items():
        terminal = price_paths[:, h]                     # prices at day h
        predictions[label] = {
            "current":    round(S0, 4),
            "p10":        round(float(np.percentile(terminal, 10)), 4),
            "median":     round(float(np.percentile(terminal, 50)), 4),
            "p90":        round(float(np.percentile(terminal, 90)), 4),
            "p10_chg%":   round((float(np.percentile(terminal, 10))  - S0) / S0 * 100, 2),
            "med_chg%":   round((float(np.percentile(terminal, 50))  - S0) / S0 * 100, 2),
            "p90_chg%":   round((float(np.percentile(terminal, 90))  - S0) / S0 * 100, 2),
        }

    return {
        "paths":       price_paths,   # used by report_generator.py for fan chart
        "predictions": predictions,
        "params": {
            "S0":            round(S0, 4),
            "mu_annual":     round(mu_annual_blended, 4),
            "mu_method":     "BLENDED (60% CAPM + 40% Historical)",
            "mu_capm":       round(mu_annual_capm, 4),
            "mu_hist":       round(mu_annual_hist, 4),
            "sigma_annual":  round(sigma_annual, 4),
            "beta_used":     round(beta_val, 4),
            "n_paths":       MONTE_CARLO_PATHS,
        },
    }


# ------------------------------------------------------------------ #
# Master function: run all statistical models at once
# ------------------------------------------------------------------ #
def compute_all_statistics(price_df: pd.DataFrame, ticker: str) -> dict:
    """
    Run every statistical model and return a combined summary dict.

    Parameters
    ----------
    price_df : pd.DataFrame
        OHLCV + Log_Return DataFrame from stock_fetcher.
    ticker : str
        Used for the Beta SPY-comparison download.

    Returns
    -------
    dict with keys:
      "volatility", "beta", "sharpe", "sortino", "drawdown",
      "regression", "monte_carlo"
    """
    beta_result = compute_beta(price_df, ticker)
    # Extract numeric beta for passing into Monte Carlo CAPM drift.
    beta_val = beta_result.get("beta")   # float or None

    return {
        "volatility":  compute_volatility(price_df),
        "beta":        beta_result,
        "sharpe":      compute_sharpe_ratio(price_df),
        "sortino":     compute_sortino_ratio(price_df),
        "drawdown":    compute_max_drawdown_calmar(price_df),
        "regression":  compute_linear_regression_predictions(price_df),
        "monte_carlo": compute_monte_carlo(price_df, beta=beta_val),
    }
