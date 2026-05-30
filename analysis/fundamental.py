"""
analysis/fundamental.py — Fundamental valuation analysis for FinBot.

All calculations use data from the `info` dict returned by
stock_fetcher.fetch_stock_data() (which internally calls yf.Ticker.info).

Five metrics are implemented:
 1. P/E Ratio              — relative valuation vs. earnings
 2. PEG Ratio              — growth-adjusted P/E
 3. Graham Number          — intrinsic value floor (Benjamin Graham)
 4. Simplified DCF         — discounted free-cash-flow intrinsic value
 5. Price-to-Book / Dividend Yield — balance-sheet and income signals
"""

import math
from config import DEFAULT_WACC, TERMINAL_GROWTH_RATE, RISK_FREE_RATE


def _to_float(val):
    """Safely coerce a value to float; return None if not possible."""
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ #
# 1. P/E Ratio (Trailing & Forward)
# ------------------------------------------------------------------ #
# Formula:
#   Trailing P/E = Current Price / Trailing EPS (last 12 months)
#   Forward  P/E = Current Price / Forward EPS  (next 12 months est.)
# Source:  Graham, B. & Dodd, D. "Security Analysis" (1934), Chapter 26.
#          Also covered in Damodaran, A. "Investment Valuation" (2012).
# Rationale: The primary multi-used to assess whether a stock is cheap
#             or expensive relative to its earnings. A P/E above the
#             sector average suggests market optimism (or overvaluation).
#             Trailing P/E uses known figures; Forward P/E uses estimates.
def compute_pe_ratios(info: dict) -> dict:
    """Return trailing and forward P/E ratios with an assessment."""
    price   = _to_float(info.get("currentPrice"))
    t_eps   = _to_float(info.get("trailingEps"))
    f_eps   = _to_float(info.get("forwardEps"))
    t_pe    = _to_float(info.get("trailingPE"))   # Use pre-computed if available
    f_pe    = _to_float(info.get("forwardPE"))

    # Fall back to manual calculation if yfinance field is missing
    if t_pe is None and price and t_eps and t_eps != 0:
        t_pe = price / t_eps
    if f_pe is None and price and f_eps and f_eps != 0:
        f_pe = price / f_eps

    def assess_pe(pe):
        if pe is None:
            return "N/A"
        if pe < 0:
            return "NEGATIVE (company losing money)"
        if pe < 15:
            return "POTENTIALLY UNDERVALUED (< 15x)"
        if pe < 25:
            return "FAIR VALUE RANGE (15–25x)"
        if pe < 40:
            return "GROWTH PREMIUM (25–40x)"
        return "EXPENSIVE / HIGH GROWTH PRICED IN (> 40x)"

    return {
        "trailing_pe":      round(t_pe, 2) if t_pe is not None else None,
        "forward_pe":       round(f_pe, 2) if f_pe is not None else None,
        "trailing_signal":  assess_pe(t_pe),
        "forward_signal":   assess_pe(f_pe),
    }


# ------------------------------------------------------------------ #
# 2. PEG Ratio (Price / Earnings-to-Growth)
# ------------------------------------------------------------------ #
# Formula:
#   PEG = (P/E) / (Annual EPS Growth Rate × 100)
# Source:  Lynch, P. "One Up on Wall Street" (1989) — popularised the PEG.
#          Also formalised by Damodaran in "Investment Valuation" (2012).
# Rationale: A PEG of 1.0 is considered "fairly valued" — the P/E matches
#             the earnings growth rate. PEG < 1 suggests undervaluation
#             relative to growth; PEG > 2 is typically considered expensive.
#             Uses Yahoo Finance's `earningsGrowth` (YoY) field.
def compute_peg_ratio(info: dict) -> dict:
    t_pe     = _to_float(info.get("trailingPE"))
    eg_rate  = _to_float(info.get("earningsGrowth"))  # decimal, e.g. 0.15 for 15%

    if t_pe is None or eg_rate is None or eg_rate <= 0:
        return {"peg": None, "signal": "N/A — insufficient data"}

    peg = t_pe / (eg_rate * 100)

    if peg < 0:
        signal = "NEGATIVE — potential distortion (loss or shrinking earnings)"
    elif peg < 1.0:
        signal = "UNDERVALUED relative to growth (PEG < 1.0)"
    elif peg < 2.0:
        signal = "FAIR to SLIGHTLY EXPENSIVE (PEG 1.0–2.0)"
    else:
        signal = "EXPENSIVE relative to growth (PEG > 2.0)"

    return {"peg": round(peg, 2), "signal": signal}


# ------------------------------------------------------------------ #
# 3. Graham Number (Intrinsic Value Floor)
# ------------------------------------------------------------------ #
# Formula:
#   Graham Number = √( 22.5 × EPS × Book Value Per Share )
#   where 22.5 = 15 (max P/E) × 1.5 (max P/B) per Graham's criteria.
# Source:  Graham, B. "The Intelligent Investor" (1949, revised 1973),
#          Chapter 14 — "Stock Selection for the Defensive Investor".
# Rationale: Provides a conservative intrinsic value floor. If the current
#             price is significantly above the Graham Number, the stock may
#             be overpriced from a value-investing standpoint.
def compute_graham_number(info: dict) -> dict:
    eps  = _to_float(info.get("trailingEps"))
    bvps = _to_float(info.get("bookValue"))        # Book Value Per Share
    price = _to_float(info.get("currentPrice"))

    if eps is None or bvps is None or eps <= 0 or bvps <= 0:
        return {"graham_number": None, "signal": "N/A — EPS or Book Value unavailable/negative"}

    graham = math.sqrt(22.5 * eps * bvps)

    if price:
        margin = (graham - price) / price * 100
        if price <= graham:
            signal = f"UNDERVALUED — Price is {abs(margin):.1f}% BELOW Graham Number"
        else:
            signal = f"OVERVALUED vs Graham — Price is {abs(margin):.1f}% ABOVE Graham Number"
    else:
        signal = "Price unavailable for comparison"

    return {"graham_number": round(graham, 2), "signal": signal}


# ------------------------------------------------------------------ #
# 4. Simplified Discounted Cash Flow (DCF)
# ------------------------------------------------------------------ #
# Formula:
#   Step 1: Project Free Cash Flow for years 1–10:
#             FCF_t = FCF_0 × (1 + g)^t,   g = revenue/earnings growth
#   Step 2: Terminal Value = FCF_10 × (1 + g_t) / (WACC - g_t)
#             where g_t = terminal growth rate (config: 3 %)
#   Step 3: Intrinsic Value per Share = (Σ PV(FCF_t) + PV(TV)) / Shares Outstanding
#             PV(FCF_t) = FCF_t / (1 + WACC)^t
# Source:  Damodaran, A. "Investment Valuation" (2nd ed., 2002), Chapter 12.
#          Gordon, M. "The Investment, Financing, and Valuation of the
#          Corporation" (1962) — Gordon Growth Model used for terminal value.
# Rationale: DCF is the theoretically correct intrinsic value method — it
#             values the business as the sum of its future cash flows
#             discounted to today's dollars. The WACC approximation is
#             taken from config.py (default 10 %).
def compute_dcf(info: dict) -> dict:
    fcf        = _to_float(info.get("freeCashflow"))              # Annual FCF in dollars
    shares     = _to_float(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
    price      = _to_float(info.get("currentPrice"))
    # Use revenue growth as a proxy for FCF growth when earnings growth is unavailable
    g_rate     = _to_float(info.get("revenueGrowth")) or _to_float(info.get("earningsGrowth")) or 0.05
    wacc       = DEFAULT_WACC
    g_terminal = TERMINAL_GROWTH_RATE

    if fcf is None or shares is None or shares == 0 or fcf <= 0:
        return {"dcf_value": None, "signal": "N/A — Free Cash Flow or share count unavailable"}

    # Clamp growth rate: negative growth uses 2 %, hyper-growth capped at 30 %
    g = max(0.02, min(float(g_rate), 0.30))

    # Project FCF for 10 years and discount
    pv_fcf = sum(
        (fcf * (1 + g) ** t) / (1 + wacc) ** t
        for t in range(1, 11)
    )

    # Terminal value (Gordon Growth Model) — discounted back 10 years
    fcf_10     = fcf * (1 + g) ** 10
    tv         = fcf_10 * (1 + g_terminal) / (wacc - g_terminal)
    pv_tv      = tv / (1 + wacc) ** 10

    intrinsic_total   = pv_fcf + pv_tv          # Enterprise value approximation
    intrinsic_per_share = intrinsic_total / shares

    if price:
        margin = (intrinsic_per_share - price) / price * 100
        if intrinsic_per_share > price:
            signal = f"UNDERVALUED — DCF suggests {margin:.1f}% upside"
        else:
            signal = f"OVERVALUED — DCF suggests {abs(margin):.1f}% downside"
    else:
        signal = "Price unavailable for margin-of-safety calculation"

    return {
        "dcf_value":        round(intrinsic_per_share, 2),
        "growth_rate_used": f"{g*100:.1f}%",
        "wacc_used":        f"{wacc*100:.1f}%",
        "signal":           signal,
    }


# ------------------------------------------------------------------ #
# 5. Price-to-Book & Dividend Yield
# ------------------------------------------------------------------ #
# Price-to-Book:
#   Formula: P/B = Current Price / Book Value Per Share
#   Source:  Graham, B. "The Intelligent Investor" (1973), Ch. 14.
#   Rationale: P/B < 1 can indicate undervaluation (stock trades below
#               accounting net-asset-value). P/B >> 1 is common for asset-
#               light/tech companies and may not signal overvaluation alone.
#
# Dividend Yield:
#   Formula: Yield = Annual Dividend Per Share / Current Price
#   Source:  Standard income-investing metric; Siegel, J. "Stocks for the
#             Long Run" (2014), Chapter 6.
#   Rationale: Yield provides a "cash return" while holding the stock.
#               A yield falling far below the risk-free rate reduces the
#               income-investing case.
def compute_pb_dividend(info: dict) -> dict:
    pb     = _to_float(info.get("priceToBook"))
    dy     = _to_float(info.get("dividendYield"))     # decimal, e.g. 0.012 = 1.2%
    price  = _to_float(info.get("currentPrice"))
    bvps   = _to_float(info.get("bookValue"))

    # Manual P/B fall-back
    if pb is None and price and bvps and bvps != 0:
        pb = price / bvps

    results = {}

    if pb is not None:
        results["price_to_book"] = round(pb, 2)
        if pb < 1.0:
            results["pb_signal"] = f"BELOW BOOK VALUE (P/B = {pb:.2f}) — value opportunity or financial stress"
        elif pb < 3.0:
            results["pb_signal"] = f"REASONABLE (P/B = {pb:.2f})"
        else:
            results["pb_signal"] = f"HIGH PREMIUM TO BOOK (P/B = {pb:.2f}) — typical for asset-light companies"
    else:
        results["price_to_book"] = None
        results["pb_signal"]     = "N/A"

    if dy is not None:
        results["dividend_yield"]   = round(dy * 100, 2)   # store as %
        results["dividend_signal"]  = (
            f"{dy*100:.2f}% yield"
            + (f" — below risk-free rate ({RISK_FREE_RATE*100:.2f}%)" if dy < RISK_FREE_RATE else " — above risk-free rate")
        )
    else:
        results["dividend_yield"]   = None
        results["dividend_signal"]  = "No dividend / data unavailable"

    return results


# ------------------------------------------------------------------ #
# Master function: run all fundamental metrics at once
# ------------------------------------------------------------------ #
def compute_all_fundamentals(info: dict) -> dict:
    """
    Run every fundamental metric and return a combined summary dict.

    Parameters
    ----------
    info : dict
        The `info` dict from stock_fetcher.fetch_stock_data().

    Returns
    -------
    dict with keys: "pe", "peg", "graham", "dcf", "pb_div"
    """
    return {
        "pe":     compute_pe_ratios(info),
        "peg":    compute_peg_ratio(info),
        "graham": compute_graham_number(info),
        "dcf":    compute_dcf(info),
        "pb_div": compute_pb_dividend(info),
    }
