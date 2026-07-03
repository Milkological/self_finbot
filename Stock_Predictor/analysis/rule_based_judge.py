"""
analysis/rule_based_judge.py — Deterministic rule-based investment judge for FinBot.

Used when the LLM pipeline is skipped (--no-llm flag or no API keys).
Evaluates FOUR explicit investment lenses and produces a composite verdict:

  FUNDAMENTAL LENS  — Is the business healthy?
    ROE, operating margin, profit margin, revenue growth, FCF yield,
    interest coverage, debt/equity

  TECHNICAL LENS    — Is the price/timing right?
    RSI, MACD, price vs SMA, ADX (directional), Bollinger %B,
    52-week position, OBV, linear-regression trend

  VALUATION LENS    — Am I overpaying?
    Trailing P/E, Forward P/E, PEG, EV/EBITDA, P/S,
    DCF vs price, Graham Number vs price, P/B, analyst target

  RISK LENS         — What is the downside exposure?
    Sortino ratio, Calmar ratio, Beta, volatility regime, earnings proximity

Each lens produces a score 0–100 and a letter grade (A/B/C/D/F).
The composite is a weighted average:
    composite = 0.30 × fundamental + 0.25 × technical
              + 0.30 × valuation  + 0.15 × risk

Composite → recommendation:
    ≥75  → STRONG BUY
    60–74 → BUY
    45–59 → HOLD
    30–44 → SELL
    <30   → STRONG SELL

All N/A / missing fields are SKIPPED (not penalised) — scores are normalised
over available points only, so ETFs and REITs without EPS are not unfairly
penalised. N/A → possible and earned both stay at 0 for that criterion.
"""

import math


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _grade(score: float) -> str:
    if score >= 83: return "A"
    if score >= 66: return "B"
    if score >= 50: return "C"
    if score >= 33: return "D"
    return "F"


def _normalise(earned: int | float, possible: int | float) -> float:
    """Convert raw points to a 0-100 score, guarding against zero-division."""
    if possible == 0:
        return 50.0          # No data → neutral
    return round(earned / possible * 100, 1)


# ------------------------------------------------------------------ #
# Lens 1 — Fundamental: "Is the business healthy?"
# ------------------------------------------------------------------ #

def _score_fundamental(info: dict, statistical: dict) -> dict:
    earned   = 0
    possible = 0
    signals  = []

    # ── ROE (Return on Equity) ────────────────────────────────── #
    roe = info.get("returnOnEquity")
    if roe is not None:
        possible += 2
        if roe > 0.15:
            pts = 2; label = "STRONG (>15%)"
        elif roe > 0.08:
            pts = 1; label = "ADEQUATE (8–15%)"
        else:
            pts = 0; label = "WEAK (<8%)"
        earned += pts
        signals.append({"metric": "ROE", "value": f"{roe*100:.1f}%", "label": label, "points": pts})

    # ── Operating Margin ──────────────────────────────────────── #
    op_margin = info.get("operatingMargins")
    if op_margin is not None:
        possible += 2
        if op_margin > 0.20:
            pts = 2; label = "STRONG (>20%)"
        elif op_margin > 0.10:
            pts = 1; label = "MODERATE (10–20%)"
        else:
            pts = 0; label = "LOW / NEGATIVE (<10%)"
        earned += pts
        signals.append({"metric": "Operating Margin", "value": f"{op_margin*100:.1f}%", "label": label, "points": pts})

    # ── Net Profit Margin ─────────────────────────────────────── #
    margin = info.get("profitMargins")
    if margin is not None:
        possible += 2
        if margin > 0.15:
            pts = 2; label = "HIGH (>15%)"
        elif margin > 0.05:
            pts = 1; label = "MODERATE (5–15%)"
        else:
            pts = 0; label = "LOW / NEGATIVE (<5%)"
        earned += pts
        signals.append({"metric": "Net Profit Margin", "value": f"{margin*100:.1f}%", "label": label, "points": pts})

    # ── Revenue Growth ────────────────────────────────────────── #
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        possible += 2
        if rev_growth > 0.10:
            pts = 2; label = "STRONG (>10%)"
        elif rev_growth >= 0:
            pts = 1; label = "POSITIVE (0–10%)"
        else:
            pts = 0; label = "DECLINING (<0%)"
        earned += pts
        signals.append({"metric": "Revenue Growth", "value": f"{rev_growth*100:.1f}%", "label": label, "points": pts})

    # ── FCF Yield (FCF ÷ Market Cap) ─────────────────────────── #
    # Replaces the old binary FCF positive/negative check.
    # FCF yield >5% = genuinely cash-generative at scale.
    fcf    = info.get("freeCashflow")
    mktcap = info.get("marketCap")
    if fcf is not None and mktcap and mktcap > 0:
        fcf_yield = fcf / mktcap
        possible += 2
        if fcf_yield > 0.05:
            pts = 2; label = f"HIGH FCF yield ({fcf_yield*100:.1f}% >5%)"
        elif fcf_yield > 0.02:
            pts = 1; label = f"MODERATE FCF yield ({fcf_yield*100:.1f}% 2–5%)"
        elif fcf_yield > 0:
            pts = 1; label = f"POSITIVE but thin ({fcf_yield*100:.1f}%)"
        else:
            pts = 0; label = f"NEGATIVE FCF yield ({fcf_yield*100:.1f}%)"
        earned += pts
        signals.append({"metric": "FCF Yield", "value": f"{fcf_yield*100:.1f}%", "label": label, "points": pts})
    elif fcf is not None:
        # No market cap — fall back to binary
        possible += 2
        if fcf > 0:
            pts = 2; label = "POSITIVE (no mktcap for yield)"
        else:
            pts = 0; label = "NEGATIVE"
        earned += pts
        signals.append({"metric": "Free Cash Flow", "value": f"${fcf:,.0f}", "label": label, "points": pts})

    # ── Interest Coverage ─────────────────────────────────────── #
    # operatingCashflow / abs(interestExpense). Complements D/E by
    # testing whether earnings can actually service the debt load.
    op_cf  = info.get("operatingCashflow")
    int_exp = info.get("interestExpense")
    if op_cf is not None and int_exp is not None and int_exp != 0:
        coverage = op_cf / abs(int_exp)
        possible += 2
        if coverage > 3.0:
            pts = 2; label = f"STRONG ({coverage:.1f}× >3.0)"
        elif coverage > 1.5:
            pts = 1; label = f"ADEQUATE ({coverage:.1f}× 1.5–3.0)"
        else:
            pts = 0; label = f"WEAK ({coverage:.1f}× <1.5) — debt service risk"
        earned += pts
        signals.append({"metric": "Interest Coverage", "value": f"{coverage:.1f}×", "label": label, "points": pts})

    # ── Debt / Equity ─────────────────────────────────────────── #
    dte = info.get("debtToEquity")
    if dte is not None:
        possible += 2
        # yfinance returns D/E as a percentage (e.g. 145 means 1.45×)
        dte_ratio = dte / 100 if dte > 10 else dte
        if dte_ratio < 1.0:
            pts = 2; label = "LOW (<1.0×)"
        elif dte_ratio < 2.5:
            pts = 1; label = "MODERATE (1.0–2.5×)"
        else:
            pts = 0; label = "HIGH (>2.5×)"
        earned += pts
        signals.append({"metric": "Debt/Equity", "value": f"{dte_ratio:.2f}×", "label": label, "points": pts})

    score = _normalise(earned, possible)
    grade = _grade(score)

    if score >= 66:
        verdict = "HEALTHY"
    elif score >= 45:
        verdict = "ADEQUATE"
    else:
        verdict = "WEAK"

    strong = len([s for s in signals if s["points"] >= 2])
    summary = (
        f"Business health score {score:.0f}/100 (Grade {grade}). "
        f"{strong} of {len(signals)} metrics are strong. "
        f"Verdict: {verdict}."
    )

    return {
        "score":    score,
        "grade":    grade,
        "verdict":  verdict,
        "signals":  signals,
        "summary":  summary,
        "earned":   earned,
        "possible": possible,
    }


# ------------------------------------------------------------------ #
# Lens 2 — Technical: "Is the price/timing right?"
# ------------------------------------------------------------------ #

def _score_technical(technical: dict, statistical: dict, info: dict) -> dict:
    earned   = 0
    possible = 0
    signals  = []

    latest = technical.get("latest", {})
    sig    = technical.get("signals", {})
    price  = latest.get("Close")

    # ── RSI ───────────────────────────────────────────────────── #
    rsi = latest.get("RSI")
    if rsi is not None:
        possible += 2
        if 30 <= rsi <= 55:
            pts = 2; label = f"ENTRY ZONE ({rsi:.1f})"
        elif rsi < 30:
            pts = 1; label = f"OVERSOLD ({rsi:.1f}) — potential bounce"
        elif rsi <= 70:
            pts = 1; label = f"ELEVATED ({rsi:.1f})"
        else:
            pts = 0; label = f"OVERBOUGHT ({rsi:.1f})"
        earned += pts
        signals.append({"metric": "RSI (14)", "value": f"{rsi:.1f}", "label": label, "points": pts})

    # ── MACD ──────────────────────────────────────────────────── #
    macd_signal = sig.get("MACD", "")
    if macd_signal:
        possible += 2
        upper = macd_signal.upper()
        if "BULLISH" in upper:
            pts = 2; label = "BULLISH crossover"
        elif "BEARISH" in upper:
            pts = 0; label = "BEARISH crossover"
        else:
            pts = 1; label = "NEUTRAL"
        earned += pts
        signals.append({"metric": "MACD", "value": macd_signal, "label": label, "points": pts})

    # ── Price vs SMA (20 / 50 / 200) ─────────────────────────── #
    # Direct test of the current price regime — independent of crossover labels.
    if price is not None:
        sma_vals = {
            "SMA_20":  latest.get("SMA_20"),
            "SMA_50":  latest.get("SMA_50"),
            "SMA_200": latest.get("SMA_200"),
        }
        available = {k: v for k, v in sma_vals.items() if v is not None}
        if available:
            possible += 2
            above = sum(1 for v in available.values() if price > v)
            total = len(available)
            if above == total:
                pts = 2; label = f"ABOVE ALL {total} SMAs — bullish regime"
            elif above >= total // 2 + 1:
                pts = 1; label = f"ABOVE {above}/{total} SMAs — mixed bullish"
            else:
                pts = 0; label = f"BELOW {total - above}/{total} SMAs — bearish regime"
            earned += pts
            sma_str = " | ".join(f"{k}={v:.2f}" for k, v in available.items())
            signals.append({"metric": "Price vs SMAs", "value": f"${price:.2f}", "label": f"{label} ({sma_str})", "points": pts})

    # ── ADX + Directional Indicators ─────────────────────────── #
    # Fixed vs old version: only award trending point if PDI > NDI
    # (bullish directional pressure), not just ADX > 25 blindly.
    adx  = latest.get("ADX")
    pdi  = latest.get("ADX_PDI")
    ndi  = latest.get("ADX_NDI")
    if adx is not None:
        possible += 2
        trend_strong = adx > 25
        bullish_dir  = (pdi is not None and ndi is not None and pdi > ndi)
        if trend_strong and bullish_dir:
            pts = 2; label = f"STRONG BULLISH TREND (ADX={adx:.1f}, PDI={pdi:.1f}>NDI={ndi:.1f})"
        elif trend_strong and not bullish_dir:
            pdi_s = f"{pdi:.1f}" if pdi is not None else "?"
            ndi_s = f"{ndi:.1f}" if ndi is not None else "?"
            pts = 0; label = f"STRONG BEARISH TREND (ADX={adx:.1f}, PDI={pdi_s}<NDI={ndi_s})"
        elif adx > 20:
            pts = 1; label = f"TRANSITIONING (ADX={adx:.1f})"
        else:
            pts = 1; label = f"RANGING / NO TREND (ADX={adx:.1f} <20) — mean-reversion regime"
        earned += pts
        signals.append({"metric": "ADX (14) + DI", "value": f"{adx:.1f}", "label": label, "points": pts})

    # ── Bollinger Band position ───────────────────────────────── #
    pct_b    = latest.get("BB_PctB")
    bb_signal = sig.get("Bollinger", "")
    if pct_b is not None:
        possible += 2
        if pct_b <= 0.3:
            pts = 2; label = f"NEAR LOWER BAND / OVERSOLD (%B={pct_b:.2f})"
        elif pct_b <= 0.6:
            pts = 2; label = f"MID-BAND ENTRY ZONE (%B={pct_b:.2f})"
        elif pct_b <= 0.85:
            pts = 1; label = f"UPPER HALF (%B={pct_b:.2f})"
        else:
            pts = 0; label = f"NEAR UPPER BAND / EXTENDED (%B={pct_b:.2f})"
        earned += pts
        signals.append({"metric": "Bollinger %B", "value": f"{pct_b:.2f}", "label": label, "points": pts})
    elif bb_signal:
        possible += 2
        upper = bb_signal.upper()
        if "OVERSOLD" in upper or "LOWER" in upper:
            pts = 2; label = "NEAR LOWER BAND"
        elif "SQUEEZE" in upper or "NEUTRAL" in upper:
            pts = 1; label = "NEUTRAL / SQUEEZE"
        elif "OVERBOUGHT" in upper or "UPPER" in upper:
            pts = 0; label = "NEAR UPPER BAND"
        else:
            pts = 1; label = bb_signal
        earned += pts
        signals.append({"metric": "Bollinger", "value": bb_signal, "label": label, "points": pts})

    # ── 52-Week Position ──────────────────────────────────────── #
    # Bonus signal: momentum zone (near 52wk high) or value-bounce zone
    # (near 52wk low). Awarded as 0 or 1 pt — directional nuance only.
    high_52 = info.get("fiftyTwoWeekHigh")
    low_52  = info.get("fiftyTwoWeekLow")
    if price is not None and high_52 and low_52:
        possible += 1
        pct_from_high = (price - high_52) / high_52 * 100
        pct_from_low  = (price - low_52)  / low_52  * 100
        if pct_from_high > -10:
            pts = 1; label = f"NEAR 52WK HIGH — momentum zone ({pct_from_high:+.1f}% from high)"
        elif pct_from_low < 15:
            pts = 1; label = f"NEAR 52WK LOW — value-bounce zone (+{pct_from_low:.1f}% from low)"
        else:
            pts = 0; label = f"MID-RANGE ({pct_from_high:+.1f}% from high, +{pct_from_low:.1f}% from low)"
        earned += pts
        signals.append({"metric": "52-Week Position", "value": f"${price:.2f}", "label": label, "points": pts})

    # ── OBV ───────────────────────────────────────────────────── #
    obv_signal = sig.get("OBV", "")
    if obv_signal:
        possible += 2
        upper = obv_signal.upper()
        if "BULLISH" in upper or "RISING" in upper:
            pts = 2; label = "BULLISH — volume confirming price"
        elif "BEARISH" in upper or "FALLING" in upper:
            pts = 0; label = "BEARISH — volume diverging"
        else:
            pts = 1; label = "NEUTRAL"
        earned += pts
        signals.append({"metric": "OBV", "value": obv_signal, "label": label, "points": pts})

    # ── Linear Regression Trend (statistical confirmation) ───── #
    # A high R² uptrend is a statistically validated directional signal,
    # not just a price-level observation. Bonus 0-1 pt.
    linreg = statistical.get("regression", {})
    lr_dir = linreg.get("trend_dir")
    lr_r2  = linreg.get("r_squared")
    if lr_dir is not None and lr_r2 is not None:
        possible += 1
        if lr_dir == "UPWARD" and lr_r2 > 0.70:
            pts = 1; label = f"CONFIRMED UPTREND (R²={lr_r2:.2f} >0.70)"
        elif lr_dir == "UPWARD":
            pts = 1; label = f"UPWARD TREND (R²={lr_r2:.2f}, weak fit)"
        else:
            pts = 0; label = f"DOWNTREND (R²={lr_r2:.2f})"
        earned += pts
        signals.append({"metric": "Linear Reg Trend", "value": f"R²={lr_r2:.2f}", "label": label, "points": pts})

    score = _normalise(earned, possible)
    grade = _grade(score)

    if score >= 66:
        verdict = "BULLISH"
    elif score >= 45:
        verdict = "NEUTRAL"
    else:
        verdict = "BEARISH"

    summary = (
        f"Technical score {score:.0f}/100 (Grade {grade}). "
        f"Trend/momentum picture is {verdict}."
    )

    return {
        "score":    score,
        "grade":    grade,
        "verdict":  verdict,
        "signals":  signals,
        "summary":  summary,
        "earned":   earned,
        "possible": possible,
    }


# ------------------------------------------------------------------ #
# Lens 3 — Valuation: "Am I overpaying?"
# ------------------------------------------------------------------ #

def _relative_multiple_pts(value: float, median: float) -> tuple:
    """
    Grade a valuation multiple against its sector-peer median.

      < 0.80× median → 2 pts (discount to peers)
      0.80–1.20×     → 1 pt  (in line)
      > 1.20×        → 0 pts (premium to peers)

    Returns (points, label).
    """
    ratio = value / median
    if ratio < 0.80:
        return 2, f"DISCOUNT to sector ({value:.1f}× vs peer median {median:.1f}×)"
    if ratio <= 1.20:
        return 1, f"IN LINE with sector ({value:.1f}× vs peer median {median:.1f}×)"
    return 0, f"PREMIUM to sector ({value:.1f}× vs peer median {median:.1f}×)"


def _score_valuation(fundamental: dict, analyst_data: dict, info: dict,
                     sector_multiples: dict = None) -> dict:
    earned   = 0
    possible = 0
    signals  = []

    price  = info.get("currentPrice")
    pe     = fundamental.get("pe", {})
    peg    = fundamental.get("peg", {})
    dcf    = fundamental.get("dcf", {})
    graham = fundamental.get("graham", {})
    pb_div = fundamental.get("pb_div", {})
    pt     = analyst_data.get("price_target") or {}
    sm     = sector_multiples or {}

    # ── Trailing P/E ──────────────────────────────────────────── #
    # Sector-relative when peer medians are available (a 30× P/E is
    # cheap for semis, expensive for banks); absolute bands otherwise.
    t_pe     = pe.get("trailing_pe")
    has_t_pe = t_pe is not None and t_pe > 0
    sec_t_pe = sm.get("trailing_pe")
    if t_pe is not None:
        possible += 2
        if t_pe < 0:
            pts = 0; label = "NEGATIVE EPS (loss-making)"
        elif sec_t_pe:
            pts, label = _relative_multiple_pts(t_pe, sec_t_pe)
        elif t_pe < 15:
            pts = 2; label = "CHEAP (<15×)"
        elif t_pe < 25:
            pts = 1; label = "FAIR (15–25×)"
        elif t_pe < 40:
            pts = 0; label = "GROWTH PREMIUM (25–40×)"
        else:
            pts = 0; label = "EXPENSIVE (>40×)"
        earned += pts
        signals.append({"metric": "Trailing P/E", "value": f"{t_pe:.1f}×", "label": label, "points": pts})

    # ── Forward P/E ───────────────────────────────────────────── #
    # More relevant than trailing for growth stocks.
    f_pe = pe.get("forward_pe")
    sec_f_pe = sm.get("forward_pe")
    if f_pe is not None and f_pe > 0:
        possible += 2
        if sec_f_pe:
            pts, label = _relative_multiple_pts(f_pe, sec_f_pe)
            label = label.replace("sector (", "sector fwd (")
        elif f_pe < 15:
            pts = 2; label = "CHEAP FORWARD (<15×)"
        elif f_pe < 25:
            pts = 1; label = "FAIR FORWARD (15–25×)"
        else:
            pts = 0; label = "EXPENSIVE FORWARD (>25×)"
        earned += pts
        signals.append({"metric": "Forward P/E", "value": f"{f_pe:.1f}×", "label": label, "points": pts})

    # ── PEG Ratio ─────────────────────────────────────────────── #
    peg_val = peg.get("peg")
    if peg_val is not None:
        possible += 2
        if peg_val < 0:
            pts = 0; label = "NEGATIVE (distorted)"
        elif peg_val < 1.0:
            pts = 2; label = "UNDERVALUED vs growth (<1.0)"
        elif peg_val < 2.0:
            pts = 1; label = "FAIR (1.0–2.0)"
        else:
            pts = 0; label = "EXPENSIVE (>2.0)"
        earned += pts
        signals.append({"metric": "PEG Ratio", "value": f"{peg_val:.2f}", "label": label, "points": pts})

    # ── EV/EBITDA ─────────────────────────────────────────────── #
    # The #1 institutional metric — capital-structure neutral, GAAP-agnostic.
    ev_ebitda = info.get("enterpriseToEbitda")
    if ev_ebitda is not None and ev_ebitda > 0:
        possible += 2
        if ev_ebitda < 10:
            pts = 2; label = "CHEAP (<10×)"
        elif ev_ebitda < 20:
            pts = 1; label = "MODERATE (10–20×)"
        else:
            pts = 0; label = "EXPENSIVE (>20×)"
        earned += pts
        signals.append({"metric": "EV/EBITDA", "value": f"{ev_ebitda:.1f}×", "label": label, "points": pts})

    # ── Price/Sales ───────────────────────────────────────────── #
    # Scored only when no trailing P/E is available (loss-making companies,
    # ETFs) to avoid double-counting revenue vs earnings metrics.
    ps = info.get("priceToSalesTrailing12Months")
    if ps is not None and not has_t_pe:
        possible += 2
        if ps < 2:
            pts = 2; label = "LOW P/S (<2×)"
        elif ps < 6:
            pts = 1; label = "MODERATE P/S (2–6×)"
        else:
            pts = 0; label = "HIGH P/S (>6×)"
        earned += pts
        signals.append({"metric": "Price/Sales (P/E N/A)", "value": f"{ps:.1f}×", "label": label, "points": pts})

    # ── DCF vs Price ──────────────────────────────────────────── #
    dcf_val = dcf.get("dcf_value")
    if dcf_val is not None and price is not None:
        possible += 2
        margin_pct = (dcf_val - price) / price * 100
        if margin_pct > 0:
            pts = 2; label = f"UNDERVALUED — DCF ${dcf_val:.2f} is {margin_pct:.0f}% above price"
        elif margin_pct > -20:
            pts = 1; label = f"NEAR FAIR VALUE — within 20% of DCF ${dcf_val:.2f}"
        else:
            pts = 0; label = f"OVERVALUED — DCF ${dcf_val:.2f} is {abs(margin_pct):.0f}% below price"
        earned += pts
        signals.append({"metric": "DCF Intrinsic", "value": f"${dcf_val:.2f}", "label": label, "points": pts})

    # ── Graham Number vs Price ────────────────────────────────── #
    graham_val = graham.get("graham_number")
    if graham_val is not None and price is not None:
        possible += 2
        margin_pct = (graham_val - price) / price * 100
        if margin_pct > 0:
            pts = 2; label = f"BELOW GRAHAM (${graham_val:.2f}) — margin of safety {margin_pct:.0f}%"
        elif margin_pct > -20:
            pts = 1; label = f"SLIGHTLY ABOVE GRAHAM (${graham_val:.2f})"
        else:
            pts = 0; label = f"WELL ABOVE GRAHAM (${graham_val:.2f}) — {abs(margin_pct):.0f}% premium"
        earned += pts
        signals.append({"metric": "Graham Number", "value": f"${graham_val:.2f}", "label": label, "points": pts})

    # ── Price-to-Book ─────────────────────────────────────────── #
    pb = pb_div.get("price_to_book")
    sec_pb = sm.get("price_to_book")
    if pb is not None:
        possible += 2
        if pb > 0 and sec_pb:
            pts, label = _relative_multiple_pts(pb, sec_pb)
        elif pb < 1.5:
            pts = 2; label = "LOW P/B (<1.5×)"
        elif pb < 3.0:
            pts = 1; label = "MODERATE P/B (1.5–3.0×)"
        else:
            pts = 0; label = "HIGH P/B (>3.0×)"
        earned += pts
        signals.append({"metric": "Price/Book", "value": f"{pb:.2f}×", "label": label, "points": pts})

    # ── Analyst Mean Target vs Price ──────────────────────────── #
    # Reduced to max 1pt (was 2pts) — analyst targets are trend-following
    # and circular; they should be a weak corroborating signal only.
    mean_target = pt.get("mean")
    if mean_target is not None and price is not None:
        possible += 1
        upside_pct = (mean_target - price) / price * 100
        if upside_pct > 5:
            pts = 1; label = f"UPSIDE — analyst target ${mean_target:.2f} ({upside_pct:+.0f}%)"
        else:
            pts = 0; label = f"AT/BELOW TARGET — ${mean_target:.2f} ({upside_pct:+.0f}%)"
        earned += pts
        signals.append({"metric": "Analyst Target", "value": f"${mean_target:.2f}", "label": label, "points": pts})

    score = _normalise(earned, possible)
    grade = _grade(score)

    if score >= 66:
        verdict = "UNDERVALUED"
    elif score >= 45:
        verdict = "FAIRLY_VALUED"
    else:
        verdict = "OVERVALUED"

    summary = (
        f"Valuation score {score:.0f}/100 (Grade {grade}). "
        f"Stock appears {verdict.replace('_', ' ')} on balance."
    )
    if sm:
        sector = info.get("sector") or "sector"
        n_peers = sm.get("n") or 0
        peers_str = f"{n_peers} largest {sector} peers" if n_peers else f"{sector} sector medians"
        summary += f" P/E, forward P/E and P/B graded relative to {peers_str}."

    return {
        "score":    score,
        "grade":    grade,
        "verdict":  verdict,
        "signals":  signals,
        "summary":  summary,
        "earned":   earned,
        "possible": possible,
        "sector_multiples": sm or None,
    }


# ------------------------------------------------------------------ #
# Lens 4 — Risk: "What is the downside exposure?"
# ------------------------------------------------------------------ #

def _score_risk(statistical: dict, analyst_data: dict, info: dict) -> dict:
    earned   = 0
    possible = 0
    signals  = []

    # ── Sortino Ratio ─────────────────────────────────────────── #
    # Better than Sharpe for risk — only penalises harmful (downside) vol.
    sortino_val = statistical.get("sortino", {}).get("sortino")
    if sortino_val is not None:
        possible += 2
        if sortino_val > 2.0:
            pts = 2; label = f"EXCELLENT downside-adjusted return (>{2.0})"
        elif sortino_val > 1.0:
            pts = 1; label = f"GOOD ({sortino_val:.2f}, >1.0)"
        else:
            pts = 0; label = f"POOR ({sortino_val:.2f}, <1.0)"
        earned += pts
        signals.append({"metric": "Sortino Ratio", "value": f"{sortino_val:.2f}", "label": label, "points": pts})

    # ── Calmar Ratio (CAGR / Max Drawdown) ───────────────────── #
    calmar_val = statistical.get("drawdown", {}).get("calmar")
    mdd_pct    = statistical.get("drawdown", {}).get("max_drawdown_pct")
    if calmar_val is not None:
        possible += 2
        if calmar_val > 1.0:
            pts = 2; label = f"STRONG return-per-drawdown ({calmar_val:.2f} >1.0)"
        elif calmar_val > 0.5:
            pts = 1; label = f"ACCEPTABLE ({calmar_val:.2f}, 0.5–1.0)"
        elif calmar_val > 0:
            pts = 0; label = f"WEAK ({calmar_val:.2f} <0.5)"
        else:
            pts = 0; label = f"NEGATIVE CAGR ({calmar_val:.2f})"
        earned += pts
        mdd_str = f" | MaxDD={mdd_pct:.1f}%" if mdd_pct is not None else ""
        signals.append({"metric": "Calmar Ratio", "value": f"{calmar_val:.2f}{mdd_str}", "label": label, "points": pts})

    # ── Beta (systematic market risk) ────────────────────────── #
    beta_val = statistical.get("beta", {}).get("beta")
    if beta_val is not None:
        possible += 2
        if 0.5 <= beta_val <= 1.5:
            pts = 2; label = f"SWEET SPOT (β={beta_val:.2f}, 0.5–1.5)"
        elif 0 < beta_val < 0.5:
            pts = 1; label = f"VERY DEFENSIVE (β={beta_val:.2f} <0.5) — low correlation"
        elif 1.5 < beta_val <= 2.0:
            pts = 1; label = f"AGGRESSIVE (β={beta_val:.2f} >1.5) — amplified swings"
        elif beta_val < 0:
            pts = 0; label = f"INVERSE (β={beta_val:.2f}) — unusual"
        else:
            pts = 0; label = f"VERY HIGH BETA (β={beta_val:.2f} >2.0) — volatile"
        earned += pts
        signals.append({"metric": "Beta (vs SPY)", "value": f"{beta_val:.2f}", "label": label, "points": pts})

    # ── Volatility Regime ─────────────────────────────────────── #
    vol_regime = statistical.get("volatility", {}).get("vol_regime")
    vol_pct    = statistical.get("volatility", {}).get("sigma_annual")
    if vol_regime:
        possible += 2
        if vol_regime == "COMPRESSING":
            pts = 2; label = "COMPRESSING — vol contracting, lower near-term risk"
        elif vol_regime == "STABLE":
            pts = 1; label = "STABLE — normal vol regime"
        else:
            pts = 0; label = "EXPANDING — vol rising, elevated near-term risk"
        earned += pts
        vol_str = f"{vol_pct*100:.1f}% annual" if isinstance(vol_pct, float) else "N/A"
        signals.append({"metric": "Volatility Regime", "value": vol_str, "label": label, "points": pts})

    # ── Earnings Proximity ────────────────────────────────────── #
    # Binary event risk: earnings releases cause outsized moves and
    # make short-term directional bets substantially higher risk.
    e5d  = analyst_data.get("earnings_within_5d",  False)
    e21d = analyst_data.get("earnings_within_21d", False)
    earn_date = analyst_data.get("next_earnings_date")
    if earn_date is not None or e5d or e21d:
        possible += 2
        if e5d:
            pts = 0; label = "EARNINGS WITHIN 5 DAYS — binary event risk, caution"
        elif e21d:
            pts = 1; label = "EARNINGS WITHIN 21 DAYS — elevated event risk"
        else:
            pts = 2; label = "NO NEAR-TERM EARNINGS — lower event risk"
        earned += pts
        date_str = str(earn_date) if earn_date else "scheduled"
        signals.append({"metric": "Earnings Proximity", "value": date_str, "label": label, "points": pts})

    score = _normalise(earned, possible)
    grade = _grade(score)

    if score >= 66:
        verdict = "LOW_RISK"
    elif score >= 45:
        verdict = "MODERATE_RISK"
    else:
        verdict = "HIGH_RISK"

    summary = (
        f"Risk score {score:.0f}/100 (Grade {grade}). "
        f"Overall risk profile: {verdict.replace('_', ' ')}."
    )

    return {
        "score":    score,
        "grade":    grade,
        "verdict":  verdict,
        "signals":  signals,
        "summary":  summary,
        "earned":   earned,
        "possible": possible,
    }


# ------------------------------------------------------------------ #
# Public entry point
# ------------------------------------------------------------------ #

def run_rule_based_judge(
    info:         dict,
    technical:    dict,
    fundamental:  dict,
    statistical:  dict,
    analyst_data: dict,
    ml_result:    dict = None,
    accuracy_context: dict = None,
) -> dict:
    """
    Run the four-lens deterministic judge and return a full result dict.

    The dict mirrors the shape expected by terminal_display and report_generator
    so they can render it without special-casing.
    """
    # Sector-peer medians for relative multiple grading (one cached
    # screener request per sector per week; None for ETFs/unknown
    # sectors, in which case absolute thresholds apply). Lazy import
    # keeps this module dependency-free for pure-dict callers/tests.
    sector_multiples = None
    try:
        from data.sector_data import get_sector_multiples
        sector_multiples = get_sector_multiples(info.get("sector"))
    except Exception:
        pass

    fund = _score_fundamental(info, statistical)
    tech = _score_technical(technical, statistical, info)
    val  = _score_valuation(fundamental, analyst_data, info, sector_multiples)
    risk = _score_risk(statistical, analyst_data, info)

    # ── Base composite (weighted) ────────────────────────────── #
    # Use dynamically adjusted weights from feedback loop if available
    _BASE_W = {"fundamental": 0.30, "technical": 0.25, "valuation": 0.30, "risk": 0.15}
    if (
        accuracy_context
        and accuracy_context.get("weights_adjusted")
        and accuracy_context.get("adjusted_weights")
    ):
        w = accuracy_context["adjusted_weights"]
        w_fund = w.get("fundamental", _BASE_W["fundamental"])
        w_tech = w.get("technical",   _BASE_W["technical"])
        w_val  = w.get("valuation",   _BASE_W["valuation"])
        w_risk = w.get("risk",        _BASE_W["risk"])
    else:
        w_fund, w_tech, w_val, w_risk = (
            _BASE_W["fundamental"], _BASE_W["technical"],
            _BASE_W["valuation"],   _BASE_W["risk"],
        )
    composite = (
        w_fund * fund["score"] +
        w_tech * tech["score"] +
        w_val  * val["score"]  +
        w_risk * risk["score"]
    )

    # ── ML score modifier (bounded ±10 pts, accuracy-gated) ────── #
    # Each horizon's contribution is scaled by that model's walk-forward
    # CV skill: a coin-flip model (≤50% accuracy) moves the composite by
    # nothing; a ≥65%-accuracy model gets full weight. This stops noisy
    # per-ticker models from swinging recommendations.
    ml_note   = ""
    ml_adjust = 0.0
    if ml_result and isinstance(ml_result, dict):
        ml_metrics = ml_result.get("metrics") or {}
        horizon_adjusts = []
        probs_shown = []
        for h in (5, 21):
            # Prefer the feedback-calibrated probability when present.
            p = ml_result.get(f"clf_{h}d_prob_up_cal")
            if not isinstance(p, float):
                p = ml_result.get(f"clf_{h}d_prob_up")
            if not isinstance(p, float):
                continue
            acc = ml_metrics.get(f"clf_{h}d_accuracy")
            if isinstance(acc, (int, float)) and not math.isnan(acc):
                skill = max(0.0, min(1.0, (acc - 0.5) / 0.15))
            else:
                skill = 0.0     # no validated skill → no influence
            horizon_adjusts.append(skill * max(-10.0, min(10.0, (p - 0.5) * 20)))
            probs_shown.append(f"{p:.0%}")
        if horizon_adjusts:
            ml_adjust = sum(horizon_adjusts) / len(horizon_adjusts)
            composite += ml_adjust
            direction = "bullish" if ml_adjust > 0 else "bearish"
            if abs(ml_adjust) >= 1.0:
                ml_note = (
                    f" ML signals lean {direction} ({'/'.join(probs_shown)} prob-up, "
                    f"weighted by validated accuracy); "
                    f"composite adjusted by {ml_adjust:+.1f}pts."
                )
            elif probs_shown:
                ml_note = " ML influence muted (models show no validated edge over coin-flip)."

    # ── Monte Carlo cross-validation (small valuation nudge) ─── #
    mc_preds = statistical.get("monte_carlo", {}).get("predictions", {})
    mc_1yr   = mc_preds.get("1 Year") or mc_preds.get("1 year")
    price    = info.get("currentPrice")
    if mc_1yr and price:
        mc_median = mc_1yr.get("median")
        if isinstance(mc_median, (int, float)) and mc_median > 0:
            mc_upside = (mc_median - price) / price
            if mc_upside > 0.15:
                val["score"] = min(100.0, val["score"] + 3)
            elif mc_upside < -0.15:
                val["score"] = max(0.0, val["score"] - 3)
            # Recompute composite with nudged val score
            composite = (
                w_fund * fund["score"] +
                w_tech * tech["score"] +
                w_val  * val["score"]  +
                w_risk * risk["score"]
            )
            composite += ml_adjust

    effective_weights = {"fundamental": w_fund, "technical": w_tech, "valuation": w_val, "risk": w_risk}

    composite = round(composite, 1)

    # ── Hard veto guard rails ─────────────────────────────────── #
    # Prevent false BUY signals on genuinely distressed situations.
    veto_triggered = False
    veto_reason    = ""

    rev_growth = info.get("revenueGrowth", 0) or 0
    fcf        = info.get("freeCashflow",  0) or 0
    if rev_growth < 0 and fcf < 0 and fund["score"] < 35:
        veto_triggered = True
        veto_reason    = "Veto: declining revenue + negative FCF + weak fundamentals"

    if (tech["verdict"] == "BEARISH"
            and fund["verdict"] == "WEAK"
            and val["verdict"] == "OVERVALUED"):
        veto_triggered = True
        veto_reason    = "Veto: triple-negative (bearish technicals + weak business + overvalued)"

    if veto_triggered:
        composite = min(composite, 44.0)   # Hard cap at top of SELL range

    # ── Composite → Recommendation ───────────────────────────── #
    if composite >= 75:
        recommendation = "STRONG BUY"
    elif composite >= 60:
        recommendation = "BUY"
    elif composite >= 45:
        recommendation = "HOLD"
    elif composite >= 30:
        recommendation = "SELL"
    else:
        recommendation = "STRONG SELL"

    if composite >= 60:
        worth = "WORTH_INVESTING"
    elif composite >= 45:
        worth = "CONDITIONAL"
    else:
        worth = "NOT_WORTH_INVESTING"

    # ── Confidence based on data completeness ────────────────── #
    total_possible = (
        fund["possible"] + tech["possible"] +
        val["possible"]  + risk["possible"]
    )
    if total_possible >= 30:
        confidence = "HIGH"
    elif total_possible >= 18:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # ── Summary ───────────────────────────────────────────────── #
    name = info.get("shortName", "")
    summary = (
        f"Rule-based composite score: {composite:.0f}/100 "
        f"(Fund {fund['score']:.0f} | Tech {tech['score']:.0f} | "
        f"Val {val['score']:.0f} | Risk {risk['score']:.0f}). "
        f"Business is {fund['verdict']}, technicals are {tech['verdict']}, "
        f"valuation is {val['verdict'].replace('_', ' ')}, "
        f"risk is {risk['verdict'].replace('_', ' ')}."
        + (f" {ml_note}" if ml_note else "")
        + (f" ⚠ {veto_reason}." if veto_triggered else "")
    )

    return {
        "composite_score":     composite,
        "recommendation":      recommendation,
        "confidence":          confidence,
        "worth_investing":     worth,
        "fundamental":         fund,
        "technical":           tech,
        "valuation":           val,
        "risk":                risk,
        "summary":             summary,
        "veto_triggered":      veto_triggered,
        "veto_reason":         veto_reason if veto_triggered else "",
        "fundamental_verdict": fund["summary"],
        "technical_verdict":   tech["summary"],
        "valuation_verdict":   val["summary"],
        "risk_verdict":        risk["summary"],
        "effective_weights":   effective_weights,
    }


# ------------------------------------------------------------------ #
# run_rule_based_analysis — mirrors get_llm_analysis() output shape
# ------------------------------------------------------------------ #

def _compute_accuracy_pct(base_clf_acc, horizon_key, vix_regime=None):
    """
    Compute per-horizon accuracy estimate for the rule-based path.

    base_clf_acc : float 0-1 from ML metrics (or None -> 0.55 default)
    horizon_key  : one of the 8 target_prices keys
    vix_regime   : "HIGH_FEAR" | "ELEVATED" | "CALM" | None
    """
    base = (base_clf_acc * 100) if isinstance(base_clf_acc, float) else 55.0
    decay = {
        "1_week": 1.00, "2_weeks": 0.95, "3_weeks": 0.90, "1_month": 0.85,
        "3_months": 0.75, "6_months": 0.65, "9_months": 0.60, "12_months": 0.55,
    }.get(horizon_key, 0.60)
    vix_penalty = {"HIGH_FEAR": -5.0, "ELEVATED": -2.0}.get(vix_regime or "CALM", 0.0)
    acc = base * decay + vix_penalty
    return round(max(30.0, min(85.0, acc)), 1)


def run_rule_based_analysis(
    info:         dict,
    technical:    dict,
    fundamental:  dict,
    statistical:  dict,
    analyst_data: dict,
    ml_result:    dict = None,
    accuracy_context: dict = None,
) -> dict:
    """
    Run the rule-based judge and return a dict whose shape matches
    the output of get_llm_analysis() so the rest of the pipeline
    (terminal_display, report_generator) works unchanged with --no-llm.
    """
    # Lazy import to avoid circular dependency (llm_analysis imports nothing from here)
    from analysis.llm_analysis import build_company_description

    description = build_company_description(info, analyst_data)
    rbj = run_rule_based_judge(info, technical, fundamental, statistical, analyst_data, ml_result, accuracy_context=accuracy_context)

    # ── Map composite score to top-level fields ───────────────────── #
    rec  = rbj.get("recommendation", "HOLD")
    conf = rbj.get("confidence", "MEDIUM")
    comp = rbj.get("composite_score", 50.0)

    if rec in ("STRONG BUY", "BUY"):
        ost = "WORTH_INVESTING";  olt = "WORTH_INVESTING"
    elif rec in ("STRONG SELL", "SELL"):
        ost = "NOT_WORTH_INVESTING"; olt = "NOT_WORTH_INVESTING"
    else:
        ost = "NEUTRAL"; olt = "NEUTRAL"

    price  = info.get("currentPrice", 0) or 0
    vix_regime = analyst_data.get("vix", {}).get("regime", "CALM")

    # ── Build target_prices from Monte Carlo / regression ─────────── #
    mc_preds  = statistical.get("monte_carlo", {}).get("predictions", {})
    reg_preds = statistical.get("regression",  {}).get("predictions", {})
    ml_acc    = (ml_result or {}).get("metrics", {})
    clf_5d    = ml_acc.get("clf_5d_accuracy")
    clf_21d   = ml_acc.get("clf_21d_accuracy")

    _mc_label = {  # target_prices key -> MC horizon label
        "1_week": "1 Week", "2_weeks": None, "3_weeks": None, "1_month": "1 Month",
        "3_months": "3 Months", "6_months": "6 Months", "9_months": "9 Months",
        "12_months": "1 Year",
    }
    _reg_label = {
        "1_week": "1 Week", "2_weeks": "2 Weeks", "3_weeks": "3 Weeks",
        "1_month": "1 Month", "3_months": "3 Months", "6_months": "6 Months",
        "9_months": "9 Months", "12_months": "1 Year",
    }

    target_prices = {}
    for key in ("1_week","2_weeks","3_weeks","1_month","3_months","6_months","9_months","12_months"):
        base_acc = clf_5d if key in ("1_week","2_weeks","3_weeks") else clf_21d
        acc = _compute_accuracy_pct(base_acc, key, vix_regime)
        mc_lbl = _mc_label.get(key)
        mc_h   = mc_preds.get(mc_lbl, {}) if mc_lbl else {}
        median = mc_h.get("median")
        if median is None:
            reg_lbl = _reg_label.get(key)
            reg_h   = reg_preds.get(reg_lbl, {}) if reg_lbl else {}
            median  = reg_h.get("price") or price
        target_prices[key] = {
            "price":        round(float(median or price), 2),
            "accuracy_pct": acc,
        }

    # ── Entry / stop from Technical lens ────────────────────────── #
    tech_latest = technical.get("latest", {})
    atr_val     = tech_latest.get("ATR") or 0
    tech_verdict = rbj.get("technical", {}).get("verdict", "NEUTRAL")
    if tech_verdict == "BULLISH":
        trade_direction = "LONG"
        entry_price     = round(price, 2)
        stop_loss       = round(price - max(atr_val * 1.5, price * 0.05), 2)
        exit_price      = target_prices["3_months"]["price"]
    elif tech_verdict == "BEARISH":
        trade_direction = "SHORT"
        entry_price     = round(price, 2)
        stop_loss       = round(price + max(atr_val * 1.5, price * 0.05), 2)
        exit_price      = target_prices["3_months"]["price"]
    else:
        trade_direction = "LONG"
        entry_price     = round(price, 2)
        stop_loss       = round(price * 0.95, 2)
        exit_price      = target_prices["3_months"]["price"]

    # ── Position sizing from Risk score ──────────────────────────── #
    risk_score = rbj.get("risk", {}).get("score", 50.0)
    if risk_score >= 66:
        position_size_pct = 1.5
    elif risk_score >= 45:
        position_size_pct = 1.0
    else:
        position_size_pct = 0.5

    # ── Bull/Bear signals from lens signals ───────────────────────── #
    def _top_signals(lens_key: str, positive: bool, n: int = 3) -> list:
        sigs = rbj.get(lens_key, {}).get("signals", [])
        if positive:
            good = [s["label"] for s in sigs if s.get("points", 0) >= 2][:n]
            return good
        else:
            bad  = [s["label"] for s in sigs if s.get("points", 0) == 0][:n]
            return bad

    key_bull = (
        _top_signals("fundamental", True, 2)
        + _top_signals("technical",   True, 2)
        + _top_signals("valuation",   True, 1)
    )[:3]
    key_bear = (
        _top_signals("fundamental", False, 1)
        + _top_signals("technical",   False, 1)
        + _top_signals("risk",        False, 2)
    )[:3]
    key_risks = [
        s["label"] for s in rbj.get("risk", {}).get("signals", [])
        if s.get("points", 0) == 0
    ][:3]

    # ── Rule-based stubs for the 12 sub-agents ───────────────────── #
    fund_score = rbj.get("fundamental", {})
    tech_score = rbj.get("technical",   {})
    val_score  = rbj.get("valuation",   {})
    risk_score_obj = rbj.get("risk",    {})

    def _stub(findings, verdict):
        return {"findings": findings, "verdict": verdict, "_source": "rule_based"}

    stub_agents = {
        "fundamental_analyst": {
            "business_health": (
                "STRONG" if fund_score.get("score",50) >= 66 else
                "ADEQUATE" if fund_score.get("score",50) >= 45 else "WEAK"
            ),
            "valuation_stance": (
                "UNDERVALUED" if val_score.get("score",50) >= 66 else
                "FAIRLY_VALUED" if val_score.get("score",50) >= 45 else "OVERVALUED"
            ),
            "capital_allocation": "ADEQUATE",
            "profitability_quality": (
                "HIGH" if fund_score.get("score",50) >= 66 else
                "MEDIUM" if fund_score.get("score",50) >= 45 else "LOW"
            ),
            "competitive_position": "AVERAGE",
            "analyst_conviction": "MODERATE",
            "short_term_verdict": ost,
            "short_term_rationale": fund_score.get("summary",""),
            "long_term_verdict": olt,
            "long_term_rationale": val_score.get("summary",""),
            "key_strengths": _top_signals("fundamental", True, 3),
            "key_concerns":  _top_signals("fundamental", False, 2),
            "findings": [s["label"] for s in fund_score.get("signals",[])[:3]],
            "verdict":  fund_score.get("summary",""),
            "_source":  "rule_based",
        },
        "sentiment_analyst": {
            "sentiment_bias": "NEUTRAL", "market_mood": "NEUTRAL",
            "narrative": "Rule-based mode — no LLM sentiment analysis.",
            "news_momentum": "STABLE", "high_impact_headlines": [],
            "sentiment_score": 0.0, "behavioural_pressure": "NEUTRAL",
            "earnings_risk_flag": analyst_data.get("earnings_within_5d", False),
            "findings": [], "verdict": "Sentiment analysis unavailable (rule-based mode).",
            "_source": "rule_based",
        },
        "news_analyst": {
            "vix_regime": analyst_data.get("vix", {}).get("regime", "N/A"),
            "macro_bias": "NEUTRAL",
            "earnings_risk": analyst_data.get("earnings_within_5d", False) or analyst_data.get("earnings_within_21d", False),
            "timing_caution": "MINOR" if analyst_data.get("earnings_within_21d", False) else "NONE",
            "override_short_term": analyst_data.get("earnings_within_5d", False),
            "override_rationale": "Earnings within 5 days" if analyst_data.get("earnings_within_5d", False) else None,
            "sector_timing": "NEUTRAL",
            "key_macro_risks": [],
            "findings": [], "verdict": "News analysis unavailable (rule-based mode).",
            "_source": "rule_based",
        },
        "technical_analyst": {
            "formula_assessment": "N/A", "formulas_used": [],
            "statistical_bias": tech_score.get("verdict","NEUTRAL"),
            "worth_investing": (
                "YES" if tech_score.get("score",50) >= 66 else
                "CONDITIONAL" if tech_score.get("score",50) >= 45 else "NO"
            ),
            "worth_investing_rationale": tech_score.get("summary",""),
            "risk_level": risk_score_obj.get("verdict","MODERATE_RISK"),
            "risk_factors": _top_signals("risk", False, 3),
            "direction": trade_direction,
            "entry_price": entry_price,
            "stop_loss":   stop_loss,
            "findings": [s["label"] for s in tech_score.get("signals",[])[:3]],
            "verdict":  tech_score.get("summary",""),
            "price_predictions": {
                "short_term": {
                    k: {"price": target_prices[k]["price"], "change_pct": None, "method": "Monte Carlo/OLS", "confidence": "LOW", "rationale": "Rule-based estimate"}
                    for k in ("1_week","2_weeks","3_weeks","1_month")
                },
                "long_term": {
                    k: {"price": target_prices[k]["price"], "change_pct": None, "method": "Monte Carlo/OLS", "confidence": "LOW", "rationale": "Rule-based estimate"}
                    for k in ("3_months","6_months","9_months","12_months")
                },
            },
            "_source": "rule_based",
        },
        "bullish_researcher": {
            "bull_stance": "STRONG" if comp >= 66 else "MODERATE" if comp >= 45 else "WEAK",
            "bull_thesis": " | ".join(key_bull) if key_bull else "No strong bull signals identified.",
            "strongest_signals": key_bull[:4],
            "growth_catalysts": [],
            "upside_scenarios": {"base": f"${target_prices['3_months']['price']}", "optimistic": f"${target_prices['6_months']['price']}"},
            "bull_confidence": conf,
            "findings": key_bull,
            "verdict": f"Rule-based bull case: {rec}",
            "_source": "rule_based",
        },
        "bearish_researcher": {
            "bear_stance": "STRONG" if comp < 35 else "MODERATE" if comp < 50 else "WEAK",
            "bear_thesis": " | ".join(key_bear) if key_bear else "No strong bear signals identified.",
            "key_risks": key_risks[:4],
            "downside_scenarios": {"base": f"${target_prices['1_month']['price']}", "pessimistic": f"${target_prices['3_months']['price']}"},
            "bear_confidence": conf,
            "findings": key_bear,
            "verdict": f"Rule-based bear case: {rec}",
            "_source": "rule_based",
        },
        "synthesizer": {
            "net_bias": (
                "BULLISH" if comp >= 60 else
                "BEARISH" if comp < 45 else "NEUTRAL"
            ),
            "consensus_strength": "STRONG" if conf == "HIGH" else "MODERATE" if conf == "MEDIUM" else "WEAK",
            "agreed_points": key_bull[:2],
            "disputed_points": [],
            "resolution": "Rule-based scoring — no LLM debate.",
            "balanced_brief": rbj.get("summary",""),
            "key_bull_signals": key_bull[:3],
            "key_bear_signals": key_bear[:3],
            "findings": [],
            "verdict": f"Rule-based synthesis — score {comp:.0f}/100: {rec}",
            "_source": "rule_based",
        },
        "momentum_trader": {
            "trade_action": "ENTER_LONG" if rec in ("STRONG BUY","BUY") and trade_direction == "LONG" else "WAIT",
            "direction": trade_direction,
            "entry_price": entry_price, "stop_loss": stop_loss,
            "target_1w": target_prices["1_week"]["price"],
            "target_2w": target_prices["2_weeks"]["price"],
            "target_3w": target_prices["3_weeks"]["price"],
            "target_1m": target_prices["1_month"]["price"],
            "position_size_pct": position_size_pct,
            "momentum_score": "MODERATE" if tech_score.get("score",50) >= 66 else "WEAK",
            "trade_rationale": f"Rule-based momentum: {tech_score.get('verdict','')}",
            "findings": [], "verdict": f"Momentum: {rec} (rule-based).",
            "_source": "rule_based",
        },
        "value_trader": {
            "trade_action": "ENTER_LONG" if rec in ("STRONG BUY","BUY") else "WAIT",
            "direction": trade_direction,
            "entry_price": entry_price, "stop_loss": stop_loss,
            "target_3m":  target_prices["3_months"]["price"],
            "target_6m":  target_prices["6_months"]["price"],
            "target_9m":  target_prices["9_months"]["price"],
            "target_12m": target_prices["12_months"]["price"],
            "position_size_pct": position_size_pct,
            "margin_of_safety_pct": None,
            "value_score": (
                "DEEP_VALUE" if val_score.get("score",50) >= 66 else
                "FAIR_VALUE" if val_score.get("score",50) >= 45 else "OVERVALUED"
            ),
            "trade_rationale": f"Rule-based valuation: {val_score.get('verdict','')}",
            "findings": [], "verdict": f"Value: {rec} (rule-based).",
            "_source": "rule_based",
        },
        "swing_trader": {
            "trade_action": "ENTER_LONG" if rec in ("STRONG BUY","BUY") else "WAIT",
            "direction": trade_direction,
            "entry_price": entry_price, "stop_loss": stop_loss,
            "target_1w": target_prices["1_week"]["price"],
            "target_2w": target_prices["2_weeks"]["price"],
            "target_3w": target_prices["3_weeks"]["price"],
            "target_1m": target_prices["1_month"]["price"],
            "position_size_pct": position_size_pct,
            "swing_setup": "NONE",
            "nearest_support": None, "nearest_resistance": None,
            "trade_rationale": "Rule-based swing: based on Bollinger position.",
            "findings": [], "verdict": f"Swing: {rec} (rule-based).",
            "_source": "rule_based",
        },
        "market_risk": {
            "market_risk_level": risk_score_obj.get("verdict","MODERATE_RISK").replace("_RISK","").replace("_"," "),
            "vol_regime_risk": statistical.get("volatility",{}).get("vol_regime","STABLE"),
            "earnings_caution": (
                "AVOID" if analyst_data.get("earnings_within_5d",False) else
                "REDUCE_SIZE" if analyst_data.get("earnings_within_21d",False) else "NONE"
            ),
            "vix_risk_flag": (
                "DANGER" if analyst_data.get("vix",{}).get("regime") == "HIGH_FEAR" else
                "CAUTION" if analyst_data.get("vix",{}).get("regime") == "ELEVATED" else "CLEAR"
            ),
            "liquidity_risk": "LOW",
            "max_position_flag": "HALF_SIZE" if risk_score_obj.get("score",50) < 45 else "FULL_SIZE",
            "timing_recommendation": (
                "AVOID" if analyst_data.get("earnings_within_5d",False) else
                "REDUCE" if analyst_data.get("earnings_within_21d",False) else "PROCEED"
            ),
            "risk_adjustments": [],
            "findings": [s["label"] for s in risk_score_obj.get("signals",[])[:3]],
            "verdict": risk_score_obj.get("summary",""),
            "_source": "rule_based",
        },
        "portfolio_risk": {
            "kelly_fraction": min(0.25, max(0.0, (comp - 50) / 100)),
            "recommended_size_pct": position_size_pct,
            "sizing_method": "FIXED",
            "scaling_factor": "FULL" if position_size_pct >= 1.5 else "HALF" if position_size_pct <= 0.5 else "NORMAL",
            "max_risk_usd_per_1k": round(position_size_pct * 10, 2),
            "validated_entry": entry_price,
            "validated_stop": stop_loss,
            "stop_distance_usd": round(abs(entry_price - stop_loss), 4),
            "stop_validation": "VALID",
            "stop_validation_note": None,
            "portfolio_exposure_ok": True,
            "sizing_rationale": f"Rule-based sizing: risk score {risk_score_obj.get('score',50):.0f}/100.",
            "findings": [],
            "verdict": f"Position size {position_size_pct}% (rule-based).",
            "_source": "rule_based",
        },
    }

    return {
        "llm_available":      False,
        "recommendation":     rec,
        "confidence":         conf,
        "overall_short_term": ost,
        "overall_long_term":  olt,
        "target_prices":      target_prices,
        "trade_direction":    trade_direction,
        "entry_price":        entry_price,
        "exit_price":         exit_price,
        "stop_loss":          stop_loss,
        "position_size_pct":  position_size_pct,
        "key_bull_case":      key_bull,
        "key_bear_case":      key_bear,
        "key_risks":          key_risks,
        "catalysts":          [],
        "summary":            rbj.get("summary", ""),
        "technical_verdict":  rbj.get("technical_verdict",  ""),
        "fundamental_verdict":rbj.get("fundamental_verdict",""),
        "valuation_verdict":  rbj.get("valuation_verdict",  ""),
        "sentiment_verdict":  "N/A (rule-based mode — no LLM sentiment analysis).",
        "macro_verdict":      "N/A (rule-based mode — no LLM macro analysis).",
        "alternative_pick":   None,
        "alternative_reason": None,
        "agents":             stub_agents,
        "providers":          {},
        "description":        description,
        "rule_based_judge":   rbj,
        "_ml_accuracy":       (ml_result or {}).get("metrics", {}),
    }


# ------------------------------------------------------------------ #
# Quick judge — lightweight two-lens score for the discovery screener
# ------------------------------------------------------------------ #

def run_quick_judge(info: dict, technical: dict) -> dict:
    """
    Score a discovery candidate using only the fundamental and technical
    lenses (no statistical models, analyst data, or ML — those need the
    full pipeline and would make a broad market scan far too slow).

    Missing data is skipped, not penalised, exactly as in the full
    judge, so thin/young tickers degrade toward a neutral 50.

    Returns {"fundamental": {...}, "technical": {...}} — each the same
    dict shape the full lens scorers produce (score / grade / verdict /
    signals).
    """
    fund = _score_fundamental(info, {})
    tech = _score_technical(technical, {}, info)
    return {"fundamental": fund, "technical": tech}
