"""
analysis/llm_analysis.py — Multi-provider, multi-team LLM analysis for FinBot.

Agent architecture (five specialist teams + the judge):

  COMPANY DESCRIPTION (no LLM — pure yfinance data)
    Description Block   Assembles company background from Yahoo Finance:
                          sector, industry, country, employees, website,
                          business summary, and last 5 news headlines.
                          Zero API cost; always available.

  ANALYST TEAM  (4 agents — run in parallel)
    A1 — Fundamental Analyst   Evaluates company health, valuation, and
                                  capital allocation: ROE, margins, FCF, D/E,
                                  P/E, PEG, DCF, Graham Number. ST & LT verdicts.
    A2 — Sentiment Analyst     Interprets pre-scored news sentiment plus
                                  recent headlines to gauge market mood and
                                  near-term behavioural pressure.
    A3 — News Analyst          Macro regime, VIX, earnings calendar, sector
                                  rotation. Flags event risks and timing overrides.
    A4 — Technical Analyst     Selects peer-reviewed formulas (absorbs old
                                  Mathematician + Quant), produces 8-horizon
                                  price targets, entry/stop levels, and a
                                  formula_assessment.

  RESEARCHER TEAM  (Bull + Bear run in parallel -> Synthesizer)
    R1 — Bullish Researcher    Reads all 4 analyst reports; argues the
                                  strongest possible bull case.
    R2 — Bearish Researcher    Reads all 4 analyst reports; argues the
                                  strongest possible bear case.
    R3 — Research Synthesizer  Receives both research briefs; resolves
                                  disagreements and produces a balanced
                                  investment brief for the Trading Team.

  TRADING TEAM  (3 agents — run in parallel)
    T1 — Momentum Trader       Entry/exit/size from ADX, MACD, OBV, RSI
                                  momentum signals.
    T2 — Value Trader          Entry/exit/size from DCF, P/E, Graham Number,
                                  PEG valuation signals.
    T3 — Swing Trader          Entry/exit/size from Fibonacci, Bollinger,
                                  pivot levels, and short-term timing signals.

  RISK MANAGEMENT TEAM  (2 agents — run in parallel)
    RM1 — Market Risk Agent    Vol regime, VIX, beta, drawdown, and
                                  earnings-proximity risk assessment.
    RM2 — Portfolio Risk Agent Kelly/ATR position sizing, stop placement,
                                  and portfolio exposure limits.

  THE JUDGE  (1 agent — serial, receives all 5 teams)
    J — The Judge              Synthesises all 5 team reports + ML signals
                                -> final recommendation, 8-horizon price
                                  targets each with an accuracy_pct,
                                  entry/stop/exit, bull/bear cases, risks,
                                  and catalysts.

All LLM agents use the ReAct prompting framework
(Thought -> Action -> Observation -> Answer).

Gracefully degrades: if no LLM credentials are present the function
returns a placeholder result so the rest of the program continues.
"""

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from json_repair import repair_json
from openai import AzureOpenAI, OpenAI, APIConnectionError, APITimeoutError
from google import genai
from google.genai import types as genai_types

from config import (
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    AZURE_ENABLED,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    GOOGLE_ENABLED,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_ENABLED,
    LLM_ENABLED,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS_AZURE,
    LLM_MAX_TOKENS_GOOGLE,
    LLM_MAX_TOKENS_DEEPSEEK,
    LLM_TIMEOUT,
    LLM_MAX_RETRIES,
    LLM_CIRCUIT_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ================================================================== #
# Per-provider circuit breaker
# ================================================================== #

def _is_connection_error(exc: Exception) -> bool:
    """
    True when *exc* looks like the provider is unreachable/hanging (as
    opposed to a per-agent content error like a JSON parse failure, which
    should NOT trip the breaker). Covers the openai SDK's connection and
    timeout classes plus common connection signatures from any provider.
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = ("connection error", "connection aborted", "connection refused",
               "connection reset", "timed out", "timeout", "getaddrinfo",
               "max retries", "name or service not known", "temporary failure",
               "failed to establish", "unavailable", "deadline exceeded")
    return any(m in text for m in markers)


class _CircuitBreaker:
    """
    Trips after LLM_CIRCUIT_THRESHOLD consecutive connection-class failures
    for one provider, after which the provider's remaining agents in the
    13-agent chain are skipped immediately instead of each re-hitting the
    dead endpoint (and paying the full timeout). Thread-safe: wave 1 runs
    four agents in parallel against the same breaker.
    """

    def __init__(self, provider: str, threshold: int = 2):
        self.provider  = provider
        self.threshold = max(1, threshold)
        self._failures = 0
        self.tripped   = False
        self.reason    = ""
        self.tripped_at = ""        # human label of the agent/wave that tripped it
        self._lock     = threading.Lock()

    def is_open(self) -> bool:
        return self.tripped

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0

    def record_failure(self, exc: Exception, where: str = "") -> None:
        with self._lock:
            if not _is_connection_error(exc):
                return          # content error — transient, don't trip
            self._failures += 1
            if self._failures >= self.threshold and not self.tripped:
                self.tripped    = True
                self.reason     = str(exc)
                self.tripped_at = where
                logger.warning("LLM provider %s circuit OPEN after %s: %s",
                               self.provider, where or "repeated failures", exc)

# ReAct system instruction injected into every agent call
_REACT_SYSTEM = (
    "You are a senior quantitative analyst operating within a multi-agent investment research "
    "system. Before producing your final JSON answer, reason through the data using the "
    "ReAct framework:\n\n"
    "THOUGHT: <step-by-step analysis of the data provided>\n"
    "ACTION:  <what you are evaluating or computing>\n"
    "OBSERVATION: <what the data reveals>\n"
    "ANSWER: <your final JSON output>\n\n"
    "Respond ONLY with valid JSON after the ANSWER: marker. "
    "No markdown, no code fences, no extra text outside the JSON object."
)


# ================================================================== #
# DESCRIPTION BLOCK -- pure yfinance, zero LLM calls
# ================================================================== #

def build_company_description(info: dict, analyst_data: dict) -> dict:
    """
    Build a company description dict entirely from Yahoo Finance data.
    No LLM call is made. Always succeeds even with partial data.
    """
    import re as _re
    name       = info.get("longName") or info.get("shortName", "Unknown")
    ticker_sym = info.get("symbol", "")
    sector     = info.get("sector", "N/A")
    industry   = info.get("industry", "N/A")
    country    = info.get("country", "N/A")
    website    = info.get("website", "N/A")
    employees  = info.get("fullTimeEmployees")
    employees_str = f"{employees:,}" if employees else "N/A"
    summary    = info.get("longBusinessSummary", "")
    news       = analyst_data.get("news", [])
    news_titles = [n.get("title", "") for n in news[:5] if n.get("title")]
    founding_year = "Unknown"
    year_match = _re.search(r'\b(1[89]\d{2}|20[0-2]\d)\b', summary[:500])
    if year_match:
        founding_year = year_match.group(0)
    return {
        "company_name":        name,
        "ticker":              ticker_sym,
        "sector":              sector,
        "industry":            industry,
        "country":             country,
        "website":             website,
        "employees":           employees_str,
        "founding_year":       founding_year,
        "business_overview":   summary,
        "recent_news":         news_titles,
        "_source":             "yfinance",
    }


# ================================================================== #
# ANALYST TEAM -- Agent A1: Fundamental Analyst
# ================================================================== #

def _build_fundamental_analyst_prompt(ticker, info, fundamental, analyst_data, statistical) -> str:
    price    = info.get("currentPrice", "N/A")
    sector   = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    pe       = fundamental.get("pe", {})
    peg      = fundamental.get("peg", {})
    pb_div   = fundamental.get("pb_div", {})
    graham   = fundamental.get("graham", {})
    dcf      = fundamental.get("dcf", {})
    fcf      = info.get("freeCashflow")
    mktcap   = info.get("marketCap")
    dte      = info.get("debtToEquity")
    roe      = info.get("returnOnEquity")
    op_margin = info.get("operatingMargins")
    rev_growth = info.get("revenueGrowth")
    buybacks   = info.get("sharesPercentSharesOut")
    rec_key    = analyst_data.get("recommendation_key", "N/A").upper()
    pt         = analyst_data.get("price_target") or {}
    recs       = analyst_data.get("recommendations", [])
    n_up   = sum(1 for r in recs[:10] if "Upgrade"   in r.get("action", ""))
    n_down = sum(1 for r in recs[:10] if "Downgrade" in r.get("action", ""))
    fcf_yield  = f"{fcf/mktcap*100:.1f}%" if fcf and mktcap else "N/A"
    dte_str    = f"{dte:.1f}" if dte is not None else "N/A"
    roe_str    = f"{roe*100:.1f}%" if roe is not None else "N/A"
    op_str     = f"{op_margin*100:.1f}%" if op_margin is not None else "N/A"
    rev_str    = f"{rev_growth*100:.1f}%" if rev_growth is not None else "N/A"
    bb_str     = f"{buybacks*100:.2f}%/yr" if buybacks is not None else "N/A"
    ev_ebitda  = info.get("enterpriseToEbitda")
    ev_str     = f"{ev_ebitda:.1f}x" if ev_ebitda else "N/A"
    return (
        f"You are a Fundamental Analyst. Assess the financial health and valuation of {ticker} "
        f"using the data below. Apply the ReAct framework before producing your JSON.\n\n"
        f"TICKER: {ticker} | PRICE: ${price} | SECTOR: {sector} | INDUSTRY: {industry}\n\n"
        f"HEALTH METRICS:\n"
        f"  ROE: {roe_str} | Operating Margin: {op_str} | Revenue Growth (YoY): {rev_str}\n"
        f"  FCF Yield: {fcf_yield} | Debt/Equity: {dte_str} | Buyback Rate: {bb_str}\n\n"
        f"VALUATION METRICS:\n"
        f"  Trailing P/E: {pe.get('trailing_pe','N/A')} ({pe.get('trailing_signal','')})\n"
        f"  Forward P/E: {pe.get('forward_pe','N/A')} ({pe.get('forward_signal','')})\n"
        f"  PEG: {peg.get('peg','N/A')} ({peg.get('signal','')})\n"
        f"  EV/EBITDA: {ev_str}\n"
        f"  DCF Intrinsic: ${dcf.get('dcf_value','N/A')} ({dcf.get('signal','')})\n"
        f"  Graham Number: ${graham.get('graham_number','N/A')} ({graham.get('signal','')})\n"
        f"  P/B: {pb_div.get('price_to_book','N/A')} | Div Yield: {pb_div.get('dividend_yield','N/A')}%\n\n"
        f"ANALYST CONSENSUS: {rec_key} | Target ${pt.get('mean','N/A')} "
        f"(Low ${pt.get('low','N/A')} / High ${pt.get('high','N/A')})\n"
        f"Last 10 actions: {n_up} upgrades, {n_down} downgrades\n\n"
        f"TASKS:\n"
        f"1. Evaluate financial health (ROE, margins, FCF, debt load).\n"
        f"2. Assess valuation vs peers (P/E, PEG, DCF, Graham Number).\n"
        f"3. Assess capital allocation quality (FCF yield, buybacks, D/E).\n"
        f"4. SHORT-TERM (1-4 wk) verdict: is the fundamental picture supportive?\n"
        f"5. LONG-TERM (3-12 mo) verdict: is the valuation attractive at this price?\n\n"
        '{"business_health":"STRONG"|"ADEQUATE"|"WEAK",'
        '"valuation_stance":"UNDERVALUED"|"FAIRLY_VALUED"|"OVERVALUED",'
        '"capital_allocation":"EXCELLENT"|"ADEQUATE"|"POOR",'
        '"profitability_quality":"HIGH"|"MEDIUM"|"LOW",'
        '"competitive_position":"LEADER"|"STRONG"|"AVERAGE"|"WEAK",'
        '"analyst_conviction":"STRONG"|"MODERATE"|"WEAK",'
        '"short_term_verdict":"WORTH_INVESTING"|"NEUTRAL"|"NOT_WORTH_INVESTING",'
        '"short_term_rationale":"<max 20 words>",'
        '"long_term_verdict":"WORTH_INVESTING"|"NEUTRAL"|"NOT_WORTH_INVESTING",'
        '"long_term_rationale":"<max 20 words>",'
        '"key_strengths":["<max 20 words>","<max 20 words>","<max 20 words>"],'
        '"key_concerns":["<max 20 words>","<max 20 words>"],'
        '"findings":["<max 20 words, number-backed>","<max 20 words>","<max 20 words>"],'
        '"verdict":"<1 sentence summarising the fundamental picture>"}'
    )


# ================================================================== #
# ANALYST TEAM -- Agent A2: Sentiment Analyst
# ================================================================== #

def _build_sentiment_analyst_prompt(ticker, info, analyst_data, sentiment: dict) -> str:
    price     = info.get("currentPrice", "N/A")
    sector    = info.get("sector", "N/A")
    news      = analyst_data.get("news", [])
    headlines = "\n".join(f"  {i+1}. {n.get('title','')}" for i, n in enumerate(news[:10]))
    score     = sentiment.get("overall_score", 0.0)
    label     = sentiment.get("label", "NEUTRAL")
    sent_summ = sentiment.get("summary", "")
    vix       = analyst_data.get("vix", {})
    e5d       = analyst_data.get("earnings_within_5d", False)
    e21d      = analyst_data.get("earnings_within_21d", False)
    return (
        f"You are a Sentiment Analyst. Interpret the pre-scored sentiment data and recent news "
        f"headlines for {ticker} to gauge market mood and behavioural pressure. Apply the ReAct "
        f"framework.\n\n"
        f"TICKER: {ticker} | PRICE: ${price} | SECTOR: {sector}\n\n"
        f"PRE-SCORED SENTIMENT (NLP model output):\n"
        f"  Score: {score:+.2f} (range -1.0 to +1.0) | Label: {label}\n"
        f"  Summary: {sent_summ or 'Not provided'}\n\n"
        f"VIX: {vix.get('value','N/A')} -> {vix.get('regime','N/A')}\n"
        f"EARNINGS WITHIN 5 DAYS: {e5d} | WITHIN 21 DAYS: {e21d}\n\n"
        f"RECENT NEWS HEADLINES (up to 10):\n"
        f"{headlines or '  No headlines available.'}\n\n"
        f"TASKS:\n"
        f"1. Interpret the numerical sentiment score in context of the headlines.\n"
        f"2. Identify the dominant narrative (fear, greed, uncertainty, euphoria).\n"
        f"3. Assess whether news sentiment confirms or conflicts with the pre-scored label.\n"
        f"4. Flag any high-impact single headlines (earnings surprise, regulatory risk, M&A).\n"
        f"5. Project the near-term behavioural impact on price.\n\n"
        f"OUTPUT only this JSON after your ReAct reasoning:\n"
        + '{"sentiment_bias":"BULLISH"|"BEARISH"|"NEUTRAL",'
        '"market_mood":"FEAR"|"GREED"|"UNCERTAINTY"|"EUPHORIA"|"NEUTRAL",'
        '"narrative":"<dominant story in 20 words>",'
        '"news_momentum":"ACCELERATING"|"STABLE"|"DECELERATING",'
        '"high_impact_headlines":["<headline <=15 words>","<headline <=15 words>"],'
        f'"sentiment_score":{score:.2f},'
        '"behavioural_pressure":"BUYING_PRESSURE"|"SELLING_PRESSURE"|"NEUTRAL",'
        f'"earnings_risk_flag":{str(e5d or e21d).lower()},'
        '"findings":["<max 20 words>","<max 20 words>","<max 20 words>"],'
        '"verdict":"<1 sentence on how sentiment will impact near-term price>"}'
    )


# ================================================================== #
# ANALYST TEAM -- Agent A3: News Analyst
# ================================================================== #

def _build_news_analyst_prompt(ticker, info, technical, statistical, analyst_data) -> str:
    sector    = info.get("sector", "N/A")
    vix       = analyst_data.get("vix", {})
    e5d       = analyst_data.get("earnings_within_5d", False)
    e21d      = analyst_data.get("earnings_within_21d", False)
    earn_date = analyst_data.get("next_earnings_date", "Unknown")
    beta      = statistical.get("beta", {})
    vol       = statistical.get("volatility", {})
    vol_pct   = f"{vol['sigma_annual']*100:.1f}%" if isinstance(vol.get("sigma_annual"), float) else "N/A"
    vol_regime = vol.get("vol_regime", "N/A")
    vol_ratio  = vol.get("vol_regime_ratio", "N/A")
    drawdown   = statistical.get("drawdown", {})
    tech_s     = technical.get("signals", {})
    pt         = analyst_data.get("price_target") or {}
    news       = analyst_data.get("news", [])
    news_str   = "\n".join(f"  {i+1}. {n.get('title','')}" for i, n in enumerate(news[:5]))
    earn_warn  = ("** EARNINGS WITHIN 5 DAYS -- binary event" if e5d
                  else "** EARNINGS WITHIN 21 DAYS" if e21d
                  else f"No near-term earnings (next: {earn_date})")
    return (
        f"You are a News Analyst specialising in macro regime identification, earnings risk, "
        f"and sector rotation. Apply the ReAct framework.\n\n"
        f"TICKER: {ticker} | SECTOR: {sector}\n\n"
        f"MACRO / REGIME DATA:\n"
        f"  VIX: {vix.get('value','N/A')} -> {vix.get('regime','N/A')}\n"
        f"  Earnings: {earn_warn}\n"
        f"  sigma_annual: {vol_pct} | Vol Regime: {vol_regime} (30d/2yr ratio={vol_ratio})\n"
        f"  Beta: {beta.get('beta','N/A')} ({beta.get('label','N/A')})\n"
        f"  MaxDD: {drawdown.get('max_drawdown_pct','N/A')}% | Calmar: {drawdown.get('calmar','N/A')}\n"
        f"  ADX: {tech_s.get('ADX','N/A')} | OBV: {tech_s.get('OBV','N/A')}\n"
        f"  Analyst Target: Mean=${pt.get('mean','N/A')} (Low=${pt.get('low','N/A')} / High=${pt.get('high','N/A')})\n\n"
        f"TOP NEWS HEADLINES:\n"
        f"{news_str or '  None available.'}\n\n"
        f"TASKS:\n"
        f"1. Classify the current VIX regime and its implications for sizing and stops.\n"
        f"2. Flag earnings risk: should traders avoid or reduce positions ahead of earnings?\n"
        f"3. Assess sector rotation: does current macro timing support or oppose a position?\n"
        f"4. Determine if a short-term timing OVERRIDE is warranted.\n"
        f"5. State your overall macro bias (RISK_ON / NEUTRAL / RISK_OFF).\n\n"
        f"OUTPUT only this JSON after your ReAct reasoning:\n"
        '{"vix_regime":"CALM"|"ELEVATED"|"HIGH_FEAR",'
        '"macro_bias":"RISK_ON"|"NEUTRAL"|"RISK_OFF",'
        '"earnings_risk":true|false,'
        '"timing_caution":"NONE"|"MINOR"|"MAJOR",'
        '"override_short_term":true|false,'
        '"override_rationale":"<20 words or null>",'
        '"sector_timing":"FAVOURABLE"|"NEUTRAL"|"UNFAVOURABLE",'
        '"key_macro_risks":["<max 20 words>","<max 20 words>"],'
        '"findings":["<max 20 words>","<max 20 words>","<max 20 words>"],'
        '"verdict":"<1 sentence on macro regime and its impact on this trade>"}'
    )


# ================================================================== #
# ANALYST TEAM -- Agent A4: Technical Analyst
# (absorbs Mathematician + Quant Analyst -- full 8-horizon math targets)
# ================================================================== #

def _build_technical_analyst_prompt(ticker, info, technical, fundamental, statistical) -> str:
    price    = info.get("currentPrice", "N/A")
    hi52     = info.get("fiftyTwoWeekHigh", "N/A")
    lo52     = info.get("fiftyTwoWeekLow",  "N/A")
    tech_l   = technical.get("latest", {})
    tech_s   = technical.get("signals", {})
    fibs     = technical.get("fibonacci", {})
    pivots   = technical.get("pivot_levels", {})
    vol      = statistical.get("volatility", {})
    beta     = statistical.get("beta", {})
    sharpe   = statistical.get("sharpe", {})
    sortino  = statistical.get("sortino", {})
    drawdown = statistical.get("drawdown", {})
    reg      = statistical.get("regression", {})
    mc       = statistical.get("monte_carlo", {})
    mc_preds = mc.get("predictions", {})
    mc_params= mc.get("params", {})
    pe_data  = fundamental.get("pe", {})
    dcf      = fundamental.get("dcf", {})
    vol_pct  = f"{vol['sigma_annual']*100:.1f}%" if isinstance(vol.get("sigma_annual"), float) else "N/A"
    atr_val  = tech_l.get("ATR")
    atr_str  = f"${atr_val:.4f}" if atr_val is not None else "N/A"
    adx_val  = tech_l.get("ADX")
    adx_str  = f"{adx_val:.1f}" if adx_val is not None else "N/A"
    fib_str  = " | ".join(f"{k}: ${v}" for k, v in list(fibs.items())[:6]) or "N/A"
    supp_str = " | ".join(f"${s}" for s in pivots.get("support_levels", [])[:4]) or "N/A"
    res_str  = " | ".join(f"${r}" for r in pivots.get("resistance_levels", [])[:4]) or "N/A"
    reg_str  = " | ".join(
        f"{h}: ${d['price']} ({d['change%']:+.1f}%)"
        for h, d in reg.get("predictions", {}).items()
    ) or "N/A"
    mc_str = " | ".join(
        f"{h}: P10=${mc.get('p10','N/A')} Med=${mc.get('median','N/A')} P90=${mc.get('p90','N/A')}"
        for h, mc in mc_preds.items()
    ) or "N/A"
    json_template = (
        '{"formula_assessment":"VALID"|"IMPROVED"|"PARTIALLY_VALID"|"INVALID",'
        '"formulas_used":[{"name":"<formula>","formula":"<expression with actual numbers>",'
        '"rationale":"<why now>","result":"<what it reveals>"}],'
        '"statistical_bias":"BULLISH"|"BEARISH"|"NEUTRAL",'
        '"worth_investing":"YES"|"NO"|"CONDITIONAL",'
        '"worth_investing_rationale":"<2 sentences on risk/reward>",'
        '"risk_level":"HIGH"|"MEDIUM"|"LOW",'
        '"risk_factors":["<specific risk>","<specific risk>","<specific risk>"],'
        '"direction":"LONG"|"SHORT",'
        '"entry_price":<float>,'
        '"stop_loss":<float>,'
        '"findings":["<max 20 words, number-backed>","<max 20 words>","<max 20 words>"],'
        '"verdict":"<1 sentence quant summary>",'
        '"price_predictions":{'
        '"short_term":{'
        '"1_week":{"price":<float>,"change_pct":<float>,"method":"<formula>",'
        '"confidence":"HIGH"|"MEDIUM"|"LOW","rationale":"<max 20 words>"},'
        '"2_weeks":{"price":<float>,"change_pct":<float>,"method":"<formula>",'
        '"confidence":"HIGH"|"MEDIUM"|"LOW","rationale":"<max 20 words>"},'
        '"3_weeks":{"price":<float>,"change_pct":<float>,"method":"<formula>",'
        '"confidence":"HIGH"|"MEDIUM"|"LOW","rationale":"<max 20 words>"},'
        '"1_month":{"price":<float>,"change_pct":<float>,"method":"<formula>",'
        '"confidence":"HIGH"|"MEDIUM"|"LOW","rationale":"<max 20 words>"}},'
        '"long_term":{'
        '"3_months":{"price":<float>,"change_pct":<float>,"method":"<formula>",'
        '"confidence":"HIGH"|"MEDIUM"|"LOW","rationale":"<max 20 words>"},'
        '"6_months":{"price":<float>,"change_pct":<float>,"method":"<formula>",'
        '"confidence":"HIGH"|"MEDIUM"|"LOW","rationale":"<max 20 words>"},'
        '"9_months":{"price":<float>,"change_pct":<float>,"method":"<formula>",'
        '"confidence":"HIGH"|"MEDIUM"|"LOW","rationale":"<max 20 words>"},'
        '"12_months":{"price":<float>,"change_pct":<float>,"method":"<formula>",'
        '"confidence":"HIGH"|"MEDIUM"|"LOW","rationale":"<max 20 words>"}}}}'
    )
    return (
        f"You are a Technical Analyst. Select the most appropriate peer-reviewed quantitative "
        f"formulas for {ticker}, apply them rigorously, derive price estimates at 8 horizons, "
        f"then produce the final recommendation with trade setup. Apply the ReAct framework.\n\n"
        f"IMPORTANT: Do NOT use Elliott Wave Theory or Harmonic patterns. "
        f"Use only established, peer-reviewed statistical and investment formulas.\n\n"
        f"TICKER: {ticker} | PRICE: ${price} | 52W High: ${hi52} | 52W Low: ${lo52}\n"
        f"RISK: sigma_annual={vol_pct} | ATR(14)={atr_str} | Beta={beta.get('beta','N/A')} ({beta.get('label','N/A')})\n"
        f"RETURN METRICS: Sharpe={sharpe.get('sharpe','N/A')} | Sortino={sortino.get('sortino','N/A')} | "
        f"MaxDD={drawdown.get('max_drawdown_pct','N/A')}% | Calmar={drawdown.get('calmar','N/A')}\n"
        f"TREND: ADX={adx_str} ({tech_s.get('ADX','N/A')}) | OBV={tech_s.get('OBV','N/A')}\n"
        f"TECHNICALS: RSI={tech_l.get('RSI','N/A')} ({tech_s.get('RSI','N/A')}) | "
        f"MACD={tech_s.get('MACD','N/A')} | MA={tech_s.get('MA_Trend','N/A')} | BB={tech_s.get('Bollinger','N/A')}\n"
        f"SMA20={tech_l.get('SMA_20','N/A')} | SMA50={tech_l.get('SMA_50','N/A')} | "
        f"SMA200={tech_l.get('SMA_200','N/A')} | EMA12={tech_l.get('EMA_12','N/A')} | EMA26={tech_l.get('EMA_26','N/A')}\n"
        f"BB_Upper={tech_l.get('BB_Upper','N/A')} | BB_Middle={tech_l.get('BB_Middle','N/A')} | "
        f"BB_Lower={tech_l.get('BB_Lower','N/A')} | BB_PctB={tech_l.get('BB_PctB','N/A')}\n"
        f"FIBONACCI: {fib_str}\n"
        f"SUPPORTS: {supp_str} | RESISTANCES: {res_str}\n"
        f"FUNDAMENTALS: P/E={pe_data.get('trailing_pe','N/A')} | Fwd P/E={pe_data.get('forward_pe','N/A')} | "
        f"DCF=${dcf.get('dcf_value','N/A')} ({dcf.get('signal','')})\n"
        f"OLS REGRESSION: {reg_str} | R2={reg.get('r_squared','N/A')} | Dir={reg.get('trend_dir','N/A')}\n"
        f"MONTE CARLO ({mc_params.get('mu_method','N/A')}): {mc_str}\n\n"
        f"AVAILABLE FORMULAS (select 3-5 most appropriate):\n"
        f"1. CAPM Expected Return: E(R) = Rf + beta x ERP -- best for 3-12 month horizons.\n"
        f"2. Z-Score Mean Reversion: Z = (P - BB_Middle) / (BB_Width/4) -- best for 1-3 wk RANGING (ADX<25).\n"
        f"3. Momentum / ROC: project from SMA slope/percent change -- best for 1-4 wk TRENDING (ADX>25).\n"
        f"4. OLS Regression Trend Extrapolation: extend pre-computed slope -- best for medium/long trend.\n"
        f"5. Volatility Price Cone: P +/- (sigma_annual / sqrt(252/h)) x P -- validates range realism.\n"
        f"6. DCF Convergence Model: interpolate current price -> DCF intrinsic over 12 months.\n"
        f"7. Sharpe-Implied Return: P x (1 + annual_return)^(h/252) -- best for 3-12 month.\n"
        f"8. Bollinger Band Mean Reversion: BB_Upper/Lower as near-term targets -- RANGING only.\n\n"
        f"REGIME GUIDANCE:\n"
        f"- ADX > 25 (TRENDING):   weight Momentum/ROC, CAPM, OLS. Avoid mean-reversion for short-term.\n"
        f"- ADX < 20 (RANGING):    weight Z-Score, Bollinger 1-4 wk. CAPM/DCF for long-term.\n"
        f"- 20 <= ADX <= 25 (MIXED): blend both; MEDIUM confidence short-term.\n\n"
        f"OUTPUT only this JSON after your ReAct reasoning:\n"
        + json_template
    )


# ================================================================== #
# RESEARCHER TEAM -- Agent R1: Bullish Researcher
# ================================================================== #

def _build_bullish_researcher_prompt(
    ticker, info,
    fundamental_analyst: dict,
    sentiment_analyst: dict,
    news_analyst: dict,
    technical_analyst: dict,
) -> str:
    price   = info.get("currentPrice", "N/A")
    sector  = info.get("sector", "N/A")
    st_tech = technical_analyst.get("price_predictions", {}).get("short_term", {})
    lt_tech = technical_analyst.get("price_predictions", {}).get("long_term", {})
    return (
        f"You are a Bullish Researcher in a two-sided research debate. Your role is to build "
        f"the STRONGEST POSSIBLE bull case for {ticker} from the analyst reports below. "
        f"Be rigorous -- only cite genuinely positive signals. Apply the ReAct framework.\n\n"
        f"TICKER: {ticker} | PRICE: ${price} | SECTOR: {sector}\n\n"
        f"=== FUNDAMENTAL ANALYST ===\n"
        f"  Health: {fundamental_analyst.get('business_health','N/A')} | "
        f"Valuation: {fundamental_analyst.get('valuation_stance','N/A')} | "
        f"Capital Alloc: {fundamental_analyst.get('capital_allocation','N/A')}\n"
        f"  ST: {fundamental_analyst.get('short_term_verdict','N/A')} -- "
        f"{fundamental_analyst.get('short_term_rationale','N/A')}\n"
        f"  LT: {fundamental_analyst.get('long_term_verdict','N/A')} -- "
        f"{fundamental_analyst.get('long_term_rationale','N/A')}\n"
        f"  Strengths: {' | '.join(fundamental_analyst.get('key_strengths',[]))}\n"
        f"  Verdict: {fundamental_analyst.get('verdict','N/A')}\n\n"
        f"=== SENTIMENT ANALYST ===\n"
        f"  Bias: {sentiment_analyst.get('sentiment_bias','N/A')} | "
        f"Mood: {sentiment_analyst.get('market_mood','N/A')} | "
        f"Pressure: {sentiment_analyst.get('behavioural_pressure','N/A')}\n"
        f"  Narrative: {sentiment_analyst.get('narrative','N/A')}\n"
        f"  Verdict: {sentiment_analyst.get('verdict','N/A')}\n\n"
        f"=== NEWS ANALYST ===\n"
        f"  VIX: {news_analyst.get('vix_regime','N/A')} | "
        f"Macro: {news_analyst.get('macro_bias','N/A')} | "
        f"Sector Timing: {news_analyst.get('sector_timing','N/A')}\n"
        f"  Override ST: {news_analyst.get('override_short_term','N/A')}\n"
        f"  Verdict: {news_analyst.get('verdict','N/A')}\n\n"
        f"=== TECHNICAL ANALYST ===\n"
        f"  Bias: {technical_analyst.get('statistical_bias','N/A')} | "
        f"Direction: {technical_analyst.get('direction','N/A')} | "
        f"Worth: {technical_analyst.get('worth_investing','N/A')}\n"
        f"  1W=${st_tech.get('1_week',{}).get('price','N/A')} | "
        f"1M=${st_tech.get('1_month',{}).get('price','N/A')} | "
        f"12M=${lt_tech.get('12_months',{}).get('price','N/A')}\n"
        f"  Verdict: {technical_analyst.get('verdict','N/A')}\n\n"
        f"OUTPUT only this JSON after your ReAct reasoning:\n"
        '{"bull_stance":"STRONG"|"MODERATE"|"WEAK",'
        '"bull_thesis":"<3-4 sentence comprehensive bull case>",'
        '"strongest_signals":["<signal with number <=20 words>","<signal <=20 words>",'
        '"<signal <=20 words>","<signal <=20 words>"],'
        '"growth_catalysts":["<catalyst <=15 words>","<catalyst <=15 words>","<catalyst <=15 words>"],'
        '"upside_scenarios":{"base":"<price scenario <=15 words>","optimistic":"<price scenario <=15 words>"},'
        '"bull_confidence":"HIGH"|"MEDIUM"|"LOW",'
        '"findings":["<max 20 words>","<max 20 words>"],'
        '"verdict":"<1 sentence bull summary>"}'
    )


# ================================================================== #
# RESEARCHER TEAM -- Agent R2: Bearish Researcher
# ================================================================== #

def _build_bearish_researcher_prompt(
    ticker, info,
    fundamental_analyst: dict,
    sentiment_analyst: dict,
    news_analyst: dict,
    technical_analyst: dict,
) -> str:
    price   = info.get("currentPrice", "N/A")
    sector  = info.get("sector", "N/A")
    st_tech = technical_analyst.get("price_predictions", {}).get("short_term", {})
    lt_tech = technical_analyst.get("price_predictions", {}).get("long_term", {})
    return (
        f"You are a Bearish Researcher in a two-sided research debate. Your role is to build "
        f"the STRONGEST POSSIBLE bear case for {ticker} from the analyst reports below. "
        f"Be rigorous -- only cite genuine risks and negatives. Apply the ReAct framework.\n\n"
        f"TICKER: {ticker} | PRICE: ${price} | SECTOR: {sector}\n\n"
        f"=== FUNDAMENTAL ANALYST ===\n"
        f"  Health: {fundamental_analyst.get('business_health','N/A')} | "
        f"Valuation: {fundamental_analyst.get('valuation_stance','N/A')}\n"
        f"  ST: {fundamental_analyst.get('short_term_verdict','N/A')} | "
        f"LT: {fundamental_analyst.get('long_term_verdict','N/A')}\n"
        f"  Concerns: {' | '.join(fundamental_analyst.get('key_concerns',[]))}\n"
        f"  Verdict: {fundamental_analyst.get('verdict','N/A')}\n\n"
        f"=== SENTIMENT ANALYST ===\n"
        f"  Bias: {sentiment_analyst.get('sentiment_bias','N/A')} | "
        f"Mood: {sentiment_analyst.get('market_mood','N/A')} | "
        f"Pressure: {sentiment_analyst.get('behavioural_pressure','N/A')}\n"
        f"  Earnings Risk: {sentiment_analyst.get('earnings_risk_flag','N/A')}\n"
        f"  Verdict: {sentiment_analyst.get('verdict','N/A')}\n\n"
        f"=== NEWS ANALYST ===\n"
        f"  VIX: {news_analyst.get('vix_regime','N/A')} | "
        f"Macro: {news_analyst.get('macro_bias','N/A')} | "
        f"Timing: {news_analyst.get('timing_caution','N/A')}\n"
        f"  Override ST: {news_analyst.get('override_short_term','N/A')} -- "
        f"{news_analyst.get('override_rationale','N/A')}\n"
        f"  Macro Risks: {' | '.join(news_analyst.get('key_macro_risks',[]))}\n"
        f"  Verdict: {news_analyst.get('verdict','N/A')}\n\n"
        f"=== TECHNICAL ANALYST ===\n"
        f"  Bias: {technical_analyst.get('statistical_bias','N/A')} | "
        f"Risk: {technical_analyst.get('risk_level','N/A')}\n"
        f"  1W=${st_tech.get('1_week',{}).get('price','N/A')} | "
        f"1M=${st_tech.get('1_month',{}).get('price','N/A')} | "
        f"12M=${lt_tech.get('12_months',{}).get('price','N/A')}\n"
        f"  Risk Factors: {' | '.join(technical_analyst.get('risk_factors',[]))}\n"
        f"  Verdict: {technical_analyst.get('verdict','N/A')}\n\n"
        f"OUTPUT only this JSON after your ReAct reasoning:\n"
        '{"bear_stance":"STRONG"|"MODERATE"|"WEAK",'
        '"bear_thesis":"<3-4 sentence comprehensive bear case>",'
        '"key_risks":["<risk with number <=20 words>","<risk <=20 words>",'
        '"<risk <=20 words>","<risk <=20 words>"],'
        '"downside_scenarios":{"base":"<scenario <=15 words>","pessimistic":"<scenario <=15 words>"},'
        '"bear_confidence":"HIGH"|"MEDIUM"|"LOW",'
        '"findings":["<max 20 words>","<max 20 words>"],'
        '"verdict":"<1 sentence bear summary>"}'
    )


# ================================================================== #
# RESEARCHER TEAM -- Agent R3: Research Synthesizer
# ================================================================== #

def _build_research_synthesizer_prompt(
    ticker, info,
    bullish_researcher: dict,
    bearish_researcher: dict,
) -> str:
    price  = info.get("currentPrice", "N/A")
    sector = info.get("sector", "N/A")
    return (
        f"You are a Research Synthesizer. Your role is to resolve the dialectical debate between "
        f"the Bull and Bear researchers for {ticker} and produce a balanced, objective brief for "
        f"the Trading Team. Apply the ReAct framework.\n\n"
        f"TICKER: {ticker} | PRICE: ${price} | SECTOR: {sector}\n\n"
        f"=== BULLISH RESEARCHER ===\n"
        f"  Stance: {bullish_researcher.get('bull_stance','N/A')} | "
        f"Confidence: {bullish_researcher.get('bull_confidence','N/A')}\n"
        f"  Thesis: {bullish_researcher.get('bull_thesis','N/A')}\n"
        f"  Strongest Signals: {' | '.join(bullish_researcher.get('strongest_signals',[]))}\n"
        f"  Catalysts: {' | '.join(bullish_researcher.get('growth_catalysts',[]))}\n"
        f"  Upside Base: {bullish_researcher.get('upside_scenarios',{}).get('base','N/A')}\n"
        f"  Verdict: {bullish_researcher.get('verdict','N/A')}\n\n"
        f"=== BEARISH RESEARCHER ===\n"
        f"  Stance: {bearish_researcher.get('bear_stance','N/A')} | "
        f"Confidence: {bearish_researcher.get('bear_confidence','N/A')}\n"
        f"  Thesis: {bearish_researcher.get('bear_thesis','N/A')}\n"
        f"  Key Risks: {' | '.join(bearish_researcher.get('key_risks',[]))}\n"
        f"  Downside Base: {bearish_researcher.get('downside_scenarios',{}).get('base','N/A')}\n"
        f"  Verdict: {bearish_researcher.get('verdict','N/A')}\n\n"
        f"TASKS:\n"
        f"1. Identify which signals both sides AGREE on (high-conviction facts).\n"
        f"2. Identify which signals are DISPUTED and explain who has the stronger argument.\n"
        f"3. Determine the NET BIAS after weighing all evidence.\n"
        f"4. Produce a concise, balanced investment brief for the Trading Team.\n\n"
        f"OUTPUT only this JSON after your ReAct reasoning:\n"
        '{"net_bias":"BULLISH"|"BEARISH"|"NEUTRAL",'
        '"consensus_strength":"STRONG"|"MODERATE"|"WEAK",'
        '"agreed_points":["<agreed fact <=20 words>","<agreed fact <=20 words>","<agreed fact <=20 words>"],'
        '"disputed_points":["<disputed point <=20 words>","<disputed point <=20 words>"],'
        '"resolution":"<who won each dispute in 20 words each>",'
        '"balanced_brief":"<3-4 sentences for the Trading Team>",'
        '"key_bull_signals":["<<=15 words>","<<=15 words>","<<=15 words>"],'
        '"key_bear_signals":["<<=15 words>","<<=15 words>","<<=15 words>"],'
        '"findings":["<max 20 words>","<max 20 words>"],'
        '"verdict":"<1 sentence on the balance of evidence>"}'
    )


# ================================================================== #
# TRADING TEAM -- Agent T1: Momentum Trader
# ================================================================== #

def _build_momentum_trader_prompt(
    ticker, info, technical, statistical,
    synthesizer: dict, news_analyst: dict, ml_result=None,
) -> str:
    price    = info.get("currentPrice", "N/A")
    tech_l   = technical.get("latest", {})
    tech_s   = technical.get("signals", {})
    vol      = statistical.get("volatility", {})
    atr_val  = tech_l.get("ATR")
    atr_str  = f"${atr_val:.4f}" if atr_val is not None else "N/A"
    adx_val  = tech_l.get("ADX")
    adx_str  = f"{adx_val:.1f}" if adx_val is not None else "N/A"
    vol_pct  = f"{vol['sigma_annual']*100:.1f}%" if isinstance(vol.get("sigma_annual"), float) else "N/A"
    ml_block = _format_ml_block(ml_result)
    return (
        f"You are a Momentum Trader. Determine the optimal momentum trade for {ticker} "
        f"based on trend and momentum signals. Apply the ReAct framework.\n\n"
        f"TICKER: {ticker} | PRICE: ${price}\n"
        f"MOMENTUM SIGNALS:\n"
        f"  RSI(14): {tech_l.get('RSI','N/A')} ({tech_s.get('RSI','N/A')}) | MACD: {tech_s.get('MACD','N/A')}\n"
        f"  ADX: {adx_str} ({tech_s.get('ADX','N/A')}) | OBV: {tech_s.get('OBV','N/A')}\n"
        f"  MA Trend: {tech_s.get('MA_Trend','N/A')} | ATR(14): {atr_str}\n"
        f"  SMA20={tech_l.get('SMA_20','N/A')} | SMA50={tech_l.get('SMA_50','N/A')} | SMA200={tech_l.get('SMA_200','N/A')}\n"
        f"  sigma_annual: {vol_pct}\n\n"
        f"RESEARCH BRIEF (Synthesizer):\n"
        f"  Net Bias: {synthesizer.get('net_bias','N/A')} | Strength: {synthesizer.get('consensus_strength','N/A')}\n"
        f"  Brief: {synthesizer.get('balanced_brief','N/A')}\n\n"
        f"NEWS ANALYST: Macro={news_analyst.get('macro_bias','N/A')} | "
        f"VIX={news_analyst.get('vix_regime','N/A')} | Override={news_analyst.get('override_short_term','N/A')}\n\n"
        f"{ml_block}\n\n"
        f"OUTPUT only this JSON after your ReAct reasoning:\n"
        '{"trade_action":"ENTER_LONG"|"ENTER_SHORT"|"WAIT"|"HOLD"|"EXIT",'
        '"direction":"LONG"|"SHORT"|"NONE",'
        '"entry_price":<float>,'
        '"stop_loss":<float>,'
        '"target_1w":<float>,'
        '"target_2w":<float>,'
        '"target_3w":<float>,'
        '"target_1m":<float>,'
        '"position_size_pct":<float>,'
        '"momentum_score":"STRONG"|"MODERATE"|"WEAK"|"ABSENT",'
        '"trade_rationale":"<max 30 words>",'
        '"findings":["<max 20 words>","<max 20 words>"],'
        '"verdict":"<1 sentence momentum trade summary>"}'
    )


# ================================================================== #
# TRADING TEAM -- Agent T2: Value Trader
# ================================================================== #

def _build_value_trader_prompt(
    ticker, info, fundamental, statistical, analyst_data,
    synthesizer: dict, news_analyst: dict,
) -> str:
    price    = info.get("currentPrice", "N/A")
    pe       = fundamental.get("pe", {})
    peg      = fundamental.get("peg", {})
    dcf      = fundamental.get("dcf", {})
    graham   = fundamental.get("graham", {})
    pb_div   = fundamental.get("pb_div", {})
    pt       = analyst_data.get("price_target") or {}
    mc_preds = statistical.get("monte_carlo", {}).get("predictions", {})
    mc_1yr   = mc_preds.get("1 Year", mc_preds.get("1 year", {}))
    mc_median_str = f"${mc_1yr.get('median','N/A')}" if mc_1yr else "N/A"
    return (
        f"You are a Value Trader. Determine whether {ticker} presents a compelling value entry "
        f"based on fundamental and valuation signals. Apply the ReAct framework.\n\n"
        f"TICKER: {ticker} | PRICE: ${price}\n"
        f"VALUATION SIGNALS:\n"
        f"  Trailing P/E: {pe.get('trailing_pe','N/A')} ({pe.get('trailing_signal','')})\n"
        f"  Forward P/E:  {pe.get('forward_pe','N/A')} | PEG: {peg.get('peg','N/A')} ({peg.get('signal','')})\n"
        f"  DCF Intrinsic: ${dcf.get('dcf_value','N/A')} ({dcf.get('signal','')})\n"
        f"  Graham Number: ${graham.get('graham_number','N/A')} ({graham.get('signal','')})\n"
        f"  P/B: {pb_div.get('price_to_book','N/A')} | Div Yield: {pb_div.get('dividend_yield','N/A')}%\n"
        f"  Analyst Target: Mean=${pt.get('mean','N/A')} (Low=${pt.get('low','N/A')} / High=${pt.get('high','N/A')})\n"
        f"  MC 1-Year Median: {mc_median_str}\n\n"
        f"RESEARCH BRIEF (Synthesizer):\n"
        f"  Net Bias: {synthesizer.get('net_bias','N/A')} | Strength: {synthesizer.get('consensus_strength','N/A')}\n"
        f"  Brief: {synthesizer.get('balanced_brief','N/A')}\n\n"
        f"NEWS ANALYST: Macro={news_analyst.get('macro_bias','N/A')} | "
        f"Earnings Risk={news_analyst.get('earnings_risk','N/A')}\n\n"
        f"OUTPUT only this JSON after your ReAct reasoning:\n"
        '{"trade_action":"ENTER_LONG"|"ENTER_SHORT"|"WAIT"|"HOLD"|"EXIT",'
        '"direction":"LONG"|"SHORT"|"NONE",'
        '"entry_price":<float>,'
        '"stop_loss":<float>,'
        '"target_3m":<float>,'
        '"target_6m":<float>,'
        '"target_9m":<float>,'
        '"target_12m":<float>,'
        '"position_size_pct":<float>,'
        '"margin_of_safety_pct":<float or null>,'
        '"value_score":"DEEP_VALUE"|"FAIR_VALUE"|"OVERVALUED",'
        '"trade_rationale":"<max 30 words>",'
        '"findings":["<max 20 words>","<max 20 words>"],'
        '"verdict":"<1 sentence value trade summary>"}'
    )


# ================================================================== #
# TRADING TEAM -- Agent T3: Swing Trader
# ================================================================== #

def _build_swing_trader_prompt(
    ticker, info, technical, statistical,
    synthesizer: dict, news_analyst: dict,
) -> str:
    price    = info.get("currentPrice", "N/A")
    tech_l   = technical.get("latest", {})
    tech_s   = technical.get("signals", {})
    fibs     = technical.get("fibonacci", {})
    pivots   = technical.get("pivot_levels", {})
    vol      = statistical.get("volatility", {})
    atr_val  = tech_l.get("ATR")
    atr_str  = f"${atr_val:.4f}" if atr_val is not None else "N/A"
    vol_pct  = f"{vol['sigma_annual']*100:.1f}%" if isinstance(vol.get("sigma_annual"), float) else "N/A"
    fib_str  = " | ".join(f"{k}: ${v}" for k, v in list(fibs.items())[:5]) or "N/A"
    supp_str = " | ".join(f"${s}" for s in pivots.get("support_levels", [])[:3]) or "N/A"
    res_str  = " | ".join(f"${r}" for r in pivots.get("resistance_levels", [])[:3]) or "N/A"
    return (
        f"You are a Swing Trader. Identify high-probability short-to-medium term swing trades "
        f"for {ticker} using Fibonacci, Bollinger, and pivot levels. Apply the ReAct framework.\n\n"
        f"TICKER: {ticker} | PRICE: ${price}\n"
        f"SWING SIGNALS:\n"
        f"  BB %B: {tech_l.get('BB_PctB','N/A')} ({tech_s.get('Bollinger','N/A')})\n"
        f"  BB Upper: ${tech_l.get('BB_Upper','N/A')} | BB Middle: ${tech_l.get('BB_Middle','N/A')} | "
        f"BB Lower: ${tech_l.get('BB_Lower','N/A')}\n"
        f"  ATR(14): {atr_str} | sigma_annual: {vol_pct}\n"
        f"  RSI: {tech_l.get('RSI','N/A')} | ADX: {tech_l.get('ADX','N/A')}\n"
        f"FIBONACCI: {fib_str}\n"
        f"SUPPORTS: {supp_str} | RESISTANCES: {res_str}\n\n"
        f"RESEARCH BRIEF:\n"
        f"  Net Bias: {synthesizer.get('net_bias','N/A')} | Brief: {synthesizer.get('balanced_brief','N/A')}\n\n"
        f"NEWS ANALYST: Macro={news_analyst.get('macro_bias','N/A')} | VIX={news_analyst.get('vix_regime','N/A')}\n\n"
        f"OUTPUT only this JSON after your ReAct reasoning:\n"
        '{"trade_action":"ENTER_LONG"|"ENTER_SHORT"|"WAIT"|"HOLD"|"EXIT",'
        '"direction":"LONG"|"SHORT"|"NONE",'
        '"entry_price":<float>,'
        '"stop_loss":<float>,'
        '"target_1w":<float>,'
        '"target_2w":<float>,'
        '"target_3w":<float>,'
        '"target_1m":<float>,'
        '"position_size_pct":<float>,'
        '"swing_setup":"REVERSAL"|"CONTINUATION"|"BREAKOUT"|"NONE",'
        '"nearest_support":<float>,'
        '"nearest_resistance":<float>,'
        '"trade_rationale":"<max 30 words>",'
        '"findings":["<max 20 words>","<max 20 words>"],'
        '"verdict":"<1 sentence swing trade summary>"}'
    )


# ================================================================== #
# RISK MANAGEMENT TEAM -- Agent RM1: Market Risk Agent
# ================================================================== #

def _build_market_risk_prompt(
    ticker, info, technical, statistical, analyst_data,
    synthesizer: dict,
    momentum_trader: dict,
    value_trader: dict,
    swing_trader: dict,
) -> str:
    price    = info.get("currentPrice", "N/A")
    sector   = info.get("sector", "N/A")
    vix      = analyst_data.get("vix", {})
    e5d      = analyst_data.get("earnings_within_5d", False)
    e21d     = analyst_data.get("earnings_within_21d", False)
    earn_date= analyst_data.get("next_earnings_date", "Unknown")
    beta     = statistical.get("beta", {})
    vol      = statistical.get("volatility", {})
    sortino  = statistical.get("sortino", {})
    drawdown = statistical.get("drawdown", {})
    vol_pct  = f"{vol['sigma_annual']*100:.1f}%" if isinstance(vol.get("sigma_annual"), float) else "N/A"
    vol_regime = vol.get("vol_regime", "N/A")
    tech_s   = technical.get("signals", {})
    earn_warn = ("** EARNINGS WITHIN 5 DAYS" if e5d
                 else "** EARNINGS WITHIN 21 DAYS" if e21d
                 else f"No near-term earnings (next: {earn_date})")
    trader_summary = (
        f"Momentum: {momentum_trader.get('trade_action','N/A')} | "
        f"Value: {value_trader.get('trade_action','N/A')} | "
        f"Swing: {swing_trader.get('trade_action','N/A')}"
    )
    return (
        f"You are a Market Risk Agent. Assess all market-level risks for {ticker} and advise "
        f"the Trading Team on whether proposed trades are within acceptable risk parameters. "
        f"Apply the ReAct framework.\n\n"
        f"TICKER: {ticker} | PRICE: ${price} | SECTOR: {sector}\n"
        f"MARKET RISKS:\n"
        f"  VIX: {vix.get('value','N/A')} -> {vix.get('regime','N/A')}\n"
        f"  Earnings: {earn_warn}\n"
        f"  sigma_annual: {vol_pct} | Vol Regime: {vol_regime} | "
        f"Beta: {beta.get('beta','N/A')} ({beta.get('label','N/A')})\n"
        f"  MaxDD: {drawdown.get('max_drawdown_pct','N/A')}% | Calmar: {drawdown.get('calmar','N/A')} | "
        f"Sortino: {sortino.get('sortino','N/A')}\n"
        f"  ADX: {tech_s.get('ADX','N/A')} | OBV: {tech_s.get('OBV','N/A')}\n\n"
        f"RESEARCH BRIEF: Net Bias={synthesizer.get('net_bias','N/A')} | "
        f"{synthesizer.get('balanced_brief','')}\n"
        f"TRADER ACTIONS: {trader_summary}\n\n"
        f"OUTPUT only this JSON after your ReAct reasoning:\n"
        '{"market_risk_level":"LOW"|"MODERATE"|"HIGH"|"EXTREME",'
        '"vol_regime_risk":"COMPRESSING"|"STABLE"|"EXPANDING",'
        '"earnings_caution":"NONE"|"REDUCE_SIZE"|"AVOID",'
        '"vix_risk_flag":"CLEAR"|"CAUTION"|"DANGER",'
        '"liquidity_risk":"LOW"|"MODERATE"|"HIGH",'
        '"max_position_flag":"FULL_SIZE"|"HALF_SIZE"|"QUARTER_SIZE"|"NO_NEW_POSITIONS",'
        '"timing_recommendation":"PROCEED"|"REDUCE"|"WAIT"|"AVOID",'
        '"risk_adjustments":["<adjustment <=15 words>","<adjustment <=15 words>"],'
        '"findings":["<max 20 words>","<max 20 words>","<max 20 words>"],'
        '"verdict":"<1 sentence on overall market risk environment>"}'
    )


# ================================================================== #
# RISK MANAGEMENT TEAM -- Agent RM2: Portfolio Risk Agent
# ================================================================== #

def _build_portfolio_risk_prompt(
    ticker, info, statistical, analyst_data,
    momentum_trader: dict,
    value_trader: dict,
    swing_trader: dict,
    market_risk: dict,
) -> str:
    price    = info.get("currentPrice", "N/A")
    vol      = statistical.get("volatility", {})
    beta     = statistical.get("beta", {})
    drawdown = statistical.get("drawdown", {})
    sortino  = statistical.get("sortino", {})
    sharpe   = statistical.get("sharpe", {})
    vix      = analyst_data.get("vix", {})
    e5d      = analyst_data.get("earnings_within_5d", False)
    e21d     = analyst_data.get("earnings_within_21d", False)
    vol_pct  = f"{vol['sigma_annual']*100:.1f}%" if isinstance(vol.get("sigma_annual"), float) else "N/A"
    m_entry = momentum_trader.get("entry_price", price)
    m_stop  = momentum_trader.get("stop_loss")
    v_entry = value_trader.get("entry_price", price)
    v_stop  = value_trader.get("stop_loss")
    s_entry = swing_trader.get("entry_price", price)
    s_stop  = swing_trader.get("stop_loss")
    earn_warn = "YES -- within 5 days" if e5d else ("YES -- within 21 days" if e21d else "No")
    return (
        f"You are a Portfolio Risk Agent. Calculate precise position sizing and validate all "
        f"stop-loss levels for {ticker} using Kelly criterion, ATR-based sizing, and portfolio "
        f"exposure rules. Apply the ReAct framework.\n\n"
        f"TICKER: {ticker} | PRICE: ${price}\n"
        f"RISK METRICS:\n"
        f"  sigma_annual: {vol_pct} | Beta: {beta.get('beta','N/A')} | Sharpe: {sharpe.get('sharpe','N/A')}\n"
        f"  Sortino: {sortino.get('sortino','N/A')} | MaxDD: {drawdown.get('max_drawdown_pct','N/A')}%\n"
        f"  VIX: {vix.get('value','N/A')} -> {vix.get('regime','N/A')}\n"
        f"  Near-term Earnings: {earn_warn}\n\n"
        f"MARKET RISK ASSESSMENT: {market_risk.get('market_risk_level','N/A')} risk | "
        f"Max Position: {market_risk.get('max_position_flag','N/A')} | "
        f"Timing: {market_risk.get('timing_recommendation','N/A')}\n\n"
        f"PROPOSED TRADE SETUPS:\n"
        f"  Momentum Trader: Entry=${m_entry} | Stop=${m_stop or 'N/A'} | "
        f"Size={momentum_trader.get('position_size_pct','N/A')}%\n"
        f"  Value Trader:    Entry=${v_entry} | Stop={v_stop or 'N/A'} | "
        f"Size={value_trader.get('position_size_pct','N/A')}%\n"
        f"  Swing Trader:    Entry=${s_entry} | Stop={s_stop or 'N/A'} | "
        f"Size={swing_trader.get('position_size_pct','N/A')}%\n\n"
        f"SIZING RULES (MAX_RISK_PER_TRADE = 1% of account):\n"
        f"  FULL SIZE (1.5%): HIGH confidence + CALM VIX + no earnings.\n"
        f"  NORMAL (1.0%):    MEDIUM confidence or ELEVATED VIX.\n"
        f"  HALF SIZE (0.5%): LOW confidence or HIGH_FEAR VIX or earnings within 5 days.\n"
        f"  Max portfolio exposure: 5% per ticker.\n\n"
        f"KELLY FORMULA: f* = (p x b - q) / b, where p=win_rate, b=win/loss_ratio, q=1-p.\n"
        f"ATR STOP: stop_distance >= 1.5 x ATR(14).\n\n"
        f"OUTPUT only this JSON after your ReAct reasoning:\n"
        '{"kelly_fraction":<float -- capped at 0.25>,'
        '"recommended_size_pct":<float -- final position size % of account>,'
        '"sizing_method":"KELLY"|"ATR"|"FIXED",'
        '"scaling_factor":"FULL"|"NORMAL"|"HALF"|"QUARTER",'
        '"max_risk_usd_per_1k":<float>,'
        '"validated_entry":<float>,'
        '"validated_stop":<float>,'
        '"stop_distance_usd":<float>,'
        '"stop_validation":"VALID"|"WIDENED"|"REJECTED",'
        '"stop_validation_note":"<note <=20 words or null>",'
        '"portfolio_exposure_ok":true|false,'
        '"sizing_rationale":"<max 30 words>",'
        '"findings":["<max 20 words>","<max 20 words>"],'
        '"verdict":"<1 sentence on final position sizing decision>"}'
    )


# ================================================================== #
# THE JUDGE
# ================================================================== #

def _build_judge_prompt(
    ticker, info, technical, fundamental, statistical, analyst_data,
    fundamental_analyst: dict,
    sentiment_analyst: dict,
    news_analyst: dict,
    technical_analyst: dict,
    bullish_researcher: dict,
    bearish_researcher: dict,
    synthesizer: dict,
    momentum_trader: dict,
    value_trader: dict,
    swing_trader: dict,
    market_risk: dict,
    portfolio_risk: dict,
    ml_result=None,
    accuracy_context: dict = None,
) -> str:
    price    = info.get("currentPrice", "N/A")
    company  = info.get("shortName", ticker)
    sector   = info.get("sector", "N/A")
    tech_l   = technical.get("latest", {})
    tech_s   = technical.get("signals", {})
    atr_val  = tech_l.get("ATR")
    atr_str  = f"${atr_val:.4f}" if isinstance(atr_val, (int, float)) else "N/A"
    pe_data  = fundamental.get("pe", {})
    dcf      = fundamental.get("dcf", {})
    mc_preds = statistical.get("monte_carlo", {}).get("predictions", {})
    mc_params= statistical.get("monte_carlo", {}).get("params", {})
    vol      = statistical.get("volatility", {})
    beta     = statistical.get("beta", {})
    sharpe   = statistical.get("sharpe", {})
    sortino  = statistical.get("sortino", {})
    drawdown = statistical.get("drawdown", {})
    vix      = analyst_data.get("vix", {})
    e5d      = analyst_data.get("earnings_within_5d", False)
    e21d     = analyst_data.get("earnings_within_21d", False)
    vol_pct  = f"{vol['sigma_annual']*100:.1f}%" if isinstance(vol.get("sigma_annual"), float) else "N/A"
    mc_str   = " | ".join(
        f"{h}: ${mc.get('median','N/A')} ({mc.get('med_chg%',0):+.1f}%)"
        for h, mc in mc_preds.items()
    )
    st_preds = technical_analyst.get("price_predictions", {}).get("short_term", {})
    lt_preds = technical_analyst.get("price_predictions", {}).get("long_term", {})
    mt_1w    = momentum_trader.get("target_1w", "N/A")
    mt_1m    = momentum_trader.get("target_1m", "N/A")
    vt_12m   = value_trader.get("target_12m", "N/A")
    sw_1w    = swing_trader.get("target_1w", "N/A")
    ml_block = _format_ml_block(ml_result)
    ml_metrics = ml_result.get("metrics", {}) if (ml_result and isinstance(ml_result, dict)) else {}
    clf_5d_acc  = ml_metrics.get("clf_5d_accuracy")
    clf_21d_acc = ml_metrics.get("clf_21d_accuracy")
    acc_5d_str  = f"{clf_5d_acc*100:.0f}%" if isinstance(clf_5d_acc, float) else "unknown"
    acc_21d_str = f"{clf_21d_acc*100:.0f}%" if isinstance(clf_21d_acc, float) else "unknown"
    fund_st  = fundamental_analyst.get("short_term_verdict", "NEUTRAL")
    fund_lt  = fundamental_analyst.get("long_term_verdict", "NEUTRAL")
    sent_b   = sentiment_analyst.get("sentiment_bias", "NEUTRAL")
    macro_b  = news_analyst.get("macro_bias", "NEUTRAL")
    tech_b   = technical_analyst.get("statistical_bias", "NEUTRAL")
    net_b    = synthesizer.get("net_bias", "NEUTRAL")
    mr_flag  = market_risk.get("timing_recommendation", "PROCEED")
    alt_ticker  = fundamental_analyst.get("alternative_pick")
    alt_reason  = fundamental_analyst.get("alternative_reason", "")
    earn_warn   = ("** EARNINGS WITHIN 5 DAYS -- high binary risk" if e5d
                   else "** Earnings within 21 days -- elevated event risk" if e21d
                   else f"No near-term earnings")
    alt_json = json.dumps(alt_ticker) if alt_ticker else "null"
    alt_r_json = json.dumps(alt_reason) if alt_reason else "null"

    # Build recent-news block for prompt (with dates)
    _news_items = analyst_data.get("news", [])[:5]
    _news_lines = []
    for _n in _news_items:
        if not isinstance(_n, dict):
            continue
        _pub = _n.get("publish_time", "")
        if isinstance(_pub, (int, float)):
            import datetime as _dt
            _pub = _dt.datetime.fromtimestamp(_pub).strftime("%Y-%m-%d")
        elif isinstance(_pub, str) and len(_pub) >= 10:
            _pub = _pub[:10]
        else:
            _pub = "Unknown date"
        _news_lines.append(
            f"  [{_pub}] {_n.get('title','')} — {_n.get('publisher','')}"
        )
    _news_block = "\n".join(_news_lines) if _news_lines else "  No recent news available."

    return (
        f"You are The Judge. You have received reports from five specialist teams. Make the "
        f"definitive, actionable investment decision for {ticker} using all available information. "
        f"Apply the ReAct framework.\n\n"
        f"TICKER: {ticker} ({company}) | SECTOR: {sector} | PRICE: ${price}\n\n"
        f"=== ANALYST TEAM ===\n"
        f"  Fundamental: Health={fundamental_analyst.get('business_health','N/A')} | "
        f"Val={fundamental_analyst.get('valuation_stance','N/A')} | ST={fund_st} | LT={fund_lt}\n"
        f"  Sentiment:   Bias={sent_b} | Mood={sentiment_analyst.get('market_mood','N/A')} | "
        f"Pressure={sentiment_analyst.get('behavioural_pressure','N/A')}\n"
        f"  News:        Macro={macro_b} | VIX={news_analyst.get('vix_regime','N/A')} | "
        f"Override={news_analyst.get('override_short_term',False)}\n"
        f"  Technical:   Bias={tech_b} | Direction={technical_analyst.get('direction','N/A')} | "
        f"Worth={technical_analyst.get('worth_investing','N/A')}\n"
        f"    Entry=${technical_analyst.get('entry_price','N/A')} | Stop=${technical_analyst.get('stop_loss','N/A')}\n"
        f"    ST: 1W=${st_preds.get('1_week',{}).get('price','N/A')} | "
        f"2W=${st_preds.get('2_weeks',{}).get('price','N/A')} | "
        f"3W=${st_preds.get('3_weeks',{}).get('price','N/A')} | "
        f"1M=${st_preds.get('1_month',{}).get('price','N/A')}\n"
        f"    LT: 3M=${lt_preds.get('3_months',{}).get('price','N/A')} | "
        f"6M=${lt_preds.get('6_months',{}).get('price','N/A')} | "
        f"9M=${lt_preds.get('9_months',{}).get('price','N/A')} | "
        f"12M=${lt_preds.get('12_months',{}).get('price','N/A')}\n\n"
        f"=== RESEARCHER TEAM ===\n"
        f"  Net Bias: {net_b} | Strength: {synthesizer.get('consensus_strength','N/A')}\n"
        f"  Bull: {bullish_researcher.get('bull_stance','N/A')} "
        f"(conf={bullish_researcher.get('bull_confidence','N/A')}) -- "
        f"{bullish_researcher.get('verdict','N/A')}\n"
        f"  Bear: {bearish_researcher.get('bear_stance','N/A')} "
        f"(conf={bearish_researcher.get('bear_confidence','N/A')}) -- "
        f"{bearish_researcher.get('verdict','N/A')}\n"
        f"  Synthesis: {synthesizer.get('balanced_brief','N/A')}\n\n"
        f"=== TRADING TEAM ===\n"
        f"  Momentum: {momentum_trader.get('trade_action','N/A')} | "
        f"Entry=${momentum_trader.get('entry_price','N/A')} | "
        f"Stop=${momentum_trader.get('stop_loss','N/A')} | 1W=${mt_1w} | 1M={mt_1m}\n"
        f"  Value:     {value_trader.get('trade_action','N/A')} | "
        f"Entry=${value_trader.get('entry_price','N/A')} | 12M=${vt_12m} | "
        f"MoS={value_trader.get('margin_of_safety_pct','N/A')}%\n"
        f"  Swing:     {swing_trader.get('trade_action','N/A')} | "
        f"Entry=${swing_trader.get('entry_price','N/A')} | "
        f"Setup={swing_trader.get('swing_setup','N/A')} | 1W={sw_1w}\n\n"
        f"=== RISK MANAGEMENT TEAM ===\n"
        f"  Market Risk: {market_risk.get('market_risk_level','N/A')} | "
        f"Flag: {market_risk.get('vix_risk_flag','N/A')} | Timing: {mr_flag}\n"
        f"  Portfolio:  Size={portfolio_risk.get('recommended_size_pct','N/A')}% | "
        f"Method={portfolio_risk.get('sizing_method','N/A')} | "
        f"Stop=${portfolio_risk.get('validated_stop','N/A')} | "
        f"Stop Dist=${portfolio_risk.get('stop_distance_usd','N/A')}\n\n"
        f"=== ML MODEL SIGNALS ===\n"
        f"{ml_block}\n"
        f"  ML 5d accuracy: {acc_5d_str} | ML 21d accuracy: {acc_21d_str}\n\n"
        f"=== EARNINGS & MARKET DATA ===\n"
        f"{earn_warn}\n"
        f"VIX: {vix.get('value','N/A')} -> {vix.get('regime','N/A')}\n\n"
        f"=== RECENT NEWS (with dates) ===\n"
        f"{_news_block}\n\n"
        f"=== RAW DATA ===\n"
        f"Technical: RSI={tech_s.get('RSI','N/A')} | MACD={tech_s.get('MACD','N/A')} | "
        f"MA={tech_s.get('MA_Trend','N/A')} | BB={tech_s.get('Bollinger','N/A')} | "
        f"ADX={tech_s.get('ADX','N/A')} | OBV={tech_s.get('OBV','N/A')} | ATR={atr_str}\n"
        f"Stats: Vol={vol_pct} | Beta={beta.get('beta','N/A')} | "
        f"Sharpe={sharpe.get('sharpe','N/A')} | Sortino={sortino.get('sortino','N/A')} | "
        f"MaxDD={drawdown.get('max_drawdown_pct','N/A')}%\n"
        f"Fundamentals: P/E={pe_data.get('trailing_pe','N/A')} | "
        f"Fwd P/E={pe_data.get('forward_pe','N/A')} | DCF=${dcf.get('dcf_value','N/A')}\n"
        f"Monte Carlo: {mc_str}\n\n"
        f"SCORING SYSTEM (compute score, then recommend):\n"
        f"  Technical bias:        BULLISH=+25 | NEUTRAL=0 | BEARISH=-25\n"
        f"  Net research bias:     BULLISH=+20 | NEUTRAL=0 | BEARISH=-20\n"
        f"  Fundamental ST:        WORTH_INVESTING=+15 | NEUTRAL=0 | NOT_WORTH_INVESTING=-15\n"
        f"  Fundamental LT:        WORTH_INVESTING=+20 | NEUTRAL=0 | NOT_WORTH_INVESTING=-20\n"
        f"  Macro bias:            RISK_ON=+15 | NEUTRAL=0 | RISK_OFF=-15\n"
        f"  Sentiment bias:        BULLISH=+10 | NEUTRAL=0 | BEARISH=-10\n"
        f"  ML score (range -40 to +40): use value shown or 0 if N/A\n"
        f"  Score > +85 -> STRONG BUY | +40 to +85 -> BUY | -40 to +40 -> HOLD | "
        f"-85 to -40 -> SELL | <-85 -> STRONG SELL\n\n"
        f"HARD OVERRIDE RULES:\n"
        f"  - news_analyst override_short_term=true -> overall_short_term MUST be NEUTRAL or NOT_WORTH_INVESTING.\n"
        f"  - macro_bias=RISK_OFF AND earnings_risk=true -> confidence MUST be LOW.\n"
        f"  - ATR STOP: |entry - stop| < {atr_str} -> widen stop to entry -/+ 1.5xATR; add note to key_risks.\n"
        f"  - LONG: stop_loss < entry_price. SHORT: stop_loss > entry_price.\n"
        f"  - TARGET CONSISTENCY: if majority of LT targets < ${price} -> overall_long_term=NOT_WORTH_INVESTING.\n"
        f"  - CRITICAL: ALL 8 target_prices MUST be populated with float values (no nulls).\n\n"
        f"ACCURACY_PCT COMPUTATION (per horizon -- include in each target_prices entry):\n"
        f"  PREFERRED BASE: if the HISTORICAL ACCURACY block below reports a realized\n"
        f"    Directional Accuracy for a horizon (N >= 5 predictions), use THAT realized\n"
        f"    accuracy as the base for that horizon -- it reflects how this system has\n"
        f"    actually performed, which beats any theoretical estimate.\n"
        f"  FALLBACK BASE (no track record): ML clf accuracy 5d={acc_5d_str}, 21d={acc_21d_str},\n"
        f"    then apply horizon decay: 1W=1.0, 2W=0.95, 3W=0.90, 1M=0.85, 3M=0.75,\n"
        f"    6M=0.65, 9M=0.60, 12M=0.55.\n"
        f"  Signal agreement bonus: (agreed_signals / total_signals - 0.5) x 10%.\n"
        f"  VIX penalty: HIGH_FEAR=-5%, ELEVATED=-2%, CALM=0%.\n"
        f"  Clamp final accuracy_pct to range [30%, 85%].\n\n"
        f"OUTPUT only this JSON after your ReAct reasoning:\n"
        '{"recommendation":"STRONG BUY"|"BUY"|"HOLD"|"SELL"|"STRONG SELL",'
        '"confidence":"HIGH"|"MEDIUM"|"LOW",'
        '"overall_short_term":"WORTH_INVESTING"|"NEUTRAL"|"NOT_WORTH_INVESTING",'
        '"overall_long_term":"WORTH_INVESTING"|"NEUTRAL"|"NOT_WORTH_INVESTING",'
        '"target_prices":{'
        '"1_week":{"price":<float>,"accuracy_pct":<float>},'
        '"2_weeks":{"price":<float>,"accuracy_pct":<float>},'
        '"3_weeks":{"price":<float>,"accuracy_pct":<float>},'
        '"1_month":{"price":<float>,"accuracy_pct":<float>},'
        '"3_months":{"price":<float>,"accuracy_pct":<float>},'
        '"6_months":{"price":<float>,"accuracy_pct":<float>},'
        '"9_months":{"price":<float>,"accuracy_pct":<float>},'
        '"12_months":{"price":<float>,"accuracy_pct":<float>}},'
        '"trade_direction":"LONG"|"SHORT",'
        '"entry_price":<float>,'
        '"exit_price":<float>,'
        '"stop_loss":<float>,'
        '"position_size_pct":<float>,'
        '"key_bull_case":["<15 words>","<15 words>","<15 words>"],'
        '"key_bear_case":["<15 words>","<15 words>","<15 words>"],'
        '"key_risks":["<15 words>","<15 words>","<15 words>"],'
        '"catalysts":["<15 words>","<15 words>"],'
        '"summary":"<2-3 sentences -- overall verdict integrating all 5 teams + ML + macro>",'
        '"technical_verdict":"<1 sentence: Is price/timing right?>",'
        '"fundamental_verdict":"<1 sentence: Is the business healthy?>",'
        '"valuation_verdict":"<1 sentence: Am I overpaying?>",'
        '"sentiment_verdict":"<1 sentence: What does market mood say?>",'
        '"macro_verdict":"<1 sentence: What does macro regime say?>",'
        f'"alternative_pick":{alt_json},'
        f'"alternative_reason":{alt_r_json},'
        '"timing_note":"<1-2 sentences: should the user enter now, wait for a specific price, or avoid until a condition changes?>","'
        'team_summaries":{"analyst_team":"<2 sentences summarising A1-A4 findings>","researcher_team":"<2 sentences: bull vs bear balance and net bias>","trading_team":"<2 sentences: what the 3 traders collectively recommend and why>","risk_team":"<1 sentence: overall risk posture and recommended size>"},'
        '"news_with_dates":[{"title":"<headline>","publisher":"<publisher>","date":"<YYYY-MM-DD>","impact":"BULLISH"|"BEARISH"|"NEUTRAL","impact_note":"<10 words on why this news matters>"}]}'
    ) + (_get_accuracy_injection(ticker, accuracy_context))


# ================================================================== #
# Feedback accuracy injection helper
# ================================================================== #

def _get_accuracy_injection(ticker: str, accuracy_context: dict | None) -> str:
    """Return formatted historical accuracy block for the Judge prompt, or empty string."""
    if not accuracy_context or not accuracy_context.get("has_data"):
        return ""
    try:
        from feedback.accuracy import format_llm_injection
        return "\n\n" + format_llm_injection(ticker, accuracy_context)
    except Exception:
        return ""


# ================================================================== #
# ML signals formatting helper
# ================================================================== #

def _format_ml_block(ml_result) -> str:
    if not ml_result or not isinstance(ml_result, dict) or ml_result.get("error"):
        return "ML SIGNALS: Not available"
    clf  = ml_result.get("classification", {})
    reg  = ml_result.get("regression", {})
    clf_5d  = clf.get("5d", {})
    clf_21d = clf.get("21d", {})
    reg_5d  = reg.get("5d", {})
    reg_21d = reg.get("21d", {})
    # Flat structure fallback
    p5  = clf_5d.get("prob_up") or ml_result.get("clf_5d_prob_up")
    p21 = clf_21d.get("prob_up") or ml_result.get("clf_21d_prob_up")
    d5  = clf_5d.get("direction") or ml_result.get("clf_5d_direction", "N/A")
    d21 = clf_21d.get("direction") or ml_result.get("clf_21d_direction", "N/A")
    r5  = reg_5d.get("predicted_return_pct") or ml_result.get("reg_5d_return_pct")
    r21 = reg_21d.get("predicted_return_pct") or ml_result.get("reg_21d_return_pct")
    if isinstance(p5, (int, float)) and isinstance(p21, (int, float)):
        _ml_score = round(max(-40.0, min(40.0, (p5 - 0.5) * 40 + (p21 - 0.5) * 40)), 1)
        ml_score_str = f"{_ml_score:+.1f}"
    else:
        ml_score_str = "N/A"
    return (
        f"ML SIGNALS (GradientBoosting):\n"
        f"  5d  CLF: {d5} (prob_up={p5 or 'N/A'}) | REG: {r5 or 'N/A'}% predicted\n"
        f"  21d CLF: {d21} (prob_up={p21 or 'N/A'}) | REG: {r21 or 'N/A'}% predicted\n"
        f"  Pre-computed ML score: {ml_score_str} (range -40 to +40)"
    )


# ================================================================== #
# Fallback dicts (used when an agent call fails)
# ================================================================== #

def _fallback_fundamental_analyst() -> dict:
    return {
        "business_health": "N/A", "valuation_stance": "N/A",
        "capital_allocation": "N/A", "profitability_quality": "N/A",
        "competitive_position": "N/A", "analyst_conviction": "N/A",
        "short_term_verdict": "NEUTRAL", "short_term_rationale": "Unavailable",
        "long_term_verdict": "NEUTRAL", "long_term_rationale": "Unavailable",
        "key_strengths": [], "key_concerns": [],
        "findings": [], "verdict": "Unavailable",
        "alternative_pick": None, "alternative_reason": None,
    }

def _fallback_sentiment_analyst() -> dict:
    return {
        "sentiment_bias": "NEUTRAL", "market_mood": "NEUTRAL",
        "narrative": "Unavailable", "news_momentum": "STABLE",
        "high_impact_headlines": [], "sentiment_score": 0.0,
        "behavioural_pressure": "NEUTRAL", "earnings_risk_flag": False,
        "findings": [], "verdict": "Unavailable",
    }

def _fallback_news_analyst() -> dict:
    return {
        "vix_regime": "N/A", "macro_bias": "NEUTRAL",
        "earnings_risk": False, "timing_caution": "NONE",
        "override_short_term": False, "override_rationale": None,
        "sector_timing": "NEUTRAL", "key_macro_risks": [],
        "findings": [], "verdict": "Unavailable",
    }

def _fallback_technical_analyst() -> dict:
    _es = {"price": None, "change_pct": None, "method": "N/A",
           "confidence": "N/A", "rationale": "Unavailable"}
    return {
        "formula_assessment": "N/A", "formulas_used": [],
        "statistical_bias": "NEUTRAL", "worth_investing": "CONDITIONAL",
        "worth_investing_rationale": "Unavailable", "risk_level": "N/A",
        "risk_factors": [], "direction": "LONG", "entry_price": None, "stop_loss": None,
        "findings": [], "verdict": "Unavailable",
        "price_predictions": {
            "short_term": {
                "1_week": _es.copy(), "2_weeks": _es.copy(),
                "3_weeks": _es.copy(), "1_month": _es.copy(),
            },
            "long_term": {
                "3_months": _es.copy(), "6_months": _es.copy(),
                "9_months": _es.copy(), "12_months": _es.copy(),
            },
        },
    }

def _fallback_bullish_researcher() -> dict:
    return {
        "bull_stance": "WEAK", "bull_thesis": "Unavailable",
        "strongest_signals": [], "growth_catalysts": [],
        "upside_scenarios": {"base": "N/A", "optimistic": "N/A"},
        "bull_confidence": "LOW", "findings": [], "verdict": "Unavailable",
    }

def _fallback_bearish_researcher() -> dict:
    return {
        "bear_stance": "WEAK", "bear_thesis": "Unavailable",
        "key_risks": [], "downside_scenarios": {"base": "N/A", "pessimistic": "N/A"},
        "bear_confidence": "LOW", "findings": [], "verdict": "Unavailable",
    }

def _fallback_synthesizer() -> dict:
    return {
        "net_bias": "NEUTRAL", "consensus_strength": "WEAK",
        "agreed_points": [], "disputed_points": [], "resolution": "Unavailable",
        "balanced_brief": "Research synthesis unavailable.", "key_bull_signals": [],
        "key_bear_signals": [], "findings": [], "verdict": "Unavailable",
    }

def _fallback_momentum_trader() -> dict:
    return {
        "trade_action": "WAIT", "direction": "NONE",
        "entry_price": None, "stop_loss": None,
        "target_1w": None, "target_2w": None, "target_3w": None, "target_1m": None,
        "position_size_pct": 0.0, "momentum_score": "ABSENT",
        "trade_rationale": "Unavailable", "findings": [], "verdict": "Unavailable",
    }

def _fallback_value_trader() -> dict:
    return {
        "trade_action": "WAIT", "direction": "NONE",
        "entry_price": None, "stop_loss": None,
        "target_3m": None, "target_6m": None, "target_9m": None, "target_12m": None,
        "position_size_pct": 0.0, "margin_of_safety_pct": None,
        "value_score": "FAIR_VALUE", "trade_rationale": "Unavailable",
        "findings": [], "verdict": "Unavailable",
    }

def _fallback_swing_trader() -> dict:
    return {
        "trade_action": "WAIT", "direction": "NONE",
        "entry_price": None, "stop_loss": None,
        "target_1w": None, "target_2w": None, "target_3w": None, "target_1m": None,
        "position_size_pct": 0.0, "swing_setup": "NONE",
        "nearest_support": None, "nearest_resistance": None,
        "trade_rationale": "Unavailable", "findings": [], "verdict": "Unavailable",
    }

def _fallback_market_risk() -> dict:
    return {
        "market_risk_level": "MODERATE", "vol_regime_risk": "STABLE",
        "earnings_caution": "NONE", "vix_risk_flag": "CLEAR",
        "liquidity_risk": "LOW", "max_position_flag": "FULL_SIZE",
        "timing_recommendation": "PROCEED", "risk_adjustments": [],
        "findings": [], "verdict": "Unavailable",
    }

def _fallback_portfolio_risk() -> dict:
    return {
        "kelly_fraction": 0.0, "recommended_size_pct": 1.0,
        "sizing_method": "FIXED", "scaling_factor": "NORMAL",
        "max_risk_usd_per_1k": 10.0, "validated_entry": None,
        "validated_stop": None, "stop_distance_usd": None,
        "stop_validation": "N/A", "stop_validation_note": None,
        "portfolio_exposure_ok": True,
        "sizing_rationale": "Default sizing -- agent unavailable.",
        "findings": [], "verdict": "Unavailable",
    }

def _error_result(msg: str) -> dict:
    """
    Return a safe placeholder result when all LLM providers fail.

    Uses "HOLD" rather than "ERROR" so that the feedback tracker
    can record a valid recommendation and the feedback loop can
    accumulate directional accuracy data even on failed LLM runs.
    The ``llm_available=False`` flag is the authoritative signal
    that no LLM output was produced.
    """
    return {
        "llm_available":      False,
        "recommendation":     "HOLD",
        "confidence":         "LOW",
        "overall_short_term": "NEUTRAL",
        "overall_long_term":  "NEUTRAL",
        "target_prices":      {},
        "trade_direction":    "NONE",
        "entry_price":        None,
        "exit_price":         None,
        "stop_loss":          None,
        "position_size_pct":  0.0,
        "key_bull_case":      [],
        "key_bear_case":      [],
        "key_risks":          [],
        "catalysts":          [],
        "summary":            msg,
        "technical_verdict":  "N/A",
        "fundamental_verdict":"N/A",
        "valuation_verdict":  "N/A",
        "sentiment_verdict":  "N/A",
        "macro_verdict":      "N/A",
        "alternative_pick":   None,
        "alternative_reason": None,
        "timing_note":        None,
        "team_summaries":     {},
        "news_with_dates":    [],
        "agents":             {},
    }


# ================================================================== #
# Agent chain runner (per provider)
# ================================================================== #

def _run_agent_chain(
    call_fn, ticker, info, technical, fundamental, statistical, analyst_data,
    ml_result=None, accuracy_context=None,
    provider_name="", breaker=None, progress_cb=None,
) -> dict:
    """
    Run the full 13-agent chain using a single provider's call function.

    Execution order (parallelism via ThreadPoolExecutor):

    STEP 1 [parallel]:   A1 Fundamental, A2 Sentiment, A3 News, A4 Technical
    STEP 2 [parallel]:   R1 Bullish, R2 Bearish  (depend on all A agents)
    STEP 3 [serial]:     R3 Synthesizer           (depends on R1 + R2)
    STEP 4 [parallel]:   T1 Momentum, T2 Value, T3 Swing, RM1 Market Risk
    STEP 5 [serial]:     RM2 Portfolio Risk        (depends on T1+T2+T3+RM1)
    STEP 6 [serial]:     Judge                     (depends on everything)

    ``breaker`` is a _CircuitBreaker: once it trips (provider unreachable),
    later agents short-circuit to their rule-based fallbacks instead of
    each re-hitting the dead endpoint. ``progress_cb(provider, stage)`` is
    invoked at the start of each wave for live UI.
    """
    sentiment = (ml_result.get("sentiment", {}) if ml_result and isinstance(ml_result, dict)
                 else {"overall_score": 0.0, "label": "NEUTRAL", "summary": ""})

    def _note(stage: str) -> None:
        if progress_cb:
            try:
                progress_cb(provider_name, stage)
            except Exception:
                pass

    def _safe_call(prompt_fn, fallback_fn, *args, _agent="agent", **kwargs):
        # Skip immediately once the provider's breaker is open.
        if breaker is not None and breaker.is_open():
            fb = fallback_fn()
            fb["_error"] = f"skipped — {provider_name} unreachable (circuit open)"
            return fb
        try:
            r = call_fn(prompt_fn(*args, **kwargs))
            if breaker is not None:
                breaker.record_success()
            return r
        except Exception as e:
            if breaker is not None:
                breaker.record_failure(e, where=_agent)
            fb = fallback_fn()
            fb["_error"] = str(e)
            return fb

    # -- STEP 1: Analyst Team (4 parallel) -------------------------
    _note("Analyst team (1/6)")
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_fund = ex.submit(_safe_call,
            _build_fundamental_analyst_prompt, _fallback_fundamental_analyst,
            ticker, info, fundamental, analyst_data, statistical)
        f_sent = ex.submit(_safe_call,
            _build_sentiment_analyst_prompt, _fallback_sentiment_analyst,
            ticker, info, analyst_data, sentiment)
        f_news = ex.submit(_safe_call,
            _build_news_analyst_prompt, _fallback_news_analyst,
            ticker, info, technical, statistical, analyst_data)
        f_tech = ex.submit(_safe_call,
            _build_technical_analyst_prompt, _fallback_technical_analyst,
            ticker, info, technical, fundamental, statistical)
    fundamental_analyst = f_fund.result()
    sentiment_analyst   = f_sent.result()
    news_analyst        = f_news.result()
    technical_analyst   = f_tech.result()

    # -- STEP 2: Researcher Team -- Bull + Bear (parallel) ----------
    _note("Researchers (2/6)")
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_bull = ex.submit(_safe_call,
            _build_bullish_researcher_prompt, _fallback_bullish_researcher,
            ticker, info, fundamental_analyst, sentiment_analyst,
            news_analyst, technical_analyst)
        f_bear = ex.submit(_safe_call,
            _build_bearish_researcher_prompt, _fallback_bearish_researcher,
            ticker, info, fundamental_analyst, sentiment_analyst,
            news_analyst, technical_analyst)
    bullish_researcher = f_bull.result()
    bearish_researcher = f_bear.result()

    # -- STEP 3: Researcher Team -- Synthesizer (serial) -----------
    _note("Synthesizer (3/6)")
    synthesizer = _safe_call(
        _build_research_synthesizer_prompt, _fallback_synthesizer,
        ticker, info, bullish_researcher, bearish_researcher,
    )

    # -- STEP 4: Trading Team + Market Risk (4 parallel) -----------
    _note("Traders + market risk (4/6)")
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_mom = ex.submit(_safe_call,
            _build_momentum_trader_prompt, _fallback_momentum_trader,
            ticker, info, technical, statistical,
            synthesizer, news_analyst, ml_result)
        f_val = ex.submit(_safe_call,
            _build_value_trader_prompt, _fallback_value_trader,
            ticker, info, fundamental, statistical, analyst_data,
            synthesizer, news_analyst)
        f_swg = ex.submit(_safe_call,
            _build_swing_trader_prompt, _fallback_swing_trader,
            ticker, info, technical, statistical,
            synthesizer, news_analyst)
        f_mr  = ex.submit(_safe_call,
            _build_market_risk_prompt, _fallback_market_risk,
            ticker, info, technical, statistical, analyst_data,
            synthesizer, _fallback_momentum_trader(), _fallback_value_trader(),
            _fallback_swing_trader())
    momentum_trader = f_mom.result()
    value_trader    = f_val.result()
    swing_trader    = f_swg.result()
    market_risk     = f_mr.result()

    # -- STEP 5: Portfolio Risk (serial) ---------------------------
    _note("Portfolio risk (5/6)")
    portfolio_risk = _safe_call(
        _build_portfolio_risk_prompt, _fallback_portfolio_risk,
        ticker, info, statistical, analyst_data,
        momentum_trader, value_trader, swing_trader, market_risk,
    )

    # -- STEP 6: The Judge (serial) --------------------------------
    _note("Judge (6/6)")
    if breaker is not None and breaker.is_open():
        judge = _error_result(
            f"Judge skipped — {provider_name} unreachable (circuit open after "
            f"{breaker.tripped_at or 'repeated failures'})."
        )
    else:
        try:
            judge = call_fn(_build_judge_prompt(
                ticker, info, technical, fundamental, statistical, analyst_data,
                fundamental_analyst, sentiment_analyst, news_analyst, technical_analyst,
                bullish_researcher, bearish_researcher, synthesizer,
                momentum_trader, value_trader, swing_trader,
                market_risk, portfolio_risk, ml_result,
                accuracy_context=accuracy_context,
            ))
        except Exception as e:
            if breaker is not None:
                breaker.record_failure(e, where="judge")
            judge = _error_result(f"Judge error: {e}")

    judge["agents"] = {
        "fundamental_analyst": fundamental_analyst,
        "sentiment_analyst":   sentiment_analyst,
        "news_analyst":        news_analyst,
        "technical_analyst":   technical_analyst,
        "bullish_researcher":  bullish_researcher,
        "bearish_researcher":  bearish_researcher,
        "synthesizer":         synthesizer,
        "momentum_trader":     momentum_trader,
        "value_trader":        value_trader,
        "swing_trader":        swing_trader,
        "market_risk":         market_risk,
        "portfolio_risk":      portfolio_risk,
    }
    judge["_ml_accuracy"] = (
        ml_result.get("metrics", {}) if (ml_result and isinstance(ml_result, dict)) else {}
    )
    # Aggregate token usage across all 13 agents + judge
    _all_results = [
        fundamental_analyst, sentiment_analyst, news_analyst, technical_analyst,
        bullish_researcher, bearish_researcher, synthesizer,
        momentum_trader, value_trader, swing_trader, market_risk, portfolio_risk,
        judge,
    ]
    judge["token_usage"] = {
        "prompt_tokens":     sum(r.get("_token_usage", {}).get("prompt", 0) for r in _all_results),
        "completion_tokens": sum(r.get("_token_usage", {}).get("completion", 0) for r in _all_results),
        "total_tokens":      sum(r.get("_token_usage", {}).get("total", 0) for r in _all_results),
    }
    return judge


# ================================================================== #
# Consensus helper
# ================================================================== #

def _consensus_ft(verdicts: list) -> str:
    counts = {"WORTH_INVESTING": 0, "NEUTRAL": 0, "NOT_WORTH_INVESTING": 0}
    for v in verdicts:
        if v in counts:
            counts[v] += 1
    return max(counts, key=counts.get) if any(counts.values()) else "NEUTRAL"


# ================================================================== #
# JSON extraction helper
# ================================================================== #

def _extract_json(raw: str) -> dict:
    """
    Robustly extract a JSON object from a raw LLM response.
    Handles ReAct preamble (THOUGHT/ACTION/OBSERVATION/ANSWER sections).
    """
    text = raw.strip()
    if not text:
        raise json.JSONDecodeError("Empty response from LLM", "", 0)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text.rstrip())
        text = text.strip()
    # Strip ReAct preamble
    answer_match = re.search(r'ANSWER\s*:\s*', text, re.IGNORECASE)
    if answer_match:
        text = text[answer_match.end():]
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        block = text[start : end + 1]
        try:
            return json.loads(block, strict=False)
        except json.JSONDecodeError:
            pass
    repaired = repair_json(text, return_objects=True)
    if isinstance(repaired, dict) and repaired:
        return repaired
    raise json.JSONDecodeError("No JSON object found in response", text, 0)


# ================================================================== #
# Azure OpenAI call
# ================================================================== #

@lru_cache(maxsize=1)
def _get_azure_client() -> AzureOpenAI:
    """Build the Azure client once and reuse it across all agent calls.

    The 13-agent chain (and each configured provider) issues many requests
    per run; the OpenAI/Azure SDK client is thread-safe and pools HTTP
    connections, so constructing it once avoids redundant setup overhead.
    """
    return AzureOpenAI(
        api_key        = AZURE_OPENAI_KEY,
        azure_endpoint = AZURE_OPENAI_ENDPOINT,
        api_version    = AZURE_OPENAI_API_VERSION,
        timeout        = LLM_TIMEOUT,
        max_retries    = LLM_MAX_RETRIES,
    )


def _call_azure(prompt: str) -> dict:
    client = _get_azure_client()
    response = client.chat.completions.create(
        model      = AZURE_OPENAI_DEPLOYMENT,
        messages   = [
            {"role": "system", "content": _REACT_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature = LLM_TEMPERATURE,
        max_tokens  = LLM_MAX_TOKENS_AZURE,
    )
    raw_text = response.choices[0].message.content.strip()
    result = _extract_json(raw_text)
    result["llm_available"] = True
    result["_provider"]     = "Azure OpenAI"
    usage = response.usage
    result["_token_usage"] = {
        "prompt":     usage.prompt_tokens if usage else 0,
        "completion": usage.completion_tokens if usage else 0,
        "total":      usage.total_tokens if usage else 0,
    }
    return result


# ================================================================== #
# Google Gemini call
# ================================================================== #

@lru_cache(maxsize=1)
def _get_google_client() -> "genai.Client":
    """Build the Gemini client once and reuse it across all agent calls.

    The google-genai SDK takes its request timeout (in MILLISECONDS) via
    HttpOptions. Wrapped in try/except so an SDK version without that field
    degrades to the default rather than crashing the whole LLM path.
    """
    try:
        return genai.Client(
            api_key      = GOOGLE_API_KEY,
            http_options = genai_types.HttpOptions(timeout=int(LLM_TIMEOUT * 1000)),
        )
    except Exception:
        return genai.Client(api_key=GOOGLE_API_KEY)


def _call_google(prompt: str) -> dict:
    client = _get_google_client()
    response = client.models.generate_content(
        model   = GOOGLE_MODEL,
        contents= prompt,
        config  = genai_types.GenerateContentConfig(
            system_instruction = _REACT_SYSTEM,
            temperature        = LLM_TEMPERATURE,
            max_output_tokens  = LLM_MAX_TOKENS_GOOGLE,
        ),
    )
    try:
        raw_text = response.text.strip()
    except ValueError as e:
        raise RuntimeError(f"Gemini response blocked by safety filters: {e}") from e
    result = _extract_json(raw_text)
    result["llm_available"] = True
    result["_provider"]     = f"Google Gemini ({GOOGLE_MODEL})"
    usage = getattr(response, "usage_metadata", None)
    result["_token_usage"] = {
        "prompt":     getattr(usage, "prompt_token_count", 0) or 0,
        "completion": getattr(usage, "candidates_token_count", 0) or 0,
        "total":      getattr(usage, "total_token_count", 0) or 0,
    }
    return result


# ================================================================== #
# DeepSeek call (OpenAI-compatible)
# ================================================================== #

@lru_cache(maxsize=1)
def _get_deepseek_client() -> OpenAI:
    """Build the DeepSeek (OpenAI-compatible) client once and reuse it."""
    return OpenAI(
        api_key     = DEEPSEEK_API_KEY,
        base_url    = DEEPSEEK_BASE_URL,
        timeout     = LLM_TIMEOUT,
        max_retries = LLM_MAX_RETRIES,
    )


def _call_deepseek(prompt: str) -> dict:
    client = _get_deepseek_client()
    response = client.chat.completions.create(
        model      = DEEPSEEK_MODEL,
        messages   = [
            {"role": "system", "content": _REACT_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature = LLM_TEMPERATURE,
        max_tokens  = LLM_MAX_TOKENS_DEEPSEEK,
    )
    choice   = response.choices[0]
    raw_text = (choice.message.content or "").strip()
    finish   = getattr(choice, "finish_reason", None)

    if finish == "length" and not raw_text:
        raise ValueError(
            f"DeepSeek hit the token limit (max_tokens={LLM_MAX_TOKENS_DEEPSEEK}) "
            "before producing any JSON. Increase LLM_MAX_TOKENS_DEEPSEEK in .env."
        )
    if not raw_text:
        raise ValueError(
            "DeepSeek returned an empty response. "
            f"finish_reason={finish!r}. Check your API key and model name."
        )
    if finish == "length":
        # Truncated mid-JSON — try to repair before giving up
        import warnings
        warnings.warn(
            f"DeepSeek response was truncated (finish_reason='length'). "
            "Consider raising LLM_MAX_TOKENS_DEEPSEEK."
        )

    result = _extract_json(raw_text)
    result["llm_available"] = True
    result["_provider"]     = f"DeepSeek ({DEEPSEEK_MODEL})"
    usage = response.usage
    result["_token_usage"] = {
        "prompt":     usage.prompt_tokens if usage else 0,
        "completion": usage.completion_tokens if usage else 0,
        "total":      usage.total_tokens if usage else 0,
    }
    return result


# ================================================================== #
# Public entry point
# ================================================================== #

def get_llm_analysis(
    ticker:       str,
    info:         dict,
    technical:    dict,
    fundamental:  dict,
    statistical:  dict,
    analyst_data: dict,
    ml_result:    dict = None,
    accuracy_context: dict = None,
    accuracy_context_map: dict = None,
    progress_cb=None,
) -> dict:
    """
    Run the full 5-team, 13-agent LLM analysis pipeline for *ticker*.

    Returns a result dict consumed by terminal_display and report_generator.
    Falls back gracefully when no LLM credentials are configured.

    Parameters
    ----------
    accuracy_context_map : dict, optional
        Maps provider name → accuracy_context dict so each chain gets its own
        historical accuracy for bias correction and prompt injection.
        Falls back to the single ``accuracy_context`` when not provided.
    """
    # Helper: select per-provider context or fall back to the shared one
    def _ctx(provider_name: str) -> dict | None:
        if accuracy_context_map and provider_name in accuracy_context_map:
            return accuracy_context_map[provider_name]
        return accuracy_context

    description = build_company_description(info, analyst_data)

    if not LLM_ENABLED:
        return {
            "llm_available":      False,
            "recommendation":     "N/A -- LLM disabled",
            "confidence":         "N/A",
            "overall_short_term": "N/A",
            "overall_long_term":  "N/A",
            "target_prices":      {},
            "trade_direction":    "N/A",
            "entry_price":        None,
            "exit_price":         None,
            "stop_loss":          None,
            "position_size_pct":  None,
            "key_bull_case":      [],
            "key_bear_case":      [],
            "key_risks":          [],
            "catalysts":          [],
            "summary":            (
                "LLM analysis unavailable. Set AZURE_OPENAI_KEY+AZURE_OPENAI_ENDPOINT "
                "or GOOGLE_API_KEY in .env to enable."
            ),
            "technical_verdict":  "See technical section.",
            "fundamental_verdict":"See fundamental section.",
            "valuation_verdict":  "See statistical section.",
            "sentiment_verdict":  "See sentiment section.",
            "macro_verdict":      "N/A",
            "agents":             {},
            "providers":          {},
            "description":        description,
        }

    # Build the list of configured providers in priority order. The first
    # successful provider in this order becomes the primary (displayed) result.
    provider_specs = []
    if AZURE_ENABLED:
        provider_specs.append(("Azure OpenAI", _call_azure))
    if GOOGLE_ENABLED:
        provider_specs.append((f"Google Gemini ({GOOGLE_MODEL})", _call_google))
    if DEEPSEEK_ENABLED:
        provider_specs.append((f"DeepSeek ({DEEPSEEK_MODEL})", _call_deepseek))

    breakers = {pname: _CircuitBreaker(pname, LLM_CIRCUIT_THRESHOLD)
                for pname, _ in provider_specs}

    def _run_provider(pname: str, call_fn) -> dict:
        """Run one provider's full 13-agent chain. Never raises."""
        breaker = breakers[pname]
        try:
            chain = _run_agent_chain(
                call_fn, ticker, info, technical, fundamental,
                statistical, analyst_data, ml_result,
                accuracy_context=_ctx(pname),
                provider_name=pname, breaker=breaker, progress_cb=progress_cb,
            )
            # A tripped breaker means the endpoint was unreachable: the
            # "chain" is all rule-based fallbacks, so mark it unavailable
            # rather than presenting fabricated LLM output.
            if breaker.is_open():
                res = _error_result(
                    f"{pname} unreachable — {breaker.reason} "
                    f"(circuit opened at {breaker.tripped_at or 'startup'})."
                )
                res["_provider"] = pname
                return res
            chain["llm_available"] = True
            chain["_provider"]     = pname
            return chain
        except Exception as e:
            logger.warning("LLM provider %s failed: %s", pname, e)
            res = _error_result(f"{pname} error: {e}")
            res["_provider"] = pname
            return res

    # Run every provider chain concurrently — they are fully independent,
    # so wall-clock collapses to the slowest single provider rather than
    # the sum of all providers. Each chain still parallelises its own agents.
    providers_result = {}
    if provider_specs:
        with ThreadPoolExecutor(max_workers=len(provider_specs)) as ex:
            futures = {
                ex.submit(_run_provider, pname, fn): pname
                for pname, fn in provider_specs
            }
            for fut in futures:
                pname = futures[fut]
                providers_result[pname] = fut.result()

    # Select the primary result: first provider (in priority order) that
    # actually produced LLM output.
    primary_result = None
    for pname, _ in provider_specs:
        result = providers_result.get(pname)
        if result and result.get("llm_available", False):
            primary_result = result
            break

    # Per-provider failure summary — surfaced to the caller (main.py prints it)
    provider_errors = {
        pname: providers_result[pname].get("summary", "Unknown error")
        for pname, _ in provider_specs
        if not providers_result.get(pname, {}).get("llm_available", False)
    }

    if primary_result is None:
        # Every provider failed. Rather than returning a HOLD/N/A stub, hand
        # off to the deterministic rule-based judge so the user still gets a
        # real, actionable analysis. Lazy import avoids a circular dependency
        # (rule_based_judge imports build_company_description from this module).
        err_msg = " | ".join(f"{p}: {m}" for p, m in provider_errors.items()) or "unknown"
        logger.warning("All LLM providers failed (%s) — falling back to rule-based judge.", err_msg)
        try:
            from analysis.rule_based_judge import run_rule_based_analysis
            fallback = run_rule_based_analysis(
                info, technical, fundamental, statistical, analyst_data, ml_result,
                accuracy_context=accuracy_context,
            )
            fallback["llm_available"]   = False
            fallback["llm_failed"]      = True
            fallback["provider_errors"] = provider_errors
            fallback["providers"]       = providers_result
            fallback["description"]     = description
            fallback["summary"] = (
                "All LLM providers were unreachable — showing the deterministic "
                "rule-based judge instead. (" + err_msg + ")"
            )
            return fallback
        except Exception as e:
            logger.error("Rule-based fallback also failed: %s", e)
            res = _error_result(f"All LLM providers failed: {err_msg}")
            res["provider_errors"] = provider_errors
            return res

    primary_result["llm_available"]   = True
    primary_result["providers"]       = providers_result
    primary_result["provider_errors"] = provider_errors
    primary_result["description"]     = description

    # Apply per-provider bias correction to each provider's target prices
    try:
        from feedback.accuracy import apply_bias_correction
        for pname, presult in providers_result.items():
            if not presult.get("llm_available", False):
                continue
            pctx = _ctx(pname) or {}
            if pctx.get("has_data"):
                presult["target_prices"] = apply_bias_correction(
                    presult.get("target_prices", {}), pctx
                )
    except Exception:
        pass

    return primary_result


# ================================================================== #
# Provider health check (used by the --doctor diagnostic)
# ================================================================== #

def ping_providers() -> list[dict]:
    """
    Send a minimal 1-token request to every configured provider to check
    reachability, WITHOUT running the 13-agent chain. Reuses the same
    cached clients (so it exercises the real timeout/endpoint config) and
    the circuit breaker's error classifier for a clean reason.

    Returns one dict per configured provider:
        {"provider": str, "ok": bool, "latency_s": float|None,
         "connection_error": bool, "detail": str}
    An empty list means no provider is configured at all.
    """
    import time

    specs = []
    if AZURE_ENABLED:
        specs.append(("Azure OpenAI", _get_azure_client, AZURE_OPENAI_DEPLOYMENT, "openai"))
    if GOOGLE_ENABLED:
        specs.append((f"Google Gemini ({GOOGLE_MODEL})", _get_google_client, GOOGLE_MODEL, "google"))
    if DEEPSEEK_ENABLED:
        specs.append((f"DeepSeek ({DEEPSEEK_MODEL})", _get_deepseek_client, DEEPSEEK_MODEL, "openai"))

    # A diagnostic should fail fast — override to a short timeout so
    # `--doctor` doesn't wait out the full LLM_TIMEOUT per dead provider.
    ping_timeout = min(LLM_TIMEOUT, 15.0)

    results = []
    for pname, client_fn, model, kind in specs:
        t0 = time.time()
        try:
            client = client_fn()
            if kind == "openai":
                # .with_options returns a shallow client copy with a shorter
                # per-request timeout and no retries.
                client.with_options(timeout=ping_timeout, max_retries=0).chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0.0,
                )
            else:  # google
                client.models.generate_content(
                    model=model,
                    contents="ping",
                    config=genai_types.GenerateContentConfig(max_output_tokens=1),
                )
            results.append({
                "provider": pname, "ok": True,
                "latency_s": round(time.time() - t0, 2),
                "connection_error": False, "detail": "reachable",
            })
        except Exception as e:
            results.append({
                "provider": pname, "ok": False, "latency_s": None,
                "connection_error": _is_connection_error(e),
                "detail": f"{type(e).__name__}: {str(e)[:160]}",
            })
    return results
