"""
data/analyst_fetcher.py — Fetches analyst recommendations, consensus price
targets, and recent news headlines for a ticker.

Data source: Yahoo Finance via the `yfinance` library.

Returns an 'analyst_data' dict consumed by both the terminal display
and the LLM prompt-builder.
"""

import yfinance as yf
import datetime
import pandas as pd


# Module-level constant — defined once rather than re-created on every call.
_ACTION_LABELS = {
    "up":   "Upgrade",
    "down": "Downgrade",
    "init": "Initiation",
    "reit": "Reiterated",
    "main": "Maintained",
}


def fetch_analyst_data(ticker: str) -> dict:
    """
    Retrieve analyst-facing data for *ticker* from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Valid Yahoo Finance ticker symbol.

    Returns
    -------
    dict with keys:
      "recommendations"   — pd.DataFrame | None  (recent Buy/Hold/Sell history)
      "price_target"      — dict | None  (mean, low, high, numberOfAnalysts)
      "recommendation_key"— str  (e.g. "buy", "hold", "sell", "strong_buy")
      "news"              — list[dict]  (up to 10 recent headlines)
      "earnings_dates"    — pd.DataFrame | None  (upcoming / recent earnings)
    """

    stock = yf.Ticker(ticker)
    result: dict = {}

    # ------------------------------------------------------------------ #
    # 1. Analyst consensus recommendation key
    # ------------------------------------------------------------------ #
    # Yahoo Finance provides a pre-aggregated consensus string such as
    # "buy", "hold", "sell", "strong_buy", "underperform".
    # ETFs don't have a quoteSummary endpoint, so stock.info may raise a
    # 404 — fall back to "N/A" in that case.
    try:
        result["recommendation_key"] = stock.info.get("recommendationKey", "N/A")
    except Exception:
        result["recommendation_key"] = "N/A"

    # ------------------------------------------------------------------ #
    # 2. Price target summary (mean / low / high / count)
    # ------------------------------------------------------------------ #
    try:
        pt = stock.analyst_price_targets  # dict-like object in newer yfinance
        if pt is not None:
            result["price_target"] = {
                "current":  pt.get("current"),
                "mean":     pt.get("mean"),
                "low":      pt.get("low"),
                "high":     pt.get("high"),
                "median":   pt.get("median"),
            }
        else:
            result["price_target"] = None
    except Exception:
        result["price_target"] = None

    # ------------------------------------------------------------------ #
    # 3. Analyst upgrade/downgrade history (last 30 actions)
    #
    # Yahoo Finance returns a DataFrame with columns:
    #   Firm, ToGrade, FromGrade, Action
    # where Action ∈ {"up", "down", "init", "reit", "main"}.
    #
    # We convert each row to a plain dict so every downstream consumer
    # (LLM prompt, terminal display, Markdown report, PDF) can use it
    # without any pandas dependency.
    # ------------------------------------------------------------------ #
    try:
        recs: pd.DataFrame = stock.upgrades_downgrades
        if recs is not None and not recs.empty:
            # Normalise index to tz-naive dates
            idx = pd.to_datetime(recs.index)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            recs.index = idx

            recs_list = []
            for dt, row in recs.head(30).iterrows():
                raw_action = str(row.get("Action", row.get("action", ""))).lower()
                recs_list.append({
                    "date":       dt.date().isoformat(),
                    "firm":       str(row.get("Firm",      row.get("firm",       "Unknown"))),
                    "to_grade":   str(row.get("ToGrade",   row.get("toGrade",    ""))).strip(),
                    "from_grade": str(row.get("FromGrade", row.get("fromGrade",  ""))).strip(),
                    "action":     _ACTION_LABELS.get(raw_action, raw_action.title()),
                })
            result["recommendations"] = recs_list
        else:
            result["recommendations"] = []
    except Exception:
        result["recommendations"] = []

    # ------------------------------------------------------------------ #
    # 4. Recent news headlines (up to 10 articles)
    # ------------------------------------------------------------------ #
    try:
        news_raw = stock.news or []
        headlines = []
        for item in news_raw[:10]:
            # yfinance >= 0.2.50 wraps fields inside a nested "content" dict;
            # older versions use flat top-level keys — support both.
            content = item.get("content") or item
            provider = content.get("provider") or {}
            url = (content.get("canonicalUrl") or content.get("clickThroughUrl") or {}).get("url", "")
            headline = {
                "title":     content.get("title", ""),
                "publisher": provider.get("displayName") or content.get("publisher", ""),
                "link":      url or item.get("link", ""),
            }
            # Publish time: new API uses ISO string "pubDate"; old API used unix int
            pub_time = content.get("pubDate") or item.get("providerPublishTime") or item.get("publishedAt", "")
            headline["publish_time"] = pub_time
            if headline["title"]:  # skip items with no title
                headlines.append(headline)
        result["news"] = headlines
    except Exception:
        result["news"] = []

    # ------------------------------------------------------------------ #
    # 5. Earnings calendar (upcoming / most recent earnings date)
    # ------------------------------------------------------------------ #
    try:
        cal = stock.earnings_dates
        if cal is not None and not cal.empty:
            idx = pd.to_datetime(cal.index)
            if idx.tz is not None:
                idx = idx.tz_convert(None)
            cal.index = idx
            result["earnings_dates"] = cal.head(4).copy()

            # Compute proximity flags: is there an earnings event within 5 / 21
            # trading days from today?  Used by agents to add caution to
            # short-term predictions around binary event risk.
            today         = pd.Timestamp(datetime.date.today())
            future_dates  = cal.index[cal.index >= today]
            if not future_dates.empty:
                next_earnings = future_dates.min()
                days_away     = (next_earnings - today).days
                result["earnings_within_5d"]  = days_away <= 7   # calendar days ≈ 5 biz days
                result["earnings_within_21d"] = days_away <= 30  # calendar days ≈ 21 biz days
                result["next_earnings_date"]  = next_earnings.date().isoformat()
            else:
                result["earnings_within_5d"]  = False
                result["earnings_within_21d"] = False
                result["next_earnings_date"]  = None
        else:
            result["earnings_dates"]     = None
            result["earnings_within_5d"]  = False
            result["earnings_within_21d"] = False
            result["next_earnings_date"]  = None
    except Exception:
        result["earnings_dates"]     = None
        result["earnings_within_5d"]  = False
        result["earnings_within_21d"] = False
        result["next_earnings_date"]  = None

    # ------------------------------------------------------------------ #
    # 6. VIX (CBOE Volatility Index) — macro regime proxy
    # ------------------------------------------------------------------ #
    # VIX > 30 → high fear / risk-off regime (widen stop-losses, reduce size)
    # VIX 20-30 → elevated uncertainty
    # VIX < 20  → calm / risk-on regime
    try:
        vix_ticker = yf.Ticker("^VIX")
        vix_hist   = vix_ticker.history(period="5d", interval="1d")
        if vix_hist is not None and not vix_hist.empty:
            vix_val = float(vix_hist["Close"].dropna().iloc[-1])
            if vix_val >= 30:
                vix_regime = "HIGH FEAR (>30) — risk-off; widen stops, reduce position size"
            elif vix_val >= 20:
                vix_regime = "ELEVATED (20-30) — uncertainty; tighter risk management"
            else:
                vix_regime = "CALM (<20) — risk-on regime; normal position sizing"
            result["vix"] = {"value": round(vix_val, 2), "regime": vix_regime}
        else:
            result["vix"] = {"value": None, "regime": "N/A"}
    except Exception:
        result["vix"] = {"value": None, "regime": "N/A"}

    return result
