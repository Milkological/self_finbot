"""
data/edgar_data.py — Point-in-time fundamentals from SEC EDGAR (free, no key).

yfinance only exposes a company's CURRENT fundamental snapshot, which makes
fundamentals useless as leak-free ML features (stamping today's numbers onto
past dates is look-ahead) and un-backtestable. SEC EDGAR's XBRL company-facts
API gives every reported figure with BOTH the period it covers ("end") AND the
date it became public ("filed"). Joining features on the **filed** date is the
whole point — a trading day only ever sees fundamentals that were already
filed by then.

Public API
----------
  get_cik(ticker)                 -> "0000000000" | None
  get_quarterly_facts(ticker)     -> tidy DataFrame (one row per fiscal quarter)
  get_fundamental_features(ticker, index) -> DataFrame aligned to a price index,
                                    leak-free (merge_asof on filed date), or None

Data source & etiquette
------------------------
  https://www.sec.gov/files/company_tickers.json      (ticker→CIK, cached)
  https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
SEC fair-access asks for a descriptive User-Agent with contact info and ≤10
req/s. We cache the CIK map and each company's facts on disk (7-day TTL under
.cache/edgar/), so a normal run makes at most one request per ticker per week.

Foreign issuers / ADRs that don't file US-GAAP XBRL simply aren't in EDGAR;
every accessor returns None for them and the caller degrades to NaN features
(which the trainer drops per-ticker via its all-NaN-column rule — so they never
become degenerate zero columns).
"""

import datetime
import json
import logging
import os
import threading
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# SEC asks for a real contact in the UA string (milkeoh@gmail.com is the
# project owner). Do not remove — requests without it get 403s.
_HEADERS = {"User-Agent": "FinBot research (milkeoh@gmail.com)"}

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "edgar"
)
_CIK_TTL_DAYS   = 30
_FACTS_TTL_DAYS = 7
_MIN_REQUEST_GAP = 0.15          # ~7 req/s ceiling, politeness

_lock = threading.Lock()
_last_request = [0.0]
_cik_map: "dict | None" = None
_feat_cache: dict = {}           # ticker -> aligned features DataFrame (per process)

# XBRL tag candidates per concept (filers use different tags for the same idea).
# First tag found with USD (or USD/shares) units wins.
_FLOW_CONCEPTS = {
    "revenue":     ["RevenueFromContractWithCustomerExcludingAssessedTax",
                    "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income":  ["NetIncomeLoss", "ProfitLoss"],
    "op_income":   ["OperatingIncomeLoss"],
}
_EPS_TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"]


# ------------------------------------------------------------------ #
# HTTP with polite throttle + disk cache
# ------------------------------------------------------------------ #

def _throttle() -> None:
    with _lock:
        gap = time.time() - _last_request[0]
        if gap < _MIN_REQUEST_GAP:
            time.sleep(_MIN_REQUEST_GAP - gap)
        _last_request[0] = time.time()


def _cache_path(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, name)


def _fresh(path: str, ttl_days: int) -> bool:
    if not os.path.isfile(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < ttl_days * 86400


def _get_json(url: str, cache_name: str, ttl_days: int) -> "dict | None":
    path = _cache_path(cache_name)
    if _fresh(path, ttl_days):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    try:
        _throttle()
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data
    except Exception as exc:
        logger.debug("EDGAR fetch failed for %s: %s", url, exc)
        # Serve a stale cache if we have one — better than nothing.
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None


# ------------------------------------------------------------------ #
# Ticker → CIK
# ------------------------------------------------------------------ #

def get_cik(ticker: str) -> "str | None":
    """Return the 10-digit zero-padded CIK for *ticker*, or None."""
    global _cik_map
    with _lock:
        cached = _cik_map
    if cached is None:
        data = _get_json("https://www.sec.gov/files/company_tickers.json",
                         "company_tickers.json", _CIK_TTL_DAYS)
        cached = {}
        if data:
            for row in data.values():
                cached[str(row["ticker"]).upper()] = str(row["cik_str"]).zfill(10)
        with _lock:
            _cik_map = cached
    return cached.get(ticker.upper().strip())


# ------------------------------------------------------------------ #
# Quarterly facts extraction
# ------------------------------------------------------------------ #

def _quarterly_series(usgaap: dict, tags: list, per_share: bool = False) -> pd.DataFrame:
    """
    Extract a clean quarterly series, MERGED across all candidate tags.

    Filers switch tags over time (e.g. NVDA moved revenue from
    RevenueFromContractWithCustomerExcludingAssessedTax to Revenues), so a
    single tag misses whole eras. We pool every candidate tag's ~80–100 day
    (single-quarter) facts and keep one row per fiscal period_end (earliest
    filing — the original 10-Q, not a later restatement).

    Returns columns [period_end, filed, value]; empty if nothing matched.
    """
    unit_key = "USD/shares" if per_share else "USD"
    recs = []
    for tag in tags:
        node = usgaap.get(tag)
        if not node:
            continue
        for r in node.get("units", {}).get(unit_key, []):
            start, end, filed = r.get("start"), r.get("end"), r.get("filed")
            val = r.get("val")
            if not (start and end and filed) or val is None:
                continue
            try:
                d0 = datetime.date.fromisoformat(start)
                d1 = datetime.date.fromisoformat(end)
            except ValueError:
                continue
            if 80 <= (d1 - d0).days <= 100:          # a single fiscal quarter
                recs.append({"period_end": d1,
                             "filed": datetime.date.fromisoformat(filed),
                             "value": float(val)})
    if not recs:
        return pd.DataFrame(columns=["period_end", "filed", "value"])
    df = pd.DataFrame(recs).sort_values(["period_end", "filed"])
    df = df.drop_duplicates(subset="period_end", keep="first").reset_index(drop=True)
    return df


def _load_usgaap(ticker: str) -> "dict | None":
    """Fetch a ticker's us-gaap fact block (cached), or None if not in EDGAR."""
    cik = get_cik(ticker)
    if not cik:
        return None
    facts = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                      f"facts_{cik}.json", _FACTS_TTL_DAYS)
    if not facts:
        return None
    return facts.get("facts", {}).get("us-gaap", {}) or None


def get_quarterly_facts(ticker: str) -> "pd.DataFrame | None":
    """
    Tidy quarterly-fundamentals DataFrame for *ticker* (one row per fiscal
    quarter, sorted by period_end):
      columns: period_end, revenue, net_income, op_income, eps_diluted,
               and a *_filed date per concept.
    None when the ticker isn't in EDGAR (foreign/ADR) or has no usable data.
    """
    usgaap = _load_usgaap(ticker)
    if not usgaap:
        return None

    frames = []
    for name, tags in (("revenue",    _FLOW_CONCEPTS["revenue"]),
                       ("net_income", _FLOW_CONCEPTS["net_income"]),
                       ("op_income",  _FLOW_CONCEPTS["op_income"]),
                       ("eps_diluted", _EPS_TAGS)):
        s = _quarterly_series(usgaap, tags, per_share=(name == "eps_diluted"))
        if not s.empty:
            frames.append(s.rename(columns={"value": name, "filed": f"{name}_filed"}))
    if not frames:
        return None
    base = frames[0]
    for f in frames[1:]:
        base = base.merge(f, on="period_end", how="outer")
    return base.sort_values("period_end").reset_index(drop=True)


# ------------------------------------------------------------------ #
# Leak-free feature construction
# ------------------------------------------------------------------ #

# NOTE: we use net-income YoY, not EPS YoY. EDGAR reports EPS as-filed, so a
# stock split (e.g. NVDA's 2024 10-for-1) makes raw EPS incomparable across
# the split and produces nonsense YoY. Revenue and net income are absolute
# dollar figures and split-immune.
FEATURE_NAMES = [
    "edgar_revenue_yoy", "edgar_ni_yoy", "edgar_revenue_accel",
    "edgar_op_margin", "edgar_net_margin",
]


def _asof(idx: pd.DatetimeIndex, dates, values) -> "list":
    """
    Point-in-time align (public_date, value) pairs onto *idx*: each row of
    idx gets the most recent value whose public_date is ≤ that row's date.
    This is the leak-free join — a value is invisible until its filing date.
    Ratios are clipped to [-5, 5] so one freak filing can't dominate scaling.
    """
    # Normalise both merge keys to nanosecond datetimes — pandas 3.x refuses
    # merge_asof across mismatched datetime resolutions (e.g. us vs s).
    d = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(list(dates)).values).astype("datetime64[ns]"),
        "v":    pd.to_numeric(pd.Series(list(values)).values, errors="coerce"),
    }).dropna(subset=["date"]).sort_values("date")
    d["v"] = d["v"].clip(-5, 5)
    if d.empty:
        return [float("nan")] * len(idx)
    left = pd.DataFrame({"date": pd.to_datetime(idx).astype("datetime64[ns]")})
    left = left.reset_index(names="pos").sort_values("date")
    merged = pd.merge_asof(left, d, on="date", direction="backward").sort_values("pos")
    return merged["v"].tolist()


def get_fundamental_features(ticker: str, index: "pd.Index") -> "pd.DataFrame | None":
    """
    DataFrame indexed like *index* (a price DatetimeIndex) with the EDGAR
    features in FEATURE_NAMES, or None if unavailable.

    Leak-free by construction: every derived metric is stamped at the date
    all its inputs became public (max of the relevant concepts' filing dates)
    and joined backward. Values before the first filing are NaN (not 0) so
    the trainer drops the columns for data-less tickers instead of learning
    a constant.
    """
    with _lock:
        if ticker in _feat_cache:
            cached = _feat_cache[ticker]
            return cached.reindex(index) if cached is not None else None

    result = None
    try:
        usgaap = _load_usgaap(ticker)
        if usgaap:
            rev = _quarterly_series(usgaap, _FLOW_CONCEPTS["revenue"])
            oi  = _quarterly_series(usgaap, _FLOW_CONCEPTS["op_income"])
            ni  = _quarterly_series(usgaap, _FLOW_CONCEPTS["net_income"])

            idx = pd.to_datetime(index)
            out = pd.DataFrame(index=index)
            any_data = False

            # Revenue YoY + acceleration — public on the quarter's filing date.
            if len(rev) >= 5:
                rev = rev.sort_values("period_end").reset_index(drop=True)
                yoy = rev["value"] / rev["value"].shift(4) - 1
                accel = yoy - yoy.shift(1)
                out["edgar_revenue_yoy"]   = _asof(idx, rev["filed"], yoy)
                out["edgar_revenue_accel"] = _asof(idx, rev["filed"], accel)
                any_data = True
            else:
                out["edgar_revenue_yoy"] = float("nan")
                out["edgar_revenue_accel"] = float("nan")

            # Net-income YoY (split-immune, sign-safe denominator).
            if len(ni) >= 5:
                ni_s = ni.sort_values("period_end").reset_index(drop=True)
                prev = ni_s["value"].shift(4)
                ni_yoy = (ni_s["value"] - prev) / prev.abs()
                out["edgar_ni_yoy"] = _asof(idx, ni_s["filed"], ni_yoy)
                any_data = True
            else:
                out["edgar_ni_yoy"] = float("nan")

            # Margins — public when BOTH numerator and revenue are filed.
            def _margin(flow: pd.DataFrame, colname: str) -> None:
                nonlocal any_data
                if flow.empty or rev.empty:
                    out[colname] = float("nan"); return
                m = rev.rename(columns={"value": "rev", "filed": "rev_filed"}).merge(
                    flow.rename(columns={"value": "num", "filed": "num_filed"}),
                    on="period_end", how="inner")
                if m.empty:
                    out[colname] = float("nan"); return
                margin = m["num"] / m["rev"].where(m["rev"] > 0)
                pub = m[["rev_filed", "num_filed"]].max(axis=1)
                out[colname] = _asof(idx, pub, margin)
                any_data = True

            _margin(oi, "edgar_op_margin")
            _margin(ni, "edgar_net_margin")

            result = out[FEATURE_NAMES] if any_data else None
    except Exception as exc:
        logger.debug("EDGAR feature build failed for %s: %s", ticker, exc)
        result = None

    with _lock:
        _feat_cache[ticker] = result
    return result
