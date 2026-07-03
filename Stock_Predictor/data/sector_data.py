"""
data/sector_data.py — Sector-median valuation multiples (free, cached).

A P/E of 30 is cheap for semiconductors and outrageous for banks, so
absolute valuation thresholds systematically punish growth sectors and
flatter deep-value ones. This module supplies each Yahoo sector's
median multiples so analysis/rule_based_judge._score_valuation() can
grade a stock against its actual peer group.

How the medians are built
-------------------------
One Yahoo screener query per sector (`yf.screen` with an EquityQuery,
key-free): the ~100 largest US stocks in that sector by market cap.
Screener quotes carry trailingPE / forwardPE / priceToBook directly, so
no per-constituent .info calls are needed — a whole sector costs a
single request. Medians of the positive values are cached to
.cache/sector_medians.json with a 7-day TTL (sector medians drift
slowly).

P/S and EV/EBITDA are NOT in screener quotes; the judge keeps absolute
thresholds for those.

Fallback
--------
If the screener is unreachable and no cache exists, a static table of
approximate sector medians (US large/mid caps, early-2026 vintage) is
used so the judge still grades relative to *something* sector-shaped.
The "source" field in the returned dict says which path served it.
"""

import datetime
import json
import logging
import os
import statistics
import threading

import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".cache", "sector_medians.json",
)
_TTL_DAYS = 7
_lock = threading.Lock()
_mem_cache: dict = {}      # sector -> result dict (per-process)

# Yahoo's 11 sector names as they appear in info["sector"] AND in the
# screener's sector field — the two vocabularies match.
_YAHOO_SECTORS = {
    "Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
    "Consumer Defensive", "Industrials", "Energy", "Basic Materials",
    "Communication Services", "Utilities", "Real Estate",
}

# Approximate US sector medians (large/mid cap, early 2026). Only used
# when both the live screen and the disk cache are unavailable.
_STATIC_MEDIANS: dict[str, dict] = {
    "Technology":             {"trailing_pe": 32.0, "forward_pe": 26.0, "price_to_book": 6.5},
    "Financial Services":     {"trailing_pe": 13.0, "forward_pe": 12.0, "price_to_book": 1.4},
    "Healthcare":             {"trailing_pe": 22.0, "forward_pe": 17.0, "price_to_book": 3.5},
    "Consumer Cyclical":      {"trailing_pe": 20.0, "forward_pe": 17.0, "price_to_book": 3.0},
    "Consumer Defensive":     {"trailing_pe": 20.0, "forward_pe": 18.0, "price_to_book": 3.5},
    "Industrials":            {"trailing_pe": 22.0, "forward_pe": 19.0, "price_to_book": 3.5},
    "Energy":                 {"trailing_pe": 13.0, "forward_pe": 12.0, "price_to_book": 1.7},
    "Basic Materials":        {"trailing_pe": 17.0, "forward_pe": 14.0, "price_to_book": 2.0},
    "Communication Services": {"trailing_pe": 19.0, "forward_pe": 15.0, "price_to_book": 2.5},
    "Utilities":              {"trailing_pe": 19.0, "forward_pe": 17.0, "price_to_book": 2.0},
    "Real Estate":            {"trailing_pe": 30.0, "forward_pe": 32.0, "price_to_book": 2.0},
}

# quote-field → result-key mapping for the medians we can compute.
_QUOTE_FIELDS = {
    "trailingPE":  "trailing_pe",
    "forwardPE":   "forward_pe",
    "priceToBook": "price_to_book",
}


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def get_sector_multiples(sector: str) -> "dict | None":
    """
    Return median valuation multiples for *sector*:

        {"trailing_pe": float|None, "forward_pe": float|None,
         "price_to_book": float|None, "n": int, "source": str}

    or None when *sector* is missing/unknown (e.g. ETFs report "N/A").
    Resolution order: process memo → fresh disk cache → live Yahoo
    screen (re-cached) → stale disk cache → static table.
    """
    if not sector or sector not in _YAHOO_SECTORS:
        return None

    with _lock:
        if sector in _mem_cache:
            return _mem_cache[sector]

    disk = _load_disk_cache()
    entry = disk.get(sector)
    if entry and _is_fresh(entry):
        result = {k: entry.get(k) for k in ("trailing_pe", "forward_pe", "price_to_book")}
        result.update({"n": entry.get("n", 0), "source": "cache"})
        return _remember(sector, result)

    live = _fetch_sector_medians(sector)
    if live:
        disk[sector] = {**live, "ts": datetime.date.today().isoformat()}
        _save_disk_cache(disk)
        return _remember(sector, {**live, "source": "yahoo_screen"})

    if entry:   # stale cache beats a hardcoded guess
        result = {k: entry.get(k) for k in ("trailing_pe", "forward_pe", "price_to_book")}
        result.update({"n": entry.get("n", 0), "source": "stale_cache"})
        return _remember(sector, result)

    static = _STATIC_MEDIANS.get(sector)
    if static:
        return _remember(sector, {**static, "n": 0, "source": "static_fallback"})
    return None


# ------------------------------------------------------------------ #
# Internals
# ------------------------------------------------------------------ #

def _remember(sector: str, result: dict) -> dict:
    with _lock:
        _mem_cache[sector] = result
    return result


def _is_fresh(entry: dict) -> bool:
    try:
        ts = datetime.date.fromisoformat(entry.get("ts", ""))
        return (datetime.date.today() - ts).days < _TTL_DAYS
    except Exception:
        return False


def _load_disk_cache() -> dict:
    try:
        if os.path.isfile(_CACHE_PATH):
            with open(_CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_disk_cache(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.debug("sector cache write failed: %s", exc)


def _fetch_sector_medians(sector: str, size: int = 100) -> "dict | None":
    """One screener request: top *size* US stocks in *sector* by mkt cap."""
    try:
        query = yf.EquityQuery("and", [
            yf.EquityQuery("eq", ["sector", sector]),
            yf.EquityQuery("eq", ["region", "us"]),
        ])
        resp = yf.screen(query, size=size,
                         sortField="intradaymarketcap", sortAsc=False)
        quotes = (resp or {}).get("quotes", [])
        if len(quotes) < 10:
            return None

        result: dict = {}
        n_used = 0
        for field, key in _QUOTE_FIELDS.items():
            # Median of positive values only — negative P/E means losses,
            # not cheapness, and would drag the peer benchmark nonsense-ward.
            vals = sorted(
                float(q[field]) for q in quotes
                if isinstance(q.get(field), (int, float)) and 0 < q[field] < 500
            )
            if len(vals) >= 10:
                result[key] = round(statistics.median(vals), 2)
                n_used = max(n_used, len(vals))
            else:
                result[key] = None
        if not any(v for v in result.values()):
            return None
        result["n"] = n_used
        return result
    except Exception as exc:
        logger.warning("sector screen failed for %s: %s", sector, exc)
        return None
