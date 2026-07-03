"""
discovery/sources.py — Candidate-ticker sources for `--discover` mode.

Every source is free and API-key-less, returns a list of Candidate
dicts, and is individually try/except-wrapped by the screener so one
dead source never kills a discovery run:

  • yfinance predefined screeners  (day gainers, most actives, …)
  • Volume-spike scan              (volume > 3× its 20-day average with
                                    positive 5-day momentum, over the
                                    full NASDAQ/NYSE listed universe)
  • 52-week-high breakouts         (close within 2% of a fresh 52-week
                                    high on above-average volume —
                                    piggybacks on the volume-scan
                                    download, zero extra requests)
  • New-listing detector           (diff of today's NASDAQ Trader
                                    symbol directory against the last
                                    cached copy — literally "new
                                    upcoming tickers": IPOs/uplistings)

A Candidate is {"symbol": str, "source": str, "note": str}.
"""

import datetime
import glob
import io
import logging
import os
import time

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# Cache lives inside the project so .gitignore's .cache/ entry covers it.
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "symbols"
)

# NASDAQ Trader symbol directories — the canonical free listing files.
_SYMBOL_FILES = {
    "nasdaqlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "otherlisted":  "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}

_PREDEFINED_SCREENS = [
    "day_gainers",
    "most_actives",
    "small_cap_gainers",
    "undervalued_growth_stocks",
    "growth_technology_stocks",
]


# ------------------------------------------------------------------ #
# Source 1 — yfinance predefined screeners
# ------------------------------------------------------------------ #

def from_yfinance_screens(count: int = 25) -> list[dict]:
    """Candidates from Yahoo's predefined screener queries."""
    candidates: list[dict] = []
    for screen in _PREDEFINED_SCREENS:
        try:
            resp = yf.screen(screen, count=count)
            quotes = (resp or {}).get("quotes", [])
            for q in quotes:
                sym = str(q.get("symbol", "")).upper().strip()
                if not sym:
                    continue
                candidates.append({
                    "symbol": sym,
                    "source": f"screen:{screen}",
                    "note":   str(q.get("shortName") or q.get("longName") or ""),
                })
        except Exception as exc:
            logger.warning("discover: screen '%s' failed: %s", screen, exc)
    return candidates


# ------------------------------------------------------------------ #
# Listed-symbol universe (shared by volume scan + new-listing detector)
# ------------------------------------------------------------------ #

def _download_symbol_file(name: str, url: str) -> "str | None":
    """
    Download a NASDAQ Trader symbol directory and cache it under
    .cache/symbols/{name}_{YYYY-MM-DD}.txt. Returns the cached path
    (today's file if it already exists) or None on failure.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    today = datetime.date.today().isoformat()
    path  = os.path.join(_CACHE_DIR, f"{name}_{today}.txt")
    if os.path.isfile(path):
        return path
    try:
        resp = requests.get(url, timeout=30,
                            headers={"User-Agent": "FinBot/1.0 (ticker discovery)"})
        resp.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        return path
    except Exception as exc:
        logger.warning("discover: could not download %s: %s", name, exc)
        return None


def _parse_symbol_file(path: str) -> pd.DataFrame:
    """Parse a pipe-delimited NASDAQ Trader file into a DataFrame."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # Last line is a "File Creation Time: ..." footer — drop it.
    lines = [ln for ln in text.splitlines() if "File Creation Time" not in ln]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)
    # nasdaqlisted uses "Symbol"; otherlisted uses "ACT Symbol".
    sym_col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
    df = df.rename(columns={sym_col: "Symbol"})
    return df.dropna(subset=["Symbol"])


def get_common_stock_universe() -> list[str]:
    """
    All plain common-stock symbols listed on NASDAQ/NYSE/AMEX:
    no ETFs, no test issues, no warrants/rights/units (symbols with
    punctuation or a 5th-letter W/R/U suffix on NASDAQ).
    """
    symbols: set[str] = set()
    for name, url in _SYMBOL_FILES.items():
        path = _download_symbol_file(name, url)
        if not path:
            continue
        try:
            df = _parse_symbol_file(path)
            if "ETF" in df.columns:
                df = df[df["ETF"].str.upper() != "Y"]
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"].str.upper() != "Y"]
            for sym in df["Symbol"]:
                sym = str(sym).strip().upper()
                if not sym.isalpha() or len(sym) > 5:
                    continue    # drops BRK.A / units / preferreds etc.
                if name == "nasdaqlisted" and len(sym) == 5 and sym[-1] in "WRU":
                    continue    # NASDAQ warrant/right/unit suffixes
                symbols.add(sym)
        except Exception as exc:
            logger.warning("discover: could not parse %s: %s", name, exc)
    return sorted(symbols)


# ------------------------------------------------------------------ #
# Source 2+3 — volume spikes & 52-week-high breakouts (one download)
# ------------------------------------------------------------------ #

def from_volume_scan(
    universe: list[str],
    chunk_size: int = 200,
    progress_cb=None,
) -> list[dict]:
    """
    Batch-download 1 year of daily bars for the whole *universe* (in
    chunks) and flag two momentum signatures from the same data:

      volume_spike   — yesterday..today volume > 3× the trailing 20-day
                       average AND the 5-day return is positive.
      52wk_breakout  — close within 2% of the period high on volume
                       > 1.5× the 20-day average (classic momentum-
                       continuation setup).
    """
    candidates: list[dict] = []
    chunks = [universe[i:i + chunk_size] for i in range(0, len(universe), chunk_size)]

    for ci, chunk in enumerate(chunks):
        if progress_cb:
            progress_cb(ci, len(chunks))
        try:
            df = yf.download(chunk, period="1y", auto_adjust=True,
                             group_by="ticker", progress=False, threads=True)
        except Exception as exc:
            logger.warning("discover: volume-scan chunk %d failed: %s", ci, exc)
            continue
        if df is None or df.empty:
            continue

        for sym in chunk:
            try:
                sub = df[sym] if isinstance(df.columns, pd.MultiIndex) else df
                close = sub["Close"].dropna()
                vol   = sub["Volume"].dropna()
                if len(close) < 30 or len(vol) < 25:
                    continue

                last_vol   = float(vol.iloc[-1])
                avg_vol_20 = float(vol.iloc[-21:-1].mean())
                ret_5d     = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0.0
                if avg_vol_20 > 0 and last_vol > 3.0 * avg_vol_20 and ret_5d > 0:
                    candidates.append({
                        "symbol": sym,
                        "source": "volume_spike",
                        "note":   f"vol {last_vol/avg_vol_20:.1f}x 20d avg, +{ret_5d*100:.1f}% 5d",
                    })

                period_high = float(close.max())
                last_close  = float(close.iloc[-1])
                if (len(close) >= 200 and period_high > 0
                        and last_close >= 0.98 * period_high
                        and avg_vol_20 > 0 and last_vol > 1.5 * avg_vol_20):
                    candidates.append({
                        "symbol": sym,
                        "source": "52wk_breakout",
                        "note":   f"within {abs(1 - last_close/period_high)*100:.1f}% of 52wk high on volume",
                    })
            except Exception:
                continue
        time.sleep(0.3)   # be polite to Yahoo between chunk requests

    return candidates


# ------------------------------------------------------------------ #
# Source 4 — new-listing detector
# ------------------------------------------------------------------ #

def from_new_listings() -> list[dict]:
    """
    Diff today's symbol directories against the most recent previously
    cached copies. Symbols that appear today but not before are fresh
    listings (IPOs, uplistings, SPAC completions) — literally "new
    upcoming tickers". On the first ever run there is no baseline, so
    this returns [] and starts detecting from the next run onward.
    """
    candidates: list[dict] = []
    today = datetime.date.today().isoformat()

    for name, url in _SYMBOL_FILES.items():
        today_path = _download_symbol_file(name, url)
        if not today_path:
            continue

        older = sorted(
            p for p in glob.glob(os.path.join(_CACHE_DIR, f"{name}_*.txt"))
            if not p.endswith(f"{name}_{today}.txt")
        )
        if not older:
            continue        # first run — baseline only
        prev_path = older[-1]

        try:
            now_syms  = set(_parse_symbol_file(today_path)["Symbol"].str.strip().str.upper())
            prev_syms = set(_parse_symbol_file(prev_path)["Symbol"].str.strip().str.upper())
        except Exception as exc:
            logger.warning("discover: new-listing diff failed for %s: %s", name, exc)
            continue

        prev_date = os.path.basename(prev_path).replace(f"{name}_", "").replace(".txt", "")
        for sym in sorted(now_syms - prev_syms):
            if sym.isalpha() and len(sym) <= 5:
                candidates.append({
                    "symbol": sym,
                    "source": "new_listing",
                    "note":   f"appeared in {name} since {prev_date}",
                })

    return candidates
