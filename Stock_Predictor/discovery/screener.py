"""
discovery/screener.py — Rank discovery candidates and update the watchlist.

Flow (run_discovery):
  1. Gather candidates from every source in discovery/sources.py
     (each isolated — a dead source is logged and skipped).
  2. Merge by symbol; a ticker flagged by several independent sources
     ranks up (multi-source confirmation is itself a signal).
  3. Hard filters via fast_info: price ≥ $2, market cap ≥ min_cap,
     ~$1M/day dollar volume. Survivors capped (rate-limit safety).
  4. quick_score(): one 1y history + one info fetch per candidate,
     then the existing rule-based fundamental + technical lenses via
     rule_based_judge.run_quick_judge() — no ML, no Monte Carlo.
     Composite = 0.40·technical + 0.35·fundamental + 0.25·confirmation.
  5. Output: rich table, ranked CSV under reports/discovery/, and the
     top N appended to watchlist.txt with provenance comments.
     tickers.txt is user-owned and never touched. Promote discoveries
     with:  python main.py --file watchlist.txt --no-llm
"""

import datetime
import logging
import os
import time

import pandas as pd
import numpy as np

from analysis.rule_based_judge import run_quick_judge
from analysis.technical import compute_all_technicals
from config import REPORTS_DIR
from data import market_data
from discovery import sources

logger = logging.getLogger(__name__)

_PROJECT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCHLIST_PATH = os.path.join(_PROJECT_DIR, "watchlist.txt")
TICKERS_PATH   = os.path.join(_PROJECT_DIR, "tickers.txt")

MAX_SCORED = 40          # cap on candidates that get a full quick_score


# ------------------------------------------------------------------ #
# Candidate gathering & filtering
# ------------------------------------------------------------------ #

def _gather_candidates(console, include_volume_scan: bool = True) -> dict[str, dict]:
    """Run all sources and merge results by symbol."""
    raw: list[dict] = []

    with console.status("[dim]Querying yfinance predefined screeners…[/dim]"):
        try:
            raw += sources.from_yfinance_screens()
        except Exception as exc:
            console.log(f"[yellow]Screener source failed: {exc}[/yellow]")

    try:
        new_listings = sources.from_new_listings()
        raw += new_listings
        if new_listings:
            console.log(f"[green]New listings detected: "
                        f"{', '.join(c['symbol'] for c in new_listings[:15])}[/green]")
    except Exception as exc:
        console.log(f"[yellow]New-listing source failed: {exc}[/yellow]")

    if include_volume_scan:
        try:
            universe = sources.get_common_stock_universe()
            if universe:
                console.log(f"[dim]Volume/breakout scan over {len(universe)} listed symbols "
                            f"(chunked — takes a few minutes)…[/dim]")
                with console.status("[dim]Scanning…[/dim]") as status:
                    def _cb(done, total):
                        status.update(f"[dim]Scanning volume/breakouts… chunk {done + 1}/{total}[/dim]")
                    raw += sources.from_volume_scan(universe, progress_cb=_cb)
        except Exception as exc:
            console.log(f"[yellow]Volume-scan source failed: {exc}[/yellow]")

    merged: dict[str, dict] = {}
    for c in raw:
        sym = c["symbol"]
        entry = merged.setdefault(sym, {"symbol": sym, "sources": [], "notes": []})
        if c["source"] not in entry["sources"]:
            entry["sources"].append(c["source"])
        if c["note"]:
            entry["notes"].append(f"{c['source']}: {c['note']}")
    return merged


def _known_symbols() -> set[str]:
    """Symbols already tracked in tickers.txt or watchlist.txt."""
    known: set[str] = set()
    for path in (TICKERS_PATH, WATCHLIST_PATH):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    token = line.split("#", 1)[0].strip().upper()
                    if token:
                        known.add(token)
        except OSError:
            pass
    return known


def _passes_hard_filters(symbol: str, min_cap: float) -> bool:
    """Cheap tradability screen via fast_info (one light request)."""
    try:
        fi = market_data.get_ticker(symbol).fast_info
        price = fi.get("lastPrice") or fi.get("last_price")
        cap   = fi.get("marketCap") or fi.get("market_cap")
        vol10 = (fi.get("tenDayAverageVolume")
                 or fi.get("ten_day_average_volume") or 0)
        if not price or price < 2.0:
            return False
        if cap is not None and cap < min_cap:
            return False
        if price * (vol10 or 0) < 1_000_000:   # ≥ $1M/day traded
            return False
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ #
# Quick scoring
# ------------------------------------------------------------------ #

def _quick_score(symbol: str, n_sources: int) -> "dict | None":
    """
    Two-lens rule-based score plus a source-confirmation component.
    Returns None when no usable price history exists.
    """
    try:
        hist = market_data.get_history(symbol, "1y")
    except Exception:
        return None
    if hist is None or hist.empty or len(hist) < 10:
        return None

    price_df = hist.copy()
    idx = pd.to_datetime(price_df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    price_df.index = idx
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in price_df.columns]
    price_df = price_df[cols].dropna()
    price_df["Log_Return"] = np.log(price_df["Close"] / price_df["Close"].shift(1))

    info = market_data.get_info(symbol)
    try:
        technical = compute_all_technicals(price_df)
    except Exception:
        technical = {}

    lenses = run_quick_judge(info, technical)
    tech_score = lenses["technical"]["score"]
    fund_score = lenses["fundamental"]["score"]
    confirmation = min(100.0, 35.0 * n_sources)

    composite = round(0.40 * tech_score + 0.35 * fund_score + 0.25 * confirmation, 1)
    return {
        "quick_score":  composite,
        "tech_score":   tech_score,
        "fund_score":   fund_score,
        "confirmation": confirmation,
        "price":        float(price_df["Close"].iloc[-1]),
        "name":         str(info.get("shortName") or ""),
        "sector":       str(info.get("sector") or "N/A"),
        "market_cap":   info.get("marketCap"),
        "history_days": len(price_df),
    }


# ------------------------------------------------------------------ #
# Outputs
# ------------------------------------------------------------------ #

def _write_csv(rows: list[dict]) -> str:
    out_dir = os.path.join(REPORTS_DIR, "discovery")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"discovery_{datetime.date.today().isoformat()}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _append_watchlist(rows: list[dict], top_n: int) -> list[str]:
    """Append the top N discoveries to watchlist.txt with provenance."""
    added: list[str] = []
    today = datetime.date.today().isoformat()
    lines = []
    for row in rows[:top_n]:
        srcs = "+".join(s.replace("screen:", "") for s in row["sources"])
        lines.append(
            f"{row['symbol']:<6} # discovered {today} via {srcs}, score {row['quick_score']}"
        )
        added.append(row["symbol"])
    if lines:
        header_needed = not os.path.isfile(WATCHLIST_PATH)
        with open(WATCHLIST_PATH, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("# FinBot discovery watchlist — promote picks with:\n"
                        "#   python main.py --file watchlist.txt --no-llm\n")
            f.write("\n".join(lines) + "\n")
    return added


# ------------------------------------------------------------------ #
# Entry point (called from main.py --discover)
# ------------------------------------------------------------------ #

def run_discovery(console, top_n: int = 5, min_cap: float = 100e6) -> None:
    from rich.table import Table

    console.print("\n[bold cyan]FinBot — Ticker Discovery[/bold cyan]\n")

    merged = _gather_candidates(console)
    console.print(f"Candidates from all sources: [bold]{len(merged)}[/bold]")
    if not merged:
        console.print("[yellow]No candidates found — all sources may be unreachable.[/yellow]")
        return

    # Skip what we already track, prioritise multi-source names, filter.
    known = _known_symbols()
    fresh = [c for s, c in merged.items() if s not in known]
    fresh.sort(key=lambda c: len(c["sources"]), reverse=True)

    survivors: list[dict] = []
    with console.status("[dim]Applying tradability filters…[/dim]") as status:
        for i, cand in enumerate(fresh):
            if len(survivors) >= MAX_SCORED:
                break
            status.update(f"[dim]Filtering {cand['symbol']} "
                          f"({i + 1}/{len(fresh)}, kept {len(survivors)})…[/dim]")
            if _passes_hard_filters(cand["symbol"], min_cap):
                survivors.append(cand)
            time.sleep(0.15)   # spread out fast_info requests

    console.print(f"Passed price/liquidity/market-cap filters: [bold]{len(survivors)}[/bold]")
    if not survivors:
        console.print("[yellow]Nothing tradable survived the filters today.[/yellow]")
        return

    # Full quick score on the survivors.
    rows: list[dict] = []
    with console.status("[dim]Scoring candidates…[/dim]") as status:
        for i, cand in enumerate(survivors):
            status.update(f"[dim]Scoring {cand['symbol']} ({i + 1}/{len(survivors)})…[/dim]")
            scored = _quick_score(cand["symbol"], len(cand["sources"]))
            if scored is None:
                continue
            rows.append({
                "symbol":  cand["symbol"],
                "sources": cand["sources"],
                "notes":   " | ".join(cand["notes"])[:300],
                **scored,
            })
            time.sleep(0.15)

    if not rows:
        console.print("[yellow]No candidate had scoreable price history.[/yellow]")
        return
    rows.sort(key=lambda r: r["quick_score"], reverse=True)

    # ── Terminal table ──────────────────────────────────────────── #
    table = Table(title=f"Discovery — {datetime.date.today().isoformat()} "
                        f"(top {min(len(rows), 15)} of {len(rows)})")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Symbol", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Tech", justify="right")
    table.add_column("Fund", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Sector")
    table.add_column("Sources")
    for i, r in enumerate(rows[:15], start=1):
        table.add_row(
            str(i), r["symbol"], f"{r['quick_score']:.0f}",
            f"{r['tech_score']:.0f}", f"{r['fund_score']:.0f}",
            f"${r['price']:.2f}", r["sector"],
            ", ".join(s.replace("screen:", "") for s in r["sources"]),
        )
    console.print(table)

    csv_path = _write_csv([
        {**r, "sources": "+".join(r["sources"])} for r in rows
    ])
    console.print(f"Full ranked list: [green]{csv_path}[/green]")

    added = _append_watchlist(rows, top_n)
    if added:
        console.print(
            f"Appended to [bold]watchlist.txt[/bold]: {', '.join(added)}\n"
            f"[dim]Run a full analysis on them with:  "
            f"python main.py --file watchlist.txt --no-llm[/dim]\n"
        )
