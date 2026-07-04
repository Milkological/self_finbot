"""
analysis/backtester.py — Historical validation of the technical lens.

Answers the question none of the live machinery can answer quickly:
**do the rule-based judge's technical scores actually predict forward
returns?** The live feedback loop only learns from predictions made
from today onward; this harness replays the scoring rules across years
of history and pools thousands of observations in minutes.

What is replayed (and what is deliberately not)
-----------------------------------------------
Only the TECHNICAL lens is replayed. Every input it needs — RSI, MACD,
SMAs, ADX/DI, Bollinger %B, OBV, 52-week position — is a past-only
rolling computation, so scoring "as of" a historical date is leak-free:
indicators are computed once over the full history and read at row i,
and the 52-week high/low is the trailing 252-bar max/min at row i.

The fundamental and valuation lenses CANNOT be honestly backtested with
free data: yfinance only provides *current* snapshots (today's P/E,
margins, DCF inputs), and stamping those onto past dates is look-ahead
bias that would manufacture beautiful fake results. The OLS-trend bonus
signal is likewise skipped (it lives in the statistical dict). So this
is a verdict on the timing half of the judge — which is also what
drives 40% of the discovery quick score.

Method
------
For every ticker in the universe: sample weekly (every 5th bar) from
bar 30 onward, score the technical lens as of that bar, then look up
the realised 21-bar and 63-bar forward returns and the same-window SPY
return (excess = ticker − SPY). Results pool across tickers into:

  • forward return / hit rate / excess return by score bucket
  • Spearman rank IC (score vs forward return)
  • a full-vs-thin split by how many signals fed each score, so we
    learn whether scores from short-history tickers (the discovery
    case) deserve the same trust as fully-informed ones

Caveats printed with the results: today's universe = survivorship bias
(delisted losers are invisible), and overlapping weekly windows inflate
n relative to truly independent samples — treat significance
accordingly.
"""

import datetime
import logging
import os

import numpy as np
import pandas as pd

from analysis.rule_based_judge import _score_technical
from analysis.technical import compute_all_technicals
from config import HISTORY_PERIOD, REPORTS_DIR
from data import market_data

logger = logging.getLogger(__name__)

SAMPLE_EVERY = 5          # score once per trading week
WARMUP_BARS  = 30         # RSI/MACD/BB/ADX are all live by ~27 bars
MIN_BARS     = 126 + 21   # ~6 months history floor + shortest horizon
HORIZONS     = (21, 63)   # forward windows in trading days
# A fully-informed score draws on 13 possible points (RSI 2, MACD 2,
# SMAs 2, ADX 2, %B 2, 52wk 1, OBV 2). Below this the score came from
# a reduced signal set (young ticker / missing data).
FULL_POSSIBLE = 13

_BUCKETS = [(-1, 40, "<40  (bearish)"),
            (40, 55, "40–54 (neutral-)"),
            (55, 70, "55–69 (neutral+)"),
            (70, 999, "≥70  (bullish)")]


# ------------------------------------------------------------------ #
# Per-ticker replay
# ------------------------------------------------------------------ #

def _prepare_price_df(symbol: str) -> "pd.DataFrame | None":
    """5y OHLCV formatted exactly like the live pipeline feeds the lens."""
    try:
        hist = market_data.get_history(symbol, HISTORY_PERIOD)
    except Exception:
        return None
    if hist is None or len(hist) < MIN_BARS:
        return None
    df = hist.copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    df.index = idx
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[cols].dropna()
    return df if len(df) >= MIN_BARS else None


def _signals_asof(ind: pd.DataFrame, i: int) -> dict:
    """Rebuild the two signal strings the lens reads, as of row *i*,
    using the exact derivations from technical.compute_all_technicals."""
    sig: dict = {}
    row = ind.iloc[i]

    macd_hist, macd_line, macd_sig = row.get("MACD_Hist"), row.get("MACD_Line"), row.get("MACD_Signal")
    if pd.notna(macd_hist) and pd.notna(macd_line) and pd.notna(macd_sig):
        if macd_hist > 0 and macd_line > macd_sig:
            sig["MACD"] = "BULLISH — MACD above Signal line"
        elif macd_hist < 0 and macd_line < macd_sig:
            sig["MACD"] = "BEARISH — MACD below Signal line"
        else:
            sig["MACD"] = "NEUTRAL / CROSSOVER"

    if "OBV" in ind.columns and i >= 20:
        obv_now, obv_then = ind["OBV"].iloc[i], ind["OBV"].iloc[i - 20]
        px_now,  px_then  = ind["Close"].iloc[i], ind["Close"].iloc[i - 20]
        if pd.notna(obv_now) and pd.notna(obv_then):
            price_up, obv_up = px_now > px_then, obv_now > obv_then
            if price_up and obv_up:
                sig["OBV"] = "CONFIRMING UPTREND — volume supports price rise"
            elif not price_up and not obv_up:
                sig["OBV"] = "CONFIRMING DOWNTREND — volume supports price decline"
            elif price_up:
                sig["OBV"] = "BEARISH DIVERGENCE — price rising but volume retreating"
            else:
                sig["OBV"] = "BULLISH DIVERGENCE — price falling but volume accumulating"
    return sig


def replay_ticker(symbol: str) -> list[dict]:
    """Score the technical lens weekly across *symbol*'s history."""
    price_df = _prepare_price_df(symbol)
    if price_df is None:
        return []
    try:
        ind = compute_all_technicals(price_df)["df"]
    except Exception as exc:
        logger.warning("backtest: indicators failed for %s: %s", symbol, exc)
        return []

    close = ind["Close"]
    # Trailing 252-bar extremes as of each row — the leak-free stand-in
    # for info["fiftyTwoWeekHigh"/"Low"].
    hi_52 = close.rolling(252, min_periods=60).max()
    lo_52 = close.rolling(252, min_periods=60).min()

    lens_cols = ("Close", "RSI", "SMA_20", "SMA_50", "SMA_200",
                 "ADX", "ADX_PDI", "ADX_NDI", "BB_PctB")
    rows: list[dict] = []
    last_scoreable = len(ind) - min(HORIZONS) - 1

    for i in range(WARMUP_BARS, last_scoreable + 1, SAMPLE_EVERY):
        latest = {}
        for col in lens_cols:
            v = ind[col].iloc[i] if col in ind.columns else None
            latest[col] = float(v) if v is not None and pd.notna(v) else None
        if latest["Close"] is None:
            continue

        info = {}
        if pd.notna(hi_52.iloc[i]) and pd.notna(lo_52.iloc[i]):
            info = {"fiftyTwoWeekHigh": float(hi_52.iloc[i]),
                    "fiftyTwoWeekLow":  float(lo_52.iloc[i])}

        technical = {"latest": latest, "signals": _signals_asof(ind, i)}
        res = _score_technical(technical, {}, info)

        row = {
            "ticker":   symbol,
            "date":     ind.index[i].date().isoformat(),
            "score":    res["score"],
            "possible": res["possible"],
        }
        base = float(close.iloc[i])
        for h in HORIZONS:
            row[f"fwd_{h}d"] = (
                float(close.iloc[i + h]) / base - 1 if i + h < len(close) else np.nan
            )
        rows.append(row)
    return rows


# ------------------------------------------------------------------ #
# Universe assembly & benchmark
# ------------------------------------------------------------------ #

def broad_universe(per_sector: int = 10) -> list[str]:
    """Top-cap US names from each Yahoo sector (one screen per sector)."""
    import yfinance as yf
    from data.sector_data import _YAHOO_SECTORS
    symbols: list[str] = []
    for sector in sorted(_YAHOO_SECTORS):
        try:
            q = yf.EquityQuery("and", [
                yf.EquityQuery("eq", ["sector", sector]),
                yf.EquityQuery("eq", ["region", "us"]),
            ])
            resp = yf.screen(q, size=per_sector,
                             sortField="intradaymarketcap", sortAsc=False)
            symbols += [str(x.get("symbol", "")).upper()
                        for x in (resp or {}).get("quotes", [])]
        except Exception as exc:
            logger.warning("backtest: broad-universe screen failed for %s: %s", sector, exc)
    return [s for s in dict.fromkeys(symbols) if s]


def _spy_forward_returns() -> "pd.DataFrame | None":
    """SPY forward returns indexed by date, for excess-return columns."""
    df = _prepare_price_df("SPY")
    if df is None:
        return None
    out = pd.DataFrame(index=pd.to_datetime(df.index).normalize())
    for h in HORIZONS:
        out[f"spy_fwd_{h}d"] = df["Close"].shift(-h).values / df["Close"].values - 1
    return out


# ------------------------------------------------------------------ #
# Entry point (called from main.py --backtest)
# ------------------------------------------------------------------ #

def run_backtest(console, tickers: list[str], broad: bool = False) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from rich.table import Table
    from scipy.stats import spearmanr

    universe = list(dict.fromkeys(t.upper() for t in tickers))
    if broad:
        with console.status("[dim]Building broad sector universe…[/dim]"):
            extra = broad_universe()
        universe = list(dict.fromkeys(universe + extra))
    console.print(f"\n[bold cyan]FinBot — Technical-Lens Backtest[/bold cyan] "
                  f"({len(universe)} tickers, weekly sampling)\n")

    # Prewarm histories through the shared cache (3 workers, polite).
    with console.status("[dim]Prewarming price histories…[/dim]"):
        with ThreadPoolExecutor(max_workers=3) as pool:
            for t in universe + ["SPY"]:
                pool.submit(market_data.get_history, t, HISTORY_PERIOD)

    all_rows: list[dict] = []
    skipped: list[str] = []
    with console.status("[dim]Replaying…[/dim]") as status:
        for k, sym in enumerate(universe):
            status.update(f"[dim]Replaying {sym} ({k + 1}/{len(universe)})…[/dim]")
            rows = replay_ticker(sym)
            if rows:
                all_rows += rows
            else:
                skipped.append(sym)

    if not all_rows:
        console.print("[yellow]No ticker had enough history to replay (≥6 months needed).[/yellow]")
        return
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])

    # Excess returns vs SPY over the identical windows.
    spy = _spy_forward_returns()
    if spy is not None:
        df = df.merge(spy, left_on="date", right_index=True, how="left")
        for h in HORIZONS:
            df[f"excess_{h}d"] = df[f"fwd_{h}d"] - df[f"spy_fwd_{h}d"]

    df["thin"] = df["possible"] < FULL_POSSIBLE

    # ── Persist the full observation set ────────────────────────── #
    out_dir = os.path.join(REPORTS_DIR, "backtest")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(
        out_dir, f"backtest_technical_{datetime.date.today().isoformat()}.csv")
    df.to_csv(csv_path, index=False)

    # ── Bucket table per horizon ────────────────────────────────── #
    for h in HORIZONS:
        fwd, exc = f"fwd_{h}d", f"excess_{h}d"
        sub = df.dropna(subset=[fwd])
        if sub.empty:
            continue
        table = Table(title=f"{h}-day forward returns by technical score "
                            f"({len(sub)} observations)")
        table.add_column("Score bucket")
        table.add_column("N", justify="right")
        table.add_column("Mean fwd", justify="right")
        table.add_column("Median fwd", justify="right")
        table.add_column("Hit rate", justify="right")
        table.add_column("Mean excess vs SPY", justify="right")
        for lo, hi, name in _BUCKETS:
            chunk = sub[(sub["score"] >= lo) & (sub["score"] < hi)]
            if chunk.empty:
                continue
            mean_exc = (f"{chunk[exc].mean() * 100:+.2f}%"
                        if exc in chunk.columns and chunk[exc].notna().any() else "n/a")
            table.add_row(
                name, str(len(chunk)),
                f"{chunk[fwd].mean() * 100:+.2f}%",
                f"{chunk[fwd].median() * 100:+.2f}%",
                f"{(chunk[fwd] > 0).mean() * 100:.0f}%",
                mean_exc,
            )
        console.print(table)

        ic, ic_p = spearmanr(sub["score"], sub[fwd])
        console.print(f"  Spearman IC (score → {h}d return): "
                      f"[bold]{ic:+.3f}[/bold] (p={ic_p:.1e})")
        for thin, tag in ((False, "full-signal scores"), (True, "thin-signal scores")):
            part = sub[sub["thin"] == thin]
            if len(part) >= 30:
                ic_t, _ = spearmanr(part["score"], part[fwd])
                console.print(f"    · {tag} (n={len(part)}): IC {ic_t:+.3f}")
        console.print()

    if skipped:
        console.print(f"[dim]Skipped (insufficient history): {', '.join(skipped)}[/dim]")
    console.print(f"Full observation CSV: [green]{csv_path}[/green]")
    console.print(
        "\n[dim]Read with care: (1) technical lens only — fundamental/valuation lenses "
        "cannot be replayed without point-in-time data; (2) today's universe = "
        "survivorship bias (delisted losers are invisible); (3) weekly samples with "
        f"{HORIZONS[-1]}-day windows overlap, so effective sample size is smaller "
        "than N suggests.[/dim]\n"
    )
