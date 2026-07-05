"""
portfolio/simulator.py — Paper-trading scoreboard over resolved predictions.

The single number that matters: *would following FinBot's BUY calls have made
money?* This replays every resolved BUY / STRONG BUY prediction as an equal-risk
paper trade — entry at the price when the call was made, exit at the resolved
actual price at a chosen horizon — and reports win rate, average return, and
excess over simply holding SPY across the same windows.

It reads ONLY data the feedback loop already produces (feedback/*_feedback.csv:
recommendation, current_price, and the resolved actual_* columns), so there's no
re-fetching beyond one SPY history pull for the benchmark. Because it needs
resolved outcomes, run `--resolve` first to settle matured predictions.

Positions are treated as independent equal-weight slices held to the horizon —
not a single compounding account — because real predictions overlap in time and
the schema stores no per-trade stop/size. That keeps the P&L honest rather than
implying a precision the data doesn't support.

    python main.py --portfolio                 # 21-day (1-month) horizon, all providers
    python main.py --portfolio --horizon 3m    # 63-day horizon
    python main.py --portfolio --provider rule_based   # LLM-free scoreboard only
"""

import glob
import logging
import os

import numpy as np
import pandas as pd

from config import FEEDBACK_DIR, HISTORY_PERIOD
from data import market_data

logger = logging.getLogger(__name__)

# Horizon label → (feedback actual column, approx calendar days for SPY match)
_HORIZONS = {
    "1w":  ("actual_1w",  7),   "2w": ("actual_2w", 14),  "3w": ("actual_3w", 21),
    "1m":  ("actual_1m",  30),  "3m": ("actual_3m", 91),  "6m": ("actual_6m", 182),
    "9m":  ("actual_9m", 273),  "12m": ("actual_12m", 365),
}

# BUY-side signals become long paper trades, weighted by conviction.
_WEIGHTS = {"STRONG BUY": 1.0, "BUY": 0.6}


def _load_feedback(provider: str | None) -> pd.DataFrame:
    """Concatenate all feedback CSVs into one frame (object dtype, NA-safe)."""
    frames = []
    for path in glob.glob(os.path.join(FEEDBACK_DIR, "*_feedback.csv")):
        try:
            df = pd.read_csv(path, dtype=str, on_bad_lines="skip", keep_default_na=False)
            frames.append(df)
        except Exception as exc:
            logger.debug("portfolio: cannot read %s: %s", path, exc)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if provider and "provider" in df.columns:
        df = df[df["provider"] == provider]
    return df


def _spy_returns(dates, cal_days: int) -> dict:
    """Map each entry date → SPY forward return over ~cal_days, for benchmarking."""
    try:
        spy = market_data.get_history("SPY", HISTORY_PERIOD)
        close = spy["Close"].copy()
        close.index = pd.to_datetime(close.index)
        if close.index.tz is not None:
            close.index = close.index.tz_convert(None)
        close = close.sort_index()
    except Exception:
        return {}
    out = {}
    for d in dates:
        try:
            d0 = pd.Timestamp(d)
            entry_idx = close.index.searchsorted(d0)
            exit_idx = close.index.searchsorted(d0 + pd.Timedelta(days=cal_days))
            if entry_idx < len(close) and exit_idx < len(close) and exit_idx > entry_idx:
                out[d] = float(close.iloc[exit_idx] / close.iloc[entry_idx] - 1)
        except Exception:
            continue
    return out


def _num(v):
    try:
        f = float(v)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


def run_portfolio(console, horizon: str = "1m", provider: str | None = None) -> None:
    from rich.table import Table

    if horizon not in _HORIZONS:
        console.print(f"[red]Unknown horizon '{horizon}'. Choose from: {', '.join(_HORIZONS)}[/red]")
        return
    actual_col, cal_days = _HORIZONS[horizon]

    console.print(f"\n[bold cyan]FinBot — Paper-Trading Scoreboard[/bold cyan]  "
                  f"(BUY signals, {horizon} horizon"
                  f"{', provider=' + provider if provider else ''})\n")

    df = _load_feedback(provider)
    if df.empty or "recommendation" not in df.columns:
        console.print("[yellow]No feedback data. Run some analyses (and --resolve) first.[/yellow]")
        return

    # Build the trade log from resolved BUY / STRONG BUY rows.
    trades = []
    for _, r in df.iterrows():
        rec = str(r.get("recommendation", "")).strip().upper()
        if rec not in _WEIGHTS:
            continue
        entry = _num(r.get("current_price"))
        exit_ = _num(r.get(actual_col))
        if entry is None or exit_ is None or entry <= 0:
            continue
        trades.append({
            "ticker": r.get("ticker", ""),
            "date":   r.get("run_date", ""),
            "rec":    rec,
            "weight": _WEIGHTS[rec],
            "entry":  entry,
            "exit":   exit_,
            "ret":    exit_ / entry - 1,
        })

    if not trades:
        console.print(f"[yellow]No resolved BUY/STRONG BUY trades at the {horizon} horizon yet.[/yellow]\n"
                      "[dim]Run `python main.py --resolve` to settle matured predictions, "
                      "then try again.[/dim]")
        return

    tdf = pd.DataFrame(trades)
    spy_map = _spy_returns(sorted(tdf["date"].unique()), cal_days)
    tdf["spy_ret"] = tdf["date"].map(spy_map)
    tdf["excess"]  = tdf["ret"] - tdf["spy_ret"]

    # ── Headline stats ──────────────────────────────────────────── #
    n = len(tdf)
    win = (tdf["ret"] > 0).mean()
    mean_ret = tdf["ret"].mean()
    med_ret = tdf["ret"].median()
    # Weighted average (conviction-weighted) return.
    wavg = float((tdf["ret"] * tdf["weight"]).sum() / tdf["weight"].sum())
    matched = tdf.dropna(subset=["spy_ret"])
    mean_excess = matched["excess"].mean() if not matched.empty else None
    spy_beat = (matched["excess"] > 0).mean() if not matched.empty else None

    summary = Table(title=f"Summary — {n} paper trades ({horizon} hold)")
    summary.add_column("Metric"); summary.add_column("Value", justify="right")
    summary.add_row("Trades", str(n))
    summary.add_row("Win rate", f"{win*100:.0f}%")
    summary.add_row("Mean return / trade", f"{mean_ret*100:+.2f}%")
    summary.add_row("Median return / trade", f"{med_ret*100:+.2f}%")
    summary.add_row("Conviction-weighted return", f"{wavg*100:+.2f}%")
    if mean_excess is not None:
        summary.add_row("Mean excess vs SPY", f"{mean_excess*100:+.2f}%")
        summary.add_row("Beat SPY (share of trades)", f"{spy_beat*100:.0f}%")
    summary.add_row("Best / worst trade",
                    f"{tdf['ret'].max()*100:+.1f}% / {tdf['ret'].min()*100:+.1f}%")
    console.print(summary)

    # ── By conviction ───────────────────────────────────────────── #
    by = Table(title="By signal strength")
    by.add_column("Signal"); by.add_column("N", justify="right")
    by.add_column("Win rate", justify="right"); by.add_column("Mean return", justify="right")
    for rec in ("STRONG BUY", "BUY"):
        sub = tdf[tdf["rec"] == rec]
        if not sub.empty:
            by.add_row(rec, str(len(sub)), f"{(sub['ret']>0).mean()*100:.0f}%",
                       f"{sub['ret'].mean()*100:+.2f}%")
    console.print(by)

    # ── Recent closed trades ────────────────────────────────────── #
    recent = Table(title="Recent closed trades")
    for c in ("Date", "Ticker", "Signal", "Entry", "Exit", "Return", "vs SPY"):
        recent.add_column(c, justify="right" if c in ("Entry","Exit","Return","vs SPY") else "left")
    for _, t in tdf.sort_values("date").tail(12).iterrows():
        ret_style = "green" if t["ret"] > 0 else "red"
        exc = f"{t['excess']*100:+.1f}%" if pd.notna(t["excess"]) else "n/a"
        recent.add_row(str(t["date"]), str(t["ticker"]), t["rec"],
                       f"${t['entry']:.2f}", f"${t['exit']:.2f}",
                       f"[{ret_style}]{t['ret']*100:+.1f}%[/{ret_style}]", exc)
    console.print(recent)

    verdict = ("[green]following BUY calls beat holding SPY[/green]"
               if (mean_excess or 0) > 0 else
               "[yellow]BUY calls did not beat SPY over this sample[/yellow]")
    console.print(f"\n[bold]Bottom line:[/bold] over {n} resolved {horizon} trades, {verdict}.")
    console.print("[dim]Equal-weight, independent holds to horizon; no stops (schema stores none). "
                  "Survivorship-free (these are your actual past calls), but a small/biased sample "
                  "until many predictions resolve.[/dim]\n")
