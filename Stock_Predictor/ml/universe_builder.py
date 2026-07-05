"""
ml/universe_builder.py — Build a broad training universe for the pooled model.

The pooled global model (ml/global_trainer.py) is only as good as the data it
pools. By default that's just the handful of tickers you've analyzed (~8, ~7k
rows), which is thin for the cross-ticker patterns the global model is meant to
learn. This module fetches a broad, liquid universe (~10 large caps per sector),
builds a features.csv for each under reports/_universe/{TICKER}/, and hands off
to the global trainer — taking the pool to ~100 tickers / ~120k rows.

Run once (it's a slow, network-heavy batch), then periodically to refresh:

    python main.py --build-universe                 # ~10/sector (~110 tickers)
    python main.py --build-universe --universe-per-sector 5   # smaller/faster

It is resumable: a ticker whose reports/_universe/{T}/features.csv is younger
than --universe-max-age-days (default 30) is skipped, so an interrupted run
picks up where it left off. Every ticker is isolated in try/except.

Reuses: backtester.broad_universe (sector screens), market_data cache,
compute_all_technicals, csv_exporter.export_features_csv (identical feature set,
incl. leak-free EDGAR fundamentals). Sentiment is neutral for universe rows —
they exist to train the price/fundamental features, not sentiment.
"""

import datetime
import logging
import os
import time

from config import REPORTS_DIR, HISTORY_PERIOD, ML_MIN_TRAIN_ROWS
from data import market_data
from data.stock_fetcher import fetch_stock_data
from analysis.technical import compute_all_technicals
from data.csv_exporter import export_features_csv

logger = logging.getLogger(__name__)

UNIVERSE_DIR = os.path.join(REPORTS_DIR, "_universe")
_NEUTRAL_SENTIMENT = {"overall_score": 0.0, "label": "NEUTRAL", "summary": ""}


def _is_fresh(path: str, max_age_days: int) -> bool:
    if not os.path.isfile(path):
        return False
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    return age_days < max_age_days


def _build_one(ticker: str) -> str:
    """Fetch + featurize one ticker into reports/_universe/{T}/features.csv."""
    price_df, info = fetch_stock_data(ticker)      # cached via market_data
    technical = compute_all_technicals(price_df)
    out_dir = os.path.join(UNIVERSE_DIR, ticker.upper())
    os.makedirs(out_dir, exist_ok=True)
    return export_features_csv(ticker, price_df, technical, info,
                               _NEUTRAL_SENTIMENT, out_dir)


def build_universe(console, per_sector: int = 10, max_age_days: int = 30,
                   force: bool = False, retrain: bool = True) -> dict:
    """
    Build/refresh the universe feature CSVs, then retrain the global model.
    Returns a summary dict.
    """
    from analysis.backtester import broad_universe
    from ml.global_trainer import train_global_models

    console.print("\n[bold cyan]FinBot — Building training universe[/bold cyan]\n")

    # Assemble target list: broad sector screen + whatever's in tickers.txt.
    with console.status("[dim]Fetching sector screens…[/dim]"):
        universe = broad_universe(per_sector=per_sector)
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tickers.txt")
    if os.path.isfile(base):
        with open(base, encoding="utf-8") as f:
            for line in f:
                tok = line.split("#", 1)[0].strip().upper()
                if tok:
                    universe.append(tok)
    universe = list(dict.fromkeys(universe))       # dedupe, keep order
    console.print(f"Universe: [bold]{len(universe)}[/bold] tickers "
                  f"(~{per_sector}/sector + tickers.txt)")

    built, skipped, failed = [], [], []
    os.makedirs(UNIVERSE_DIR, exist_ok=True)
    with console.status("[dim]Building feature files…[/dim]") as status:
        for i, tkr in enumerate(universe, 1):
            fpath = os.path.join(UNIVERSE_DIR, tkr, "features.csv")
            if not force and _is_fresh(fpath, max_age_days):
                skipped.append(tkr)
                continue
            status.update(f"[dim]{i}/{len(universe)}  building {tkr} "
                          f"(built {len(built)}, skipped {len(skipped)}, failed {len(failed)})…[/dim]")
            try:
                _build_one(tkr)
                built.append(tkr)
            except Exception as exc:
                logger.debug("universe build failed for %s: %s", tkr, exc)
                failed.append(tkr)
            time.sleep(0.2)        # polite pacing on top of the cache/throttles

    console.print(f"[green]Built {len(built)}[/green] · "
                  f"[dim]skipped {len(skipped)} (fresh)[/dim] · "
                  f"[yellow]failed {len(failed)}[/yellow]")
    if failed:
        console.print(f"[dim]Failed (delisted/no data): {', '.join(failed[:20])}"
                      f"{'…' if len(failed) > 20 else ''}[/dim]")

    result = {"built": built, "skipped": skipped, "failed": failed}
    if retrain:
        console.print("\n[dim]Retraining pooled global model on the expanded universe…[/dim]")
        gres = train_global_models()
        result["global"] = gres
        if gres.get("status") == "trained":
            m = gres["metrics"]
            console.print(
                f"[bold green]Global model retrained[/bold green] on "
                f"{gres['rows']} pooled rows from {len(gres['tickers'])} tickers."
            )
            for h in (5, 21):
                console.print(f"  {h:>2}d CV accuracy: {m.get(f'clf_{h}d_accuracy')} "
                              f"(±{m.get(f'clf_{h}d_accuracy_std')})")
        else:
            console.print("[yellow]Global retrain: insufficient pooled data.[/yellow]")
    return result
