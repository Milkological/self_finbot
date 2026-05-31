"""
feedback/resolver.py — Resolve pending predictions by fetching actual prices.

For each unresolved row in feedback/{TICKER}_feedback.csv, checks whether
enough calendar days have passed for each horizon (1W=7d, 2W=14d, etc.).
When a horizon is resolvable, fetches the actual close price via yfinance
and fills in the outcome columns (actual_Xh, dir_correct_Xh, pct_error_Xh,
ml_Xd_correct, rec_correct).

Called automatically at the start of run_pipeline() and also via --resolve flag.
"""

import os
import datetime
import pandas as pd
import yfinance as yf

from config import FEEDBACK_DIR


# Calendar days after run_date when each horizon is considered resolvable.
# We use calendar days (not trading days) so the resolver works on weekends too.
# A 1-week horizon becomes resolvable 7 calendar days after the run date, etc.
HORIZON_CALENDAR_DAYS = {
    "1w":  7,
    "2w":  14,
    "3w":  21,
    "1m":  31,
    "3m":  93,
    "6m":  186,
    "9m":  279,
    "12m": 365,
}

# Suffixes for column access
HORIZONS = list(HORIZON_CALENDAR_DAYS.keys())


def _csv_path(ticker: str) -> str:
    return os.path.join(FEEDBACK_DIR, f"{ticker.upper()}_feedback.csv")


def _fetch_price_on_or_before(ticker: str, target_date: datetime.date) -> float | None:
    """
    Fetch the closing price on target_date or the most recent trading day before it.
    Uses a small window (target_date - 7d → target_date + 1d) to handle weekends/holidays.
    """
    start = (target_date - datetime.timedelta(days=7)).isoformat()
    end   = (target_date + datetime.timedelta(days=1)).isoformat()
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                         progress=False, show_errors=False)
        if df.empty:
            return None
        # Normalise column names if yfinance returns MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def resolve_pending(ticker: str) -> int:
    """
    Resolve any unresolved horizon columns for *ticker*.

    Returns the number of rows that had at least one new outcome filled in.
    Silently skips if the CSV doesn't exist yet.
    """
    path = _csv_path(ticker)
    if not os.path.isfile(path):
        return 0

    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return 0

    if df.empty:
        return 0

    today       = datetime.date.today()
    rows_updated = 0

    for idx, row in df.iterrows():
        try:
            run_date = datetime.date.fromisoformat(str(row["run_date"]))
        except (ValueError, TypeError):
            continue

        current_price_str = row.get("current_price", "")
        try:
            current_price = float(current_price_str)
        except (ValueError, TypeError):
            continue

        if current_price <= 0:
            continue

        row_changed = False

        for h in HORIZONS:
            # Skip if already resolved
            actual_col = f"actual_{h}"
            if str(row.get(actual_col, "")).strip() not in ("", "nan"):
                continue

            # Check if enough calendar days have passed
            days_needed = HORIZON_CALENDAR_DAYS[h]
            resolve_after = run_date + datetime.timedelta(days=days_needed)
            if today < resolve_after:
                continue

            # Fetch actual price
            actual_price = _fetch_price_on_or_before(ticker, resolve_after)
            if actual_price is None:
                continue

            # pct_error = (actual - predicted) / predicted * 100
            target_col = f"target_{h}"
            target_str = str(row.get(target_col, "")).strip()
            try:
                target_price = float(target_str)
                if target_price > 0:
                    pct_error = round((actual_price - target_price) / target_price * 100, 2)
                    df.at[idx, f"pct_error_{h}"] = pct_error

                    # Directional accuracy: did price move in the right direction from current?
                    predicted_up = target_price > current_price
                    actual_up    = actual_price > current_price
                    df.at[idx, f"dir_correct_{h}"] = int(predicted_up == actual_up)
                else:
                    df.at[idx, f"pct_error_{h}"]   = ""
                    df.at[idx, f"dir_correct_{h}"]  = ""
            except (ValueError, TypeError):
                df.at[idx, f"pct_error_{h}"]   = ""
                df.at[idx, f"dir_correct_{h}"]  = ""

            df.at[idx, actual_col] = round(actual_price, 4)
            row_changed = True

        # ── ML correctness (resolve at 1W / 1M horizons) ──────────── #
        # 5-day ML: resolve when 1W horizon resolves (closest proxy)
        if str(row.get("ml_5d_correct", "")).strip() in ("", "nan"):
            actual_1w_str = str(df.at[idx, "actual_1w"]).strip()
            if actual_1w_str not in ("", "nan"):
                try:
                    actual_1w   = float(actual_1w_str)
                    ml_5d_dir   = str(row.get("ml_5d_direction", "")).strip().upper()
                    actual_up   = actual_1w > current_price
                    predicted_up = (ml_5d_dir == "UP")
                    df.at[idx, "ml_5d_correct"] = int(predicted_up == actual_up)
                    row_changed = True
                except (ValueError, TypeError):
                    pass

        # 21-day ML: resolve when 1M horizon resolves
        if str(row.get("ml_21d_correct", "")).strip() in ("", "nan"):
            actual_1m_str = str(df.at[idx, "actual_1m"]).strip()
            if actual_1m_str not in ("", "nan"):
                try:
                    actual_1m   = float(actual_1m_str)
                    ml_21d_dir  = str(row.get("ml_21d_direction", "")).strip().upper()
                    actual_up   = actual_1m > current_price
                    predicted_up = (ml_21d_dir == "UP")
                    df.at[idx, "ml_21d_correct"] = int(predicted_up == actual_up)
                    row_changed = True
                except (ValueError, TypeError):
                    pass

        # ── Overall recommendation correctness (resolve at 1M) ─────── #
        if str(row.get("rec_correct", "")).strip() in ("", "nan"):
            actual_1m_str = str(df.at[idx, "actual_1m"]).strip()
            if actual_1m_str not in ("", "nan"):
                try:
                    actual_1m  = float(actual_1m_str)
                    rec        = str(row.get("recommendation", "")).strip().upper()
                    price_rose = actual_1m > current_price
                    if rec in ("BUY", "STRONG BUY"):
                        df.at[idx, "rec_correct"] = int(price_rose)
                    elif rec in ("SELL", "STRONG SELL"):
                        df.at[idx, "rec_correct"] = int(not price_rose)
                    else:
                        df.at[idx, "rec_correct"] = ""   # HOLD — ambiguous
                    row_changed = True
                except (ValueError, TypeError):
                    pass

        if row_changed:
            rows_updated += 1

    if rows_updated > 0:
        df.to_csv(path, index=False)

    return rows_updated


def resolve_all(tickers: list[str]) -> dict[str, int]:
    """Resolve pending outcomes for every ticker in the list."""
    results = {}
    for t in tickers:
        results[t] = resolve_pending(t)
    return results
