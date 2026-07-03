"""
feedback/backfiller.py — Historical backfill for the feedback loop.

Solves the cold-start problem: the feedback loop needs N resolved
predictions before bias correction and ML calibration activate, but
you would normally have to wait N real runs to accumulate that data.

The backfill replays predictions over the last 20% of features.csv.
Because the PERSISTED models are fitted on ALL labelled rows (see
ml/trainer.py — evaluation is walk-forward CV, deployment uses full
history), replaying them here would be in-sample and inflate the
bootstrap accuracy stats. So the backfiller fits its own temporary
replay models on the FIRST 80% only, keeping the replayed 20%
genuinely out-of-sample. Actual prices at all 8 horizons are derived
by looking ahead in the price DataFrame.

Each synthetic row is tagged  mode="backfill" / provider="backfill"
so it can be filtered separately in accuracy.py if desired, but by
default it is included alongside real run rows when computing stats.

Usage
-----
    from feedback.backfiller import backfill_historical_predictions, needs_backfill

    if needs_backfill(ticker):
        n = backfill_historical_predictions(ticker, trainer_result, csv_path, price_df)
        print(f"Backfilled {n} historical rows for {ticker}")

CLI:
    python main.py --backfill --ticker SPY
    python main.py --backfill --file tickers.txt
"""

import csv
import datetime
import logging
import os
import uuid

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)

from config import FEEDBACK_DIR, FEEDBACK_MIN_SAMPLES, BACKFILL_MIN_TEST_ROWS, ML_MODELS_DIR
from feedback.tracker import COLUMNS
from ml.trainer import _GB_PARAMS, _RF_PARAMS

logger = logging.getLogger(__name__)

# Mapping: feedback horizon label → business days to look ahead
_HORIZON_BIZ_DAYS: dict[str, int] = {
    "1w":  5,
    "2w":  10,
    "3w":  15,
    "1m":  21,
    "3m":  63,
    "6m":  126,
    "9m":  189,
    "12m": 252,
}


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def needs_backfill(ticker: str) -> bool:
    """
    Return True if the feedback CSV for *ticker* has fewer resolved
    rows than FEEDBACK_MIN_SAMPLES, indicating that bias correction
    and calibration cannot yet activate.
    """
    path = _csv_path(ticker)
    if not os.path.isfile(path):
        return True
    try:
        df = pd.read_csv(path, dtype=str)
        if df.empty:
            return True
        # Count rows that have at least the 1w actual resolved
        resolved = df["actual_1w"].replace("", pd.NA).dropna() if "actual_1w" in df.columns else pd.Series(dtype=str)
        return len(resolved) < FEEDBACK_MIN_SAMPLES
    except Exception:
        return True


def backfill_historical_predictions(
    ticker:         str,
    trainer_result: dict,
    features_csv:   str,
    price_df:       "pd.DataFrame",
) -> int:
    """
    Replay trained models across the test split of ``features_csv``
    and write resolved feedback rows to the feedback CSV.

    Parameters
    ----------
    ticker         : Ticker symbol.
    trainer_result : Dict returned by ml.trainer.load_or_train().
    features_csv   : Path to the features.csv produced by csv_exporter.
    price_df       : OHLCV DataFrame (DatetimeIndex) from yfinance.
                     Must cover at least the same period as features_csv.

    Returns
    -------
    int : Number of new rows written (0 if nothing new was added).
    """
    models  = trainer_result.get("models", {})
    feature_cols = trainer_result.get("features", [])

    if not models or not feature_cols:
        logger.warning("Backfill skipped for %s: no trained models available.", ticker)
        return 0

    # ── Load features CSV ──────────────────────────────────────── #
    try:
        feat_df = pd.read_csv(features_csv, index_col=0, parse_dates=True)
    except Exception as exc:
        logger.error("Backfill: cannot read features CSV %s: %s", features_csv, exc)
        return 0

    # Require at least both 5d and 21d labels
    label_cols = ["direction_5d", "return_5d", "direction_21d", "return_21d"]
    existing_labels = [c for c in label_cols if c in feat_df.columns]
    if len(existing_labels) < 4:
        logger.warning("Backfill: features CSV missing label columns for %s.", ticker)
        return 0

    feat_df = feat_df.dropna(subset=label_cols)
    if len(feat_df) == 0:
        return 0

    # ── Test split: last 20% of labelled rows ──────────────────── #
    split_idx = int(len(feat_df) * 0.80)
    test_df   = feat_df.iloc[split_idx:].copy()

    if len(test_df) < BACKFILL_MIN_TEST_ROWS:
        logger.info(
            "Backfill: only %d test-split rows for %s (need %d) — skipping.",
            len(test_df), ticker, BACKFILL_MIN_TEST_ROWS,
        )
        return 0

    # ── Load existing feedback to avoid duplicate run_dates ───── #
    existing_dates: set[str] = set()
    csv_path = _csv_path(ticker)
    if os.path.isfile(csv_path):
        try:
            ex = pd.read_csv(csv_path, dtype=str, on_bad_lines="skip")
            if "run_date" in ex.columns:
                # Only consider backfill rows to avoid blocking real runs
                if "provider" in ex.columns:
                    bf = ex[ex["provider"] == "backfill"]
                    existing_dates = set(bf["run_date"].dropna().tolist())
                else:
                    existing_dates = set(ex["run_date"].dropna().tolist())
        except Exception:
            pass

    # ── Normalise price_df index for forward-price lookup ──────── #
    price_idx = pd.to_datetime(price_df.index).normalize()
    if price_idx.tz is not None:
        price_idx = price_idx.tz_localize(None)
    price_df = price_df.copy()
    price_df.index = price_idx
    price_dates = sorted(price_df.index.tolist())

    # ── Prepare feature columns available in test_df ───────────── #
    avail_feat_cols = [c for c in feature_cols if c in test_df.columns]

    # ── Fit temporary replay models on the FIRST 80% only ──────── #
    # The persisted models in trainer_result are fitted on all rows,
    # so replaying them over the last 20% would be in-sample and
    # inflate the bootstrap stats. Refit here (same hyper-parameters)
    # so the replayed window stays genuinely out-of-sample. Backfill
    # only runs on cold-start, so the extra fits are a one-off cost.
    train_df = feat_df.iloc[:split_idx]
    X_tr = train_df[avail_feat_cols].ffill().bfill().fillna(0.0)
    replay_models: dict = {}
    try:
        for h in (5, 21):
            y_clf = train_df[f"direction_{h}d"].values
            y_reg = train_df[f"return_{h}d"].values
            replay_models[f"clf_{h}d"]    = GradientBoostingClassifier(**_GB_PARAMS).fit(X_tr, y_clf)
            replay_models[f"clf_{h}d_rf"] = RandomForestClassifier(**_RF_PARAMS).fit(X_tr, y_clf)
            replay_models[f"reg_{h}d"]    = GradientBoostingRegressor(**_GB_PARAMS).fit(X_tr, y_reg)
            replay_models[f"reg_{h}d_rf"] = RandomForestRegressor(**_RF_PARAMS).fit(X_tr, y_reg)
    except Exception as exc:
        logger.warning("Backfill: could not fit replay models for %s: %s", ticker, exc)
        return 0
    models = replay_models

    # ── Replay model on each test-split row ────────────────────── #
    new_rows: list[dict] = []

    for date, row in test_df.iterrows():
        date_norm = pd.Timestamp(date).normalize()
        if date_norm.tz is not None:
            date_norm = date_norm.tz_localize(None)

        run_date_str = date_norm.strftime("%Y-%m-%d")
        if run_date_str in existing_dates:
            continue

        current_price = float(row.get("Close", np.nan))
        if np.isnan(current_price) or current_price <= 0:
            continue

        # ── Build feature vector for this row ──────────────────── #
        x_vals = row[avail_feat_cols].fillna(0.0).values.reshape(1, -1)
        x_df   = pd.DataFrame(x_vals, columns=avail_feat_cols)

        # ── Run inference ──────────────────────────────────────── #
        preds: dict = {}
        for h in (5, 21):
            clf_gb = models.get(f"clf_{h}d")
            clf_rf = models.get(f"clf_{h}d_rf")
            reg_gb = models.get(f"reg_{h}d")
            reg_rf = models.get(f"reg_{h}d_rf")

            if clf_gb is not None:
                prob_gb = float(clf_gb.predict_proba(x_df)[0][1])
                if clf_rf is not None:
                    prob_rf = float(clf_rf.predict_proba(x_df)[0][1])
                    prob_up = (prob_gb + prob_rf) / 2.0
                else:
                    prob_up = prob_gb
                preds[f"clf_{h}d_prob_up"]  = round(prob_up, 4)
                preds[f"clf_{h}d_direction"] = "UP" if prob_up >= 0.5 else "DOWN"
            else:
                preds[f"clf_{h}d_prob_up"]  = None
                preds[f"clf_{h}d_direction"] = "N/A"

            if reg_gb is not None:
                ret_gb = float(reg_gb.predict(x_df)[0]) * 100
                if reg_rf is not None:
                    ret_rf = float(reg_rf.predict(x_df)[0]) * 100
                    preds[f"reg_{h}d_return_pct"] = round((ret_gb + ret_rf) / 2.0, 2)
                else:
                    preds[f"reg_{h}d_return_pct"] = round(ret_gb, 2)

        # ── Resolve actual prices at all 8 horizons ────────────── #
        actuals: dict[str, float | None] = {}
        for label, biz_days in _HORIZON_BIZ_DAYS.items():
            idx = _find_future_price_idx(date_norm, biz_days, price_dates)
            if idx is not None:
                actuals[label] = float(price_df["Close"].iloc[idx])
            else:
                actuals[label] = None

        # ── ML correctness at 5d (≈1w) and 21d (≈1m) horizons ─── #
        actual_dir_5d  = int(float(row["direction_5d"]))
        actual_dir_21d = int(float(row["direction_21d"]))
        pred_dir_5d  = 1 if preds.get("clf_5d_direction")  == "UP" else 0
        pred_dir_21d = 1 if preds.get("clf_21d_direction") == "UP" else 0
        ml_5d_correct  = 1 if pred_dir_5d  == actual_dir_5d  else 0
        ml_21d_correct = 1 if pred_dir_21d == actual_dir_21d else 0

        # ── Build the feedback row ──────────────────────────────── #
        run_id = f"{ticker.upper()}_{date_norm.strftime('%Y%m%d')}_backfill_{uuid.uuid4().hex[:6]}"

        feedback_row: dict = {col: "" for col in COLUMNS}
        feedback_row.update({
            "run_id":         run_id,
            "ticker":         ticker.upper(),
            "run_date":       run_date_str,
            "run_timestamp":  f"{run_date_str}T00:00:00",
            "mode":           "backfill",
            "provider":       "backfill",
            "current_price":  round(current_price, 4),
            "recommendation": "N/A",
            "confidence":     "N/A",
            # ML signals
            "ml_5d_direction":       preds.get("clf_5d_direction",  "N/A"),
            "ml_5d_prob_up":         preds.get("clf_5d_prob_up",    ""),
            "ml_21d_direction":      preds.get("clf_21d_direction", "N/A"),
            "ml_21d_prob_up":        preds.get("clf_21d_prob_up",   ""),
            "ml_5d_expected_return": preds.get("reg_5d_return_pct", ""),
            "ml_21d_expected_return":preds.get("reg_21d_return_pct",""),
            # ML correctness
            "ml_5d_correct":  ml_5d_correct,
            "ml_21d_correct": ml_21d_correct,
        })

        # Actual prices and direction correctness per horizon
        for label, actual_price in actuals.items():
            if actual_price is None:
                continue
            feedback_row[f"actual_{label}"] = round(actual_price, 4)
            feedback_row[f"dir_correct_{label}"] = (
                1 if actual_price > current_price else 0
            )
            # pct_error is target-based; backfill has no LLM target so leave blank

        new_rows.append(feedback_row)

    if not new_rows:
        return 0

    # ── Append to feedback CSV ──────────────────────────────────── #
    _append_rows(csv_path, new_rows)
    logger.info("Backfilled %d historical rows for %s.", len(new_rows), ticker)
    return len(new_rows)


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _csv_path(ticker: str) -> str:
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    return os.path.join(FEEDBACK_DIR, f"{ticker.upper()}_feedback.csv")


def _find_future_price_idx(
    base_date: "pd.Timestamp",
    biz_days_ahead: int,
    sorted_dates: list,
) -> "int | None":
    """
    Find the index in sorted_dates that is approximately biz_days_ahead
    trading days after base_date.  Returns None if not enough future
    data exists (e.g., recent rows where 9m/12m actuals are unavailable).
    """
    try:
        start_pos = None
        for i, d in enumerate(sorted_dates):
            if pd.Timestamp(d).normalize() >= base_date:
                start_pos = i
                break
        if start_pos is None:
            return None
        target_pos = start_pos + biz_days_ahead
        if target_pos < len(sorted_dates):
            return target_pos
        return None
    except Exception:
        return None


def _append_rows(csv_path: str, rows: list[dict]) -> None:
    """Write rows to the feedback CSV, creating header if needed."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
