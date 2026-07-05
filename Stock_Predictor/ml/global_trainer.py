"""
ml/global_trainer.py — Pooled cross-ticker ("global") ML models.

Per-ticker models train on ~1,000 rows and overfit easily; tickers with
short history can't train at all. The global model stacks the labelled
rows of EVERY ticker's most recent features.csv into one pooled dataset
and trains a single GBM+RF pair per horizon. All FEATURE_COLS are
scale-independent (ratios, oscillators, z-scores), so rows from
different tickers are directly comparable.

Deliberately excluded: current market-cap / sector snapshots — stamping
today's values onto historical rows would be look-ahead (the company was
smaller in the past), the same flaw the removed sentiment feature had.

Artifacts live under models/_GLOBAL/ (same layout as per-ticker dirs).
Training is invoked via `python main.py --retrain-global`; the predictor
blends global with per-ticker probabilities automatically whenever the
artifacts exist, and serves global-only predictions for tickers whose
own history is too short to train on.

Validation matches ml/trainer.py: walk-forward TimeSeriesSplit over the
pooled rows sorted by date, with a gap ≥ horizon to prevent the
forward-looking labels of late train rows overlapping the test window.
"""

import glob
import json
import logging
import os
import pickle
import re

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from config import ML_MODELS_DIR, REPORTS_DIR
from ml.trainer import FEATURE_COLS, _GB_PARAMS, _RF_PARAMS

logger = logging.getLogger(__name__)

GLOBAL_DIR = os.path.join(ML_MODELS_DIR, "_GLOBAL")

# Lazily-loaded process-wide cache of the global model set.
_loaded: "dict | None" = None
_load_attempted = False


# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #

def _latest_features_csvs(reports_dir: str = REPORTS_DIR) -> dict[str, str]:
    """
    Map each ticker to its most recent features.csv, pooling two sources:
      • per-run reports/{T}_{DATE}/features.csv (your analyzed tickers)
      • reports/_universe/{T}/features.csv (the broad training universe
        built by `python main.py --build-universe`)
    The universe copies only fill in tickers not already covered by a
    (fresher, sentiment-bearing) per-run report.
    """
    out: dict[str, str] = {}
    # Per-run reports first (dated dirs) — later dates win via sorted().
    for path in sorted(glob.glob(os.path.join(reports_dir, "*", "features.csv"))):
        dirname = os.path.basename(os.path.dirname(path))
        m = re.match(r"(.+)_(\d{4}-\d{2}-\d{2})$", dirname)
        if not m:
            continue
        out[m.group(1).upper()] = path
    # Universe copies — only add tickers not already present.
    for path in sorted(glob.glob(os.path.join(reports_dir, "_universe", "*", "features.csv"))):
        tkr = os.path.basename(os.path.dirname(path)).upper()
        out.setdefault(tkr, path)
    return out


def train_global_models(reports_dir: str = REPORTS_DIR, min_rows: int = 300) -> dict:
    """
    Stack every ticker's labelled rows and train pooled models.

    Returns {"status": "trained"|"insufficient_data", "tickers": [...],
             "rows": int, "metrics": {...}}.
    """
    csvs = _latest_features_csvs(reports_dir)
    frames = []
    for tkr, path in csvs.items():
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception as exc:
            logger.warning("global: cannot read %s: %s", path, exc)
            continue
        label_cols = ["direction_5d", "direction_21d", "return_5d", "return_21d"]
        existing = [c for c in label_cols if c in df.columns]
        if existing:
            df = df.dropna(subset=existing, how="all")
        if df.empty:
            continue
        df["_ticker"] = tkr
        frames.append(df)

    if not frames:
        return {"status": "insufficient_data", "tickers": [], "rows": 0, "metrics": {}}

    pooled = pd.concat(frames)
    pooled = pooled.sort_index(kind="stable")   # chronological across tickers

    # Features must exist in every source CSV (older CSVs may predate
    # newly added feature columns) so the pooled matrix has no phantom
    # all-NaN columns for some tickers.
    feat_cols = [
        c for c in FEATURE_COLS
        if all(c in f.columns for f in frames)
    ]
    X_check = pooled[feat_cols].ffill().bfill()
    canonical_cols = list(X_check.dropna(axis=1, how="all").columns)

    if len(pooled) < min_rows or not canonical_cols:
        return {"status": "insufficient_data",
                "tickers": sorted(csvs), "rows": len(pooled), "metrics": {}}

    os.makedirs(GLOBAL_DIR, exist_ok=True)
    models: dict = {}
    metrics: dict = {"tickers": sorted({f["_ticker"].iloc[0] for f in frames}),
                     "pooled_rows": len(pooled)}

    for h in (5, 21):
        target_clf = f"direction_{h}d"
        target_reg = f"return_{h}d"
        mask = pooled[target_clf].notna() & pooled[target_reg].notna()
        sub  = pooled.loc[mask]

        X = sub[canonical_cols].ffill().bfill().fillna(0.0)
        y_clf = sub[target_clf].values
        y_reg = sub[target_reg].values

        fold_accs, fold_maes = [], []
        n_splits = min(5, max(2, len(X) // 200))
        try:
            # gap scaled by the number of tickers: rows from N tickers
            # share each calendar date, so h days of label overlap spans
            # roughly h*N pooled rows.
            n_tickers = max(1, sub["_ticker"].nunique())
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=h * n_tickers)
            for tr_idx, te_idx in tscv.split(X):
                if len(tr_idx) < 100 or len(te_idx) == 0:
                    continue
                cv_gb = GradientBoostingClassifier(**_GB_PARAMS).fit(X.iloc[tr_idx], y_clf[tr_idx])
                cv_rf = RandomForestClassifier(**_RF_PARAMS).fit(X.iloc[tr_idx], y_clf[tr_idx])
                probs = (cv_gb.predict_proba(X.iloc[te_idx])[:, 1] +
                         cv_rf.predict_proba(X.iloc[te_idx])[:, 1]) / 2.0
                fold_accs.append(accuracy_score(y_clf[te_idx], (probs >= 0.5).astype(int)))

                cv_reg = GradientBoostingRegressor(**_GB_PARAMS).fit(X.iloc[tr_idx], y_reg[tr_idx])
                fold_maes.append(mean_absolute_error(y_reg[te_idx], cv_reg.predict(X.iloc[te_idx])))
        except Exception as cv_err:
            logger.warning("global: walk-forward CV failed for %dd: %s", h, cv_err)

        models[f"clf_{h}d"]    = GradientBoostingClassifier(**_GB_PARAMS).fit(X, y_clf)
        models[f"clf_{h}d_rf"] = RandomForestClassifier(**_RF_PARAMS).fit(X, y_clf)
        models[f"reg_{h}d"]    = GradientBoostingRegressor(**_GB_PARAMS).fit(X, y_reg)
        models[f"reg_{h}d_rf"] = RandomForestRegressor(**_RF_PARAMS).fit(X, y_reg)

        metrics[f"clf_{h}d_accuracy"]        = round(float(np.mean(fold_accs)), 4) if fold_accs else float("nan")
        metrics[f"clf_{h}d_accuracy_std"]    = round(float(np.std(fold_accs)), 4)  if fold_accs else float("nan")
        metrics[f"clf_{h}d_fold_accuracies"] = [round(a, 4) for a in fold_accs]
        metrics[f"reg_{h}d_mae"]             = round(float(np.mean(fold_maes)), 6) if fold_maes else float("nan")
    metrics["validation"] = "walk_forward_cv_pooled"

    for name, model in models.items():
        with open(os.path.join(GLOBAL_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)
    with open(os.path.join(GLOBAL_DIR, "features.json"), "w") as f:
        json.dump(canonical_cols, f)
    with open(os.path.join(GLOBAL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    global _loaded, _load_attempted
    _loaded, _load_attempted = None, False   # force reload on next predict

    return {"status": "trained", "tickers": metrics["tickers"],
            "rows": len(pooled), "metrics": metrics}


# ------------------------------------------------------------------ #
# Loading (used by ml/predictor.py)
# ------------------------------------------------------------------ #

def load_global_models() -> "dict | None":
    """
    Return {"models": {...}, "features": [...], "metrics": {...}} for the
    pooled model set, or None when it hasn't been trained. Cached for the
    life of the process.
    """
    global _loaded, _load_attempted
    if _load_attempted:
        return _loaded
    _load_attempted = True

    try:
        models = {}
        for name in ("clf_5d", "clf_21d", "reg_5d", "reg_21d"):
            path = os.path.join(GLOBAL_DIR, f"{name}.pkl")
            if not os.path.exists(path):
                return None
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        for name in ("clf_5d_rf", "clf_21d_rf", "reg_5d_rf", "reg_21d_rf"):
            path = os.path.join(GLOBAL_DIR, f"{name}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    models[name] = pickle.load(f)

        with open(os.path.join(GLOBAL_DIR, "features.json")) as f:
            features = json.load(f)
        metrics = {}
        metrics_path = os.path.join(GLOBAL_DIR, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)

        _loaded = {"models": models, "features": features, "metrics": metrics}
    except Exception as exc:
        logger.warning("global: could not load pooled models: %s", exc)
        _loaded = None
    return _loaded
