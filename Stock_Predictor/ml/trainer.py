"""
ml/trainer.py — Train (or load) gradient-boosted ML models for FinBot.

For each ticker four models are persisted under  models/{TICKER}/ :

  classifier_5d.pkl   — GradientBoostingClassifier  (direction ≤ 5 days)
  classifier_21d.pkl  — GradientBoostingClassifier  (direction ≤ 21 days)
  regressor_5d.pkl    — GradientBoostingRegressor   (% return ≤ 5 days)
  regressor_21d.pkl   — GradientBoostingRegressor   (% return ≤ 21 days)

Alongside each model set we write:
  features.json  — ordered list of feature column names used at training time
  metrics.json   — accuracy / MAE scores from the hold-out test split

Time-series split:  first 80 % of rows → train,  last 20 % → test.
No shuffle is applied: leakage-free chronological validation.

Auto-retraining: if any model file is older than ML_RETRAIN_DAYS
or missing, all four models are retrained from scratch.
"""

import json
import os
import pickle
import datetime
import logging

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from config import ML_MODELS_DIR, ML_MIN_TRAIN_ROWS, ML_RETRAIN_DAYS

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Feature columns consumed by both trainer and predictor
# ------------------------------------------------------------------ #

FEATURE_COLS = [
    # Returns & intraday range
    "Log_Return", "Close_pct_change", "High_Low_Range",
    # 52-week position
    "dist_52w_high", "dist_52w_low",
    # Normalised MA ratios: (Close/MA) - 1  (price-scale independent)
    "SMA_20_ratio", "SMA_50_ratio", "SMA_200_ratio",
    "EMA_12_ratio", "EMA_26_ratio",
    # Oscillators (already bounded / normalised)
    "RSI", "BB_PctB", "BB_Width",
    # Normalised MACD
    "MACD_Line_norm", "MACD_Signal_norm", "MACD_Hist_norm",
    # Rolling aggregates
    "RSI_5d_mean", "RSI_21d_mean",
    "Vol_5d", "Vol_21d",
    "Momentum_5d", "Momentum_21d",
    # Lagged returns (temporal context)
    "Log_Return_lag1", "Log_Return_lag2", "Log_Return_lag3",
    # Volume context
    "Volume_ratio",
    # Sentiment
    "sentiment_score",
    # Regime features
    "ADX",
    "vol_regime_ratio",
    # Macro context (VIX)
    "vix_level",
    "vix_5d_change",
    # Earnings event risk
    "earnings_within_14d",
    # Mean-reversion signal
    "price_zscore_20d",
]

# GradientBoosting hyper-parameters (conservative — limited data)
_GB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42,
)

# RandomForest hyper-parameters (ensemble companion to GBM)
_RF_PARAMS = dict(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def load_or_train(ticker: str, csv_path: str, force_retrain: bool = False) -> dict:
    """
    Return a dict with 4 trained models and metadata.

    If persisted models exist and are fresh (< ML_RETRAIN_DAYS old)
    they are loaded from disk; otherwise the CSV is read and the
    models are (re)trained.

    Parameters
    ----------
    ticker        : Ticker symbol, used to locate the models/ sub-dir.
    csv_path      : Absolute path to the features.csv produced by
                    csv_exporter.export_features_csv().
    force_retrain : When True, retrain even if models are fresh.

    Returns
    -------
    {
        "models":   { "clf_5d", "clf_21d", "reg_5d", "reg_21d" }
        "features": list[str]
        "metrics":  { ... }
        "models_dir": str
        "trained_on_rows": int
        "status":   "loaded" | "trained" | "insufficient_data"
    }
    """
    models_dir = os.path.join(ML_MODELS_DIR, ticker.upper())
    os.makedirs(models_dir, exist_ok=True)

    if not force_retrain and _models_are_fresh(models_dir):
        result = _load_models(models_dir)
        if result:
            # If the stored feature set no longer matches the current FEATURE_COLS
            # (e.g. new features were added in a code update), silently retrain so
            # the user never has to pass --retrain manually after an upgrade.
            stored_features = set(result.get("features", []))
            current_features = set(FEATURE_COLS)
            if not current_features.issubset(stored_features):
                logger.info(
                    "Ticker %s: feature set changed (%d stored vs %d current) — retraining.",
                    ticker,
                    len(stored_features),
                    len(current_features),
                )
            else:
                result["status"] = "loaded"
                return result

    return train_models(ticker, csv_path, models_dir)


def train_models(ticker: str, csv_path: str, models_dir: str | None = None) -> dict:
    """
    Read ``csv_path``, train four models from scratch, persist them.

    Returns the same dict shape as load_or_train().
    """
    if models_dir is None:
        models_dir = os.path.join(ML_MODELS_DIR, ticker.upper())
        os.makedirs(models_dir, exist_ok=True)

    df = _load_csv(csv_path)
    if df is None or len(df) < ML_MIN_TRAIN_ROWS:
        logger.warning(
            "Ticker %s: only %d training rows (need %d) — skipping ML.",
            ticker,
            len(df) if df is not None else 0,
            ML_MIN_TRAIN_ROWS,
        )
        return {
            "models": {}, "features": [], "metrics": {},
            "models_dir": models_dir, "trained_on_rows": 0,
            "status": "insufficient_data",
        }

    # Columns that exist in this CSV
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    # Determine a single canonical feature set for ALL horizon models.
    # Drop columns that are entirely NaN across the full dataset so every
    # model is trained on the same feature space and predictions are consistent.
    X_check = df[feat_cols].ffill().bfill()
    X_check = X_check.dropna(axis=1, how="all")
    canonical_cols = list(X_check.columns)

    # ── Classifier: direction (binary 0 / 1) ───────────────────── #
    models  = {}
    metrics = {}

    for h in (5, 21):
        target_clf = f"direction_{h}d"
        target_reg = f"return_{h}d"

        # Drop rows where label is NaN (last h rows)
        mask = df[target_clf].notna() & df[target_reg].notna()
        sub  = df.loc[mask].copy()

        # Use canonical_cols (same for all horizons) so every model
        # expects the exact same feature vector at inference time.
        X = sub[canonical_cols].ffill().bfill().fillna(0.0)

        y_clf = sub[target_clf].values
        y_reg = sub[target_reg].values

        split = int(len(X) * 0.80)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_clf_tr, y_clf_te = y_clf[:split], y_clf[split:]
        y_reg_tr, y_reg_te = y_reg[:split], y_reg[split:]

        # Classifier — GBM
        clf_gb = GradientBoostingClassifier(**_GB_PARAMS)
        clf_gb.fit(X_tr, y_clf_tr)

        # Classifier — RandomForest (ensemble companion)
        clf_rf = RandomForestClassifier(**_RF_PARAMS)
        clf_rf.fit(X_tr, y_clf_tr)

        # Ensemble prediction: average probabilities from GBM and RF
        if len(X_te) > 0:
            gb_probs = clf_gb.predict_proba(X_te)[:, 1]
            rf_probs = clf_rf.predict_proba(X_te)[:, 1]
            ens_probs = (gb_probs + rf_probs) / 2.0
            ens_preds = (ens_probs >= 0.5).astype(int)
            clf_acc = accuracy_score(y_clf_te, ens_preds)
        else:
            clf_acc = float("nan")

        # Regressor — GBM
        reg_gb = GradientBoostingRegressor(**_GB_PARAMS)
        reg_gb.fit(X_tr, y_reg_tr)

        # Regressor — RandomForest (ensemble companion)
        reg_rf = RandomForestRegressor(**_RF_PARAMS)
        reg_rf.fit(X_tr, y_reg_tr)

        reg_mae = mean_absolute_error(y_reg_te, reg_gb.predict(X_te)) if len(X_te) > 0 else float("nan")

        models[f"clf_{h}d"]    = clf_gb   # GBM remains primary (for compatibility)
        models[f"clf_{h}d_rf"] = clf_rf
        models[f"reg_{h}d"]    = reg_gb
        models[f"reg_{h}d_rf"] = reg_rf
        metrics[f"clf_{h}d_accuracy"] = round(clf_acc, 4)
        metrics[f"reg_{h}d_mae"]      = round(reg_mae, 6)
        metrics[f"used_cols_{h}d"]    = canonical_cols

        # Feature importances — top 10 GBM features sorted by importance desc
        importances = clf_gb.feature_importances_
        top_idx = importances.argsort()[::-1][:10]
        metrics[f"top_features_clf_{h}d"] = [
            {"feature": canonical_cols[i], "importance": round(float(importances[i]), 4)}
            for i in top_idx
        ]
        reg_importances = reg_gb.feature_importances_
        top_reg_idx = reg_importances.argsort()[::-1][:10]
        metrics[f"top_features_reg_{h}d"] = [
            {"feature": canonical_cols[i], "importance": round(float(reg_importances[i]), 4)}
            for i in top_reg_idx
        ]

    # Persist — save canonical_cols (what models were actually trained on)
    _save_models(models_dir, models, canonical_cols, metrics)

    return {
        "models":          models,
        "features":        canonical_cols,
        "metrics":         metrics,
        "models_dir":      models_dir,
        "trained_on_rows": len(df),
        "status":          "trained",
    }


# ------------------------------------------------------------------ #
# Persistence helpers
# ------------------------------------------------------------------ #

def _save_models(models_dir: str, models: dict, feat_cols: list, metrics: dict) -> None:
    for name, model in models.items():
        path = os.path.join(models_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.debug("Saved model: %s", path)

    with open(os.path.join(models_dir, "features.json"), "w") as f:
        json.dump(feat_cols, f)

    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _load_models(models_dir: str) -> dict | None:
    expected = ["clf_5d", "clf_21d", "reg_5d", "reg_21d"]
    rf_optional = ["clf_5d_rf", "clf_21d_rf", "reg_5d_rf", "reg_21d_rf"]
    models = {}
    for name in expected:
        path = os.path.join(models_dir, f"{name}.pkl")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        except Exception:
            return None
    # Load RF companions if available (non-fatal if absent for backward compat)
    for name in rf_optional:
        path = os.path.join(models_dir, f"{name}.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    models[name] = pickle.load(f)
            except Exception:
                pass

    feats_path   = os.path.join(models_dir, "features.json")
    metrics_path = os.path.join(models_dir, "metrics.json")

    feat_cols = []
    metrics   = {}
    if os.path.exists(feats_path):
        with open(feats_path) as f:
            feat_cols = json.load(f)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {
        "models":          models,
        "features":        feat_cols,
        "metrics":         metrics,
        "models_dir":      models_dir,
        "trained_on_rows": 0,   # unknown when loading
    }


def _models_are_fresh(models_dir: str) -> bool:
    """Return True if all four model files exist and are within ML_RETRAIN_DAYS."""
    cutoff = datetime.datetime.now() - datetime.timedelta(days=ML_RETRAIN_DAYS)
    for name in ["clf_5d", "clf_21d", "reg_5d", "reg_21d"]:
        path = os.path.join(models_dir, f"{name}.pkl")
        if not os.path.exists(path):
            return False
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        if mtime < cutoff:
            return False
    return True


def _load_csv(csv_path: str) -> pd.DataFrame | None:
    """Load features.csv and drop rows where all label cols are NaN."""
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Rows where both label targets are NaN are inference-only rows
        label_cols = ["direction_5d", "direction_21d", "return_5d", "return_21d"]
        existing   = [c for c in label_cols if c in df.columns]
        if existing:
            df = df.dropna(subset=existing, how="all")
        return df
    except Exception as exc:
        logger.error("Could not load CSV %s: %s", csv_path, exc)
        return None
