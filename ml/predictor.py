"""
ml/predictor.py — Run inference against trained FinBot ML models.

Constructs a single-row feature vector from the latest data snapshot
(last row of the CSV) and returns predictions from all four models.

Expected models (loaded from models/{TICKER}/ by trainer.load_or_train):
  clf_5d  — GradientBoostingClassifier  → probability of price up in 5 days
  clf_21d — GradientBoostingClassifier  → probability of price up in 21 days
  reg_5d  — GradientBoostingRegressor   → expected % return over 5 days
  reg_21d — GradientBoostingRegressor   → expected % return over 21 days
"""

import os
import json
import logging

import numpy as np
import pandas as pd

from ml.trainer import FEATURE_COLS

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def predict(csv_path: str, trainer_result: dict) -> dict:
    """
    Load the latest feature row from ``csv_path`` and run inference.

    Parameters
    ----------
    csv_path       : Absolute path to the features.csv written by
                     csv_exporter.export_features_csv().
    trainer_result : Dict returned by ml.trainer.load_or_train().

    Returns
    -------
    {
        "clf_5d_prob_up":    float   # 0–1 probability of UP in 5 trading days
        "clf_21d_prob_up":   float   # 0–1 probability of UP in 21 trading days
        "clf_5d_direction":  str     # "UP" | "DOWN"
        "clf_21d_direction": str     # "UP" | "DOWN"
        "reg_5d_return_pct": float   # predicted % change over 5 days
        "reg_21d_return_pct":float   # predicted % change over 21 days
        "metrics":           dict    # accuracy / MAE from training
        "trained_on_rows":   int
        "status":            str     # "ok" | "insufficient_data" | "error"
    }
    """
    status = trainer_result.get("status", "")
    if status == "insufficient_data" or not trainer_result.get("models"):
        return _empty(status or "insufficient_data")

    models  = trainer_result["models"]
    metrics = trainer_result.get("metrics", {})
    trained_rows = trainer_result.get("trained_on_rows", 0)

    # Build feature vector using the feature list from training time.
    # This must match what the persisted models were trained on.
    train_features = trainer_result.get("features") or FEATURE_COLS
    x = _build_feature_vector(csv_path, train_features)
    if x is None:
        return _empty("error")

    result = {
        "metrics":         metrics,
        "trained_on_rows": trained_rows,
        "status":          "ok",
    }

    for h in (5, 21):
        clf = models.get(f"clf_{h}d")
        reg = models.get(f"reg_{h}d")

        if clf is not None:
            prob_up = float(clf.predict_proba(x)[0][1])
            direction = "UP" if prob_up >= 0.5 else "DOWN"
            result[f"clf_{h}d_prob_up"]   = round(prob_up, 4)
            result[f"clf_{h}d_direction"] = direction
        else:
            result[f"clf_{h}d_prob_up"]   = None
            result[f"clf_{h}d_direction"] = "N/A"

        if reg is not None:
            ret_pct = float(reg.predict(x)[0]) * 100
            result[f"reg_{h}d_return_pct"] = round(ret_pct, 2)
        else:
            result[f"reg_{h}d_return_pct"] = None

    return result


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _build_feature_vector(csv_path: str, feature_cols: list) -> "np.ndarray | None":
    """Read the last non-NaN feature row and return a (1, n_features) array."""
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except Exception as exc:
        logger.error("predict: cannot read CSV %s: %s", csv_path, exc)
        return None

    # Use only feature columns that exist in this CSV
    feat_cols = [c for c in feature_cols if c in df.columns]
    if not feat_cols:
        return None

    X = df[feat_cols]
    # Forward-fill then fallback to 0.0 for any residual NaN
    X = X.ffill().bfill().fillna(0.0)

    # Take the very last row (today's snapshot).
    # Return as a DataFrame (not a numpy array) so scikit-learn can
    # match the named features the model was trained with.
    return X.iloc[[-1]]


def _empty(status: str) -> dict:
    return {
        "clf_5d_prob_up":     None,
        "clf_21d_prob_up":    None,
        "clf_5d_direction":   "N/A",
        "clf_21d_direction":  "N/A",
        "reg_5d_return_pct":  None,
        "reg_21d_return_pct": None,
        "metrics":            {},
        "trained_on_rows":    0,
        "status":             status,
    }
