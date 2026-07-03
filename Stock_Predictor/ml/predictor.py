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
import math
import pandas as pd

from ml.trainer import FEATURE_COLS
from ml import global_trainer

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
    has_local = bool(trainer_result.get("models")) and status != "insufficient_data"

    # Pooled cross-ticker models (models/_GLOBAL/, trained via
    # --retrain-global). Blended with the per-ticker models when both
    # exist; used alone for tickers whose history is too short to train.
    global_set = global_trainer.load_global_models()

    if not has_local and not global_set:
        return _empty(status or "insufficient_data")

    models  = trainer_result.get("models", {}) if has_local else {}
    metrics = trainer_result.get("metrics", {}) if has_local else {}
    trained_rows = trainer_result.get("trained_on_rows", 0)

    # Build feature vector using the feature list from training time.
    # This must match what the persisted models were trained on.
    x = None
    if has_local:
        train_features = trainer_result.get("features") or FEATURE_COLS
        x = _build_feature_vector(csv_path, train_features)
        if x is None and not global_set:
            return _empty("error")

    x_global = _build_feature_vector(csv_path, global_set["features"]) if global_set else None

    if not has_local:
        # Short-history ticker: serve pure global predictions and surface
        # the pooled CV metrics so the judge's accuracy gating still works.
        metrics = dict(global_set.get("metrics", {}))

    result = {
        "metrics":         metrics,
        "trained_on_rows": trained_rows,
        "status":          "ok" if has_local else "ok_global",
        "global_blend":    bool(global_set),
    }

    cal_path = os.path.join(os.path.dirname(csv_path), "..", "..", "models",
                            os.path.basename(os.path.dirname(csv_path)), "calibration.json")
    cal_path = os.path.normpath(cal_path)

    for h in (5, 21):
        prob_local  = _ensemble_prob(models, x, h) if x is not None else None
        prob_global = _ensemble_prob(global_set["models"], x_global, h) if (global_set and x_global is not None) else None

        prob_up = _blend(
            prob_local,  _metric_acc(metrics, h) if has_local else None,
            prob_global, _metric_acc(global_set.get("metrics", {}), h) if global_set else None,
        )

        if prob_up is not None:
            prob_up = round(prob_up, 4)

            # Calibrate using empirical lookup table if it exists
            cal_prob_up = _apply_calibration(prob_up, cal_path, h)

            # Decision threshold: tuned per-horizon from resolved feedback
            # (stored in calibration.json) — falls back to 0.5.
            threshold = _tuned_threshold(cal_path, h)
            direction = "UP" if cal_prob_up >= threshold else "DOWN"

            # Confidence label based on distance from the decision boundary
            confidence_margin = abs(cal_prob_up - threshold)
            if confidence_margin > 0.20:
                confidence = "HIGH"
            elif confidence_margin > 0.10:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            result[f"clf_{h}d_prob_up"]         = prob_up
            result[f"clf_{h}d_prob_up_cal"]     = round(cal_prob_up, 4)
            result[f"clf_{h}d_direction"]       = direction
            result[f"clf_{h}d_confidence"]      = confidence
        else:
            result[f"clf_{h}d_prob_up"]         = None
            result[f"clf_{h}d_prob_up_cal"]     = None
            result[f"clf_{h}d_direction"]       = "N/A"
            result[f"clf_{h}d_confidence"]      = "N/A"

        ret_local  = _ensemble_return(models, x, h) if x is not None else None
        ret_global = _ensemble_return(global_set["models"], x_global, h) if (global_set and x_global is not None) else None
        ret_pct = _blend(
            ret_local,  _metric_acc(metrics, h) if has_local else None,
            ret_global, _metric_acc(global_set.get("metrics", {}), h) if global_set else None,
        )
        result[f"reg_{h}d_return_pct"] = round(ret_pct, 2) if ret_pct is not None else None

    return result


# ------------------------------------------------------------------ #
# Blending helpers
# ------------------------------------------------------------------ #

def _ensemble_prob(models: dict, x, h: int) -> "float | None":
    """GBM+RF averaged probability-of-up for horizon *h* (None if absent)."""
    clf_gb = models.get(f"clf_{h}d")
    if clf_gb is None:
        return None
    prob = float(clf_gb.predict_proba(x)[0][1])
    clf_rf = models.get(f"clf_{h}d_rf")
    if clf_rf is not None:
        prob = (prob + float(clf_rf.predict_proba(x)[0][1])) / 2.0
    return prob


def _ensemble_return(models: dict, x, h: int) -> "float | None":
    """GBM+RF averaged expected % return for horizon *h* (None if absent)."""
    reg_gb = models.get(f"reg_{h}d")
    if reg_gb is None:
        return None
    ret = float(reg_gb.predict(x)[0]) * 100
    reg_rf = models.get(f"reg_{h}d_rf")
    if reg_rf is not None:
        ret = (ret + float(reg_rf.predict(x)[0]) * 100) / 2.0
    return ret


def _metric_acc(metrics: dict, h: int) -> "float | None":
    """Walk-forward CV accuracy for horizon *h* (None when missing/NaN)."""
    acc = metrics.get(f"clf_{h}d_accuracy")
    if isinstance(acc, (int, float)) and not math.isnan(acc):
        return float(acc)
    return None


def _blend(v_local, acc_local, v_global, acc_global) -> "float | None":
    """
    Accuracy-weighted blend of the per-ticker and pooled predictions.
    Weight = CV accuracy edge over coin-flip (floored at a small epsilon
    so a model is never zeroed out entirely on tiny samples). Falls back
    to whichever value exists, or a 50/50 mix when accuracies are unknown.
    """
    if v_local is None and v_global is None:
        return None
    if v_local is None:
        return v_global
    if v_global is None:
        return v_local
    w_local  = max((acc_local  or 0.5) - 0.5, 0.02)
    w_global = max((acc_global or 0.5) - 0.5, 0.02)
    return (w_local * v_local + w_global * v_global) / (w_local + w_global)


def _tuned_threshold(cal_path: str, horizon: int) -> float:
    """Per-horizon decision threshold from calibration.json (default 0.5)."""
    try:
        if os.path.isfile(cal_path):
            with open(cal_path) as f:
                cal = json.load(f)
            t = cal.get(f"threshold_{horizon}d")
            if isinstance(t, (int, float)) and 0.3 <= t <= 0.7:
                return float(t)
    except Exception:
        pass
    return 0.5


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _apply_calibration(raw_prob: float, cal_path: str, horizon: int) -> float:
    """
    Look up the calibrated probability for ``raw_prob`` from
    ``calibration.json`` if it exists.  Falls back to ``raw_prob``
    when the file is absent or the lookup fails.

    calibration.json format (written by feedback/accuracy.py):
        {
          "5d":  [{"lo": 0.3, "hi": 0.4, "empirical": 0.42}, ...],
          "21d": [...]
        }
    """
    try:
        if not os.path.isfile(cal_path):
            return raw_prob
        with open(cal_path) as f:
            cal = json.load(f)
        key = f"{horizon}d"
        buckets = cal.get(key, [])
        for b in buckets:
            if b["lo"] <= raw_prob < b["hi"]:
                return float(b["empirical"])
    except Exception:
        pass
    return raw_prob


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
        "clf_5d_prob_up":       None,
        "clf_5d_prob_up_cal":   None,
        "clf_21d_prob_up":      None,
        "clf_21d_prob_up_cal":  None,
        "clf_5d_direction":     "N/A",
        "clf_21d_direction":    "N/A",
        "clf_5d_confidence":    "N/A",
        "clf_21d_confidence":   "N/A",
        "reg_5d_return_pct":    None,
        "reg_21d_return_pct":   None,
        "metrics":              {},
        "trained_on_rows":      0,
        "status":               status,
    }
