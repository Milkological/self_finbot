"""
feedback/accuracy.py — Compute historical accuracy statistics from feedback CSV.

get_context(ticker, mode) reads the resolved rows and returns:
  - per-horizon directional accuracy %
  - mean signed bias (pct_error) per horizon
  - bias correction factors (applied to LLM targets when N >= MIN_SAMPLES)
  - adjusted lens weights for the rule-based judge (when N >= MIN_SAMPLES)
  - ML accuracy (5d / 21d forward accuracy)
  - overall recommendation hit rate

Called before analysis so the results can be injected into LLM prompts
and used to reweight the rule-based judge.
"""

import os
import json
import statistics
import pandas as pd

from config import FEEDBACK_DIR, FEEDBACK_MIN_SAMPLES, FEEDBACK_MAX_BIAS_CORRECTION, CALIBRATION_MIN_SAMPLES, ML_MODELS_DIR

HORIZONS = ["1w", "2w", "3w", "1m", "3m", "6m", "9m", "12m"]

# Original rule-based lens weights
_BASE_WEIGHTS = {
    "fundamental": 0.30,
    "technical":   0.25,
    "valuation":   0.30,
    "risk":        0.15,
}


def _csv_path(ticker: str) -> str:
    return os.path.join(FEEDBACK_DIR, f"{ticker.upper()}_feedback.csv")


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return f if pd.notna(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def get_context(ticker: str, mode: str = "llm", provider: str | None = None) -> dict:
    """
    Load the feedback CSV and compute accuracy statistics.

    Parameters
    ----------
    ticker : str
    mode   : "llm" or "no_llm" — used to filter rows and compute lens accuracy

    Returns
    -------
    dict with keys:
        horizon_stats        : per-horizon accuracy %, bias, N, correction_factor
        ml_accuracy          : {"5d": float|None, "21d": float|None}
        rec_hit_rate         : float|None  (0-1)
        adjusted_weights     : dict (same keys as _BASE_WEIGHTS)
        weights_adjusted     : bool  (True if N >= MIN_SAMPLES)
        total_resolved       : int
        has_data             : bool
    """
    empty = {
        "horizon_stats":    {h: {"n": 0, "dir_accuracy": None, "mean_bias": None,
                                 "correction_factor": 1.0} for h in HORIZONS},
        "ml_accuracy":       {"5d": None, "21d": None},
        "rec_hit_rate":      None,
        "adjusted_weights":  dict(_BASE_WEIGHTS),
        "weights_adjusted":  False,
        "total_resolved":    0,
        "has_data":          False,
    }

    path = _csv_path(ticker)
    if not os.path.isfile(path):
        return empty

    try:
        # on_bad_lines='skip' tolerates rows written by older versions that had
        # a different column count (e.g. before the 'provider' column was added).
        df = pd.read_csv(path, dtype=str, on_bad_lines="skip")
    except Exception:
        return empty

    if df.empty:
        return empty

    # ── Deduplicate same-day rows (safety net for existing CSVs) ──── #
    # Keep only the first row per (run_date, provider) pair so that
    # running the program multiple times on the same day doesn't inflate
    # sample counts with non-independent data points.
    dedup_cols = [c for c in ["run_date", "provider"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="first").copy()

    # ── Filter by provider if specified (backward-compatible) ─────── #
    if provider and "provider" in df.columns:
        df = df[df["provider"].str.strip() == provider.strip()]
        if df.empty:
            return empty

    # ── Per-horizon stats ─────────────────────────────────────────── #
    horizon_stats = {}
    total_resolved = 0

    for h in HORIZONS:
        actual_col  = f"actual_{h}"
        dir_col     = f"dir_correct_{h}"
        err_col     = f"pct_error_{h}"

        if actual_col not in df.columns:
            horizon_stats[h] = {"n": 0, "dir_accuracy": None,
                                 "mean_bias": None, "correction_factor": 1.0}
            continue

        resolved = df[df[actual_col].str.strip().replace("nan", "") != ""].copy()
        n = len(resolved)

        if n == 0:
            horizon_stats[h] = {"n": 0, "dir_accuracy": None,
                                 "mean_bias": None, "correction_factor": 1.0}
            continue

        total_resolved = max(total_resolved, n)

        # Directional accuracy
        dir_vals = [_safe_int(v) for v in resolved[dir_col] if dir_col in resolved.columns]
        dir_vals = [v for v in dir_vals if v is not None]
        dir_acc  = round(sum(dir_vals) / len(dir_vals) * 100, 1) if dir_vals else None

        # Signed bias — use the MEDIAN, which is robust to the occasional
        # huge outlier a volatile horizon produces (a single +120% error
        # would drag the mean and swing every future target).
        err_vals = [_safe_float(v) for v in resolved[err_col] if err_col in resolved.columns]
        err_vals = [v for v in err_vals if v is not None]
        median_bias = round(statistics.median(err_vals), 2) if err_vals else None

        # Bias correction factor (only when N >= MIN_SAMPLES)
        correction_factor = 1.0
        if n >= FEEDBACK_MIN_SAMPLES and median_bias is not None:
            # median_bias is in %; convert to fraction.
            # positive bias = targets undershot → multiply up; negative = overshot → down
            raw_correction = median_bias / 100.0
            clamped = max(-FEEDBACK_MAX_BIAS_CORRECTION,
                          min(FEEDBACK_MAX_BIAS_CORRECTION, raw_correction))
            # Shrink toward 1.0 on thin samples: at N == MIN_SAMPLES only half
            # of the correction is applied; full strength once N >= 2*MIN_SAMPLES.
            # Prevents 5 data points from moving targets as hard as 20+ do.
            shrink = min(1.0, n / (2.0 * FEEDBACK_MIN_SAMPLES))
            correction_factor = round(1.0 + clamped * shrink, 4)

        horizon_stats[h] = {
            "n":                n,
            "dir_accuracy":     dir_acc,
            # key kept as "mean_bias" for backward compatibility with callers;
            # value is now the median signed error.
            "mean_bias":        median_bias,
            "correction_factor": correction_factor,
        }

    # ── ML accuracy ───────────────────────────────────────────────── #
    ml_5d_vals  = [_safe_int(v) for v in df.get("ml_5d_correct",  pd.Series(dtype=str))
                   if str(v).strip() not in ("", "nan")]
    ml_21d_vals = [_safe_int(v) for v in df.get("ml_21d_correct", pd.Series(dtype=str))
                   if str(v).strip() not in ("", "nan")]

    ml_5d_acc  = round(sum(ml_5d_vals)  / len(ml_5d_vals)  * 100, 1) if ml_5d_vals  else None
    ml_21d_acc = round(sum(ml_21d_vals) / len(ml_21d_vals) * 100, 1) if ml_21d_vals else None

    # ── Recommendation hit rate ───────────────────────────────────── #
    rec_vals = [_safe_int(v) for v in df.get("rec_correct", pd.Series(dtype=str))
                if str(v).strip() not in ("", "nan")]
    rec_hit_rate = round(sum(rec_vals) / len(rec_vals), 3) if rec_vals else None

    # ── Dynamic lens weight adjustment (no_llm mode) ─────────────── #
    adjusted_weights = dict(_BASE_WEIGHTS)
    weights_adjusted = False

    if total_resolved >= FEEDBACK_MIN_SAMPLES and "rec_correct" in df.columns:
        # For each lens, compare its score direction against actual outcome.
        # A lens score > 60 is "bullish signal". If price rose, it was correct.
        lens_map = {
            "fundamental": "score_fundamental",
            "technical":   "score_technical",
            "valuation":   "score_valuation",
            "risk":        "score_risk",
        }
        lens_accuracies = {}
        for lens, score_col in lens_map.items():
            if score_col not in df.columns or "rec_correct" not in df.columns:
                continue
            rows_with_data = df[
                (df[score_col].str.strip().replace("nan", "") != "") &
                (df["rec_correct"].str.strip().replace("nan", "") != "")
            ].copy()
            if len(rows_with_data) < FEEDBACK_MIN_SAMPLES:
                continue
            correct_count = 0
            total_count   = 0
            for _, r in rows_with_data.iterrows():
                score = _safe_float(r[score_col])
                rec_c = _safe_int(r["rec_correct"])
                if score is None or rec_c is None:
                    continue
                # Score > 60 = bullish lens verdict
                lens_bullish  = score > 60
                # rec_correct=1 means the directional call was right
                outcome_right = bool(rec_c)
                correct_count += int(lens_bullish == outcome_right)
                total_count   += 1
            if total_count >= FEEDBACK_MIN_SAMPLES:
                lens_accuracies[lens] = correct_count / total_count

        if lens_accuracies:
            # Scale each weight by accuracy relative to 50% baseline,
            # capped at ±50% of original weight, then renormalize.
            new_weights = {}
            for lens, base_w in _BASE_WEIGHTS.items():
                acc = lens_accuracies.get(lens)
                if acc is None:
                    new_weights[lens] = base_w
                else:
                    # Scale factor: 1.0 at 50% accuracy, 1.5 at 100%, 0.5 at 0%
                    scale  = 0.5 + acc
                    cap_hi = base_w * 1.50
                    cap_lo = base_w * 0.50
                    new_weights[lens] = round(max(cap_lo, min(cap_hi, base_w * scale)), 4)

            total_w = sum(new_weights.values())
            if total_w > 0:
                adjusted_weights = {k: round(v / total_w, 4) for k, v in new_weights.items()}
                weights_adjusted = True

    return {
        "horizon_stats":    horizon_stats,
        "ml_accuracy":      {"5d": ml_5d_acc, "21d": ml_21d_acc},
        "rec_hit_rate":     rec_hit_rate,
        "adjusted_weights": adjusted_weights,
        "weights_adjusted": weights_adjusted,
        "total_resolved":   total_resolved,
        "has_data":         total_resolved > 0,
    }


def build_calibration_table(ticker: str) -> dict:
    """
    Build a probability calibration lookup table from resolved feedback rows
    and save it to  models/{TICKER}/calibration.json.

    Buckets the raw ``ml_Xd_prob_up`` values into ranges and computes the
    empirical direction-accuracy per bucket.  The predictor then replaces
    raw model probabilities with these empirical values, giving
    calibrated confidence estimates.

    Activated only when resolved rows >= CALIBRATION_MIN_SAMPLES.

    Returns
    -------
    dict  — calibration table written (or existing) keyed by "5d" / "21d".
            Returns empty dict when there is insufficient data.
    """
    path = _csv_path(ticker)
    if not os.path.isfile(path):
        return {}

    try:
        # on_bad_lines="skip" matches get_context/resolver so a single stray
        # malformed row (e.g. the "saw 73 fields" corruption in some CSVs)
        # no longer makes calibration silently throw and never build.
        df = pd.read_csv(path, dtype=str, on_bad_lines="skip")
    except Exception:
        return {}

    if df.empty:
        return {}

    # Only use resolved rows (have ml_Xd_correct filled)
    def _has_col(col):
        return col in df.columns

    result = {}

    buckets_def = [
        (0.0, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01),
    ]

    for h_label, prob_col, correct_col in [
        ("5d",  "ml_5d_prob_up",  "ml_5d_correct"),
        ("21d", "ml_21d_prob_up", "ml_21d_correct"),
    ]:
        if not (_has_col(prob_col) and _has_col(correct_col)):
            continue

        sub = df[[prob_col, correct_col]].copy()
        sub[prob_col]    = pd.to_numeric(sub[prob_col],    errors="coerce")
        sub[correct_col] = pd.to_numeric(sub[correct_col], errors="coerce")
        sub = sub.dropna()

        if len(sub) < CALIBRATION_MIN_SAMPLES:
            continue

        buckets = []
        for lo, hi in buckets_def:
            mask  = (sub[prob_col] >= lo) & (sub[prob_col] < hi)
            chunk = sub[mask]
            if len(chunk) >= 3:                      # need at least 3 data points
                empirical = round(float(chunk[correct_col].mean()), 4)
                buckets.append({"lo": lo, "hi": hi, "n": len(chunk), "empirical": empirical})

        if buckets:
            result[h_label] = buckets

    if result:
        cal_path = os.path.join(ML_MODELS_DIR, ticker.upper(), "calibration.json")
        os.makedirs(os.path.dirname(cal_path), exist_ok=True)
        try:
            with open(cal_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Could not save calibration table: %s", exc)

    return result



def apply_bias_correction(targets: dict, accuracy_context: dict) -> dict:
    """
    Apply per-horizon bias correction factors to LLM price targets.

    Parameters
    ----------
    targets          : dict — llm_result["target_prices"]
                       keys: "1_week", "2_weeks", ..., "12_months"
    accuracy_context : dict — from get_context()

    Returns corrected targets dict (same structure). Adds "bias_corrected": True
    to each corrected horizon entry.
    """
    if not targets or not accuracy_context.get("has_data"):
        return targets

    # Map target_prices keys → horizon stat keys
    key_map = {
        "1_week":   "1w",
        "2_weeks":  "2w",
        "3_weeks":  "3w",
        "1_month":  "1m",
        "3_months": "3m",
        "6_months": "6m",
        "9_months": "9m",
        "12_months":"12m",
    }

    corrected = {}
    stats = accuracy_context.get("horizon_stats", {})

    for tk, hk in key_map.items():
        entry = targets.get(tk)
        if entry is None:
            corrected[tk] = entry
            continue

        h_stat = stats.get(hk, {})
        cf     = h_stat.get("correction_factor", 1.0)
        n      = h_stat.get("n", 0)

        if n >= FEEDBACK_MIN_SAMPLES and cf != 1.0:
            price = entry.get("price")
            if isinstance(price, (int, float)) and price > 0:
                corrected[tk] = {**entry,
                                  "price":          round(price * cf, 2),
                                  "price_raw":      price,
                                  "bias_corrected": True,
                                  "correction_factor": cf}
            else:
                corrected[tk] = entry
        else:
            corrected[tk] = entry

    return corrected


def format_llm_injection(ticker: str, accuracy_context: dict) -> str:
    """
    Format the historical accuracy block to inject into the LLM Judge prompt.
    Returns an empty string if there is no data yet.
    """
    if not accuracy_context.get("has_data"):
        return (
            f"\n[HISTORICAL ACCURACY FOR {ticker}]\n"
            "No prior predictions on record for this ticker yet. "
            "This is the first or an early prediction — no calibration data available.\n"
        )

    lines = [f"\n[HISTORICAL ACCURACY FOR {ticker}]"]
    lines.append(
        "You have made predictions for this ticker before. "
        "Use this data to calibrate your price targets and confidence levels.\n"
    )

    stats = accuracy_context.get("horizon_stats", {})
    horizon_labels = {
        "1w": "1 Week", "2w": "2 Weeks", "3w": "3 Weeks", "1m": "1 Month",
        "3m": "3 Months", "6m": "6 Months", "9m": "9 Months", "12m": "12 Months",
    }

    lines.append("Horizon | N Predictions | Directional Accuracy | Mean Price Bias")
    lines.append("--------|--------------|---------------------|-----------------")
    for h, label in horizon_labels.items():
        s = stats.get(h, {})
        n   = s.get("n", 0)
        acc = s.get("dir_accuracy")
        bias= s.get("mean_bias")
        cf  = s.get("correction_factor", 1.0)

        if n == 0:
            lines.append(f"{label} | 0 | No data | No data")
            continue

        acc_str  = f"{acc:.0f}%" if acc is not None else "N/A"
        if bias is not None:
            bias_dir = "overshot (too high)" if bias < 0 else "undershot (too low)"
            bias_str = f"{abs(bias):.1f}% {bias_dir}"
        else:
            bias_str = "N/A"
        cf_note = f" [correction {cf:.3f}x applied]" if cf != 1.0 and n >= FEEDBACK_MIN_SAMPLES else ""
        lines.append(f"{label} | {n} | {acc_str} | {bias_str}{cf_note}")

    ml = accuracy_context.get("ml_accuracy", {})
    rec = accuracy_context.get("rec_hit_rate")
    lines.append("")
    if ml.get("5d") is not None:
        lines.append(f"ML 5-day forward accuracy:  {ml['5d']:.0f}%")
    if ml.get("21d") is not None:
        lines.append(f"ML 21-day forward accuracy: {ml['21d']:.0f}%")
    if rec is not None:
        lines.append(f"Overall recommendation hit rate: {rec*100:.0f}%")

    # ── Grounding: anchor accuracy_pct to REALIZED accuracy ───────── #
    lines.append(
        f"\nINSTRUCTION (GROUNDING): For every horizon with N >= {FEEDBACK_MIN_SAMPLES} "
        "predictions, use that horizon's realized Directional Accuracy above as the BASE "
        "for its accuracy_pct — NOT the ML test-split accuracy. Then apply only the small "
        "VIX/agreement adjustments and the standard [30%, 85%] clamp. Horizons with no "
        "track record fall back to the ML-derived base."
    )

    # ── Caution rules tied to the live recommendation hit rate ────── #
    if rec is not None and rec < 0.40:
        lines.append(
            f"TRACK-RECORD CAUTION: live recommendation hit rate is {rec*100:.0f}% "
            "(below 40%) — cap confidence at LOW and prefer HOLD unless the signals "
            "strongly align."
        )
    elif rec is not None and rec < 0.50:
        lines.append(
            f"TRACK-RECORD CAUTION: live recommendation hit rate is {rec*100:.0f}% "
            "(below 50%) — cap confidence at MEDIUM."
        )

    # Flag individual horizons that have been worse than a coin flip.
    weak = [
        horizon_labels[h] for h in horizon_labels
        if stats.get(h, {}).get("n", 0) >= FEEDBACK_MIN_SAMPLES
        and (stats.get(h, {}).get("dir_accuracy") is not None)
        and stats[h]["dir_accuracy"] < 45.0
    ]
    if weak:
        lines.append(
            "LOW-ACCURACY HORIZONS (realized directional accuracy <45%): "
            + ", ".join(weak)
            + " — assign these LOW confidence and widen or avoid their targets."
        )

    lines.append(
        "\nINSTRUCTION: Calibrate your price targets using the bias data above. "
        "If your targets consistently overshoot, be more conservative. "
        "If they consistently undershoot, be more aggressive. "
        "Do NOT simply repeat past errors.\n"
    )

    return "\n".join(lines)
