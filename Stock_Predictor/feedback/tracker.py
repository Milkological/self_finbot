"""
feedback/tracker.py — Save a prediction row to feedback/{TICKER}_feedback.csv.

Called at the END of run_pipeline() after all analysis is complete.
One CSV per ticker; outcome columns start blank and are filled by resolver.py.
"""

import os
import csv
import uuid
import datetime
import pandas as pd
from config import FEEDBACK_DIR


# ── Column order (must match resolver and accuracy modules) ────────── #
COLUMNS = [
    # identity
    "run_id", "ticker", "run_date", "run_timestamp", "mode", "provider",
    # core prediction
    "current_price", "recommendation", "confidence",
    # LLM price targets
    "target_1w", "target_2w", "target_3w", "target_1m",
    "target_3m", "target_6m", "target_9m", "target_12m",
    # ML signals
    "ml_5d_direction", "ml_5d_prob_up", "ml_21d_direction", "ml_21d_prob_up",
    "ml_5d_expected_return", "ml_21d_expected_return",
    # rule-based scores (no-llm mode)
    "score_fundamental", "score_technical", "score_valuation",
    "score_risk", "score_composite",
    # LLM verdicts (llm mode)
    "tech_verdict", "fundamental_verdict", "valuation_verdict", "sentiment_verdict",
    # news & sentiment
    "headline_1", "headline_2", "headline_3",
    "sentiment_score", "sentiment_label",
    "key_catalysts", "key_bull_case", "key_bear_case",
    # volume
    "avg_volume_5d", "volume_ratio_20d",
    # options
    "put_call_ratio", "options_iv_avg",
    # short interest
    "short_ratio", "short_pct_float",
    # ── outcomes (filled by resolver) ─────────────────────────────── #
    "actual_1w", "actual_2w", "actual_3w", "actual_1m",
    "actual_3m", "actual_6m", "actual_9m", "actual_12m",
    "dir_correct_1w", "dir_correct_2w", "dir_correct_3w", "dir_correct_1m",
    "dir_correct_3m", "dir_correct_6m", "dir_correct_9m", "dir_correct_12m",
    "pct_error_1w", "pct_error_2w", "pct_error_3w", "pct_error_1m",
    "pct_error_3m", "pct_error_6m", "pct_error_9m", "pct_error_12m",
    "ml_5d_correct", "ml_21d_correct", "rec_correct",
]


def _csv_path(ticker: str) -> str:
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    return os.path.join(FEEDBACK_DIR, f"{ticker.upper()}_feedback.csv")


def save_sentiment_history(ticker: str, sentiment: dict) -> str:
    """
    Append today's headline-sentiment score to
    feedback/sentiment_history_{TICKER}.csv (one row per calendar day).

    This builds a genuine per-day sentiment archive over time so that
    sentiment can eventually be reintroduced as an ML feature without
    look-ahead bias (historical CSV rows currently have no real
    sentiment — see ml/trainer.py FEATURE_COLS note).

    Returns the path to the CSV file. Same-day reruns overwrite the
    existing row for today rather than duplicating it.
    """
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    path  = os.path.join(FEEDBACK_DIR, f"sentiment_history_{ticker.upper()}.csv")
    today = datetime.date.today().isoformat()
    row   = {
        "date":            today,
        "sentiment_score": float(sentiment.get("overall_score", 0.0)),
        "sentiment_label": str(sentiment.get("label", "NEUTRAL")),
    }

    if os.path.isfile(path):
        df = pd.read_csv(path, dtype={"date": str})
        df = df[df["date"] != today]                    # upsert today's row
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)
    return path


def save_prediction(
    ticker:       str,
    mode:         str,          # "llm" or "no_llm"
    price_df:     "pd.DataFrame",
    info:         dict,
    llm_result:   dict,
    ml_result:    dict,
    analyst_data: dict,
    options_data: dict,
    provider:     str = "",     # e.g. "Azure OpenAI", "Google Gemini (...)", "DeepSeek (...)", "rule_based"
) -> str:
    """
    Append one prediction row to feedback/{TICKER}_feedback.csv.

    Returns the path to the CSV file.
    """
    now       = datetime.datetime.now()
    run_id    = f"{ticker.upper()}_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_date  = now.date().isoformat()
    run_ts    = now.isoformat(timespec="seconds")

    # ── Current price ────────────────────────────────────────────── #
    current_price = info.get("currentPrice") or float(price_df["Close"].iloc[-1])

    # ── LLM price targets ─────────────────────────────────────────── #
    targets    = llm_result.get("target_prices", {})
    target_map = {
        "target_1w":  targets.get("1_week",   {}).get("price"),
        "target_2w":  targets.get("2_weeks",  {}).get("price"),
        "target_3w":  targets.get("3_weeks",  {}).get("price"),
        "target_1m":  targets.get("1_month",  {}).get("price"),
        "target_3m":  targets.get("3_months", {}).get("price"),
        "target_6m":  targets.get("6_months", {}).get("price"),
        "target_9m":  targets.get("9_months", {}).get("price"),
        "target_12m": targets.get("12_months",{}).get("price"),
    }

    # ── ML signals ───────────────────────────────────────────────── #
    ml = ml_result or {}

    # ── Rule-based scores ─────────────────────────────────────────── #
    rbj = llm_result.get("rule_based_judge") or {}
    score_fundamental = rbj.get("fundamental", {}).get("score") if rbj else None
    score_technical   = rbj.get("technical",   {}).get("score") if rbj else None
    score_valuation   = rbj.get("valuation",   {}).get("score") if rbj else None
    score_risk        = rbj.get("risk",        {}).get("score") if rbj else None
    score_composite   = rbj.get("composite_score") if rbj else None

    # In no_llm mode the judge result is at the top level of llm_result
    if mode == "no_llm":
        score_fundamental = llm_result.get("rule_based_judge", {}).get("fundamental", {}).get("score")
        score_technical   = llm_result.get("rule_based_judge", {}).get("technical",   {}).get("score")
        score_valuation   = llm_result.get("rule_based_judge", {}).get("valuation",   {}).get("score")
        score_risk        = llm_result.get("rule_based_judge", {}).get("risk",        {}).get("score")
        score_composite   = llm_result.get("rule_based_judge", {}).get("composite_score")
        # run_rule_based_analysis stores the judge at top level differently
        if score_fundamental is None:
            score_fundamental = llm_result.get("fundamental", {}).get("score") if isinstance(llm_result.get("fundamental"), dict) else None
            score_technical   = llm_result.get("technical",   {}).get("score") if isinstance(llm_result.get("technical"),   dict) else None
            score_valuation   = llm_result.get("valuation",   {}).get("score") if isinstance(llm_result.get("valuation"),   dict) else None
            score_risk        = llm_result.get("risk",        {}).get("score") if isinstance(llm_result.get("risk"),        dict) else None
            score_composite   = llm_result.get("composite_score")

    # ── LLM verdicts ──────────────────────────────────────────────── #
    tech_verdict        = llm_result.get("technical_verdict",   "")
    fundamental_verdict = llm_result.get("fundamental_verdict", "")
    valuation_verdict   = llm_result.get("valuation_verdict",   "")
    sentiment_verdict   = llm_result.get("sentiment_verdict",   "")

    # ── News & sentiment ──────────────────────────────────────────── #
    news_list = analyst_data.get("news", [])
    headline_1 = news_list[0]["title"] if len(news_list) > 0 else ""
    headline_2 = news_list[1]["title"] if len(news_list) > 1 else ""
    headline_3 = news_list[2]["title"] if len(news_list) > 2 else ""

    sentiment       = ml.get("sentiment", {}) or {}
    sentiment_score = sentiment.get("overall_score", "")
    sentiment_label = sentiment.get("label", "")

    catalysts  = llm_result.get("catalysts",    []) or []
    bull_case  = llm_result.get("key_bull_case",[]) or []
    bear_case  = llm_result.get("key_bear_case",[]) or []
    key_catalysts = "; ".join(str(c) for c in catalysts[:5])
    key_bull_case = "; ".join(str(b) for b in bull_case[:3])
    key_bear_case = "; ".join(str(b) for b in bear_case[:3])

    # ── Volume ────────────────────────────────────────────────────── #
    avg_volume_5d = volume_ratio_20d = ""
    if price_df is not None and "Volume" in price_df.columns and len(price_df) >= 5:
        vol_series    = price_df["Volume"].dropna()
        avg_5d        = vol_series.iloc[-5:].mean()
        avg_20d       = vol_series.iloc[-20:].mean() if len(vol_series) >= 20 else avg_5d
        avg_volume_5d = round(avg_5d, 0)
        volume_ratio_20d = round(avg_5d / avg_20d, 3) if avg_20d > 0 else ""

    # ── Options & shorts ──────────────────────────────────────────── #
    opt            = options_data or {}
    put_call_ratio = opt.get("put_call_ratio", "")
    options_iv_avg = opt.get("options_iv_avg", "")
    short_ratio    = info.get("shortRatio",            "")
    short_pct_float= info.get("shortPercentOfFloat",   "")

    # ── Assemble row ──────────────────────────────────────────────── #
    row = {
        "run_id":         run_id,
        "ticker":         ticker.upper(),
        "run_date":       run_date,
        "run_timestamp":  run_ts,
        "mode":           mode,
        "provider":       provider,
        "current_price":  current_price,
        "recommendation": llm_result.get("recommendation", ""),
        "confidence":     llm_result.get("confidence", ""),
        **target_map,
        "ml_5d_direction":      ml.get("clf_5d_direction",    ""),
        "ml_5d_prob_up":        ml.get("clf_5d_prob_up",      ""),
        "ml_21d_direction":     ml.get("clf_21d_direction",   ""),
        "ml_21d_prob_up":       ml.get("clf_21d_prob_up",     ""),
        "ml_5d_expected_return":  ml.get("reg_5d_return_pct", ""),
        "ml_21d_expected_return": ml.get("reg_21d_return_pct",""),
        "score_fundamental":    score_fundamental if score_fundamental is not None else "",
        "score_technical":      score_technical   if score_technical   is not None else "",
        "score_valuation":      score_valuation   if score_valuation   is not None else "",
        "score_risk":           score_risk        if score_risk        is not None else "",
        "score_composite":      score_composite   if score_composite   is not None else "",
        "tech_verdict":         tech_verdict,
        "fundamental_verdict":  fundamental_verdict,
        "valuation_verdict":    valuation_verdict,
        "sentiment_verdict":    sentiment_verdict,
        "headline_1":           headline_1,
        "headline_2":           headline_2,
        "headline_3":           headline_3,
        "sentiment_score":      sentiment_score,
        "sentiment_label":      sentiment_label,
        "key_catalysts":        key_catalysts,
        "key_bull_case":        key_bull_case,
        "key_bear_case":        key_bear_case,
        "avg_volume_5d":        avg_volume_5d,
        "volume_ratio_20d":     volume_ratio_20d,
        "put_call_ratio":       put_call_ratio,
        "options_iv_avg":       options_iv_avg,
        "short_ratio":          short_ratio,
        "short_pct_float":      short_pct_float,
        # outcomes — blank at prediction time
        "actual_1w": "", "actual_2w": "", "actual_3w": "", "actual_1m": "",
        "actual_3m": "", "actual_6m": "", "actual_9m": "", "actual_12m": "",
        "dir_correct_1w": "", "dir_correct_2w": "", "dir_correct_3w": "", "dir_correct_1m": "",
        "dir_correct_3m": "", "dir_correct_6m": "", "dir_correct_9m": "", "dir_correct_12m": "",
        "pct_error_1w": "", "pct_error_2w": "", "pct_error_3w": "", "pct_error_1m": "",
        "pct_error_3m": "", "pct_error_6m": "", "pct_error_9m": "", "pct_error_12m": "",
        "ml_5d_correct": "", "ml_21d_correct": "", "rec_correct": "",
    }

    path = _csv_path(ticker)
    file_exists = os.path.isfile(path)

    # ── Same-day upsert: replace existing row for (run_date, provider) #
    # If a row already exists for today + this provider, overwrite it
    # with the latest analysis rather than appending a duplicate.
    # This ensures same-day reruns always reflect the most recent output.
    if file_exists:
        try:
            existing_df = pd.read_csv(path, dtype=str)
            mask = (
                (existing_df["run_date"].str.strip() == run_date) &
                (existing_df["provider"].str.strip() == str(provider).strip())
            )
            if mask.any():
                # Drop the stale row(s) and rewrite the whole file, then
                # fall through to append the fresh row below.
                existing_df = existing_df[~mask]
                existing_df.to_csv(path, index=False, columns=COLUMNS)
                file_exists = os.path.isfile(path)
        except Exception:
            pass  # malformed CSV edge-case: fall through and append normally

    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow(row)

    return path
