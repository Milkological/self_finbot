"""
data/sentiment_fetcher.py — Headline sentiment scoring.

Scoring order:
  1. Configured LLM provider(s) — Azure, then Google (dormant when no keys).
  2. VADER — a fully-offline lexicon scorer (vaderSentiment). This is what
     makes the pipeline work with ZERO API keys: without it, headline
     sentiment was a permanent 0.0 stub in --no-llm mode.
  3. Neutral stub — only when there are no headlines to score at all.

The return shape is identical across all three paths so the ML feature,
the sentiment history archive, and the display never need to care which
scorer ran.
"""

import json
import logging
from json_repair import repair_json

logger = logging.getLogger(__name__)

from config import (
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    AZURE_ENABLED,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    GOOGLE_ENABLED,
    LLM_ENABLED,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS_AZURE,
    LLM_MAX_TOKENS_GOOGLE,
)


# ------------------------------------------------------------------ #
# Prompt
# ------------------------------------------------------------------ #

_SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. "
    "Given a list of news headlines about a stock, rate the overall market "
    "sentiment expressed in those headlines on a scale from -1.0 (very bearish) "
    "to +1.0 (very bullish), with 0.0 being neutral. "
    "Return ONLY a JSON object with exactly these three keys:\n"
    "  overall_score  : float between -1.0 and 1.0\n"
    "  label          : one of POSITIVE, NEUTRAL, NEGATIVE\n"
    "  summary        : one concise sentence explaining the sentiment\n"
    "Do NOT include any other text outside the JSON object."
)


def _build_prompt(headlines: list[dict]) -> str:
    if not headlines:
        return "No headlines provided."
    lines = "\n".join(
        f"- {h.get('title', '')}" for h in headlines[:20]
    )
    return f"Analyse the following news headlines:\n\n{lines}"


# ------------------------------------------------------------------ #
# LLM callers
# ------------------------------------------------------------------ #

def _call_azure(prompt: str) -> dict | None:
    """Call Azure OpenAI; return parsed dict or None on failure."""
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content or ""
        return _parse(raw)
    except Exception:
        return None


def _call_google(prompt: str) -> dict | None:
    """Call Google Gemini; return parsed dict or None on failure."""
    try:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=GOOGLE_API_KEY)
        resp = client.models.generate_content(
            model=GOOGLE_MODEL,
            contents=f"{_SYSTEM_PROMPT}\n\n{prompt}",
            config=gtypes.GenerateContentConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=256,
                safety_settings=[
                    gtypes.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
                    gtypes.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
                    gtypes.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    gtypes.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ],
            ),
        )
        raw = resp.text or ""
        return _parse(raw)
    except Exception:
        return None


def _parse(raw: str) -> dict | None:
    """Extract and validate the JSON dict from LLM output."""
    try:
        data = json.loads(repair_json(raw))
        score = float(data.get("overall_score", 0.0))
        score = max(-1.0, min(1.0, score))
        label = str(data.get("label", "NEUTRAL")).upper()
        if label not in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
            label = "NEUTRAL"
        return {
            "overall_score": score,
            "label":         label,
            "summary":       str(data.get("summary", "")),
        }
    except Exception:
        return None


# ------------------------------------------------------------------ #
# VADER — offline lexicon fallback (no API keys required)
# ------------------------------------------------------------------ #

# Lazy module-level singleton: the lexicon loads from disk once. None when
# vaderSentiment isn't installed, in which case we degrade to the neutral stub.
_vader = None
_vader_tried = False

# VADER's published per-text thresholds (Hutto & Gilbert 2014) for counting
# individual positive/negative headlines in the summary.
_POS_T, _NEG_T = 0.05, -0.05
# The AGGREGATE label uses a slightly wider band: financial headlines are terse
# and a near-zero average shouldn't read as a directional signal.
_AGG_T = 0.15


# Finance-domain lexicon augmentation. VADER's stock lexicon is tuned for
# social media and misreads financial jargon ("beats", "miss", "downgrade"),
# so we inject valence for the terms that actually move headlines. Values are
# on VADER's ~[-4, +4] scale and updated into the analyzer's lexicon at init.
_FINANCE_LEXICON = {
    # strongly bullish
    "beats": 2.6, "beat": 2.2, "crushes": 3.2, "crushed": 2.8, "surge": 3.0,
    "surges": 3.0, "surging": 3.0, "soar": 3.2, "soars": 3.2, "soaring": 3.2,
    "rally": 2.4, "rallies": 2.4, "jumps": 2.4, "jumped": 2.4, "upgrade": 2.8,
    "upgraded": 2.8, "upgrades": 2.8, "outperform": 2.6, "record": 2.0,
    "breakthrough": 2.6, "raises": 1.8, "raised": 1.8, "tops": 2.0, "topped": 2.0,
    "bullish": 2.8, "buyback": 1.8, "dividend": 1.2, "profit": 1.6, "profitable": 2.0,
    "growth": 1.4, "expansion": 1.4, "momentum": 1.2, "accelerate": 1.6,
    # strongly bearish
    "miss": -2.4, "misses": -2.4, "missed": -2.4, "plunge": -3.2, "plunges": -3.2,
    "plummet": -3.2, "plummets": -3.2, "tumble": -2.8, "tumbles": -2.8,
    "slump": -2.6, "slumps": -2.6, "downgrade": -2.8, "downgraded": -2.8,
    "downgrades": -2.8, "underperform": -2.4, "cuts": -2.0, "cut": -1.8,
    "slashes": -2.6, "slashed": -2.6, "warns": -2.0, "warning": -2.0,
    "bearish": -2.8, "bankruptcy": -3.6, "bankrupt": -3.6, "default": -2.8,
    "probe": -1.8, "lawsuit": -2.0, "investigation": -1.8, "fraud": -3.4,
    "layoffs": -2.2, "layoff": -2.2, "recall": -2.0, "halts": -1.8, "halt": -1.8,
    "loss": -1.8, "losses": -1.8, "decline": -1.6, "declines": -1.6, "weak": -1.8,
    "sinks": -2.6, "sink": -2.6, "crash": -3.2, "selloff": -2.4, "dilution": -1.8,
}


def _get_vader():
    global _vader, _vader_tried
    if not _vader_tried:
        _vader_tried = True
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            analyzer.lexicon.update(_FINANCE_LEXICON)   # domain calibration
            _vader = analyzer
        except Exception as exc:
            logger.debug("VADER unavailable (%s) — sentiment will be neutral.", exc)
            _vader = None
    return _vader


def _call_vader(headlines: list[dict]) -> dict | None:
    """Score each headline title with VADER; average the compound scores."""
    analyzer = _get_vader()
    if analyzer is None:
        return None
    titles = [str(h.get("title", "")).strip() for h in headlines[:20]]
    titles = [t for t in titles if t]
    if not titles:
        return None

    compounds = [analyzer.polarity_scores(t)["compound"] for t in titles]
    avg = sum(compounds) / len(compounds)
    n_pos = sum(1 for c in compounds if c >= _POS_T)
    n_neg = sum(1 for c in compounds if c <= _NEG_T)

    if avg >= _AGG_T:
        label = "POSITIVE"
    elif avg <= -_AGG_T:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "overall_score": round(max(-1.0, min(1.0, avg)), 4),
        "label":         label,
        "summary":       (f"VADER (offline): {len(titles)} headlines, "
                          f"{n_pos} positive / {n_neg} negative (avg {avg:+.2f})."),
    }


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

_NEUTRAL: dict = {"overall_score": 0.0, "label": "NEUTRAL", "summary": ""}


def score_headlines(headlines: list[dict]) -> dict:
    """
    Score a list of news headline dicts (each should have a 'title' key).

    Tries the configured LLM provider(s) first (Azure, then Google), then
    the offline VADER scorer, then a neutral stub only when there are no
    headlines. This means real sentiment is produced with zero API keys.

    Returns
    -------
    {
        "overall_score": float  # -1.0 .. +1.0
        "label":         str    # POSITIVE | NEUTRAL | NEGATIVE
        "summary":       str    # one-sentence explanation
    }
    """
    if not headlines:
        return _NEUTRAL

    # 1. LLM providers (dormant without keys)
    if LLM_ENABLED:
        prompt = _build_prompt(headlines)
        if AZURE_ENABLED:
            result = _call_azure(prompt)
            if result:
                return result
        if GOOGLE_ENABLED:
            result = _call_google(prompt)
            if result:
                return result

    # 2. Offline VADER fallback — the LLM-free path
    result = _call_vader(headlines)
    if result:
        return result

    # 3. Nothing worked (VADER not installed and no LLM)
    return _NEUTRAL
