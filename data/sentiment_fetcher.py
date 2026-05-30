"""
data/sentiment_fetcher.py — LLM-based headline sentiment scoring.

Sends a batch of news headlines to the configured LLM provider(s) and
returns a normalised sentiment score, label, and brief summary.

Falls back to a neutral stub when no LLM credentials are configured
or when every LLM call fails.
"""

import json
from json_repair import repair_json

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
# Public API
# ------------------------------------------------------------------ #

_NEUTRAL: dict = {"overall_score": 0.0, "label": "NEUTRAL", "summary": ""}


def score_headlines(headlines: list[dict]) -> dict:
    """
    Score a list of news headline dicts (each should have a 'title' key).

    Tries Azure first, then Google.  Returns a neutral stub if both
    providers are unavailable or all calls fail.

    Returns
    -------
    {
        "overall_score": float  # -1.0 .. +1.0
        "label":         str    # POSITIVE | NEUTRAL | NEGATIVE
        "summary":       str    # one-sentence explanation
    }
    """
    if not LLM_ENABLED or not headlines:
        return _NEUTRAL

    prompt = _build_prompt(headlines)

    if AZURE_ENABLED:
        result = _call_azure(prompt)
        if result:
            return result

    if GOOGLE_ENABLED:
        result = _call_google(prompt)
        if result:
            return result

    return _NEUTRAL
