# sentiment_analyzer.py
# Classifies the sentiment of a text string as positive, negative, or neutral.
#
# WHY VADER?
#   VADER (Valence Aware Dictionary and sEntiment Reasoner) is purpose-built
#   for short social media text. It handles:
#     - Slang and abbreviations  ("lol", "omg")
#     - ALL CAPS emphasis        ("GREAT stock!!!")
#     - Punctuation boosting     ("amazing!!!" scores higher than "amazing")
#     - Emoticons                (":)" adds positive signal)
#   This makes it a much better fit for Reddit comments than general-purpose
#   NLP models, and it runs entirely offline with no API cost.
#
# THRESHOLD REFERENCE:
#   Recommended by Hutto & Gilbert (2014), the VADER authors:
#     compound >= +0.05  → positive
#     compound <= -0.05  → negative
#     otherwise          → neutral

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Module-level singleton analyzer.
# WHY at module level?
#   Initializing SentimentIntensityAnalyzer loads the VADER lexicon from disk.
#   Creating it once at import time avoids redundant I/O on every function call,
#   which matters when analyzing thousands of Reddit comments.
# ---------------------------------------------------------------------------
_analyzer = SentimentIntensityAnalyzer()

# Compound score thresholds from the VADER paper.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def analyze_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of the given text using VADER and returns
    a human-readable label alongside the raw compound score.

    The compound score is a normalized, weighted sum of all individual word
    valence scores, ranging from -1.0 (most negative) to +1.0 (most positive).

    Args:
        text: A string to analyze (e.g. a Reddit comment or post body).

    Returns:
        A dict with two keys:
          - "label": "positive", "negative", or "neutral"
          - "score": float compound score, rounded to 4 decimal places

        For empty or non-string input, returns {"label": "neutral", "score": 0.0}
        rather than raising, so a single bad comment doesn't crash the pipeline.
    """
    # Guard: treat missing/invalid input as neutral rather than propagating errors.
    # This is safe because the caller (aggregator) checks the label, not the score.
    if not text or not isinstance(text, str):
        return {"label": "neutral", "score": 0.0}

    # polarity_scores() returns a dict: {"neg": float, "neu": float, "pos": float, "compound": float}
    # We only use "compound" because it combines all three dimensions into one score.
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    # Apply the published threshold rules to assign a human-readable label.
    if compound >= POSITIVE_THRESHOLD:
        label = "positive"
    elif compound <= NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return {"label": label, "score": round(compound, 4)}
