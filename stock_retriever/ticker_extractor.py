# ticker_extractor.py
# Extracts stock ticker symbols from free-form text such as Reddit post titles,
# post bodies, and comments.
#
# WHY two formats?
#   Reddit users write tickers in two common ways:
#     1. Dollar-prefixed:  $AAPL, $TSLA  (explicit intent signal)
#     2. Bare uppercase:   AAPL, TSLA    (common in body text)
#   Supporting both maximises recall without sacrificing too much precision.
#
# WHY a blocklist?
#   The bare uppercase pattern (2-5 capital letters) would match common English
#   words like "I", "THE", "CEO", "BUY", "FOR". The blocklist filters these out
#   to reduce false positives in the final CSV.

import re
from typing import List

# ---------------------------------------------------------------------------
# Blocklist of uppercase words that are NOT stock tickers.
# Includes: common English words, Reddit slang, and finance/market jargon
# that happen to match the 2-5 uppercase letter pattern.
# Dollar-prefixed matches ($BUY) are NOT filtered — the $ already signals
# the user intended a ticker reference.
# ---------------------------------------------------------------------------
TICKER_BLOCKLIST = {
    # Single/two-letter articles, pronouns, prepositions
    "A", "I", "IT", "BE", "DO", "GO", "AM", "IS", "IN", "ON", "AT", "BY",
    "TO", "OF", "OR", "AS", "AN", "UP", "US", "WE", "HE", "ME", "MY",
    "SO", "NO", "OK", "IF",
    # Common three-letter words
    "THE", "AND", "FOR", "ARE", "NOT", "BUT", "ALL", "CAN", "GET", "GOT",
    "HAD", "HAS", "HIM", "HIS", "HOW", "ITS", "LET", "MAN", "NEW", "NOW",
    "OLD", "ONE", "OUR", "OUT", "OWN", "PUT", "SAY", "SEE", "SET", "TWO",
    "WAS", "WAY", "WHO", "WHY", "WIN", "YET", "YOU", "ANY", "DAY", "DID",
    "END", "FEW", "HER", "MAY", "OFF", "USE", "TOO", "TOP",
    # Finance / market jargon
    "CEO", "CFO", "COO", "CTO", "IPO", "ETF", "ATH", "ATL", "EPS", "EMA",
    "SMA", "RSI", "MACD", "NYSE", "SEC", "USD", "USA", "GDP", "CPI",
    "FED", "IMF", "ECB",
    # Reddit trading slang
    "BUY", "SELL", "HOLD", "LONG", "SHORT", "YOLO", "MOON", "PUMP", "DUMP",
    "BULL", "BEAR", "FOMO", "FUD", "DD", "TD", "TLDR", "IMO", "IMHO",
    "TBH", "FYI", "LOL", "IRA", "EDIT", "NEWS", "POST", "LINK", "SALE",
}

# ---------------------------------------------------------------------------
# Compiled regex patterns (compiled once at module level for performance).
# ---------------------------------------------------------------------------

# Pattern 1 — dollar-prefixed ticker: $AAPL, $TSLA
#   \$     — literal dollar sign
#   [A-Z]{1,5} — 1 to 5 uppercase ASCII letters
#   \b     — word boundary (so $AAPLL doesn't match as $AAPL)
DOLLAR_TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b')

# Pattern 2 — bare uppercase word: AAPL, TSLA (minimum 2 letters to skip "A", "I")
#   \b     — word boundary on both sides (not a substring of a longer word)
#   [A-Z]{2,5} — 2 to 5 uppercase ASCII letters
BARE_TICKER_PATTERN = re.compile(r'\b([A-Z]{2,5})\b')


def extract_tickers(text: str) -> List[str]:
    """
    Scans the given text for stock ticker symbols and returns a deduplicated,
    alphabetically sorted list of matches.

    Two formats are recognised:
      - Dollar-prefixed: $AAPL  (blocklist is NOT applied — $ signals intent)
      - Bare uppercase:  AAPL   (blocklist IS applied to reduce false positives)

    Args:
        text: A raw string to scan. Can be a post title, body, or comment.

    Returns:
        A sorted list of unique ticker strings, e.g. ["AAPL", "TSLA"].
        Returns an empty list for empty, None, or non-string input.
    """
    # Guard: return empty list for falsy or non-string input to avoid
    # AttributeError / TypeError propagating up the pipeline.
    if not text or not isinstance(text, str):
        return []

    found: set = set()

    # --- Pass 1: dollar-prefixed tickers ---
    # These are trusted as intentional ticker references, so we skip the blocklist.
    for match in DOLLAR_TICKER_PATTERN.finditer(text):
        found.add(match.group(1))  # group(1) captures only the letters, not the $

    # --- Pass 2: bare uppercase words ---
    # Apply the blocklist here because plain uppercase words have high false-positive rate.
    for match in BARE_TICKER_PATTERN.finditer(text):
        ticker = match.group(1)
        if ticker not in TICKER_BLOCKLIST:
            found.add(ticker)

    # Sort for deterministic, readable output
    return sorted(found)
