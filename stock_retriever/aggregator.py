# aggregator.py
# Takes the list of raw posts from reddit_scraper and produces a structured
# dict that groups ticker mentions with their sentiment data.
#
# HOW it works:
#   For every text segment (title, body, each comment) in every post:
#     1. extract_tickers()   → find all tickers mentioned in that text
#     2. analyze_sentiment() → score the sentiment of that text
#     3. Record the mention under each ticker found
#
# WHY analyze at the text-segment level (not post level)?
#   A single post may say "I love $AAPL but $TSLA is terrible."
#   Scoring the whole post as one unit would produce a misleading mixed sentiment
#   for both tickers. By analyzing each sentence/comment individually, the
#   sentiment is more accurately attributed to the specific ticker being discussed.
#
# NOTE on comment length cap:
#   Comments are stored with a max of 300 characters per entry to keep the
#   CSV cells readable and file size manageable while still capturing the gist.

from typing import Any, Dict, List

from sentiment_analyzer import analyze_sentiment
from ticker_extractor import extract_tickers

# Maximum number of characters to store per comment snippet in the CSV.
# Long Reddit comments often include off-topic rambling; the first 300 chars
# usually capture the key sentiment statement.
MAX_COMMENT_LENGTH = 300


def _get_all_texts(post: Dict[str, Any]) -> List[str]:
    """
    Extracts all text segments from a post as a flat list for individual analysis.

    WHY split into segments instead of concatenating everything?
      Each segment (title, body, comment) may contain different tickers with
      different sentiments. Keeping them separate allows more accurate sentiment
      attribution per ticker per segment.

    Args:
        post: A post dict as returned by reddit_scraper.fetch_posts().

    Returns:
        A list of non-empty strings: [title, body, comment1, comment2, ...]
        Empty strings are filtered out to skip wasted processing.
    """
    # Start with title and body (always present, may be empty string)
    texts = [post.get("title", ""), post.get("body", "")]
    # Extend with every comment in the thread
    texts.extend(post.get("comments", []))
    # Filter out empty/whitespace-only strings to avoid pointless analysis
    return [t for t in texts if t and t.strip()]


def aggregate_by_ticker(posts: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """
    Processes all posts and their comments, extracts tickers from each text
    segment, scores the segment's sentiment, and accumulates the results
    grouped by ticker symbol.

    Args:
        posts: A list of post dicts from reddit_scraper.fetch_posts().
               Can be an empty list (returns empty dict).

    Returns:
        A dict keyed by ticker symbol. Each value is a dict:
        {
            "dates":             set of "YYYY-MM-DD" strings,
            "total":             int  — total number of text segments mentioning this ticker,
            "positive_count":    int  — segments scored as positive,
            "positive_comments": list[str] — the text of positive segments (capped length),
            "negative_count":    int  — segments scored as negative,
            "negative_comments": list[str] — the text of negative segments (capped length),
        }

        Example:
        {
            "AAPL": {
                "dates": {"2026-05-28", "2026-05-30"},
                "total": 5,
                "positive_count": 3,
                "positive_comments": ["Great earnings...", ...],
                "negative_count": 1,
                "negative_comments": ["Way overvalued..."],
            }
        }
    """
    # Plain dict — keys are ticker strings, values are accumulator sub-dicts.
    aggregated: Dict[str, Dict] = {}

    for post in posts:
        post_date = post.get("date", "unknown")
        texts = _get_all_texts(post)

        for text in texts:
            # Step 1: Find all ticker symbols mentioned in this text segment
            tickers = extract_tickers(text)
            if not tickers:
                # No tickers in this segment — nothing to record, move on
                continue

            # Step 2: Score the sentiment of this segment once.
            # We reuse the same sentiment result for all tickers found in the
            # same segment (since the sentiment applies to the whole text).
            sentiment = analyze_sentiment(text)
            label = sentiment["label"]

            # Step 3: Accumulate data for each ticker found in this segment
            for ticker in tickers:
                # Initialise the accumulator dict the first time we see this ticker
                if ticker not in aggregated:
                    aggregated[ticker] = {
                        "dates":             set(),
                        "total":             0,
                        "positive_count":    0,
                        "positive_comments": [],
                        "negative_count":    0,
                        "negative_comments": [],
                    }

                entry = aggregated[ticker]

                # Record the date this ticker was seen (set deduplicates automatically)
                entry["dates"].add(post_date)

                # Increment the overall mention counter
                entry["total"] += 1

                # Store a capped-length snippet in the appropriate sentiment bucket.
                # Neutral mentions are counted in "total" but not stored in either
                # comment list, since they add no directional signal.
                snippet = text[:MAX_COMMENT_LENGTH]
                if label == "positive":
                    entry["positive_count"] += 1
                    entry["positive_comments"].append(snippet)
                elif label == "negative":
                    entry["negative_count"] += 1
                    entry["negative_comments"].append(snippet)

    return aggregated
