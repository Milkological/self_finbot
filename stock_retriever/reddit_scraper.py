# reddit_scraper.py
# Fetches posts and their full comment trees from a given subreddit using PRAW
# (Python Reddit API Wrapper). Only posts created within the last N days are returned.
#
# WHY PRAW?
#   PRAW is the official Reddit API wrapper. It handles OAuth2 authentication,
#   rate limiting, and pagination automatically, which keeps this code simple
#   and compliant with Reddit's API terms of service.
#
# WHY fetch from both 'new' and 'hot' feeds?
#   - 'new'  → captures the most recently posted content chronologically
#   - 'hot'  → captures highly upvoted/engaged content that may have been posted
#              days ago but is still actively commented on
#   Together they maximize coverage of relevant posts within the time window.
#
# WHY read-only mode?
#   We only need to read posts and comments — not post, vote, or message.
#   Read-only PRAW requires only client_id + client_secret (no username/password),
#   which is simpler and exposes fewer credentials.

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import praw

from config import get_reddit_credentials


def _is_within_days(created_utc: float, days: int) -> bool:
    """
    Determines whether a Reddit post falls within the lookback window.

    Compares the post's UTC creation timestamp against a cutoff time calculated
    as `now (UTC) minus days`. All comparisons are done in UTC to avoid
    timezone-related off-by-one errors.

    Args:
        created_utc: Unix epoch timestamp from the Reddit post (float seconds).
        days: Number of days to look back from now.

    Returns:
        True if the post was created within the last `days` days, else False.
    """
    # Calculate the cutoff as an aware datetime (timezone.utc prevents naive/aware mixing)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    # Convert the raw Unix timestamp to an aware UTC datetime for comparison
    post_time = datetime.fromtimestamp(created_utc, tz=timezone.utc)
    return post_time >= cutoff


def _collect_comments(submission) -> List[str]:
    """
    Collects all comment body text from a Reddit submission's comment tree.

    Reddit's API returns some comments as 'MoreComments' placeholder objects
    rather than actual Comment objects. replace_more() expands these placeholders
    by making additional API requests so the full comment tree is available.

    WHY limit=0 for replace_more?
      limit=0 expands ALL placeholders, ensuring no comments are missed.
      For very large threads this can be slow; you can reduce to limit=5 or
      limit=None (Reddit's default subset) for faster runs at the cost of coverage.

    Args:
        submission: A PRAW Submission object with a .comments attribute.

    Returns:
        A flat list of non-empty, non-deleted comment body strings.
    """
    # Expand all "load more comments" placeholders in the comment tree
    submission.comments.replace_more(limit=0)

    comments = []
    # .list() flattens the nested comment tree into a single iterable of Comment objects
    for comment in submission.comments.list():
        body = getattr(comment, "body", "")
        # Reddit marks deleted/removed comments with these sentinel strings — skip them
        if body and body not in ("[deleted]", "[removed]"):
            comments.append(body)
    return comments


def fetch_posts(subreddit_name: str, days: int = 7) -> List[Dict[str, Any]]:
    """
    Fetches posts from the specified subreddit that were created within the
    last `days` days, along with all their comments.

    The function deduplicates posts that appear in both the 'new' and 'hot'
    feeds using a set of seen post IDs.

    Args:
        subreddit_name: The subreddit name WITHOUT the 'r/' prefix
                        (e.g. "stockstobuytoday", not "r/stockstobuytoday").
        days: How many days back to search. Defaults to 7.

    Returns:
        A list of post dicts. Each dict has the shape:
        {
            "post_id":  str,   — Reddit's unique post ID
            "title":    str,   — Post title
            "body":     str,   — Post body text (empty string if link-only post)
            "date":     str,   — Creation date in "YYYY-MM-DD" (UTC)
            "comments": [str]  — All comment bodies from the thread
        }
    """
    # Load credentials from .env and initialise the PRAW Reddit client
    creds = get_reddit_credentials()

    # read_only=True means we do not need a username/password — safer and simpler
    # for a scrape-only use case.
    reddit = praw.Reddit(
        client_id=creds["client_id"],
        client_secret=creds["client_secret"],
        user_agent=creds["user_agent"],
    )

    subreddit = reddit.subreddit(subreddit_name)

    seen_ids: set = set()   # Tracks already-processed post IDs to skip duplicates
    results: List[Dict[str, Any]] = []

    # Fetch from 'new' (chronological) first, then 'hot' (engagement-ranked).
    # limit=500 is the maximum Reddit allows per API call.
    feeds = [
        subreddit.new(limit=500),
        subreddit.hot(limit=500),
    ]

    for feed in feeds:
        for submission in feed:
            # Skip duplicate posts that appeared in a previous feed
            if submission.id in seen_ids:
                continue
            seen_ids.add(submission.id)

            # Discard posts that fall outside the time window
            if not _is_within_days(submission.created_utc, days):
                # 'new' is sorted newest-first, so once we find an old post,
                # all subsequent posts in this feed will also be too old — stop early.
                break

            # Format the creation date as a readable string (UTC) for the CSV
            post_date = datetime.fromtimestamp(
                submission.created_utc, tz=timezone.utc
            ).strftime("%Y-%m-%d")

            print(f"  Fetching: [{post_date}] {submission.title[:70]}...")

            comments = _collect_comments(submission)

            results.append({
                "post_id": submission.id,
                "title":   submission.title,
                "body":    submission.selftext or "",   # selftext is None for link posts
                "date":    post_date,
                "comments": comments,
            })

    print(f"Fetched {len(results)} posts from r/{subreddit_name} (last {days} days).")
    return results
