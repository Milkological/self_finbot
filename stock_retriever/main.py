# main.py
# Entry point for the Reddit stock ticker scraper pipeline.
#
# HOW the pipeline works (3 steps):
#   1. FETCH   — Connect to Reddit via PRAW and download posts + comments
#                from r/stockstobuytoday created in the last 7 days.
#   2. ANALYSE — For each text segment (title, body, comment), extract
#                stock tickers and score the sentiment. Aggregate by ticker.
#   3. OUTPUT  — Write the aggregated results to a dated CSV file.
#
# HOW to run:
#   1. pip install -r requirements.txt
#   2. Copy .env.example to .env and fill in your Reddit API credentials
#   3. python main.py
#
# The output CSV is written to the same directory as this script,
# named "reddit_stocks_YYYY-MM-DD.csv" (today's date).

import sys
from datetime import date

from aggregator import aggregate_by_ticker
from csv_writer import write_csv
from reddit_scraper import fetch_posts

# ---------------------------------------------------------------------------
# Configuration — change these constants to adjust behaviour without
# modifying any pipeline logic.
# ---------------------------------------------------------------------------

# The subreddit to scrape (without the "r/" prefix)
SUBREDDIT = "stockstobuytoday"

# How many days back to include posts (7 = last full week)
DAYS_BACK = 7

# Output file name uses today's date so each run produces a unique, dated file
OUTPUT_FILE = f"reddit_stocks_{date.today().isoformat()}.csv"


def main() -> None:
    """
    Orchestrates the full scrape → analyse → export pipeline.

    Prints progress to stdout at each stage so the user can follow along
    without having to inspect log files. Exits with a non-zero code on
    fatal errors (missing credentials, no posts found) so this script can
    be used safely in automated/scheduled runs.
    """
    print("--- Reddit Stock Ticker Scraper ---")
    print(f"Subreddit : r/{SUBREDDIT}")
    print(f"Lookback  : last {DAYS_BACK} days (from {date.today().isoformat()})")
    print(f"Output    : {OUTPUT_FILE}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Fetch posts and comments from Reddit
    # ------------------------------------------------------------------
    print("[1/3] Fetching posts from Reddit...")
    try:
        posts = fetch_posts(SUBREDDIT, days=DAYS_BACK)
    except EnvironmentError as e:
        # Raised by config.get_reddit_credentials() when .env is missing/incomplete.
        # Print a clear, actionable error rather than a raw traceback.
        print(f"\nCONFIGURATION ERROR: {e}")
        print("Please copy .env.example to .env and fill in your Reddit API credentials.")
        sys.exit(1)

    if not posts:
        # No posts found in the window — could be a new/empty subreddit or
        # an API issue. Exit cleanly rather than writing an empty CSV.
        print("No posts found within the time window. Exiting.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 2: Extract tickers and aggregate sentiment
    # ------------------------------------------------------------------
    print(f"\n[2/3] Extracting tickers and analysing sentiment ({len(posts)} posts)...")
    aggregated = aggregate_by_ticker(posts)

    if not aggregated:
        # Posts were found but no recognisable tickers were extracted.
        print("No stock tickers detected in the fetched posts. Exiting.")
        sys.exit(0)

    print(f"  Found {len(aggregated)} unique ticker(s).")

    # ------------------------------------------------------------------
    # Step 3: Write results to CSV
    # ------------------------------------------------------------------
    print(f"\n[3/3] Writing CSV to {OUTPUT_FILE}...")
    write_csv(aggregated, OUTPUT_FILE)

    print("\nDone.")


# Standard Python entry-point guard — allows this module to be imported
# in tests without triggering the pipeline automatically.
if __name__ == "__main__":
    main()
