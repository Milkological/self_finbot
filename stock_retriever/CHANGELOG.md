# Changelog

All notable changes to this project will be recorded in this file.

---

## [1.0.0] - 2026-05-31

### Added

- `requirements.txt` — declares all Python package dependencies (`praw`, `vaderSentiment`, `python-dotenv`, `pytest`)
- `.env.example` — template showing the three Reddit API credential variables required, with step-by-step instructions for obtaining them
- `config.py` — loads Reddit API credentials from a `.env` file using `python-dotenv`; raises a clear `EnvironmentError` if any variable is missing
- `ticker_extractor.py` — extracts stock ticker symbols from free-form text using regex; supports both dollar-prefixed (`$AAPL`) and plain uppercase (`TSLA`) formats; filters out common English words and finance jargon via a blocklist
- `sentiment_analyzer.py` — wraps the VADER `SentimentIntensityAnalyzer` to classify text as `positive`, `negative`, or `neutral` using published compound score thresholds
- `reddit_scraper.py` — authenticates with Reddit via PRAW in read-only mode; fetches posts and full comment trees from a given subreddit; filters to posts created within a configurable lookback window (default: 7 days); deduplicates across `new` and `hot` feeds
- `aggregator.py` — processes all fetched posts and comments; extracts tickers and scores sentiment per text segment; groups results by ticker into a structured dict with date, total count, positive count/comments, and negative count/comments
- `csv_writer.py` — serialises the aggregated ticker data to a UTF-8 CSV file; one row per ticker sorted alphabetically; multi-value fields (dates, comments) joined with ` | `
- `main.py` — orchestrates the full pipeline (fetch → analyse → write); outputs a dated CSV file (`reddit_stocks_YYYY-MM-DD.csv`); prints step-by-step progress to console
- `tests/test_ticker_extractor.py` — 15 unit tests covering dollar-prefixed tickers, bare uppercase tickers, blocklist filtering, deduplication, sorting, edge cases (empty string, None, lowercase)
- `tests/test_sentiment_analyzer.py` — 15 unit tests covering positive/negative/neutral classification, return value structure, score range validation, edge cases
- `tests/test_aggregator.py` — 14 unit tests covering ticker grouping, sentiment counting, date collection, result structure, comment snippet length cap
- `tests/test_csv_writer.py` — 14 unit tests covering file creation, header correctness, row count, field values, alphabetical ordering, multi-value separator joining, empty input handling
- `README.md` — full project documentation including setup guide, usage instructions, output format description, and configuration reference
