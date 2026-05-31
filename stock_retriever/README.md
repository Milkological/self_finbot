# Reddit Stock Ticker Scraper

Scrapes posts and comments from a subreddit, extracts stock ticker symbols, analyses sentiment, and writes the results to a CSV file. Built entirely with free, offline tools — no paid APIs required.

---

## How It Works

The pipeline runs in three stages:

```
Reddit (PRAW)  →  Ticker + Sentiment Analysis  →  CSV Output
```

1. **Fetch** — Connects to Reddit via PRAW and downloads posts and their full comment trees from `r/stockstobuytoday`. Only posts created within the last 7 days are included. Both the `new` (chronological) and `hot` (trending) feeds are scraped to maximise coverage.

2. **Analyse** — Each text segment (post title, post body, each comment) is scanned independently:
   - **Ticker extraction** — Detects stock tickers in two formats: dollar-prefixed (`$AAPL`) and plain uppercase (`TSLA`). A blocklist filters out common English words and finance jargon (e.g. `BUY`, `CEO`, `IPO`) that would otherwise be false positives.
   - **Sentiment scoring** — VADER scores each segment as `positive`, `negative`, or `neutral` based on the compound score of all words in the text.
   - Results are **aggregated by ticker** — all mentions of `AAPL` across all posts are combined into one entry.

3. **Output** — The aggregated data is written to a dated CSV file (`reddit_stocks_YYYY-MM-DD.csv`) with one row per ticker.

---

## Output CSV Format

| Column | Description |
|---|---|
| `Ticker` | Stock ticker symbol (e.g. `AAPL`) |
| `Date Mentioned` | All dates the ticker appeared, joined by ` \| ` |
| `Total Mentions` | Total number of text segments that mentioned this ticker |
| `Positive Count` | Number of those segments scored as positive |
| `Positive Comments` | The text of positive segments, joined by ` \| ` |
| `Negative Count` | Number of those segments scored as negative |
| `Negative Comments` | The text of negative segments, joined by ` \| ` |

Rows are sorted alphabetically by ticker. Multi-value fields (dates, comments) are joined with ` | ` inside a single cell.

---

## Project Structure

```
stock_retriever/
├── main.py                  # Entry point — runs the full pipeline
├── config.py                # Loads Reddit API credentials from .env
├── reddit_scraper.py        # Fetches posts and comments via PRAW
├── ticker_extractor.py      # Extracts ticker symbols using regex
├── sentiment_analyzer.py    # Scores sentiment using VADER
├── aggregator.py            # Groups mentions and sentiment by ticker
├── csv_writer.py            # Writes results to a CSV file
├── requirements.txt         # Python package dependencies
├── .env.example             # Template for your Reddit credentials
└── tests/
    ├── test_ticker_extractor.py
    ├── test_sentiment_analyzer.py
    ├── test_aggregator.py
    └── test_csv_writer.py
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create your Reddit API credentials

1. Log into Reddit and go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Click **"Create Another App"**
3. Set the type to **"script"**
4. Fill in any name (e.g. `stock_scraper`) and set the redirect URI to `http://localhost:8080`
5. Click **"Create app"**
6. Copy the **client ID** (the short string shown directly under your app name) and the **secret**

### 3. Create your `.env` file

Copy the example file and fill in your credentials:

```bash
copy .env.example .env
```

Open `.env` and replace the placeholder values:

```
REDDIT_CLIENT_ID=your_actual_client_id
REDDIT_CLIENT_SECRET=your_actual_client_secret
REDDIT_USER_AGENT=script:stock_scraper:v1.0 (by /u/your_reddit_username)
```

> **Important:** Never commit your `.env` file to source control. It contains your private API credentials.

---

## Usage

```bash
python main.py
```

The script prints progress to the console as it runs:

```
--- Reddit Stock Ticker Scraper ---
Subreddit : r/stockstobuytoday
Lookback  : last 7 days (from 2026-05-31)
Output    : reddit_stocks_2026-05-31.csv

[1/3] Fetching posts from Reddit...
  Fetching: [2026-05-30] AAPL looking strong ahead of earnings...
  Fetching: [2026-05-29] Why I'm buying TSLA this week...
  ...
Fetched 42 posts from r/stockstobuytoday (last 7 days).

[2/3] Extracting tickers and analysing sentiment (42 posts)...
  Found 18 unique ticker(s).

[3/3] Writing CSV to reddit_stocks_2026-05-31.csv...
CSV written to: C:\...\reddit_stocks_2026-05-31.csv
  18 unique tickers recorded.

Done.
```

The output CSV is saved in the same folder as `main.py`.

---

## Running the Tests

All unit tests run without Reddit credentials (no network connection needed):

```bash
python -m pytest tests/ -v
```

Expected output: **58 tests passed**.

---

## Configuration

To change the target subreddit or lookback window, edit the constants at the top of `main.py`:

```python
SUBREDDIT = "stockstobuytoday"   # subreddit name, without "r/"
DAYS_BACK = 7                    # how many days back to search
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `praw` | Reddit API wrapper — handles authentication, rate limiting, pagination |
| `vaderSentiment` | Sentiment analysis tuned for social media text |
| `python-dotenv` | Loads credentials from `.env` file into environment variables |
| `pytest` | Test framework for running the unit tests |
