# tests/test_aggregator.py
# Unit tests for aggregator.aggregate_by_ticker().
#
# WHAT is being tested:
#   - Empty input returns an empty dict
#   - Extracted tickers appear as keys in the result
#   - Total, positive, and negative counts are correctly tallied
#   - Both tickers from a multi-ticker post each get their own entry
#   - Dates from multiple posts are collected in the "dates" set
#   - The "dates" field is a set (not a list)
#   - Positive/negative comment snippets are stored in the right lists
#   - Every result entry has all required keys
#
# WHY mock data instead of real Reddit posts?
#   Tests must be runnable offline and without credentials. Using fixture dicts
#   that match the exact shape of reddit_scraper.fetch_posts() output allows
#   thorough testing without any network dependency.

import sys
import os

# Add the parent directory (stock_retriever/) to sys.path so that
# the test can import aggregator without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from aggregator import aggregate_by_ticker


# ---------------------------------------------------------------------------
# Fixtures — reusable mock post data (shape matches reddit_scraper output)
# ---------------------------------------------------------------------------

@pytest.fixture
def positive_aapl_post():
    """A single post with unmistakably positive language about $AAPL."""
    return [{
        "post_id": "abc123",
        "title": "$AAPL absolutely incredible earnings today",
        "body": "Best quarterly results ever, love this company!",
        "date": "2026-05-28",
        "comments": [
            "Amazing results! Buying more $AAPL for sure.",
            "This is why I hold long term — fantastic company.",
        ],
    }]


@pytest.fixture
def negative_tsla_post():
    """A single post with unmistakably negative language about TSLA."""
    return [{
        "post_id": "def456",
        "title": "TSLA is terribly overvalued and disappointing",
        "body": "Horrible miss on deliveries, I hate this stock now",
        "date": "2026-05-29",
        "comments": [
            "Worst stock ever, avoid TSLA at all costs.",
            "Terrible management decisions, sell everything.",
        ],
    }]


@pytest.fixture
def multi_ticker_post():
    """A post mentioning two different tickers — both should appear in results."""
    return [{
        "post_id": "ghi789",
        "title": "Comparing $AAPL vs MSFT this quarter",
        "body": "Both are strong but $AAPL leads on revenue",
        "date": "2026-05-30",
        "comments": [],
    }]


@pytest.fixture
def multi_day_nvda_posts():
    """Two posts on different days both mentioning NVDA — dates should both be recorded."""
    return [
        {
            "post_id": "p1",
            "title": "$NVDA earnings beat all expectations",
            "body": "Great quarter for NVDA, buying more",
            "date": "2026-05-27",
            "comments": [],
        },
        {
            "post_id": "p2",
            "title": "Should I buy NVDA at this price?",
            "body": "NVDA still looks strong after the run-up",
            "date": "2026-05-30",
            "comments": [],
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAggregateByTicker:
    """Tests for the aggregate_by_ticker() function."""

    def test_empty_posts_returns_empty_dict(self):
        # No posts — nothing to aggregate — must return an empty dict.
        result = aggregate_by_ticker([])
        assert result == {}

    def test_ticker_key_present_in_result(self, positive_aapl_post):
        # AAPL is mentioned in the post — it must appear as a key in the result.
        result = aggregate_by_ticker(positive_aapl_post)
        assert "AAPL" in result

    def test_total_count_at_least_one(self, positive_aapl_post):
        # The ticker was mentioned — total must be >= 1.
        result = aggregate_by_ticker(positive_aapl_post)
        assert result["AAPL"]["total"] >= 1

    def test_positive_sentiment_increments_positive_count(self, positive_aapl_post):
        # The post is clearly positive — positive_count should be > 0 for AAPL.
        result = aggregate_by_ticker(positive_aapl_post)
        assert result["AAPL"]["positive_count"] >= 1

    def test_negative_sentiment_increments_negative_count(self, negative_tsla_post):
        # The post is clearly negative — negative_count should be > 0 for TSLA.
        result = aggregate_by_ticker(negative_tsla_post)
        assert result["TSLA"]["negative_count"] >= 1

    def test_two_tickers_from_one_post_both_present(self, multi_ticker_post):
        # A post mentioning both AAPL and MSFT must produce entries for both.
        result = aggregate_by_ticker(multi_ticker_post)
        assert "AAPL" in result
        assert "MSFT" in result

    def test_dates_from_multiple_posts_both_recorded(self, multi_day_nvda_posts):
        # NVDA appears on two different days — both dates must be in the set.
        result = aggregate_by_ticker(multi_day_nvda_posts)
        assert "2026-05-27" in result["NVDA"]["dates"]
        assert "2026-05-30" in result["NVDA"]["dates"]

    def test_dates_field_is_a_set(self, multi_day_nvda_posts):
        # The "dates" field must be a set so deduplication is automatic.
        result = aggregate_by_ticker(multi_day_nvda_posts)
        assert isinstance(result["NVDA"]["dates"], set)

    def test_positive_comments_list_populated_for_positive_post(self, positive_aapl_post):
        # Positive-sentiment text segments should be stored in positive_comments.
        result = aggregate_by_ticker(positive_aapl_post)
        assert len(result["AAPL"]["positive_comments"]) >= 1

    def test_negative_comments_list_populated_for_negative_post(self, negative_tsla_post):
        # Negative-sentiment text segments should be stored in negative_comments.
        result = aggregate_by_ticker(negative_tsla_post)
        assert len(result["TSLA"]["negative_comments"]) >= 1

    def test_result_entry_has_all_required_keys(self, positive_aapl_post):
        # Every ticker entry must have all six required fields.
        result = aggregate_by_ticker(positive_aapl_post)
        required_keys = {
            "dates", "total",
            "positive_count", "positive_comments",
            "negative_count", "negative_comments",
        }
        assert required_keys.issubset(result["AAPL"].keys())

    def test_total_is_integer(self, positive_aapl_post):
        # The total count must be an integer, not a float or string.
        result = aggregate_by_ticker(positive_aapl_post)
        assert isinstance(result["AAPL"]["total"], int)

    def test_positive_comments_are_strings(self, positive_aapl_post):
        # Each item in positive_comments must be a string (text snippet).
        result = aggregate_by_ticker(positive_aapl_post)
        for comment in result["AAPL"]["positive_comments"]:
            assert isinstance(comment, str)

    def test_comment_snippets_capped_at_300_chars(self):
        # A comment longer than 300 characters should be truncated to 300 chars
        # when stored in the positive_comments list (see MAX_COMMENT_LENGTH).
        long_comment = "This $AAPL stock is absolutely fantastic! " * 20  # >300 chars
        posts = [{
            "post_id": "x1",
            "title": "$AAPL is great",
            "body": "",
            "date": "2026-05-30",
            "comments": [long_comment],
        }]
        result = aggregate_by_ticker(posts)
        # If AAPL was positive, check snippet length
        if result.get("AAPL") and result["AAPL"]["positive_comments"]:
            for snippet in result["AAPL"]["positive_comments"]:
                assert len(snippet) <= 300
