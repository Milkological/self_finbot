# tests/test_ticker_extractor.py
# Unit tests for ticker_extractor.extract_tickers().
#
# WHAT is being tested:
#   - Dollar-prefixed tickers ($AAPL) are always extracted
#   - Bare uppercase tickers (TSLA) are extracted when not in the blocklist
#   - Blocklisted words (BUY, IPO, CEO) are NOT extracted
#   - Results are deduplicated and alphabetically sorted
#   - Edge cases: empty string, None, lowercase text, single letters
#
# WHY no mocking needed here?
#   extract_tickers() is a pure function (no I/O, no side effects).
#   It takes a string and returns a list, so tests just call it directly.

import sys
import os

# Add the parent directory (stock_retriever/) to sys.path so that
# the test can import ticker_extractor without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ticker_extractor import extract_tickers


class TestExtractTickers:
    """Tests for the extract_tickers() function."""

    def test_dollar_prefixed_ticker_is_extracted(self):
        # $AAPL is an explicit ticker reference — must always be returned.
        result = extract_tickers("I think $AAPL is going to moon this week")
        assert "AAPL" in result

    def test_bare_uppercase_ticker_is_extracted(self):
        # Plain uppercase ticker like TSLA should be detected from normal text.
        result = extract_tickers("TSLA earnings coming up next week, watch closely")
        assert "TSLA" in result

    def test_multiple_tickers_extracted_from_one_string(self):
        # Both $NVDA and MSFT appear in the same text — both should be returned.
        result = extract_tickers("$NVDA and MSFT are both looking bullish today")
        assert "NVDA" in result
        assert "MSFT" in result

    def test_blocklisted_word_buy_not_extracted(self):
        # "BUY" is in the blocklist and should NOT appear as a ticker.
        result = extract_tickers("Everyone should BUY this stock right now")
        assert "BUY" not in result

    def test_blocklisted_word_ipo_not_extracted(self):
        # "IPO" is finance jargon in the blocklist — should be filtered out.
        result = extract_tickers("The IPO is happening next month")
        assert "IPO" not in result

    def test_blocklisted_word_ceo_not_extracted(self):
        # "CEO" is a job title in the blocklist — should be filtered out.
        result = extract_tickers("The CEO announced record profits today")
        assert "CEO" not in result

    def test_empty_string_returns_empty_list(self):
        # An empty string has no tickers — must return [] without raising.
        result = extract_tickers("")
        assert result == []

    def test_none_input_returns_empty_list(self):
        # None is not a valid string — must return [] without raising TypeError.
        result = extract_tickers(None)
        assert result == []

    def test_lowercase_text_not_extracted(self):
        # Tickers are uppercase only; lowercase "aapl" and "tsla" must not match.
        result = extract_tickers("aapl and tsla are popular stocks to watch")
        assert result == []

    def test_deduplicated_results(self):
        # The same ticker mentioned three times should appear exactly once.
        result = extract_tickers("$AAPL AAPL is the best stock, buy $AAPL now")
        assert result.count("AAPL") == 1

    def test_results_are_sorted_alphabetically(self):
        # The returned list must be sorted so output is deterministic and readable.
        result = extract_tickers("$TSLA and $AAPL and $MSFT look good today")
        assert result == sorted(result)

    def test_ticker_extracted_from_realistic_reddit_text(self):
        # Simulate a typical Reddit post sentence — ticker must still be found.
        result = extract_tickers("Just loaded up on more $GME — diamond hands, to the moon!")
        assert "GME" in result

    def test_single_letter_a_not_extracted(self):
        # "A" is in the blocklist (also too short for bare pattern at min=2).
        result = extract_tickers("A lot of people are watching this stock")
        assert "A" not in result

    def test_ticker_adjacent_to_punctuation_extracted(self):
        # A ticker followed by punctuation (comma, period) should still match
        # because the word boundary \b handles this correctly.
        result = extract_tickers("Strong buy on NVDA, and also on AMD.")
        assert "NVDA" in result
        assert "AMD" in result

    def test_dollar_prefixed_bypasses_blocklist(self):
        # If a user explicitly writes $BUY, it should be treated as a ticker
        # and returned, because the dollar sign signals intentional ticker use.
        result = extract_tickers("Looking at $BUY as a ticker today")
        assert "BUY" in result
