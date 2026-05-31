# tests/test_sentiment_analyzer.py
# Unit tests for sentiment_analyzer.analyze_sentiment().
#
# WHAT is being tested:
#   - Clearly positive text returns label "positive"
#   - Clearly negative text returns label "negative"
#   - Neutral/factual text returns label "neutral"
#   - Return dict always has "label" and "score" keys
#   - Score is a float in the valid VADER range [-1.0, 1.0]
#   - Edge cases: empty string, None input
#
# WHY no mocking?
#   analyze_sentiment() is a pure function — it wraps the VADER library which
#   is deterministic (same input always produces the same output). No I/O or
#   external state is involved, so we test it end-to-end directly.

import sys
import os

# Add the parent directory (stock_retriever/) to sys.path so that
# the test can import sentiment_analyzer without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_analyzer import analyze_sentiment, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD


class TestAnalyzeSentiment:
    """Tests for the analyze_sentiment() function."""

    def test_clearly_positive_text_returns_positive_label(self):
        # Enthusiastic, upbeat text should cross the positive threshold.
        result = analyze_sentiment("This stock is absolutely amazing, best buy ever!")
        assert result["label"] == "positive"

    def test_clearly_negative_text_returns_negative_label(self):
        # Strongly negative text should cross the negative threshold.
        result = analyze_sentiment("This stock is terrible, worst investment ever, avoid it completely")
        assert result["label"] == "negative"

    def test_neutral_factual_text_returns_neutral_label(self):
        # Plain factual text with no emotional language should score as neutral.
        result = analyze_sentiment("The stock traded at 150 dollars today on high volume")
        assert result["label"] == "neutral"

    def test_empty_string_returns_neutral(self):
        # Empty string input should return neutral without raising an exception.
        result = analyze_sentiment("")
        assert result["label"] == "neutral"

    def test_empty_string_returns_zero_score(self):
        # The score for empty input should be exactly 0.0 (our default).
        result = analyze_sentiment("")
        assert result["score"] == 0.0

    def test_none_input_returns_neutral(self):
        # None input must be handled gracefully — return neutral, not raise TypeError.
        result = analyze_sentiment(None)
        assert result["label"] == "neutral"

    def test_none_input_returns_zero_score(self):
        # None input should also produce a score of 0.0.
        result = analyze_sentiment(None)
        assert result["score"] == 0.0

    def test_return_value_contains_label_key(self):
        # The returned dict must always contain the "label" key.
        result = analyze_sentiment("Stocks are interesting")
        assert "label" in result

    def test_return_value_contains_score_key(self):
        # The returned dict must always contain the "score" key.
        result = analyze_sentiment("Stocks are interesting")
        assert "score" in result

    def test_score_is_float(self):
        # The score value must be a float (VADER compound score type).
        result = analyze_sentiment("I love this stock so much")
        assert isinstance(result["score"], float)

    def test_score_within_vader_range(self):
        # VADER compound score is always bounded between -1.0 and +1.0.
        result = analyze_sentiment("Outstanding gains today, incredible performance!!!")
        assert -1.0 <= result["score"] <= 1.0

    def test_label_is_one_of_three_valid_values(self):
        # The label must always be one of the three defined categories.
        result = analyze_sentiment("Some random text about stocks and investing")
        assert result["label"] in ("positive", "negative", "neutral")

    def test_positive_score_is_above_threshold(self):
        # When the label is "positive", the compound score must be >= POSITIVE_THRESHOLD.
        result = analyze_sentiment("Fantastic earnings, buying more immediately!")
        if result["label"] == "positive":
            assert result["score"] >= POSITIVE_THRESHOLD

    def test_negative_score_is_below_threshold(self):
        # When the label is "negative", the compound score must be <= NEGATIVE_THRESHOLD.
        result = analyze_sentiment("Horrible results, company is going bankrupt, sell everything")
        if result["label"] == "negative":
            assert result["score"] <= NEGATIVE_THRESHOLD

    def test_score_is_rounded_to_4_decimal_places(self):
        # The score should be rounded to 4 decimal places as specified in the function.
        result = analyze_sentiment("This is a great investment opportunity")
        # Check that the score has at most 4 decimal places
        assert result["score"] == round(result["score"], 4)
