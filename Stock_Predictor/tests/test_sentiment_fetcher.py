"""
tests/test_sentiment_fetcher.py
--------------------------------
Unit tests for data/sentiment_fetcher.py.

WHY these tests exist:
  score_headlines() calls a live LLM (Azure OpenAI or Google Gemini) to
  score the sentiment of news headlines.  In a test environment we must not
  make real LLM calls (expensive, flaky, need credentials).  These tests
  verify that:
    - When no LLM is configured, a neutral stub is returned immediately.
    - Empty headline lists return the neutral stub without making any call.
    - When _call_azure returns a valid parsed dict, score_headlines passes
      it through unchanged.
    - The returned dict always contains exactly the three expected keys:
      overall_score, label, summary.

HOW mocking works here:
  We patch the module-level boolean flags (LLM_ENABLED, AZURE_ENABLED,
  GOOGLE_ENABLED) inside data.sentiment_fetcher to control which code path
  executes, and patch _call_azure to inject a synthetic LLM response without
  making a network call.
"""

import unittest
from unittest.mock import patch

from data.sentiment_fetcher import score_headlines


# ---------------------------------------------------------------------------
# Constant defining the expected output shape
# ---------------------------------------------------------------------------

REQUIRED_KEYS = ("overall_score", "label", "summary")

# A minimal list of fake news headlines for testing the 'happy path'.
SAMPLE_HEADLINES = [
    {"title": "Company beats earnings estimates by 20%"},
    {"title": "Stock hits all-time high after strong guidance"},
]

# What a well-formed LLM response looks like after _call_azure parses it.
FAKE_LLM_RESPONSE = {
    "overall_score": 0.6,
    "label":         "POSITIVE",
    "summary":       "Strong earnings beat drives bullish sentiment.",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoreHeadlines(unittest.TestCase):
    """Tests for score_headlines() — LLM-based headline sentiment scoring."""

    def test_no_llm_returns_neutral_stub(self):
        # When LLM_ENABLED is False (no API keys configured), the function
        # must short-circuit immediately and return the neutral default.
        # Patching the flag at the module level simulates a keyless environment.
        with patch("data.sentiment_fetcher.LLM_ENABLED", False):
            result = score_headlines(SAMPLE_HEADLINES)
            self.assertAlmostEqual(result["overall_score"], 0.0, places=4)
            self.assertEqual(result["label"], "NEUTRAL")

    def test_empty_headlines_returns_neutral(self):
        # An empty headline list means there is nothing to score.
        # Returning neutral avoids a wasted LLM call and prevents the
        # sentiment feature from penalising tickers with no recent news.
        with patch("data.sentiment_fetcher.LLM_ENABLED", True), \
             patch("data.sentiment_fetcher.AZURE_ENABLED", True):
            result = score_headlines([])
            self.assertAlmostEqual(result["overall_score"], 0.0, places=4)
            self.assertEqual(result["label"], "NEUTRAL")

    def test_required_keys_always_present(self):
        # Regardless of the code path taken, the output dict must always
        # contain overall_score, label, and summary so callers can depend
        # on these keys without defensive checks.
        with patch("data.sentiment_fetcher.LLM_ENABLED", False):
            result = score_headlines(SAMPLE_HEADLINES)
            for key in REQUIRED_KEYS:
                self.assertIn(key, result, msg=f"Key '{key}' missing from result")

    def test_azure_response_passed_through(self):
        # When AZURE_ENABLED is True and _call_azure returns a valid dict,
        # score_headlines must return that dict directly without modification.
        with patch("data.sentiment_fetcher.LLM_ENABLED", True), \
             patch("data.sentiment_fetcher.AZURE_ENABLED", True), \
             patch("data.sentiment_fetcher.GOOGLE_ENABLED", False), \
             patch("data.sentiment_fetcher._call_azure", return_value=FAKE_LLM_RESPONSE):
            result = score_headlines(SAMPLE_HEADLINES)
            self.assertAlmostEqual(result["overall_score"], FAKE_LLM_RESPONSE["overall_score"])
            self.assertEqual(result["label"],   FAKE_LLM_RESPONSE["label"])
            self.assertEqual(result["summary"], FAKE_LLM_RESPONSE["summary"])

    def test_falls_back_to_neutral_when_azure_fails(self):
        # When Azure is enabled but _call_azure returns None (network error,
        # quota exceeded, malformed JSON), and Google is disabled,
        # the function must fall back to the neutral stub rather than raising.
        with patch("data.sentiment_fetcher.LLM_ENABLED", True), \
             patch("data.sentiment_fetcher.AZURE_ENABLED", True), \
             patch("data.sentiment_fetcher.GOOGLE_ENABLED", False), \
             patch("data.sentiment_fetcher._call_azure", return_value=None):
            result = score_headlines(SAMPLE_HEADLINES)
            self.assertAlmostEqual(result["overall_score"], 0.0, places=4)
            self.assertEqual(result["label"], "NEUTRAL")

    def test_label_is_one_of_three_valid_values(self):
        # The label must always be one of the three strings that downstream
        # code maps to display colours and score adjustments.
        with patch("data.sentiment_fetcher.LLM_ENABLED", False):
            result = score_headlines(SAMPLE_HEADLINES)
            self.assertIn(result["label"], ("POSITIVE", "NEUTRAL", "NEGATIVE"))

    def test_overall_score_within_bounds(self):
        # overall_score must be clamped to [-1.0, +1.0] by the _parse function.
        with patch("data.sentiment_fetcher.LLM_ENABLED", True), \
             patch("data.sentiment_fetcher.AZURE_ENABLED", True), \
             patch("data.sentiment_fetcher.GOOGLE_ENABLED", False), \
             patch("data.sentiment_fetcher._call_azure", return_value=FAKE_LLM_RESPONSE):
            result = score_headlines(SAMPLE_HEADLINES)
            self.assertGreaterEqual(result["overall_score"], -1.0)
            self.assertLessEqual(result["overall_score"],     1.0)


if __name__ == "__main__":
    unittest.main()
