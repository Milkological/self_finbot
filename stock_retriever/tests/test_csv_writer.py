# tests/test_csv_writer.py
# Unit tests for csv_writer.write_csv().
#
# WHAT is being tested:
#   - A CSV file is created on disk when there is data to write
#   - The CSV header row matches CSV_HEADERS exactly
#   - One data row is written per ticker
#   - Specific field values (Total Mentions, comments) are correct
#   - Tickers are written in alphabetical order
#   - No file is created when aggregated data is empty
#
# WHY use pytest's tmp_path fixture?
#   tmp_path provides a temporary directory unique to each test run.
#   This means tests never leave CSV files in the project folder and
#   never interfere with each other, even when run in parallel.

import csv
import os
import sys

# Add the parent directory (stock_retriever/) to sys.path so that
# the test can import csv_writer without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from csv_writer import CSV_HEADERS, MULTI_VALUE_SEPARATOR, write_csv


# ---------------------------------------------------------------------------
# Fixtures — reusable mock aggregated data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_aggregated():
    """
    A minimal but complete aggregated data dict representing two tickers.
    Shape matches the output of aggregator.aggregate_by_ticker().
    """
    return {
        "AAPL": {
            "dates":             {"2026-05-28", "2026-05-30"},
            "total":             5,
            "positive_count":    3,
            "positive_comments": ["Great stock!", "Best company ever"],
            "negative_count":    1,
            "negative_comments": ["Too expensive right now"],
        },
        "TSLA": {
            "dates":             {"2026-05-29"},
            "total":             2,
            "positive_count":    1,
            "positive_comments": ["Love the EV market direction"],
            "negative_count":    1,
            "negative_comments": ["Overvalued at this price"],
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWriteCsv:
    """Tests for the write_csv() function."""

    def test_csv_file_is_created(self, sample_aggregated, tmp_path):
        # After calling write_csv, the output file must exist on disk.
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        assert output.exists()

    def test_csv_header_row_matches_expected_headers(self, sample_aggregated, tmp_path):
        # The first row must contain exactly the columns defined in CSV_HEADERS.
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        with open(output, encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers == CSV_HEADERS

    def test_csv_row_count_equals_ticker_count(self, sample_aggregated, tmp_path):
        # There should be exactly one data row per ticker (plus one header row).
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        with open(output, encoding="utf-8") as f:
            all_rows = list(csv.reader(f))
        # Subtract 1 for the header row
        assert len(all_rows) - 1 == len(sample_aggregated)

    def test_aapl_ticker_present_in_csv(self, sample_aggregated, tmp_path):
        # AAPL was in the input data — it must appear in the CSV content.
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        content = output.read_text(encoding="utf-8")
        assert "AAPL" in content

    def test_tsla_ticker_present_in_csv(self, sample_aggregated, tmp_path):
        # TSLA was in the input data — it must appear in the CSV content.
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        content = output.read_text(encoding="utf-8")
        assert "TSLA" in content

    def test_total_mentions_value_is_correct(self, sample_aggregated, tmp_path):
        # AAPL's Total Mentions was set to 5 in the fixture — verify it in CSV.
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {row["Ticker"]: row for row in reader}
        assert rows["AAPL"]["Total Mentions"] == "5"

    def test_positive_count_value_is_correct(self, sample_aggregated, tmp_path):
        # AAPL's Positive Count was set to 3 in the fixture — verify it in CSV.
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {row["Ticker"]: row for row in reader}
        assert rows["AAPL"]["Positive Count"] == "3"

    def test_negative_count_value_is_correct(self, sample_aggregated, tmp_path):
        # AAPL's Negative Count was set to 1 in the fixture — verify it in CSV.
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {row["Ticker"]: row for row in reader}
        assert rows["AAPL"]["Negative Count"] == "1"

    def test_positive_comment_text_in_csv(self, sample_aggregated, tmp_path):
        # At least one positive comment text must appear in the CSV cell for AAPL.
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {row["Ticker"]: row for row in reader}
        assert "Great stock!" in rows["AAPL"]["Positive Comments"]

    def test_negative_comment_text_in_csv(self, sample_aggregated, tmp_path):
        # At least one negative comment text must appear in the CSV cell for AAPL.
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {row["Ticker"]: row for row in reader}
        assert "Too expensive right now" in rows["AAPL"]["Negative Comments"]

    def test_tickers_are_in_alphabetical_order(self, sample_aggregated, tmp_path):
        # Rows must be sorted alphabetically by ticker (AAPL before TSLA).
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            tickers = [row["Ticker"] for row in reader]
        assert tickers == sorted(tickers)

    def test_empty_aggregated_does_not_create_file(self, tmp_path):
        # When there is no data, write_csv should NOT create a file on disk.
        output = tmp_path / "empty_output.csv"
        write_csv({}, str(output))
        assert not output.exists()

    def test_multiple_dates_joined_with_separator(self, sample_aggregated, tmp_path):
        # AAPL has two dates — they should appear joined by MULTI_VALUE_SEPARATOR.
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {row["Ticker"]: row for row in reader}
        # Both dates must be in the "Date Mentioned" cell
        date_cell = rows["AAPL"]["Date Mentioned"]
        assert "2026-05-28" in date_cell
        assert "2026-05-30" in date_cell

    def test_multiple_positive_comments_joined_with_separator(self, sample_aggregated, tmp_path):
        # AAPL has two positive comments — they should be joined in one cell.
        output = tmp_path / "output.csv"
        write_csv(sample_aggregated, str(output))
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {row["Ticker"]: row for row in reader}
        pos_cell = rows["AAPL"]["Positive Comments"]
        assert "Great stock!" in pos_cell
        assert "Best company ever" in pos_cell
