# csv_writer.py
# Serialises the aggregated ticker data into a CSV file on disk.
#
# WHY one row per ticker?
#   The user requested an aggregated view — how many times a ticker was talked
#   about, with its positive/negative counts and the actual comments. One row
#   per ticker satisfies this by collapsing all mentions into a single summary.
#
# WHY " | " as the multi-value separator?
#   Fields like "Date Mentioned" and "Positive Comments" may contain multiple
#   values (several dates, several comment snippets). Joining them with " | "
#   keeps the CSV valid (no extra columns) while remaining human-readable in
#   a spreadsheet. Commas were avoided to prevent conflicts with the CSV delimiter.
#
# WHY newline="" in open()?
#   Python's csv module writes its own line endings (\r\n on Windows). Opening
#   the file with newline="" prevents Python's universal newline translation
#   from doubling the line breaks, which would produce blank rows in Excel.

import csv
import os
from typing import Dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The exact column names and order for the output CSV.
# These match the user's original specification.
CSV_HEADERS = [
    "Ticker",
    "Date Mentioned",
    "Total Mentions",
    "Positive Count",
    "Positive Comments",
    "Negative Count",
    "Negative Comments",
]

# String used to join multiple values within one CSV cell.
# e.g. "2026-05-28 | 2026-05-30" or "Great stock! | Best buy ever"
MULTI_VALUE_SEPARATOR = " | "


def write_csv(aggregated: Dict[str, Dict], output_path: str) -> None:
    """
    Writes the aggregated ticker sentiment data to a CSV file at `output_path`.

    Multi-value fields (dates, comment lists) are joined into a single string
    with MULTI_VALUE_SEPARATOR so each ticker occupies exactly one row.

    Args:
        aggregated: Dict returned by aggregator.aggregate_by_ticker().
                    Keys are ticker strings; values are accumulator dicts.
        output_path: File path (absolute or relative) where the CSV will be saved.
                     The parent directory must already exist.

    Returns:
        None. Side effect: writes a UTF-8 encoded CSV file to disk and prints
        a confirmation message.

    If `aggregated` is empty, no file is written and a message is printed instead.
    """
    if not aggregated:
        # Nothing to write — avoid creating an empty/header-only CSV that could
        # mislead the user into thinking the run succeeded with zero results.
        print("No ticker data to write — CSV will not be created.")
        return

    # Sort tickers alphabetically so the CSV is easy to scan in a spreadsheet
    sorted_tickers = sorted(aggregated.keys())

    # newline="" is required by the csv module (see module docstring above)
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)

        # Write the header row first
        writer.writeheader()

        for ticker in sorted_tickers:
            data = aggregated[ticker]

            # Sort dates chronologically so "earliest → latest" reads left to right
            dates_str = MULTI_VALUE_SEPARATOR.join(sorted(data["dates"]))

            # Join comment snippets into one cell string.
            # Each snippet was already capped at MAX_COMMENT_LENGTH in aggregator.py.
            pos_comments_str = MULTI_VALUE_SEPARATOR.join(data["positive_comments"])
            neg_comments_str = MULTI_VALUE_SEPARATOR.join(data["negative_comments"])

            writer.writerow({
                "Ticker":            ticker,
                "Date Mentioned":    dates_str,
                "Total Mentions":    data["total"],
                "Positive Count":    data["positive_count"],
                "Positive Comments": pos_comments_str,
                "Negative Count":    data["negative_count"],
                "Negative Comments": neg_comments_str,
            })

    # Use abspath so the printed path is always unambiguous regardless of cwd
    print(f"CSV written to: {os.path.abspath(output_path)}")
    print(f"  {len(sorted_tickers)} unique tickers recorded.")
