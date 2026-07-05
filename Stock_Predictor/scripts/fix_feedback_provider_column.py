"""
scripts/fix_feedback_provider_column.py — one-off repair for legacy feedback CSVs.

Some feedback CSVs were created before the `provider` column was added to the
schema. Their HEADER is missing `provider` (72 labels), even though nearly all
their data rows were later written WITH the provider value (73 fields). pandas
then throws "Expected 72 fields, saw 73" and --doctor flags a schema mismatch.

This script repairs the structure without touching correct data:
  • inserts the `provider` header label at its canonical index (after `mode`)
  • pads the handful of genuinely-old 72-field rows with an empty provider
  • leaves every already-correct 73-field row byte-for-byte unchanged
  • writes back atomically and only after verifying the result parses to the
    canonical 73-column schema

Idempotent: files already at the full width are reported and skipped.

Run:  python scripts/fix_feedback_provider_column.py
"""

import csv
import glob
import os
import sys

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config import FEEDBACK_DIR
from feedback.tracker import COLUMNS

PROVIDER_IDX = COLUMNS.index("provider")   # canonical position (after `mode`)
EXPECTED = len(COLUMNS)                     # 73


def _repair_file(path: str) -> str:
    """Return a status string describing what was done to *path*."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return "empty — skipped"

    header = rows[0]
    if "provider" in header and len(header) == EXPECTED:
        return f"already OK ({len(rows)-1} rows)"

    # 1. Header: insert the missing provider label.
    if "provider" not in header:
        header = header[:PROVIDER_IDX] + ["provider"] + header[PROVIDER_IDX:]

    # 2. Data rows: pad only the short (pre-provider) rows.
    fixed_rows = [header]
    padded = 0
    for r in rows[1:]:
        if len(r) == EXPECTED - 1:
            r = r[:PROVIDER_IDX] + [""] + r[PROVIDER_IDX:]
            padded += 1
        fixed_rows.append(r)

    # 3. Write atomically to a temp file, verify, then replace.
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(fixed_rows)

    # Guard: the repaired file must parse to exactly the canonical width.
    check = pd.read_csv(tmp, dtype=str, keep_default_na=False)
    if list(check.columns) != COLUMNS:
        os.remove(tmp)
        missing = set(COLUMNS) - set(check.columns)
        extra = set(check.columns) - set(COLUMNS)
        return (f"ABORTED — repaired columns don't match schema "
                f"(missing={sorted(missing)}, extra={sorted(extra)})")

    os.replace(tmp, path)
    return f"FIXED — header +provider, {padded} legacy row(s) padded, {len(rows)-1} rows total"


def main() -> int:
    paths = sorted(glob.glob(os.path.join(FEEDBACK_DIR, "*_feedback.csv")))
    if not paths:
        print("No feedback CSVs found.")
        return 0

    fixed = 0
    print(f"Canonical schema: {EXPECTED} columns (provider at index {PROVIDER_IDX}).\n")
    for p in paths:
        status = _repair_file(p)
        if status.startswith("FIXED"):
            fixed += 1
        print(f"  {os.path.basename(p):26} {status}")
    print(f"\nDone. {fixed} file(s) repaired, {len(paths) - fixed} already OK / skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
