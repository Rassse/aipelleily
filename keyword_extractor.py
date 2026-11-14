#!/usr/bin/env python3
"""
Mortgage CSV keyword extractor.

1. Searches a directory (optionally recursively) for CSV files.
2. Keeps only files that look like the mortgage event log (header check).
3. Extracts keywords:
   - categorical keywords (event types, property types, rejection reasons, ...)
   - text keywords from the `note` column (frequent words/bigrams)
   - distinctive keywords per event type (TF-IDF)

Standard library only. Usage examples:

    python keyword_extractor.py                       # search current directory
    python keyword_extractor.py -d ./data -r          # search ./data recursively
    python keyword_extractor.py -d ./data -t 20 -o keywords.json
    python keyword_extractor.py -f mortgage_events.csv   # only this file name/pattern
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_COLUMNS = {"mortgage_customer_id", "timestamp", "event_type", "note"}

CATEGORICAL_COLUMNS = [
    "event_type",
    "property_type",
    "collateral_type",
    "collateral_status",
    "decision",
    "rejection_reason",
    "expense_type",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "with", "this", "which", "their", "them", "they", "his", "her", "s",
    "eur", "amount", "related",
}


# --------------------------------------------------------------------------- #
# Step 1: find the file(s)
# --------------------------------------------------------------------------- #
def find_csv_files(directory: Path, pattern: str, recursive: bool):
    """Return all files in `directory` matching `pattern` (default *.csv)."""
    if not directory.is_dir():
        sys.exit(f"Directory not found: {directory}")
    globber = directory.rglob if recursive else directory.glob
    return sorted(p for p in globber(pattern) if p.is_file())


def looks_like_mortgage_csv(path: Path) -> bool:
    """Check the header row for the columns we need."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            header = next(csv.reader(fh), [])
    except (OSError, UnicodeDecodeError, csv.Error):
        return False
    return REQUIRED_COLUMNS.issubset({h.strip() for h in header})


# --------------------------------------------------------------------------- #
# Step 2: extract keywords
# --------------------------------------------------------------------------- #
def tokenize(text: str):
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def bigrams(tokens):
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def extract_keywords(rows, top_n: int):
    categorical = {col: Counter() for col in CATEGORICAL_COLUMNS}
    word_counts = Counter()
    bigram_counts = Counter()
    docs = defaultdict(Counter)  # event_type -> token counts (for TF-IDF)
    customers = set()

    for row in rows:
        customers.add(row.get("mortgage_customer_id", ""))
        for col in CATEGORICAL_COLUMNS:
            value = (row.get(col) or "").strip()
            if value:
                categorical[col][value] += 1

        tokens = tokenize(row.get("note") or "")
        word_counts.update(tokens)
        bigram_counts.update(bigrams(tokens))
        docs[(row.get("event_type") or "unknown")].update(tokens)

    # TF-IDF, treating each event type as one document
    n_docs = len(docs)
    df = Counter()
    for counts in docs.values():
        df.update(counts.keys())

    distinctive = {}
    for event, counts in docs.items():
        total = sum(counts.values()) or 1
        scored = {
            tok: (c / total) * (math.log((n_docs + 1) / (df[tok] + 1)) + 1)
            for tok, c in counts.items()
        }
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        distinctive[event] = [{"keyword": k, "score": round(s, 4)} for k, s in ranked]

    return {
        "rows": len(rows),
        "customers": len(customers - {""}),
        "categorical_keywords": {
            col: [{"keyword": k, "count": v} for k, v in cnt.most_common(top_n)]
            for col, cnt in categorical.items()
            if cnt
        },
        "top_words": [{"keyword": k, "count": v} for k, v in word_counts.most_common(top_n)],
        "top_phrases": [{"keyword": k, "count": v} for k, v in bigram_counts.most_common(top_n)],
        "distinctive_by_event_type": distinctive,
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def print_report(path: Path, result: dict):
    line = "=" * 70
    print(f"\n{line}\nFile: {path}\nRows: {result['rows']}  |  Customers: {result['customers']}\n{line}")

    print("\n-- Categorical keywords --")
    for col, items in result["categorical_keywords"].items():
        print(f"  {col}: " + ", ".join(f"{i['keyword']} ({i['count']})" for i in items))

    print("\n-- Top words in notes --")
    print("  " + ", ".join(f"{i['keyword']} ({i['count']})" for i in result["top_words"]))

    print("\n-- Top phrases in notes --")
    print("  " + ", ".join(f"{i['keyword']} ({i['count']})" for i in result["top_phrases"]))

    print("\n-- Distinctive keywords per event type (TF-IDF) --")
    for event, items in result["distinctive_by_event_type"].items():
        print(f"  {event}: " + ", ".join(i["keyword"] for i in items))


def main():
    parser = argparse.ArgumentParser(description="Find mortgage event CSVs and extract keywords.")
    parser.add_argument("-d", "--directory", default=".", help="directory to search (default: current)")
    parser.add_argument("-f", "--filename", default="*.csv",
                        help="file name or glob pattern (default: *.csv)")
    parser.add_argument("-r", "--recursive", action="store_true", help="search subdirectories too")
    parser.add_argument("-t", "--top", type=int, default=10, help="keywords per list (default: 10)")
    parser.add_argument("-o", "--output", help="optional path for JSON output")
    args = parser.parse_args()

    directory = Path(args.directory).expanduser().resolve()
    print(f"Searching {directory} for '{args.filename}'"
          f"{' (recursive)' if args.recursive else ''} ...")

    candidates = find_csv_files(directory, args.filename, args.recursive)
    if not candidates:
        sys.exit("No CSV files found.")

    matches = [p for p in candidates if looks_like_mortgage_csv(p)]
    skipped = len(candidates) - len(matches)
    print(f"Found {len(candidates)} CSV file(s); {len(matches)} match the mortgage format"
          f"{f', {skipped} skipped' if skipped else ''}.")
    if not matches:
        sys.exit("No file has the expected columns: " + ", ".join(sorted(REQUIRED_COLUMNS)))

    all_results = {}
    for path in matches:
        result = extract_keywords(load_rows(path), args.top)
        all_results[str(path)] = result
        print_report(path, result)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON saved to {out}")


if __name__ == "__main__":
    main()

