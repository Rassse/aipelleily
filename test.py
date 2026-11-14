import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from exa_py import Exa
from keyword_extractor import (
    extract_keywords,
    find_csv_files,
    load_rows,
    looks_like_mortgage_csv,
)

exa = Exa()  # reads EXA_API_KEY from the environment

TOP_N = 5          # keywords per list from the extractor
MAX_QUERIES = 20   # cap API calls
DATA_DIR = "./data"


def get_keywords(directory=DATA_DIR, recursive=True):
    """Return a de-duplicated list of search keywords from the mortgage CSV(s)."""
    files = find_csv_files(Path(directory).resolve(), "*.csv", recursive)
    files = [p for p in files if looks_like_mortgage_csv(p)]
    if not files:
        raise SystemExit("No mortgage CSV found.")

    keywords = []
    for path in files:
        result = extract_keywords(load_rows(path), TOP_N)

        # Generic categorical labels (safe to send to a third-party API)
        for col in ("event_type", "rejection_reason", "collateral_type", "property_type"):
            keywords += [i["keyword"] for i in result["categorical_keywords"].get(col, [])]

        # Phrases from free-text notes: may contain personal data, so they are
        # left out by default. Uncomment to include them.
        # keywords += [i["keyword"] for i in result["top_phrases"]]

    return list(dict.fromkeys(k.replace("_", " ") for k in keywords))


def main():
    keywords = get_keywords()[:MAX_QUERIES]
    output = {}

    for kw in keywords:
        query = kw if "mortgage" in kw.lower() else f"mortgage {kw}"
        try:
            results = exa.search(
                query,
                type="auto",
                num_results=5,
                contents={"highlights": True},
            )
        except Exception as e:
            print(f"Skipping '{query}': {e}")
            continue

        output[query] = [
            {"url": r.url, "highlights": r.highlights} for r in results.results
        ]
        print(f"\n=== {query} ===")
        for source in results.results:
            print(source.url, source.highlights)
        time.sleep(0.5)  # be gentle with rate limits

    Path("search_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSaved to search_results.json")


if __name__ == "__main__":
    main()

