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

TOP_N = 5


def get_keywords(directory=".", recursive=True):
    """Return a de-duplicated list of search keywords from the mortgage CSV(s)."""
    files = find_csv_files(Path(directory).resolve(), "*.csv", recursive)
    files = [p for p in files if looks_like_mortgage_csv(p)]
    if not files:
        raise SystemExit("No mortgage CSV found.")

    keywords = []
    for path in files:
        result = extract_keywords(load_rows(path), TOP_N)

        # Categorical values are generic labels, so they are safe to search on
        for col in ("event_type", "rejection_reason", "collateral_type", "property_type"):
            keywords += [i["keyword"] for i in result["categorical_keywords"].get(col, [])]

        # Phrases from the notes
        keywords += [i["keyword"] for i in result["top_phrases"]]

    return list(dict.fromkeys(k.replace("_", " ") for k in keywords))


for kw in get_keywords("./data"):
    query = f"mortgage {kw}"
    results = exa.search(
        query,
        type="auto",
        contents={"highlights": True},
    )
    print(f"\n=== {query} ===")
    for source in results.results:
        print(source.url, source.highlights)