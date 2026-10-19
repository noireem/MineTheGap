"""Clean AAPL FY2020 and IBM FY2020/FY2024 so DEI extraction can be checked by hand.

IBM's Item 1 narrative lives in Exhibit 13, not the 10-K wrapper. Apple's FY2020 10-K
has no diversity/inclusion language; the cleaned text is still written so that can be verified.

Run from the project root:
    python scripts/clean_check_filings.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.data_loader import (
    KEYWORDS,
    cleaned_filing_text,
    extract,
    narrative_html,
    read_downloaded_filing,
)

TARGETS = [
    ("AAPL", 2020, "0000320193-20-000096"),
    ("IBM", 2020, "0001558370-21-001489"),
    ("IBM", 2024, "0000051143-25-000015"),
]
OUT = config.ROOT / "data" / "processed" / "check_filings"


def filing_path(ticker: str, accession: str) -> Path:
    matches = list(config.FILINGS_DIR.rglob(f"{ticker}/10-K/{accession}/full-submission.txt"))
    if not matches:
        raise FileNotFoundError(f"{ticker} {accession}")
    return matches[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for ticker, fy, accession in TARGETS:
        path = filing_path(ticker, accession)
        raw = path.read_text(encoding="utf-8", errors="ignore")
        html = narrative_html(raw, path)
        text = cleaned_filing_text(html or "")
        note = ""
        if not KEYWORDS.search(text):
            note = (
                "\n[NOTE] No diversity/inclusion/DEI keyword hits after cleaning. "
                "This filing may simply not discuss DEI.\n"
            )
        cleaned_path = OUT / f"{ticker}_{fy}_cleaned.txt"
        cleaned_path.write_text(
            f"# {ticker} FY{fy}  accession {accession}\n"
            f"# source {path}\n"
            f"{note}\n{text}\n",
            encoding="utf-8",
        )
        got = read_downloaded_filing(path)
        rows = extract(got[1], ticker, got[0]) if got else []
        hits = pd.DataFrame(rows)
        hit_path = OUT / f"{ticker}_{fy}_passages.csv"
        hits.to_csv(hit_path, index=False)
        print(f"{ticker} FY{fy}: cleaned {len(text):,} chars -> {cleaned_path.name}; "
              f"{len(hits)} DEI passages -> {hit_path.name}")
        if not hits.empty:
            frames.append(hits)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(OUT / "recovered_passages.csv", index=False)
        cands_path = config.CANDIDATES
        if cands_path.exists():
            old = pd.read_csv(cands_path)
            keep = ~((old["company"] == "IBM") & (old["fiscal_year"].isin([2020, 2024])))
            merged = pd.concat([old[keep], combined], ignore_index=True)
            merged.to_csv(cands_path, index=False)
            print(f"Updated {cands_path} ({len(old)} -> {len(merged)} rows)")
    else:
        print("No DEI passages recovered (AAPL FY2020 is expected to be empty).")


if __name__ == "__main__":
    main()
