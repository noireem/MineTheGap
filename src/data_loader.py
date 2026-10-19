"""Load and parse the raw inputs: the company list, SEC 10-K filings, and the BLS spending file.
 
Filing layout expected (sec-edgar-downloader):
    <FILINGS_DIR>/**/<TICKER>/10-K/<accession>/full-submission.txt
"""


import re
from bs4 import BeautifulSoup
from src.config import BLS_CSV, FILINGS_DIR
from pathlib import Path
from datetime import date
import pandas as pd



UNIVERSE = [
    dict(ticker="TGT", company="Target", exchange="NYSE", sector="Retail", cik="27419",
         original_request="Target", swapped=False, swap_reason=""),
    dict(ticker="MSFT", company="Microsoft", exchange="NASDAQ", sector="Technology", cik="789019",
         original_request="Microsoft", swapped=False, swap_reason=""),
    dict(ticker="AAPL", company="Apple", exchange="NASDAQ", sector="Technology", cik="320193",
         original_request="private company on the original list", swapped=True,
         swap_reason="Student-chosen public replacement for a private company on the original list"),
    dict(ticker="PG", company="Procter & Gamble", exchange="NYSE", sector="Consumer Goods", cik="80424",
         original_request="P&G", swapped=False, swap_reason=""),
    dict(ticker="MCD", company="McDonald's", exchange="NYSE", sector="Restaurants", cik="63908",
         original_request="Chick-fil-A", swapped=True,
         swap_reason="Chick-fil-A is privately held; McDonald's is a public quick-service peer"),
    dict(ticker="IBM", company="IBM", exchange="NYSE", sector="Technology", cik="51143",
         original_request="IBM", swapped=False, swap_reason=""),
    dict(ticker="COST", company="Costco", exchange="NASDAQ", sector="Retail", cik="909832",
         original_request="Costco", swapped=False, swap_reason=""),
    dict(ticker="JPM", company="JPMorgan Chase", exchange="NYSE", sector="Financials", cik="19617",
         original_request="JPMC", swapped=False, swap_reason=""),
    dict(ticker="TSLA", company="Tesla", exchange="NASDAQ", sector="Automotive", cik="1318605",
         original_request="private company on the original list", swapped=True,
         swap_reason="Student-chosen public replacement for a private company on the original list"),
    dict(ticker="META", company="Meta Platforms", exchange="NASDAQ", sector="Technology", cik="1326801",
         original_request="Meta", swapped=False, swap_reason=""),
    dict(ticker="WMT", company="Walmart", exchange="NASDAQ", sector="Retail", cik="104169",
         original_request="Walmart", swapped=False, swap_reason=""),
    dict(ticker="AMZN", company="Amazon", exchange="NASDAQ", sector="Retail", cik="1018724",
         original_request="Amazon", swapped=False, swap_reason=""),
]
 
 
PUBLIC_EXCHANGES = {"NYSE", "NASDAQ"}

def validate_universe(rows: list) -> list:
    """Return a list of problems (empty = usable). Enforces: public, unique, explained swaps."""
    problems = []
    tickers = [r["ticker"] for r in rows]
    if len(set(tickers)) != len(tickers):
        problems.append("duplicate tickers")
    for r in rows:
        if r["exchange"] not in PUBLIC_EXCHANGES:
            problems.append(f"{r['ticker']}: not on a public exchange")
        if not str(r["cik"]).isdigit():
            problems.append(f"{r['ticker']}: CIK must be digits")
        if r["swapped"] and not r["swap_reason"]:
            problems.append(f"{r['ticker']}: swapped but no reason")
    return problems
 
KEYWORDS = re.compile(r"\b(diversity|inclusion|inclusive(?!\s+of\b)|DEI|belonging|racial|African[- ]American|"
                      r"underrepresented|Black(?! Friday|[-\u2010-\u2013 ]Scholes))\b", re.IGNORECASE)
 
 


def fiscal_year_of(period_end):
    """
    Returns selected accession FY(Fiscal Year)
    
    """
    return period_end.year - 1 if period_end.month <= 2 else  period_end.year #Edge case: Jan/Feb FY endings belong to previous year

def build_inventory(filings_dir: Path = FILINGS_DIR) -> pd.DataFrame:
    """Table of downloaded 10-Ks: ticker, fiscal year, accession, path, size.

    Walks <filings_dir>/**/<TICKER>/10-K/<accession>/full-submission.txt.
    """
    rows = []
    filings_dir = Path(filings_dir)

    for path in sorted(filings_dir.rglob("full-submission.txt")):
        if "10-K" not in path.parts:
            continue
        ticker = path.parts[path.parts.index("10-K") - 1].upper()
        raw = path.read_text(encoding="utf-8", errors="ignore")
        period_date = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{4})(\d{2})(\d{2})", raw)

        if not period_date:
            print(f"Warning: No period data found for {ticker} \n path:{path}")
            continue
        period = date(*map(int, period_date.groups()))
        rows.append(dict(
            ticker=ticker,
            fiscal_year=fiscal_year_of(period),
            period=period,
            accession=path.parent.name,
            path=str(path),
            size_mb=round(path.stat().st_size / 1e6, 2),
        ))
    return pd.DataFrame(rows, columns=["ticker", "fiscal_year", "period", "accession", "path", "size_mb"])


def ten_K(raw):
    for doc in re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", raw, flags=re.S):
        if re.search(r"<TYPE>\s*10-K\s*$", doc, flags=re.M):
            m = re.search(r"<TEXT>(.*?)(?:</TEXT>|$)",doc, flags=re.S)
            return m.group(1) if m else doc
    return None


BLOCKS = ["p", "div", "li", "td"]

def block_texts(soup):
    out=[]
    for el in soup.find_all(BLOCKS):
        if el.find(BLOCKS):
            continue
        t=re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        t=re.sub(r"\s+([,.;:])", r"\1", t)
        if t:
            out.append(t)
    return out



def section_of(text, current):
    s = text.lower()
    if re.match(r"item\s*1a\b", s): return "Item 1A"
    if re.match(r"item\s*(1b|2)\b", s): return "other"
    if re.match(r"item\s*1(?!\w)", s): return "Item 1"
    return current

def _drop_non_narrative(soup):
    for t in soup(["script", "style"]):
        t.decompose()
    hidden = [t for t in soup.find_all(True) if (t.name or "").lower().endswith("hidden")]
    for t in hidden:
        t.decompose()


def extract(raw, company, fy):
    # full-submission.txt wraps the 10-K in <DOCUMENT> tags; build_candidates
    # already unwraps that and passes the HTML body.
    html = ten_K(raw) if re.search(r"<DOCUMENT>", raw) else raw
    if html is None:
        return []
    soup = BeautifulSoup(html, "lxml")
    _drop_non_narrative(soup)
    rows, seen, section = [], set(), "unknown"
    for text in block_texts(soup):
        section = section_of(text, section)
        if 80 <= len(text) <= 2000 and KEYWORDS.search(text) and text not in seen:
            seen.add(text)
            rows.append(dict(company=company, fiscal_year=fy, section=section,
                             text=text, label="", labeler=""))
    return rows

NARRATIVE_TYPES = ("10-K", "EX-13")  # IBM puts Item 1 in the annual report (Exhibit 13)


def document_html_by_type(raw: str) -> dict:
    """First (longest if duplicated) HTML body for each <TYPE> in a full-submission.txt."""
    found = {}
    for doc in re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", raw, flags=re.S):
        tm = re.search(r"<TYPE>\s*(\S+)", doc)
        if not tm:
            continue
        dtype = tm.group(1).strip()
        text = re.search(r"<TEXT>(.*?)(?:</TEXT>|$)", doc, flags=re.S)
        html = text.group(1) if text else doc
        if dtype not in found or len(html) > len(found[dtype]):
            found[dtype] = html
    return found


def narrative_html(raw: str, submission=None):
    """10-K body plus Exhibit 13 when present (cleaned of nothing yet; still HTML)."""
    chunks = []
    primary = submission.parent / "primary-document.html" if submission is not None else None
    docs = document_html_by_type(raw)
    if primary is not None and primary.exists():
        chunks.append(primary.read_text(encoding="utf-8", errors="ignore"))
    elif "10-K" in docs:
        chunks.append(docs["10-K"])
    if "EX-13" in docs:
        chunks.append(docs["EX-13"])
    return "\n".join(chunks) if chunks else None


def cleaned_filing_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    _drop_non_narrative(soup)
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))


def read_downloaded_filing(submission: Path):
    """(fiscal_year, html) for one sec-edgar-downloader folder, or None if unreadable.
    The fiscal year comes from the filing's own 'CONFORMED PERIOD OF REPORT', not the folder name.
    Uses primary-document.html when present, else the 10-K document; appends EX-13."""
    raw = submission.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{4})(\d{2})(\d{2})", raw)
    if not m:
        return None
    fy = fiscal_year_of(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    html = narrative_html(raw, submission)
    return None if not html else (fy, html)

def build_candidates(filings_dir: Path = FILINGS_DIR) -> pd.DataFrame:
    """Extract passages from every downloaded 10-K: <dir>/**/<TICKER>/10-K/<accession>/full-submission.txt.
    If a company has two filings for the same fiscal year, the larger document is kept."""
    best = {}
    for sub in sorted(Path(filings_dir).rglob("full-submission.txt")):
        parts = sub.parts
        if "10-K" not in parts:
            continue
        ticker = parts[parts.index("10-K") - 1].upper()
        got = read_downloaded_filing(sub)
        if got is None:
            continue
        fy, html = got
        if (ticker, fy) not in best or len(html) > len(best[(ticker, fy)]):
            best[(ticker, fy)] = html
    if not best:
        raise FileNotFoundError(f"No <TICKER>/10-K/<accession>/full-submission.txt files under {filings_dir}")
    frames = [pd.DataFrame(extract(h, t, fy)) for (t, fy), h in sorted(best.items())]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["company", "fiscal_year", "section", "text", "label", "labeler"])
    return pd.concat(frames, ignore_index=True)

def load_bls(path: Path = BLS_CSV) -> pd.DataFrame:
    """Parse the block-structured BLS export into category, year, avg_spend,
    aggregate_millions, share_of_us_aggregate. Tolerates stray quotes and a malformed header."""
    category, header, rows = None, None, []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.replace('"', "").strip()
        if line.startswith("Subcategory:"):
            category, header = line.split(":", 1)[1].strip(), None
        elif line.startswith("Year,"):
            header = [h.strip() for h in line.split(",") if h.strip()]
        elif category and header and re.match(r"^\d{4},", line):
            # BLS footnote markers like "1354(6)" are stripped. Some rows carry an extra
            # suppression field ("-(5)"), so the aggregate and its share are read from the END
            # of the row, where the header says they always are.
            parts = [re.sub(r"\(\d+\)", "", p).strip() for p in line.split(",")]
            if not (header[-1].startswith("Share of Aggregate") and header[-2].startswith("Aggregate")):
                continue
            try:
                rows.append(dict(category=category, year=int(parts[0]), avg_spend=float(parts[2]),
                                 aggregate_millions=float(parts[-2]),
                                 share_of_us_aggregate=float(parts[-1])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["category", "year", "avg_spend", "aggregate_millions",
                                       "share_of_us_aggregate"])
 
 
def spending_categories(bls: pd.DataFrame) -> pd.DataFrame:
    """Drop the grand total and the net-assets line, which are not spending categories."""
    skip = {"Total average annual expenditures", "Net change in total assets and liabilities"}
    return bls[~bls["category"].isin(skip)]
