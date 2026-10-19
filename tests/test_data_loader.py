from bs4 import BeautifulSoup

from src.data_loader import block_texts, extract, ten_K

KEPT = ("We are committed to diversity and inclusion across our workforce and the communities "
        "we serve, and we report progress each year.")
RISK = ("Our diversity focus might not resonate with some guests, which could damage our "
        "reputation and brand over time.")


def filing(body_html):
    """Made-up full-submission.txt: an exhibit comes first, the 10-K second."""
    return ("<DOCUMENT>\n<TYPE>EX-99\n<TEXT>exhibit</TEXT>\n</DOCUMENT>\n"
            f"<DOCUMENT>\n<TYPE>10-K\n<TEXT>\n<html><body>{body_html}</body></html>\n</TEXT>\n</DOCUMENT>")


def test_keyword_paragraph_is_kept_and_tagged_with_its_section():
    raw = filing(f"<div>Item 1. Business</div><div>{KEPT}</div>"
                 "<div>Our headquarters are in a large building in a mid-sized American city downtown.</div>")
    rows = extract(raw, "TGT", 2020)
    assert len(rows) == 1                                   # the irrelevant paragraph is dropped
    assert rows[0]["section"] == "Item 1" and rows[0]["company"] == "TGT"


def test_duplicate_paragraph_is_dropped():
    raw = filing(f"<div>{RISK}</div><div>{RISK}</div>")
    assert len(extract(raw, "TGT", 2020)) == 1


def test_text_inside_script_tags_is_ignored():
    raw = filing(f"<div>{KEPT}</div><script>{KEPT} but this is javascript, not a passage.</script>")
    assert len(extract(raw, "TGT", 2020)) == 1


def test_paragraph_split_by_inline_tags_stays_whole():
    raw = filing("<div><span>We are committed to </span><b>diversity and inclusion</b>"
                 "<span> across our workforce and the communities we serve, and we report progress.</span></div>")
    rows = extract(raw, "TGT", 2020)
    assert len(rows) == 1 and rows[0]["text"].endswith("progress.")


def test_finds_the_10k_even_when_an_exhibit_comes_first():
    assert ten_K(filing("<div>body</div>")) is not None


def test_no_10k_document_returns_nothing():
    assert extract("<DOCUMENT>\n<TYPE>8-K\n<TEXT>x</TEXT>\n</DOCUMENT>", "TGT", 2020) == []


def test_block_texts_returns_every_block_not_just_the_first():
    soup = BeautifulSoup("<div>A</div><div>B</div><div>C</div>", "lxml")
    assert block_texts(soup) == ["A", "B", "C"]

def test_black_scholes_and_black_friday_are_not_dei_hits():
    from src.data_loader import KEYWORDS
    assert not KEYWORDS.search("We value options using the Black-Scholes model.")
    assert not KEYWORDS.search("Sales rise around Black Friday.")
    assert KEYWORDS.search("We support Black-owned suppliers.")
    assert not KEYWORDS.search("The purchase price was $1.1 billion, inclusive of contingent consideration.")
    assert KEYWORDS.search("We strive to provide programs that are culturally relevant and inclusive.")


def test_extract_accepts_unwrapped_html_from_build_candidates():
    html = f"<html><body><div>Item 1. Business</div><div>{KEPT}</div></body></html>"
    rows = extract(html, "TGT", 2024)
    assert len(rows) == 1 and rows[0]["company"] == "TGT"


def test_build_candidates_returns_a_dataframe(tmp_path):
    from src.data_loader import build_candidates

    sub = tmp_path / "TGT" / "10-K" / "0000027419-21-000010"
    sub.mkdir(parents=True)
    (sub / "full-submission.txt").write_text(
        "CONFORMED PERIOD OF REPORT:\t20210130\n"
        "<DOCUMENT>\n<TYPE>10-K\n<TEXT>\n"
        f"<html><body><div>Item 1. Business</div><div>{KEPT}</div></body></html>\n"
        "</TEXT>\n</DOCUMENT>",
        encoding="utf-8",
    )
    df = build_candidates(tmp_path)
    assert list(df.columns) == ["company", "fiscal_year", "section", "text", "label", "labeler"]
    assert len(df) == 1 and df.loc[0, "company"] == "TGT" and int(df.loc[0, "fiscal_year"]) == 2020


def test_build_candidates_uses_exhibit_13_when_10k_has_no_dei(tmp_path):
    from src.data_loader import build_candidates

    sub = tmp_path / "IBM" / "10-K" / "0001558370-21-001489"
    sub.mkdir(parents=True)
    (sub / "full-submission.txt").write_text(
        "CONFORMED PERIOD OF REPORT:\t20201231\n"
        "<DOCUMENT>\n<TYPE>10-K\n<TEXT>\n"
        "<html><body><div>Item 1. Business</div>"
        "<div>IBM is a technology company with operations worldwide and a long operating history.</div>"
        "</body></html>\n</TEXT>\n</DOCUMENT>\n"
        "<DOCUMENT>\n<TYPE>EX-13\n<TEXT>\n"
        f"<html><body><div>Item 1. Business</div><div>{KEPT}</div></body></html>\n"
        "</TEXT>\n</DOCUMENT>",
        encoding="utf-8",
    )
    df = build_candidates(tmp_path)
    assert len(df) == 1 and df.loc[0, "company"] == "IBM"


def test_build_inventory_reads_conformed_period(tmp_path):
    from datetime import date
    from src.data_loader import build_inventory

    sub = tmp_path / "AAPL" / "10-K" / "0000320193-24-000123"
    sub.mkdir(parents=True)
    (sub / "full-submission.txt").write_text(
        "CONFORMED PERIOD OF REPORT:\t20240928\n", encoding="utf-8"
    )
    inv = build_inventory(tmp_path)
    assert list(inv["ticker"]) == ["AAPL"]
    assert list(inv["fiscal_year"]) == [2024]
    assert inv.loc[0, "period"] == date(2024, 9, 28)