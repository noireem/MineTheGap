import csv
from pathlib import Path

from src.rubric import highlight_html, score_passage

SEED = Path(__file__).resolve().parent.parent / "data" / "seed_passages.csv"


def test_seed_examples_get_their_labels():
    with open(SEED, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    for row in rows:
        assert score_passage(row["text"]).label == row["label"], row["source"]


def test_score_sign_matches_class():
    rows = list(csv.DictReader(open(SEED, newline="", encoding="utf-8")))
    by_label = {r["label"]: score_passage(r["text"]).score for r in rows}
    assert by_label["Celebrator"] > 0
    assert by_label["Depreciator"] < 0
    assert by_label["Tolerator"] == 0


def test_no_cues_is_unclear_not_a_guess():
    r = score_passage("Our headquarters are located in Minneapolis.")
    assert r.label == "Unclear" and r.score == 0.0


def test_empty_text_does_not_crash():
    assert score_passage("").label == "Unclear"


def test_highlight_escapes_html():
    text = "<script>alert(1)</script> We must comply with the law."
    r = score_passage(text)
    out = highlight_html(text, r.matches)
    assert "<script>" not in out
    assert "<mark" in out
