"""MineTheGap web app (Streamlit). This file only draws the UI; the logic lives in src/.

Run from the project root:  streamlit run app/app.py
Pipeline commands:          python -m src.pipeline --help
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.data_loader import UNIVERSE, load_bls, spending_categories  # noqa: E402
from src.evaluate import primary_labels  # noqa: E402
from src.nlp_engine import TfidfLR  # noqa: E402
from src.rubric import highlight_html, score_passage  # noqa: E402
from src.viz import market_bar, score_scatter  # noqa: E402


EXAMPLES = {
    "Citi (Celebrator example)": (
        "The privileges we enjoy by working for Citi come with responsibilities. "
        "While elected officials and community activists must do their part, so must we. "
        "One important thing we can do is to show the costs of racial inequality "
        "through objective analysis."
    ),
    "Generic boilerplate (Tolerator example)": (
        "The XYZ098 Company is an Equal Opportunity Employer and does not discriminate "
        "based on race, color, or gender. We are committed to complying with all "
        "applicable federal and state labor laws regarding fair employment practices."
    ),
    "Target (Depreciator example)": (
        "Our continued focus on diversity, equity and inclusion and our portfolio of "
        "relevant brands and accessible product assortments might not resonate with some "
        "of our guests, which could damage our reputation and brand."
    ),
}


@st.cache_resource
def _trained_tfidf():
    """TF-IDF model trained on the hand labels, if enough exist. Otherwise None."""
    if not config.LABELED.exists():
        return None
    df = primary_labels(pd.read_csv(config.LABELED))
    if len(df) < 30 or df["label"].nunique() < 2:
        return None
    return TfidfLR().fit(df["text"], df["label"])


def _paste_tab():
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state["passage"] = text
    passage = st.text_area("Passage", key="passage", height=180)
    if not passage.strip():
        return
    result = score_passage(passage)
    c1, c2 = st.columns(2)
    c1.metric("Reading", result.label)
    c2.metric("Score (-1 to 1)", f"{result.score:+.2f}")
    st.caption(f"Cues found: {result.counts['Celebrator']} Celebrator, "
               f"{result.counts['Tolerator']} Tolerator, {result.counts['Depreciator']} Depreciator.")
    if result.label == "Unclear":
        st.info("No rubric cues were found, so the rubric makes no call on this passage.")
    model = _trained_tfidf()
    if model is not None:
        st.metric("Trained model reading (TF-IDF)", str(model.predict([passage])[0]))
    else:
        st.caption("Trained model not shown: it needs 30+ hand-labeled passages in "
                   "data/passages_labeled.csv.")
    st.markdown("**Cues highlighted**")
    st.markdown(highlight_html(passage, result.matches), unsafe_allow_html=True)


def _market_tab():
    if not config.BLS_CSV.exists():
        st.info("BLS file not found at data/processed/BLS_Consumer_Survey_Clean.csv.")
        return
    cats = spending_categories(load_bls(config.BLS_CSV))
    year = st.selectbox("Year", sorted(cats["year"].unique(), reverse=True), key="market_year")
    st.plotly_chart(market_bar(cats, year), use_container_width=True)
    st.caption("Source: BLS Consumer Expenditure Survey, reference person Black or African American. "
               "Aggregates are survey estimates, not the McKinsey $910B projection.")


def _lens_tab():
    if not (config.SCORES_CSV.exists() and config.MERGED_CSV.exists()):
        st.info("No company scores yet. Run: python -m src.pipeline build, revenue, evaluate, then link.")
        return
    scores, merged = pd.read_csv(config.SCORES_CSV), pd.read_csv(config.MERGED_CSV)
    models = sorted(scores["model"].unique())
    model = st.selectbox("Scoring model", models, index=len(models) - 1, key="lens_model")
    company = st.selectbox("Company", sorted(scores["company"].unique()), key="lens_company")
    rows = scores[(scores["model"] == model) & (scores["company"] == company)]
    st.dataframe(rows[["fiscal_year", "class", "dei_score", "n_passages"]], hide_index=True)
    cik = next((u["cik"] for u in UNIVERSE if u["ticker"] == company), None)
    if cik:
        st.markdown(f"[Filings on EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                    f"&CIK={cik}&type=10-K)")
    if config.CANDIDATES.exists():
        cands = pd.read_csv(config.CANDIDATES)
        for _, r in cands[cands["company"] == company].head(5).iterrows():
            res = score_passage(r["text"])
            st.markdown(f"**FY{r['fiscal_year']} ({r['section']})**")
            st.markdown(highlight_html(r["text"], res.matches), unsafe_allow_html=True)
    m = merged[merged["model"] == model]
    st.plotly_chart(score_scatter(m), use_container_width=True)
    st.caption(f"Exploratory: {len(m)} companies. Association, not causation.")


def render_ui():
    st.set_page_config(page_title="MineTheGap", layout="centered")
    st.title("MineTheGap")
    st.caption("Does a company's DEI language show commitment, compliance, or risk framing? "
               "It describes language, not a company's intent.")
    tab1, tab2, tab3 = st.tabs(["Paste a statement", "Market Map", "Company Lens"])
    with tab1:
        _paste_tab()
    with tab2:
        _market_tab()
    with tab3:
        _lens_tab()


render_ui()