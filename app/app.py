"""MineTheGap checkpoint app: paste a DEI statement, see the rubric's reading of it.

Run from the project root:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rubric import highlight_html, score_passage  # noqa: E402

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

st.set_page_config(page_title="MineTheGap", layout="centered")
st.title("MineTheGap")
st.caption(
    "Does a company's DEI language show commitment, compliance, or risk framing? "
    "Paste a passage from a 10-K and see how the keyword rubric reads it."
)

st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state["passage"] = text

passage = st.text_area("Passage", key="passage", height=180)

if passage.strip():
    result = score_passage(passage)
    c1, c2 = st.columns(2)
    c1.metric("Reading", result.label)
    c2.metric("Score (-1 to 1)", f"{result.score:+.2f}")
    st.caption(
        f"Cues found: {result.counts['Celebrator']} Celebrator, "
        f"{result.counts['Tolerator']} Tolerator, {result.counts['Depreciator']} Depreciator."
    )
    if result.label == "Unclear":
        st.info("No rubric cues were found, so the rubric makes no call on this passage.")
    st.markdown("**Cues highlighted**")
    st.markdown(highlight_html(passage, result.matches), unsafe_allow_html=True)

st.divider()
st.caption(
    "Baseline version: keyword rubric only. It describes language, not a company's intent. "
    "The trained embedding model will be added after the passages are labeled."
)
