# The ROI of DEI: Exploring Black Consumer Power & Corporate DEI Performance

> **A Socio-Technical Analysis of the "Black Dollar" and Corporate Effectiveness**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## Executive Summary
This project audits the feedback loop between the **Black consumer market** (Social Subsystem) and **corporate economic performance** (Technical Subsystem).

Current economic models often underprioritize the power of the Black consumer input, creating a "Socio-Technical Gap" between corporate DEI promises and actual market effectiveness. Citigroup estimates that the failure to integrate Black consumers into the technical subsystem has cost the US economy roughly **$16 trillion over the past 20 years**.

**Goal:** To quantify the "Black Dollar" as a market input and analyze if high DEI sentiment in corporate reporting correlates with better capture of this demographic.

## Core Research Questions
1. **Visualization:** How is the **$910 billion** in Black consumer spending distributed across the economy? 
2.  **Leverage Analysis:** Which sectors (e.g., Housing, Essentials) and specific entities (e.g., Blackstone, P&G) hold the highest leverage over this wealth? 
3. **NLP Classification:** Can Natural Language Processing effectively classify corporations based on 10-K filings into "Celebrators," "Tolerators," or "Depreciators"? 
4. **Impact:** Is there a statistically significant correlation between a company's DEI Sentiment Score and its revenue growth?

## 🛠 Methodology

### Part 1: Visualizing the Impact
* **Objective:** Map the distribution of Black consumer spending and wealth accumulation.
* **Actions:**
    * Create market heat maps for spending breakdowns (Living Necessities vs. Modern Essentials). 
    * Analyze the spike in "Net House Wealth" from 2019-2022. 

### Part 2: NLP Sentiment Analysis
* **Objective:** Scrape and analyze S&P 500 10-K filings and ESG reports to score corporate sentiment.
* **Classification Framework:** 
    * 🟢 **Celebrators (Coupled):** Use active voice ("We must"), ownership language, and specific actions.
    * 🟡 **Tolerators (Decoupled):** Use boilerplate compliance language ("Complying with laws") and passive commitment.
    * 🔴 **Depreciators (Rejection):** Frame diversity as a liability, litigation risk, or use defensive tones.

### Part 3: Analytical Linkage
* **Objective:** Regression modeling to test the relationship between sentiment and financial volatility.
* **Model:** $Y (Revenue Growth) \sim X_1 (Sentiment Score) + X_2 (Sector Control)$ 
* **Hypothesis:** Companies with higher DEI Sentiment Scores will display lower revenue volatility. 

## 💻 Tech Stack
* **Language:** Python
* **IDE:** VS Code (Cursor for debugging)
* **Data & NLP:** Jupyter Notebooks, Pandas, NLP Libraries (Spacy/NLTK/HuggingFace)
* **Visualization:** Streamlit, MapBox/Tableau
* **CI/CD:** Git

## 📊 Data Sources
* **McKinsey Institute for Black Economic Mobility:** For wealth and spending gap data. 
* **Federal Reserve Survey of Consumer Finances:** For asset accumulation data. 
* **SEC EDGAR Database:** For scraping public 10-K annual reports. 

## ⚠️ Scope & Disclaimer
This project is an exploration of sentiments, history, and data based on my interests as a societally focused data scientist.

* **Not Political:** This is not a political statement, nor is it a moral judgment on whether DEI "should" exist. 
* **Not Defamation:** This project does not seek to defame or misrepresent any organization; it strictly analyzes public text data. 
* **Empowerment Focus:** The goal is to remind systemically undervalued people of their economic worth and to provide data-driven follow-ups to historical disparities. 

# Checkpoint 2

**What this is:** a small, testable first version of the individual personal Project. Paste a passage from a 10-K and the
keyword rubric tells you whether it reads as Celebrator, Tolerator, or Depreciator, and
highlights the words that drove the call. This is the non-AI baseline that the trained
model will later be compared against.

## Run it

```
pip install streamlit pandas scikit-learn pytest
streamlit run app/app.py          # the app
python -m pytest                  # 12 tests
python -m src.evaluate data/seed_passages.csv   # baseline score on labeled passages
```

## What is in here

| Path | Purpose |
|---|---|
| `src/rubric.py` | The keyword rubric (cues come from the project brief) |
| `src/evaluate.py` | Macro-F1 and confusion matrix against your hand labels |
| `app/app.py` | One-page Streamlit app |
| `data/seed_passages.csv` | The 3 examples from the brief, already labeled |
| `data/labeling_template.csv` | Blank file for the ~100 passages still to label |
| `src/market.py` | Market-performance skeleton: volatility math works; EDGAR revenue pull and correlation are TODO |
| `data/companies_template.csv` | Blank file for the 12 companies (name, ticker, CIK, sector) |
| `data/revenue_template.csv` | Blank file for annual revenue per company |
| `tests/` | Unit tests for the rubric, the app, and the volatility math |

## Not done yet 

- The 12-company sample is not fixed and no 10-K passages are extracted yet.
- Only 3 passages are labeled (the brief's examples), so the 1.00 macro-F1 on them is a
  smoke test, not a result. The rubric cues were written from those same examples.
- The market side is a skeleton: no revenue is pulled yet and the DEI-score-to-volatility
  correlation is not written (see the TODOs in `src/market.py`).
- The embedding classifier has not been trained. It needs the labeled passages first.
