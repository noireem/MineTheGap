## How to access and replicate the results

This section is for instructors and reviewers. It reproduces every number in the paper from the files in this repository. No API keys are needed.

### What you need

- Python 3.13 (the version used for the reported results) and `pip`.
- Internet access for the first run only: the embedding and FinBERT models download about 0.5 GB of weights from Hugging Face.
- All commands run from the repository root.

### Quick path: reproduce the reported results (no re-downloading)

The extracted passages, my labels, and the revenue data are committed, so you can skip data collection:

| File | What it is |
|---|---|
| `data/processed/passages_candidates.csv` | 111 candidate DEI passages extracted from the 10-Ks |
| `data/processed/passages_labeled.csv` | my 87 hand labels (columns: `company, fiscal_year, section, text, label, labeler`) |
| `data/processed/revenue_data_clean.csv` | annual revenue from SEC XBRL, FY2019-FY2025 |
| `data/processed/BLS_Consumer_Survey_Clean.csv` | BLS Consumer Expenditure Survey data for the Market Map |

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-models.txt                   # sentence-transformers, transformers, torch

python -m src.pipeline evaluate    # model comparison  -> reports/results.json, reports/figures/
python -m src.pipeline link        # scores + Spearman -> data/processed/company_scores.csv,
                                   #                      merged_dataset_final.csv, and "linkage" in results.json
streamlit run app/app.py           # the web app (Market Map, Company Lens, paste-a-statement)
pytest tests/                      # automated tests
```

### Where each number in the paper comes from

| In the paper | Where to find it |
|---|---|
| Macro-F1 with 95% CI and kappa, per model (Table: model comparison) | `reports/results.json`, keys `keyword`, `tfidf_lr`, `embedding_lr`, `finbert_zero_shot` |
| Confusion matrices and per-class F1 | same file, `confusion` and `report` under each model |
| Spearman rho, p-value, 95% CI (Table: link to volatility) | `reports/results.json`, key `linkage` |
| Company scores and passage counts per company-year | `data/processed/company_scores.csv` |
| Score vs. volatility figure | `data/processed/merged_dataset_final.csv`, filtered to `model == embedding_lr` |
| 111 candidates, 87 labeled passages, class counts 23 / 38 / 26 | row counts and `label` column of the two CSVs above |
| Extraction rules and validation | `reports/extraction_decisions.md` |

Expected headline values: macro-F1 0.21 (keyword), 0.38 (TF-IDF), 0.51 (embeddings), 0.38 (FinBERT); Spearman rho -0.25 (embeddings) and -0.11 (keyword).

### Full path: rebuild from the raw SEC filings (optional)

1. **Download the 10-Ks.** `notebooks/01_data_extraction.ipynb` uses `sec-edgar-downloader` (SEC requires an email address in the request header). Filings are saved to `data/raw/sec-edgar-filings/<TICKER>/10-K/<accession>/full-submission.txt`. The raw filings are not committed because of their size. The downloader filters by *filing* date, so use filing windows 2021 and 2025 for AMZN, IBM, JPM, MCD, META, TSLA, TGT, WMT and 2020 and 2024 for AAPL, COST, MSFT, PG (these give FY2020 and FY2024).
2. **Extract candidates:** `python -m src.pipeline build` prints a company-by-fiscal-year count table and writes `passages_candidates.csv`. Fiscal year comes from each filing's period-of-report date; January and February year-ends count toward the prior year.
3. **Label:** the labels are one person's judgment and cannot be regenerated, so use the committed `passages_labeled.csv`. The rubric is in the paper (Table: labeling rubric). `label` must be exactly `Celebrator`, `Tolerator`, or `Depreciator`; other values are dropped.
4. **Revenue:** `python -m src.pipeline revenue --email you@example.com`
5. Continue with `evaluate` and `link` as above.

### Reproducibility notes

- **Tested environment:** Python 3.13, numpy 2.4.0, scipy 1.16.3, scikit-learn 1.8.0, sentence-transformers 6.1.0, transformers 5.20.1, torch 2.14.1.
- **Models:** `sentence-transformers/all-MiniLM-L6-v2` (frozen embeddings) and `ProsusAI/finbert` (zero-shot), downloaded automatically from Hugging Face.
- **Determinism:** cross-validation folds are grouped by company and deterministic, and both bootstrap procedures use a fixed seed (0). Small numeric differences are still possible across library versions or hardware, and the embedding models depend on the weights Hugging Face serves at download time.
- **No leakage:** no company appears in both training and test folds, and each company's DEI score comes from a model trained only on the other companies' labels.
- **Scale:** this is an exploratory pilot (12 companies, 87 labeled passages, one labeler). `evaluate` prints a warning when there are fewer than 100 labels.