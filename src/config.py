"""Paths shared by the whole project (everything is relative to the repo root)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILINGS_DIR = ROOT / "data" / "raw" / "sec-edgar-filings" / "sec-edgar-filings"  # sec-edgar-downloader output
INVENTORY_CSV = ROOT / "data" / "processed" / "filing_inventory.csv"
CANDIDATES = ROOT / "data" / "processed" / "passages_candidates.csv"
LABELED = ROOT / "data" / "processed" / "passages_labeled.csv"
REVENUE_CSV = ROOT / "data" / "processed" / "revenue_data_clean.csv"
SCORES_CSV = ROOT / "data" / "processed" / "company_scores.csv"
MERGED_CSV = ROOT / "data" / "processed" / "merged_dataset_final.csv"
BLS_CSV = ROOT / "data" / "processed" / "BLS_Consumer_Survey_Clean.csv"
RESULTS_JSON = ROOT / "reports" / "results.json"
FIG_DIR = ROOT / "reports" / "figures"
