"""Market-performance side of the project (skeleton).

Plan: pull annual revenue per company, turn it into revenue volatility, then
correlate that with each company's DEI score.

Done:  yoy_growth, revenue_volatility (small, tested)
TODO:  fetch_revenue, link_score_to_volatility
"""
import numpy as np
import pandas as pd


def fetch_revenue(cik: str, user_agent: str) -> pd.DataFrame:
    """TODO: return a DataFrame with columns fiscal_year, revenue_usd for one company.

    Hints:
    - Endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
      (CIK zero-padded to 10 digits). Send a User-Agent with your name and email.
    - Revenue tag varies by company. Try, in order: "Revenues",
      "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet".
    - Keep annual 10-K values only (form == "10-K", fp == "FY"), one value per fiscal year.
    - Banks and insurers use different revenue tags. Check one early.
    """
    raise NotImplementedError("Write the EDGAR revenue pull here.")


def yoy_growth(revenue: pd.Series) -> pd.Series:
    """Year-over-year revenue growth. `revenue` is ordered by fiscal year."""
    return revenue.astype(float).pct_change().dropna()


def revenue_volatility(revenue: pd.Series, min_points: int = 3) -> float:
    """Standard deviation of YoY growth. Returns NaN if there are too few growth values.

    Note: with ~6 years of data this number is noisy. Say so in the write-up.
    """
    growth = yoy_growth(revenue)
    if len(growth) < min_points:
        return float("nan")
    return float(np.std(growth, ddof=1))


def link_score_to_volatility(df: pd.DataFrame):
    """TODO: Spearman correlation between DEI score and revenue volatility.

    Expect df with one row per company and columns: company, dei_score, volatility.
    Suggested: scipy.stats.spearmanr, plus a bootstrap 95% CI (resample companies).
    H0: rho = 0.  H1: rho < 0 (higher score, lower volatility).
    Decide before looking at results whether FY2020 (COVID) stays in the revenue series.
    """
    raise NotImplementedError("Write the correlation + bootstrap CI here.")
