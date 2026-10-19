"""Market-performance side of the project: annual revenue, revenue volatility, and the
Spearman link between a company's DEI score and its volatility."""
import warnings
from datetime import date

import numpy as np
import pandas as pd

from src.data_loader import UNIVERSE, fiscal_year_of

# ASC 606 (post-~2018) first, then older total-revenue tags. Companies often switch
# tags; taking only the longest single series drops recent years for AAPL/AMZN/TGT.
REV_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "NetSales",
]


def parse_revenue(facts: dict) -> pd.DataFrame:
    """Annual revenue from a companyfacts JSON. Keeps 10-K full-year (350-380 day) values,
    one per fiscal year. Tags are merged (ASC 606 + legacy Revenues); the latest filing
    wins, and on the same filing the earlier tag in REV_TAGS wins. Fiscal year = calendar
    year of the period end, except Jan/Feb period ends (retailers) count as the prior year."""
    us = facts.get("facts", {}).get("us-gaap", {})
    rank = {tag: i for i, tag in enumerate(REV_TAGS)}
    rows = []
    for tag in REV_TAGS:
        for u in us.get(tag, {}).get("units", {}).get("USD", []):
            if u.get("form") != "10-K" or u.get("fp") != "FY" or "start" not in u:
                continue
            start, end = date.fromisoformat(u["start"]), date.fromisoformat(u["end"])
            if not 350 <= (end - start).days <= 380:
                continue
            rows.append((fiscal_year_of(end), float(u["val"]), u.get("filed", ""), rank[tag]))
    if not rows:
        return pd.DataFrame(columns=["fiscal_year", "revenue_usd"])
    df = pd.DataFrame(rows, columns=["fiscal_year", "revenue_usd", "filed", "tag_rank"])
    df = (df.sort_values(["filed", "tag_rank"], ascending=[True, False])
            .drop_duplicates("fiscal_year", keep="last")
            .sort_values("fiscal_year")[["fiscal_year", "revenue_usd"]]
            .reset_index(drop=True))
    return df


def fetch_revenue(cik: str, user_agent: str) -> pd.DataFrame:
    """Download one company's XBRL facts from SEC and return annual revenue."""
    import requests
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    return parse_revenue(resp.json())


def build_revenue_table(user_agent: str, universe: list = UNIVERSE) -> pd.DataFrame:
    import time
    frames = []
    for row in universe:
        df = fetch_revenue(row["cik"], user_agent)
        df.insert(0, "company", row["ticker"])
        frames.append(df)
        time.sleep(0.2)          # SEC asks for <= 10 requests per second
    return pd.concat(frames, ignore_index=True)


def volatility_table(revenue: pd.DataFrame, first_year: int = 2019, last_year: int = 2025) -> pd.DataFrame:
    """Per-company volatility of YoY revenue growth over [first_year, last_year]."""
    out = []
    for company, g in revenue[revenue.fiscal_year.between(first_year, last_year)].groupby("company"):
        g = g.sort_values("fiscal_year")
        out.append(dict(company=company, n_years=len(g),
                        volatility=revenue_volatility(g["revenue_usd"])))
    return pd.DataFrame(out, columns=["company", "n_years", "volatility"])


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


def link_score_to_volatility(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Spearman rho between dei_score and volatility (one row per company).
    H0: rho = 0, H1: rho < 0. Bootstrap 95% CI resamples companies."""
    from scipy.stats import spearmanr
    df = df.dropna(subset=["dei_score", "volatility"])
    if len(df) < 4:
        raise ValueError(f"Only {len(df)} companies with both values; need at least 4.")
    x, y = df["dei_score"].to_numpy(), df["volatility"].to_numpy()
    if len(set(x)) < 2 or len(set(y)) < 2:
        raise ValueError("DEI score or volatility is identical for every company, so a rank "
                         "correlation is undefined.")
    rho, p = spearmanr(x, y, alternative="less")
    rng = np.random.default_rng(seed)
    boots = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            i = rng.integers(0, len(df), len(df))
            if len(set(x[i])) < 2 or len(set(y[i])) < 2:
                continue
            boots.append(spearmanr(x[i], y[i])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_companies": int(len(df)), "spearman_rho": float(rho), "p_one_sided": float(p),
            "ci95": [float(lo), float(hi)], "n_boot_used": len(boots)}