import math

import pandas as pd

from src.market import parse_revenue, revenue_volatility, yoy_growth


def _facts(*points):
    """companyfacts-shaped JSON. Each point is (tag, start, end, val, filed)."""
    us = {}
    for tag, start, end, val, filed in points:
        us.setdefault(tag, {"units": {"USD": []}})
        us[tag]["units"]["USD"].append(
            {"form": "10-K", "fp": "FY", "start": start, "end": end, "val": val, "filed": filed}
        )
    return {"facts": {"us-gaap": us}}


def test_parse_revenue_merges_legacy_and_asc606_tags():
    # AAPL-style: Revenues through 2017, ASC 606 from 2018. Picking only the longest
    # tag used to drop the recent years.
    facts = _facts(
        ("Revenues", "2016-09-25", "2017-09-30", 229_234_000_000, "2017-11-03"),
        ("RevenueFromContractWithCustomerExcludingAssessedTax",
         "2017-10-01", "2018-09-29", 265_595_000_000, "2018-11-05"),
        ("RevenueFromContractWithCustomerExcludingAssessedTax",
         "2023-10-01", "2024-09-28", 391_035_000_000, "2024-11-01"),
    )
    df = parse_revenue(facts).set_index("fiscal_year")["revenue_usd"]
    assert df.loc[2017] == 229_234_000_000
    assert df.loc[2018] == 265_595_000_000
    assert df.loc[2024] == 391_035_000_000


def test_yoy_growth_values():
    g = yoy_growth(pd.Series([100, 110, 121]))
    assert [round(x, 3) for x in g] == [0.1, 0.1]


def test_steady_growth_has_zero_volatility():
    revenue = pd.Series([100 * 1.1**i for i in range(7)])
    assert revenue_volatility(revenue) < 1e-9


def test_bumpy_revenue_is_more_volatile_than_steady():
    steady = pd.Series([100, 105, 110, 116, 122, 128])
    bumpy = pd.Series([100, 130, 90, 140, 95, 150])
    assert revenue_volatility(bumpy) > revenue_volatility(steady)


def test_too_few_points_returns_nan():
    assert math.isnan(revenue_volatility(pd.Series([100, 110, 120])))
