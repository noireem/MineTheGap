import math

import pandas as pd

from src.market import revenue_volatility, yoy_growth


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
