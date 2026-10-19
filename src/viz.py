"""Chart builders (plotly) used by the Streamlit app."""
import pandas as pd


def market_bar(cats: pd.DataFrame, year: int):
    """Horizontal bar of aggregate Black-household spending by category for one year."""
    import plotly.express as px
    d = cats[cats["year"] == year].sort_values("aggregate_millions")
    return px.bar(d, x="aggregate_millions", y="category", orientation="h",
                  labels={"aggregate_millions": "Aggregate spending ($ millions)", "category": ""},
                  title=f"Where Black households' spending went, {year}")


def score_scatter(merged: pd.DataFrame):
    """DEI score vs revenue volatility, one point per company."""
    import plotly.express as px
    return px.scatter(merged, x="dei_score", y="volatility", text="company",
                      labels={"dei_score": "DEI score", "volatility": "Revenue volatility"})