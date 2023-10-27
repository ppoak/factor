import pandas as pd


def ret(
    price: pd.DataFrame,
    period: int = 20
):
    return price / price.shift(period) - 1

def hist_percent(
    price: pd.DataFrame,
):
    minp = price.cummin()
    maxp = price.cummax()
    return (price - minp) / (maxp - minp)