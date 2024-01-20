import numpy as np
import pandas as pd
from factor import get_data


def get_logsize(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    shares = get_data(qtd_uri, "circulation_a", start=start, stop=stop)
    price = get_data(qtd_uri, "close", start=start, stop=stop)
    adjfactor = get_data(qtd_uri, "adjfactor", start=start, stop=stop)
    return np.log(shares * price * adjfactor)

def get_momentum_20d(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    price = get_data(qtd_uri, "close", start=start, stop=stop)
    adjfactor = get_data(qtd_uri, "adjfactor", start=start, stop=stop)
    price *= adjfactor
    return price / price.shift(20) - 1

def get_volatility_20d(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    price = get_data(qtd_uri, "close", start=start, stop=stop)
    adjfactor = get_data(qtd_uri, "adjfactor", start=start, stop=stop)
    price *= adjfactor
    returns = price / price.shift(1) - 1
    return returns.rolling(20).std()

def get_ep(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    fin_uri = '/home/data/financial'
    shares = get_data(qtd_uri, "circulation_a", start=start, stop=stop)
    price = get_data(qtd_uri, "close", start=start, stop=stop)
    adjfactor = get_data(qtd_uri, "adjfactor", start=start, stop=stop)
    value = price * adjfactor * shares
    net_profit = get_data(fin_uri, 'net_profit', start=start, stop=stop)
    net_profit = net_profit.reindex(price.index).ffill()
    return net_profit / value
