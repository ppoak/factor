import factor
import numpy as np
import pandas as pd


def get_logsize(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    shares = factor.get_data(qtd_uri, "circulation_a", start=start, stop=stop)
    price = factor.get_data(qtd_uri, "close", start=start, stop=stop)
    adjfactor = factor.get_data(qtd_uri, "adjfactor", start=start, stop=stop)
    return np.log(shares * price * adjfactor)

def get_momentum_20d(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    rollback = factor.get_trading_days_rollback(qtd_uri, start, 21)
    price = factor.get_data(qtd_uri, "close", start=rollback, stop=stop)
    adjfactor = factor.get_data(qtd_uri, "adjfactor", start=rollback, stop=stop)
    price *= adjfactor
    return (price / price.shift(20) - 1).loc[start:stop]

def get_volatility_20d(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    rollback = factor.get_trading_days_rollback(qtd_uri, start, 22)
    price = factor.get_data(qtd_uri, "close", start=rollback, stop=stop)
    adjfactor = factor.get_data(qtd_uri, "adjfactor", start=rollback, stop=stop)
    price *= adjfactor
    returns = price / price.shift(1) - 1
    return returns.rolling(20).std().loc[start:stop]

def get_ep(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    fin_uri = '/home/data/financial'
    rollback = factor.get_trading_days_rollback(qtd_uri, start, 250)
    trading_days = factor.get_trading_days(qtd_uri, rollback, stop)
    shares = factor.get_data(qtd_uri, "circulation_a", start=start, stop=stop)
    price = factor.get_data(qtd_uri, "close", start=start, stop=stop)
    adjfactor = factor.get_data(qtd_uri, "adjfactor", start=start, stop=stop)
    value = price * adjfactor * shares
    net_profit = factor.get_data(fin_uri, 'net_profit', start=rollback, stop=stop)
    net_profit = net_profit.reindex(trading_days).ffill()
    return (net_profit / value).loc[start:stop]
