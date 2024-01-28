"""
Referring to BARRA-USE5, more explaination at: https://zhuanlan.zhihu.com/p/31412967
"""

import numpy as np
import factor as ft
import pandas as pd
from joblib import Parallel, delayed


QTD_URI = '/home/data/quotes-day'
IDXQTD_URI = '/home/data/index-quotes-day'
FIN_URI = '/home/data/financial'


def get_logsize(
    start: str, stop: str,
) -> pd.DataFrame:
    shares = ft.get_data(QTD_URI, "circulation_a", start=start, stop=stop)
    price = ft.get_data(QTD_URI, "close", start=start, stop=stop)
    adjfactor = ft.get_data(QTD_URI, "adjfactor", start=start, stop=stop)
    return -np.log(shares * price * adjfactor).loc[start:stop]

def get_beta(start: str, stop: str) -> pd.DataFrame:
    rollback = ft.get_trading_days_rollback(QTD_URI, start, 253)
    price = ft.get_data(QTD_URI, "close", start=rollback, stop=stop)
    index_price = ft.get_data(IDXQTD_URI, "close", pool="000985.XSHG", start=rollback, stop=stop).squeeze()
    adjfactor = ft.get_data(QTD_URI, "adjfactor", start=rollback, stop=stop)
    price *= adjfactor
    returns = price.pct_change(fill_method=None).iloc[1:]
    index_returns = index_price.pct_change(fill_method=None).iloc[1:]

    def _g_b(x):
        if x.shape[0] < 252:
            return pd.Series(index=x.columns)
        idxrt = index_returns.loc[x.index]
        return x.apply(lambda y:
            ((y - y.mean()) * (idxrt - idxrt.mean())).ewm(halflife=63).mean().iloc[-1] /
            ((idxrt - idxrt.mean()) ** 2).ewm(halflife=63).mean().iloc[-1]
        )
    
    beta = Parallel(n_jobs=-1, backend="loky")(delayed(_g_b)(x) for x in returns.rolling(252))
    return pd.concat(beta[252:], axis=1, keys=returns.index[252:]).T.loc[start:stop]

def get_momentum_20d(
    start: str, stop: str,
) -> pd.DataFrame:
    rollback = ft.get_trading_days_rollback(QTD_URI, start, 21)
    price = ft.get_data(QTD_URI, "close", start=rollback, stop=stop)
    adjfactor = ft.get_data(QTD_URI, "adjfactor", start=rollback, stop=stop)
    price *= adjfactor
    return -(price / price.shift(20) - 1).loc[start:stop]

def get_volatility_20d(
    start: str, stop: str,
) -> pd.DataFrame:
    rollback = ft.get_trading_days_rollback(QTD_URI, start, 22)
    price = ft.get_data(QTD_URI, "close", start=rollback, stop=stop)
    adjfactor = ft.get_data(QTD_URI, "adjfactor", start=rollback, stop=stop)
    price *= adjfactor
    returns = price / price.shift(1) - 1
    return -returns.rolling(20).std().loc[start:stop]

def get_ep(
    start: str, stop: str,
) -> pd.DataFrame:
    rollback = ft.get_trading_days_rollback(QTD_URI, start, 250)
    trading_days = ft.get_trading_days(QTD_URI, rollback, stop)
    shares = ft.get_data(QTD_URI, "circulation_a", start=start, stop=stop)
    price = ft.get_data(QTD_URI, "close", start=start, stop=stop)
    adjfactor = ft.get_data(QTD_URI, "adjfactor", start=start, stop=stop)
    value = price * adjfactor * shares
    net_profit = ft.get_data(FIN_URI, 'net_profit', start=rollback, stop=stop)
    net_profit = net_profit.reindex(trading_days).ffill()
    return (net_profit / value).loc[start:stop]

def get_bp(
    start: str, stop: str,
) -> pd.DataFrame:
    rollback = ft.get_trading_days_rollback(QTD_URI, start, 250)
    trading_days = ft.get_trading_days(QTD_URI, rollback, stop)
    shares = ft.get_data(QTD_URI, "circulation_a", start=start, stop=stop)
    price = ft.get_data(QTD_URI, "close", start=start, stop=stop)
    adjfactor = ft.get_data(QTD_URI, "adjfactor", start=start, stop=stop)
    value = price * adjfactor * shares
    totol_equity = ft.get_data(FIN_URI, 'total_equity', start=rollback, stop=stop)
    totol_equity = totol_equity.reindex(trading_days).ffill()
    return (totol_equity / value).loc[start:stop]

def get_roa(
    start: str, stop: str,
) -> pd.DataFrame:
    rollback = ft.get_trading_days_rollback(QTD_URI, start, 250)
    trading_days = ft.get_trading_days(QTD_URI, rollback, stop)
    net_profit = ft.get_data(FIN_URI, 'net_profit', start=rollback, stop=stop)
    net_profit = net_profit.reindex(trading_days).ffill()
    total_asset = ft.get_data(FIN_URI, 'total_assets', start=rollback, stop=stop)
    total_asset = total_asset.reindex(trading_days).ffill()
    return (net_profit / total_asset).loc[start:stop]

def get_roe(
    start: str, stop: str,
) -> pd.DataFrame:
    rollback = ft.get_trading_days_rollback(QTD_URI, start, 250)
    trading_days = ft.get_trading_days(QTD_URI, rollback, stop)
    net_profit = ft.get_data(FIN_URI, 'net_profit', start=rollback, stop=stop)
    net_profit = net_profit.reindex(trading_days).ffill()
    total_equity = ft.get_data(FIN_URI, 'total_equity', start=rollback, stop=stop)
    total_equity = total_equity.reindex(trading_days).ffill()
    return (net_profit / total_equity).loc[start:stop]

def get_current_asset_ratio(
    start: str, stop: str,
) -> pd.DataFrame:
    rollback = ft.get_trading_days_rollback(QTD_URI, start, 250)
    trading_days = ft.get_trading_days(QTD_URI, rollback, stop)
    current_assets = ft.get_data(FIN_URI, 'current_assets', start=rollback, stop=stop)
    current_assets = current_assets.reindex(trading_days).ffill()
    total_asset = ft.get_data(FIN_URI, 'total_assets', start=rollback, stop=stop)
    total_asset = total_asset.reindex(trading_days).ffill()
    return (current_assets / total_asset).loc[start:stop]
