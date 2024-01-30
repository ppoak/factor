"""
Referring to BARRA-USE5, more explaination at: https://zhuanlan.zhihu.com/p/31412967
About how to construct the regression and how to solve, refer to https://zhuanlan.zhihu.com/p/39922829
"""

import quool
import numpy as np
import factor as ft
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed


BARRA_URI = "/home/data/barra"
QTD_URI = '/home/data/quotes-day'
IDXQTD_URI = '/home/data/index-quotes-day'
IDS_URI = "/home/data/industry-info"
FIN_URI = '/home/data/financial'

YEAR = 252
MONTH = 21

INDEX_CODE = "000985.XSHG"
CODE_LEVEL = "order_book_id"
DATE_LEVEL = "date"
FACTORS = ["industry", "logsize", "beta", 
    "momentum", "volatility", "nonlinear_size",
    "bp",
]


def get_industry(start: str, stop: str) -> pd.DataFrame:
    return ft.get_data(IDS_URI, "third_industry_name", start=start, stop=stop)

def get_logsize(start: str, stop: str) -> pd.DataFrame:
    shares = ft.get_data(QTD_URI, "circulation_a", start=start, stop=stop)
    price = ft.get_data(QTD_URI, "close", start=start, stop=stop)
    adjfactor = ft.get_data(QTD_URI, "adjfactor", start=start, stop=stop)
    return -np.log(shares * price * adjfactor).loc[start:stop]

def get_beta(start: str, stop: str) -> pd.DataFrame:
    rollback = ft.get_trading_days_rollback(QTD_URI, start, YEAR + 1)
    price = ft.get_data(QTD_URI, "close", start=rollback, stop=stop)
    index_price = ft.get_data(IDXQTD_URI, "close", pool=INDEX_CODE, start=rollback, stop=stop).squeeze()
    adjfactor = ft.get_data(QTD_URI, "adjfactor", start=rollback, stop=stop)
    price *= adjfactor
    returns = price.pct_change(fill_method=None).iloc[1:]
    index_returns = index_price.pct_change(fill_method=None).iloc[1:]

    def _g_b(x):
        if x.shape[0] < YEAR:
            return pd.Series(index=x.columns, name=x.index[-1])
        idxrt = index_returns.loc[x.index]
        res = x.apply(lambda y:
            ((y - y.mean()) * (idxrt - idxrt.mean())).ewm(halflife=YEAR // 2).mean().iloc[-1] /
            ((idxrt - idxrt.mean()) ** 2).ewm(halflife=YEAR // 2).mean().iloc[-1]
        )
        res.name = x.index[-1]
        return res
    
    beta = Parallel(n_jobs=-1, backend="loky")(delayed(_g_b)(x) for x in returns.rolling(YEAR))
    return pd.concat(beta, axis=1).T.loc[start:stop]

def get_momentum(start: str, stop: str) -> pd.DataFrame:
    rollback = ft.get_trading_days_rollback(QTD_URI, start, 2 * YEAR + 1)
    price = ft.get_data(QTD_URI, "close", start=rollback, stop=stop)
    adjfactor = ft.get_data(QTD_URI, "adjfactor", start=rollback, stop=stop)
    price *= adjfactor
    logreturns = np.log(price / price.shift(1)).iloc[1:]
    def _g_m(x):
        if x.shape[0] < 2 * YEAR:
            return pd.Series(index=x.columns, name=x.index[-1])
        return x.shift(MONTH).ewm(halflife=YEAR // 2).sum().iloc[-1]
    momentum = Parallel(n_jobs=-1, backend="loky")(delayed(_g_m)(x) for x in logreturns.rolling(2 * YEAR))
    return pd.concat(momentum, axis=1).T.loc[start:stop]

def get_datsd(start: str, stop: str) -> pd.DataFrame:
    rollback = ft.get_trading_days_rollback(QTD_URI, start, YEAR + 1)
    price = ft.get_data(QTD_URI, "close", start=rollback, stop=stop)
    adjfactor = ft.get_data(QTD_URI, "adjfactor", start=rollback, stop=stop)
    price *= adjfactor
    logreturns = np.log(price / price.shift(1)).iloc[1:]
    return logreturns.ewm(halflife=2 * MONTH).std()

def get_cmra(start: str, stop: str) -> pd.DataFrame:
    rollback = ft.get_trading_days_rollback(QTD_URI, start, YEAR + 1)
    price = ft.get_data(QTD_URI, "close", start=rollback, stop=stop)
    adjfactor = ft.get_data(QTD_URI, "adjfactor", start=rollback, stop=stop)
    price *= adjfactor
    logreturns = np.log(price / price.shift(1)).iloc[1:]
    cmra = Parallel(n_jobs=-1, backend="loky")(delayed(lambda x: x.cumsum().max() - x.cumsum().min())(x) for x in logreturns.rolling(YEAR))
    return pd.concat(cmra, axis=1, keys=logreturns.index).T.loc[start:stop]

def get_hsigma(start: str, stop: str):
    rollback = ft.get_trading_days_rollback(QTD_URI, start, YEAR + 1)
    price = ft.get_data(QTD_URI, "close", start=rollback, stop=stop)
    index_price = ft.get_data(IDXQTD_URI, "close", pool=INDEX_CODE, start=rollback, stop=stop).squeeze()
    adjfactor = ft.get_data(QTD_URI, "adjfactor", start=rollback, stop=stop)
    price *= adjfactor
    returns = price.pct_change(fill_method=None).iloc[1:]
    index_returns = index_price.pct_change(fill_method=None).iloc[1:]
    beta = get_beta(rollback, stop)
    size = np.exp(-get_logsize(rollback, stop))
    hsigma = (returns - beta.mul(index_returns, axis=0)).rolling(YEAR).std()
    def _decolinear(x, y):
        x = x.dropna()
        y = y.dropna()
        idx = x.index.intersection(y.index)
        x = x.loc[idx]
        y = y.loc[idx]
        if x.empty or y.empty:
            return pd.Series(index=y.index, name=y.name)
        model = sm.OLS(y, sm.add_constant(x)).fit()
        resid = model.resid
        resid.name = y.name
        return resid
    hsigma = Parallel(n_jobs=-1, backend="loky")(delayed(_decolinear)
        (pd.concat([size.iloc[i], beta.iloc[i]], axis=1, keys=["size", "beta"]), hsigma.iloc[i]) for i in range(len(hsigma))
    )
    return pd.concat(hsigma, axis=1).T.loc[start:stop]

def get_volatility(start: str, stop: str) -> pd.DataFrame:
    datsd = ft.zscore(get_datsd(start, stop))
    cmra = ft.zscore(get_cmra(start, stop))
    hsigma = ft.zscore(get_hsigma(start, stop))
    return (0.74 * datsd + 0.16 * cmra + 0.1 * hsigma).loc[start:stop]

def get_nonlinear_size(start: str, stop: str) -> pd.DataFrame:
    size = -get_logsize(start, stop)
    tsize = size ** 3
    def _decolinear(x, y):
        x = x.dropna()
        y = y.dropna()
        idx = x.index.intersection(y.index)
        x = x.loc[idx]
        y = y.loc[idx]
        if x.empty or y.empty:
            return pd.Series(index=y.index, name=y.name)
        model = sm.OLS(y, sm.add_constant(x)).fit()
        resid = model.resid
        resid.name = y.name
        return resid
    nlsize = Parallel(n_jobs=-1, backend='loky')(delayed(_decolinear)
        (size.iloc[i], tsize.iloc[i]) for i in range(tsize.shape[0])
    )
    return ft.zscore(ft.stdoutlier(pd.concat(nlsize, axis=1).T.loc[start:stop], 3))

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

def regression(start: str, stop: str, ptype: str = "open"):
    rollback = ft.get_trading_days_rollback(QTD_URI, start, 1)
    qtd = quool.PanelTable(QTD_URI)
    qtd_data = qtd.read([ptype, "adjfactor"], start=rollback, stop=stop)
    price = qtd_data[ptype] * qtd_data["adjfactor"]
    returns = (price.groupby(CODE_LEVEL).shift(-1) / price - 1)
    
    barra_table = quool.PanelTable(BARRA_URI)
    factors = barra_table.read(FACTORS, start=start, stop=stop)

    size = np.sqrt(np.exp(factors["logsize"]))
    size = size.groupby(CODE_LEVEL, group_keys=False).apply(lambda x: x / x.sum())

    def _reg(x, y, sz):
        ids = pd.get_dummies(x["industry"]).drop(None, axis=1)
        sty = x[FACTORS[1:]]
        x = sm.add_constant(pd.concat([ids.iloc[:, :-1], sty], axis=1).iloc[:, :-1])
        x.columns = ["country"] + x.columns[1:].tolist()
        v = np.diag(sz)
        si = [sz[ids[col]].sum() for col in ids.columns]
        r = np.eye(ids.columns.size + sty.columns.size)
        r = np.concatenate([
            r[:ids.columns.size],
            [[0] + [-si[i] / si[-1] for i in range(len(si) - 1)] + [0] * sty.columns.size], 
            r[ids.columns.size:]
        ], axis=0)
        return pd.DataFrame(
            r @ np.linalg.inv(r.T @ x.values.T @ v @ x.values @ r) @ r.T @ x.values.T @ v,
            index=x.columns, columns=x.index
        )

    factor_return = Parallel(n_jobs=1, backend="loky")(delayed(_reg)(
            factors.xs(date, level=DATE_LEVEL), 
            returns.xs(date, level=DATE_LEVEL), 
            size.xs(date, level=DATE_LEVEL)
        ) for date in factors.index.get_level_values(DATE_LEVEL).unique()
    )
