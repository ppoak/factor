import quool
import numpy as np
import pandas as pd


class RelocateStrategy(quool.Strategy):

    params = (("ratio", 0.95), )

    def __init__(self) -> None:
        self.holdings = pd.Series(
            np.zeros(len(self.datas)), 
            index=[d._name for d in self.datas], name='holdings'
        )

    def next(self):
        target = pd.Series(dict([(d._name, d.portfolio[0]) for d in self.datas]), name='target')
        dec = target[target < self.holdings]
        inc = target[target > self.holdings]
        for d in dec.index:
            self.order_target_percent(data=d, target=target.loc[d] * (1 - self.p.ratio), name=d)
        for i in inc.index:
            self.order_target_percent(data=i, target=target.loc[i] * (1 - self.p.ratio), name=i)
        self.holdings = target


def ic(
    factor: pd.DataFrame | pd.Series,
    price: pd.DataFrame | pd.Series,
    code_index: str = "order_book_id",
    date_index: str = "date",
    buy_col: str = 'close',
    sell_col: str = 'close',
):
    if isinstance(factor, pd.DataFrame) and isinstance(factor.index, pd.MultiIndex):
        factor = factor.stack()
    elif not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("factor must be a DataFrame or with MultiIndex")

    buy_price = price[buy_col] if isinstance(price, pd.DataFrame) else price
    sell_price = price[sell_col] if isinstance(price, pd.DataFrame) else price
    buy_price = buy_price.loc[factor.index]
    sell_price = sell_price.loc[factor.index].groupby(level=code_index).shift(-1)
    ret = (sell_price - buy_price) / buy_price
    ic = factor.groupby(level=date_index).apply(lambda x: x.corr(ret.loc[x.index]))
    return ic

def vector_backtest(
    factor: pd.DataFrame | pd.Series,
    price: pd.DataFrame | pd.Series,
    code_index: str = "order_book_id",
    date_index: str = "date",
    buy_col: str = 'close',
    sell_col: str = 'close',
    ngroup: int = 10,
    commision: float = 0.005,
):
    if isinstance(factor, pd.DataFrame) and isinstance(factor.index, pd.MultiIndex):
        factor = factor.stack()
    elif not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("factor must be a DataFrame or with MultiIndex")
    groups = factor.groupby(date_index, group_keys=False).apply(
        lambda x: pd.qcut(x, ngroup, labels=False)) + 1
    
    turnover = []
    profit = []
    for n in range(1, ngroup + 1):
        group = groups[groups == n]
        group = group.groupby(date_index, group_keys=False).apply(
            lambda x: x / x.sum()
        )
        relocator = quool.Relocator(price, code_index, date_index, buy_col, sell_col, commision)
        turnover.append(relocator.turnover(group))
        profit.append(relocator.profit(group))
    
    turnover = pd.concat(turnover, axis=1, keys=[f"Group{i}" for i in range(1, ngroup + 1)])
    profit = pd.concat(profit, axis=1, keys=[f"Group{i}" for i in range(1, ngroup + 1)])
    profit.iloc[0] = 0
    
    return profit, turnover

def event_backtest(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    code_index: str = 'order_book_id',
    date_index: str = 'date',
    ngroup: int = 10,
    commision: float = 0.005,
):
    if isinstance(factor, pd.DataFrame) and isinstance(factor.index, pd.DatetimeIndex):
        factor = factor.stack()
    elif not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("factor must be a DataFrame or with MultiIndex")
    groupers = factor.groupby(level=date_index, group_keys=False).apply(
        lambda x: pd.qcut(x, ngroup, labels=False)
    ) + 1
    results = []
    for n in range(ngroup):
        group = (groupers == n).index
        data = pd.concat([prices, group], axis=1)
        backtrader = quool.BackTrader(data, code_index, date_index)
        results.append(backtrader.run(RelocateStrategy, commision=commision))

    return results