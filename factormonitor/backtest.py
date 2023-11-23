import numpy as np
import pandas as pd
import backtrader as bt
import dataforge as forge
from .tools import price2ret


class RelocateData(bt.feeds.PandasData):
    lines = ('portfolio', )
    params = (('portfolio', -1), )


class RelocateStrategy(forge.Strategy):

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


def vector_backtest(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    span: int | str,
    buy_col: str = 'close',
    sell_col: str = 'close',
    ngroup: int = 10,
    commision: float = 0.005,
):
    ret = price2ret(prices, span, buy_col, sell_col).loc[factor.index].values
    ranks = factor.values.argsort(axis=1)
    num_in_group = int(len(factor.columns) / ngroup)

    turnover = []
    groupret = []
    for n in range(ngroup):
        group = ranks[:, n * num_in_group:(n + 1) * num_in_group]
        group = np.sort(group, axis=1)
        turnover.append(np.concatenate([
            np.full(group.shape[1], np.nan), 
            (group[1:] - group[:-1] != 0).sum(axis=1) / group.shape[1] / 2
        ], axis=0))

        indexer = np.repeat(np.arange(group.shape[0]), group.shape[1])
        groupret.append(
            np.nanmean(ret[indexer, group.reshape(-1)], axis=1) -
            commision * turnover
        )
    
    turnover = pd.DataFrame(turnover, index=ret.index, 
        columns=[f'group{i}' for i in range(ngroup)])
    groupret = pd.DataFrame(groupret, index=ret.index,
        columns=[f'group{i}' for i in range(ngroup)])
    
    return groupret, turnover

def event_backtest(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    span: int | str,
    buy_col: str = 'close',
    sell_col: str = 'close',
    ngroup: int = 10,
    commision: float = 0.005,
):
    factor = factor.stack()
    groupers = pd.qcut(factor, ngroup, labels=False)
    for n in range(ngroup):
        group = (groupers == n).index
        data = pd.concat([prices, group], axis=1)
        data.groupby()

