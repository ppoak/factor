import numpy as np
import pandas as pd
import dataforge as forge
import factormonitor as fm


class Momentum(fm.Factor):

    def __init__(self, close: pd.Series):
        super().__init__(
            name = 'momentum',
            table = forge.AssetTable('/home/data/factor/'),
            close = close,
        )
    
    def compute(
        self, 
        span: int = 22,
    ):
        self.close: pd.Series
        self.factor = self.close / self.close.groupby(level='order_book_id').shift(span) - 1
        self.factor = self.factor.dropna()
        return self.factor
        

class TurnoverMomentum(fm.Factor):

    def __init__(self, close: pd.Series, turnover: pd.Series):
        super().__init__(
            name = 'turnover_momentum',
            table = forge.AssetTable('/home/data/factor/'),
            close = close,
            turnover = turnover,
        )

    def compute(
        self,
        span: int = 22,
    ):
        self.turnover: pd.Series; self.close: pd.Series
        ret = self.close / self.close.groupby(level='order_book_id').shift(1) - 1
        self.factor = ret.groupby(level='order_book_id', group_keys=False).apply(
            lambda x: (x * self.turnover.loc[x.index] + 1)
                .rolling(span).apply(np.prod) - 1
        )
        self.factor = self.factor.dropna()
        return self.factor