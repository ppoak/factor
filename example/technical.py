import numpy as np
import pandas as pd
import dataforge as forge
from .base import BackTestFactor


class Momentum(BackTestFactor):

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
        self.factor = self.close / self.close.groupby(level=self.code_index).shift(span) - 1
        self.factor = self.factor.dropna().sort_index()
        return self
        

class TurnoverMomentum(BackTestFactor):

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
        ret = self.close / self.close.groupby(level=self.code_index).shift(1) - 1
        self.factor = ret.groupby(level=self.code_index, group_keys=False).apply(
            lambda x: (x * self.turnover.loc[x.index] + 1)
                .rolling(span).apply(np.prod) - 1
        )
        self.factor = self.factor.dropna().sort_index()
        return self


class Std(BackTestFactor):

    def __init__(self, close: pd.Series):
        super().__init__(
            name = 'std',
            table = forge.AssetTable('/home/data/factor/'),
            close = close,
        )

    def compute(
        self,
        span: int = 22,
    ):
        self.close: pd.Series
        self.factor = self.close.groupby(level=self.code_index).rolling(span).std().droplevel(0)
        self.factor = self.factor.dropna().sort_index()
        return self

