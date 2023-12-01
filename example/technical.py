import quool
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import BackTestFactor
from joblib import Parallel, delayed


class Momentum(BackTestFactor):

    def __init__(self, close: pd.Series):
        super().__init__(
            name = 'momentum',
            table = quool.AssetTable('/home/data/factor/'),
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
            table = quool.AssetTable('/home/data/factor/'),
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
            table = quool.AssetTable('/home/data/factor/'),
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


class Id1Std(BackTestFactor):

    def __init__(self, close: pd.Series, index_close: pd.Series):
        super().__init__(
            name = 'id1_std',
            table = quool.AssetTable('/home/data/factor/'),
            close = close,
            index_close = index_close,
        )
    
    def _reg(self, r:pd.Series, ir: pd.Series):
        ir = ir.loc(axis=0)[:, r.index.get_level_values(self.date_index)]
        if r.isna().any() or ir.isna().any():
            return np.nan
        m = sm.OLS(r.values, sm.add_constant(ir.values)).fit()
        return m.resid.std()
    
    def _compute(
        self, r: pd.Series, ir: pd.Series, span: int,
    ):
        return r.rolling(span).apply(self._reg, kwargs={"ir": ir})

    def compute(
        self,
        span: int = 22,
    ):
        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=True)

        self.close: pd.Series; self.index_close: pd.Series
        ret = self.close / self.close.groupby(level=self.code_index).shift(1) - 1
        index_ret = self.index_close.pct_change()
        self.factor = ret.to_frame('ret').groupby(
            level=self.code_index, group_keys=False).parallel_apply(
            self._compute, ir=index_ret, span=span
        )
        self.factor = self.factor.dropna().sort_index()
        return self

