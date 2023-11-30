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

    def _group_compute(
        self, span: int, name: str,
        group_idx: pd.MultiIndex, 
        ret: pd.Series, 
        index_ret: pd.Series
    ):
        r = ret.loc[group_idx].droplevel(self.code_index)
        ir = index_ret.droplevel(self.code_index).loc[r.index]
        res = []
        for i in range(0, len(r) - span):
            r_ = r.iloc[i:i + span]
            ir_ = ir.iloc[i:i + span]
            if np.isnan(ir_.values).any() or np.isnan(r_.values).any():
                res.append(np.nan)
                continue
            r_ = sm.add_constant(r_)
            m = sm.OLS(ir_, r_).fit()
            res.append(m.resid.std())
        res = pd.Series(res, index=pd.MultiIndex.from_product([[name], r.index[span:]]))
        return res

    def compute(
        self,
        span: int = 22,
    ):
        self.close: pd.Series; self.index_close: pd.Series
        ret = self.close / self.close.groupby(level=self.code_index).shift(1) - 1
        index_ret = self.index_close.pct_change()
        groups = ret.groupby(level=self.code_index)
        factor = Parallel(n_jobs=1, backend='loky')(delayed(self._group_compute)
            (span, name, idx, ret, index_ret) for name, idx in groups.groups.items()
        )
        self.factor = pd.concat(factor)
        self.factor = self.factor.dropna().sort_index()
        return self

