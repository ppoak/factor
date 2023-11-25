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
        self.factor = self.close / self.close.groupby(level=self.code_index).shift(span) - 1
        self.factor = self.factor.dropna().sort_index()
        return self
        

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
        ret = self.close / self.close.groupby(level=self.code_index).shift(1) - 1
        self.factor = ret.groupby(level='order_book_id', group_keys=False).apply(
            lambda x: (x * self.turnover.loc[x.index] + 1)
                .rolling(span).apply(np.prod) - 1
        )
        self.factor = self.factor.dropna().sort_index()
        return self


if __name__ == "__main__":
    quotesday = forge.AssetTable("/home/data/quotes-day")
    df = quotesday.read("close, adjfactor, turnover")
    # ohlcv = quotesday.read("open, high, low, close, volume", start="20200101", stop="20231031")

    print("-" * 15 + " Computing momentum ... " + "-" * 15)
    momentum = Momentum(df["close"] * df["adjfactor"])
    momentum.compute(span=22).deextreme().standarize().save("momentum_span22")
    print("-" * 15 + " Computing turnover momentum ... " + "-" * 15)
    turnover_momentum = TurnoverMomentum(df["close"] * df["adjfactor"], df["turnover"])
    turnover_momentum.compute(span=22).deextreme().standarize().save("momentum_span22")
