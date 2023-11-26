import numpy as np
import pandas as pd
import dataforge as forge
import factormonitor as fm


class BackTestFactor(fm.Factor):

    def vector_backtest(
        self, 
        ngroup: int = 5,
        span: int = 22, 
        start: str = None,
        stop: str = None,
        buy_col: str = 'close', 
        sell_col: str = 'close', 
        commision: float = 0.005
    ):
        qdt = forge.AssetTable('/home/data/quotes-day/')
        start = self.factor.index.get_level_values(self.date_index).min()
        stop = self.factor.index.get_level_values(self.date_index).max()
        price = qdt.read('open, high, low, close, volume', start=start, stop=stop)
        return super().vector_backtest(
            price, span, start, stop, ngroup, 
            buy_col, sell_col, commision
        )


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
        self.factor = ret.groupby(level='order_book_id', group_keys=False).apply(
            lambda x: (x * self.turnover.loc[x.index] + 1)
                .rolling(span).apply(np.prod) - 1
        )
        self.factor = self.factor.dropna().sort_index()
        return self


if __name__ == "__main__":
    quotesday = forge.AssetTable("/home/data/quotes-day")
    df = quotesday.read("close, adjfactor, turnover")

    # print("-" * 15 + " Computing momentum ... " + "-" * 15)
    momentum = Momentum(df["close"] * df["adjfactor"])
    momentum.read(start="20200101", stop="20231031", span=22)
    result = momentum.vector_backtest(span=22)
    # momentum.compute(span=22).deextreme().standarize().save("momentum_span22")
    # print("-" * 15 + " Computing turnover momentum ... " + "-" * 15)
    # turnover_momentum = TurnoverMomentum(df["close"] * df["adjfactor"], df["turnover"])
    # turnover_momentum.compute(span=22).deextreme().standarize().save("turnover_momentum_span22")
    