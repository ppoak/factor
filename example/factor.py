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
        