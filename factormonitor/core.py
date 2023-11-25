import pandas as pd
import dataforge as forge
from .backtest import (
    vector_backtest,
    event_backtest,
)


class Factor:
    
    def __init__(
        self, 
        name: str,
        table: forge.AssetTable,
        **datasource,
    ) -> None:
        self.name = name
        self.table = table
        self.factor = None
        for name, data in datasource.items():
            setattr(self, name, data)
    
    def compute(
        self, 
        **kwargs,
    ):
        raise NotImplementedError
    
    def save(self, name: str):
        self.table.create()
        factor = self.factor.to_frame(name)
        if not self.table.fragments:
            self.table._write_fragment(factor)
        elif not name in self.table.columns:
            self.table.add(factor)
        else:
            self.table.update(factor)
    
    def vector_backtest(
        self,
        price: pd.DataFrame,
        ngroup: int = 10,
        code_index: str = 'order_book_id',
        date_index: str = 'date',
        buy_col: str = 'close',
        sell_col: str = 'close',
        commision: float = 0.005,
    ):
        profit, turnover = vector_backtest(
            self.factor, 
            price, code_index, date_index, 
            buy_col, sell_col, ngroup, commision
        )
        return profit, turnover
    
    def __str__(self):
        return f'Factor {self.name}'
    
    def __repr__(self) -> str:
        return self.__str__()