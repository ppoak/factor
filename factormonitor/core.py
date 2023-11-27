import pandas as pd
import dataforge as forge
from .backtest import (
    vector_backtest,
    event_backtest,
    ic,
)


class Factor:
    
    def __init__(
        self, 
        name: str,
        table: forge.AssetTable,
        code_index: str = 'order_book_id',
        date_index: str = "date",
        **datasource,
    ) -> None:
        self.name = name
        self.table = table
        self.code_index = code_index
        self.date_index = date_index
        self.standarized = False
        self.deextremed = False
        self.factor: pd.DataFrame | pd.Series = None
        self.datasource = datasource
        for name, data in datasource.items():
            setattr(self, name, data)
    
    def compute(
        self, 
        **kwargs,
    ):
        raise NotImplementedError
    
    def read(
        self,
        code: str = None,
        start: str = None,
        stop: str = None,
        **kwargs,
    ):
        self.factor = self.table.read(
            self.name + '_' + '_'.join([f"{key}{value}" for key, value in kwargs.items()]),
            code = code, start = start, stop = stop,
        )
        self.standarized, self.deextremed = True, True
        return self

    def deextreme(
        self,
        method: str = "std",
    ):
        if method == "std":
            self.factor = self.factor.groupby(self.date_index, group_keys=False).apply(
                lambda x: x.clip(x.mean() - 3 * x.std(), x.mean() + 3 * x.std()))
        self.deextremed = True
        return self

    def standarize(
        self,
        method: str = "zscore",
    ):
        if method == "zscore":
            self.factor = self.factor.groupby(self.date_index, group_keys=False).apply(
                lambda x: (x - x.mean()) / x.std()
            )
        self.standarized = True
        return self
    
    def save(self, **kwargs):
        self.table.create()
        name = self.name + ('_' + '_'.join([f"{key}{value}" 
            for key, value in kwargs.items()]) if kwargs else '')
        factor = self.factor.to_frame(name) if \
            isinstance(self.factor, pd.Series) else self.factor
        
        if not self.table.fragments:
            self.table._write_fragment(factor)
        
        elif isinstance(self.factor, pd.Series):
            if name not in self.table.columns:
                self.table.add(factor)
            else:
                self.table.update(factor)
        
        elif isinstance(self.factor, pd.DataFrame):
            incol = self.factor.columns.isin(self.table.columns)
            if incol.sum() > 0:
                self.table.update(self.factor[self.factor.columns[incol]])
            if incol.sum() < len(self.factor.columns):
                self.table.add(self.factor[self.factor.columns[~incol]])

    def ic(
        self,
        price: pd.DataFrame,
        span: int = 20,
        start: str = None,
        stop: str = None,
        buy_col: str = 'close',
        sell_col: str = 'close',
    ):
        reloacte_date = self.factor.index.get_level_values(self.date_index).unique()[::span]
        relocate_factor = self.factor[
            (self.factor.index.get_level_values(self.date_index) >= forge.parse_date(start)) &
            (self.factor.index.get_level_values(self.date_index) <= forge.parse_date(stop)) &
            self.factor.index.get_level_values(self.date_index).isin(reloacte_date)
        ]
        ics = []     
        if isinstance(self.factor, pd.Series):
            ics.append(ic(relocate_factor, price, 
                self.code_index, self.date_index, buy_col, sell_col))

        elif isinstance(self.factor, pd.DataFrame):
            for factor in self.factor.columns:
                ics.append(ic(relocate_factor[factor], price,
                    self.code_index, self.date_index, buy_col, sell_col))
        
        return pd.concat(ics, axis=1, keys=self.factor.columns)
    
    def vector_backtest(
        self,
        price: pd.DataFrame,
        span: int = 20,
        start: str = None,
        stop: str = None,
        ngroup: int = 5,
        buy_col: str = 'close',
        sell_col: str = 'close',
        commision: float = 0.005,
    ):
        profit, turnover = {}, {}
        reloacte_date = self.factor.index.get_level_values(self.date_index).unique()[::span]
        relocate_factor = self.factor[
            (self.factor.index.get_level_values(self.date_index) >= forge.parse_date(start)) &
            (self.factor.index.get_level_values(self.date_index) <= forge.parse_date(stop)) &
            self.factor.index.get_level_values(self.date_index).isin(reloacte_date)
        ]
        if isinstance(self.factor, pd.Series):
            profit[self.name], turnover[self.name] = vector_backtest(
                relocate_factor, price, self.code_index, self.date_index, 
                buy_col, sell_col, ngroup, commision
            )
        else:
            for factor in self.factor.columns:
                profit[factor], turnover[factor] = vector_backtest(
                    relocate_factor[factor], price, self.code_index, self.date_index, 
                    buy_col, sell_col, ngroup, commision
                )
        return profit, turnover
    
    def __str__(self):
        datasource_keys = list(self.datasource.keys())
        return (
            f'Factor {self.name}\n'
            f'\tdata source: {datasource_keys}\n'
            f'\tstandarized: {self.standarized}; deextremed: {self.deextremed}\n'
            f'\ttarget table: {self.table}'
        )
    
    def __repr__(self) -> str:
        return self.__str__()