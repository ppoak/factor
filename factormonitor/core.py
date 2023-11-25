import pandas as pd
import dataforge as forge


class Factor:
    
    def __init__(
        self, 
        name: str,
        table: forge.AssetTable,
        *data: list[pd.Series]
    ) -> None:
        self.name = name
        self.table = table
        self.factor = None
    
    def compute(
        self, 
        start: str = None, 
        stop: str = None,
        **kwargs,
    ):
        raise NotImplementedError
    
    def save(self):
        self.table.create()
        if not self.table.fragments:
            self.table._write_fragment(self.factor)
        elif not self.name in self.table.columns:
            self.table.add(self.factor)
        else:
            self.table.update(self.factor)
    
    def __str__(self):
        return f'Factor {self.name}'
    
    def __repr__(self) -> str:
        return self.__str__()