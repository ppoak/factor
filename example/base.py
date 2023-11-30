import quool
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
        qdt = quool.AssetTable('/home/data/quotes-day/')
        start = self.factor.index.get_level_values(self.date_index).min()
        stop = self.factor.index.get_level_values(self.date_index).max()
        price = qdt.read('open, high, low, close, volume', start=start, stop=stop)
        return super().vector_backtest(
            price, span, start, stop, ngroup, 
            buy_col, sell_col, commision
        )
    
    def ic(
        self, 
        span: int = 20, 
        start: str = None, 
        stop: str = None, 
        buy_col: str = 'close', 
        sell_col: str = 'close'
    ):
        qdt = quool.AssetTable('/home/data/quotes-day/')
        start = self.factor.index.get_level_values(self.date_index).min()
        stop = self.factor.index.get_level_values(self.date_index).max()
        price = qdt.read('open, high, low, close', start=start, stop=stop)
        return super().ic(price, span, start, stop, buy_col, sell_col)
