import numpy as np
import pandas as pd


def price2ret(
    prices: pd.DataFrame | pd.Series,
    span: 'str | int', 
    code_index: str = 'order_book_id',
    date_index: str = 'date',
    buy_col: str = 'close', 
    sell_col: str = 'close', 
    method: str = 'algret',
    lag: int = 1,
):
    if isinstance(prices, pd.DataFrame) and isinstance(prices.index, pd.MultiIndex):
        if isinstance(span, int):
            if span > 0:
                sell_price = prices.loc[:, sell_col]
                buy_price = prices.groupby(pd.Grouper(level=code_index))\
                    .shift(span).loc[:, buy_col]
            else:
                sell_price = prices.groupby(pd.Grouper(level=code_index))\
                    .shift(span - lag).loc[:, sell_col]
                buy_price = prices.groupby(pd.Grouper(level=code_index))\
                    .shift(-lag).loc[:, buy_col]

        else:
            if '-' in str(span):
                if isinstance(span, str):
                    span = span.strip('-')
                else:
                    span = - span
                sell_price = prices.groupby(level=code_index).shift(-lag).groupby([
                    pd.Grouper(level=date_index, freq=span, label='left'),
                    pd.Grouper(level=code_index)
                ]).last().loc[:, sell_col]
                buy_price = prices.groupby(level=code_index).shift(-lag).groupby([
                    pd.Grouper(level=date_index, freq=span, label='left'),
                    pd.Grouper(level=code_index)
                ]).first().loc[:, buy_col]

            else:
                sell_price = prices.groupby([
                    pd.Grouper(level=date_index, freq=span, label='right'),
                    pd.Grouper(level=code_index)
                ]).last().loc[:, sell_col]
                buy_price = prices.groupby([
                    pd.Grouper(level=date_index, freq=span, label='right'),
                    pd.Grouper(level=code_index)
                ]).first().loc[:, buy_col]

    elif isinstance(prices, pd.Series) and isinstance(prices.index, pd.MultiIndex):
        if isinstance(span, int):
            if span > 0:
                sell_price = prices
                buy_price = prices.groupby(pd.Grouper(level=code_index)).shift(span)
            else:
                sell_price = prices.groupby(pd.Grouper(level=code_index)).shift(span - lag)
                buy_price = prices.groupby(pd.Grouper(level=code_index)).shift(-lag)

        else:
            if '-' in str(span):
                if isinstance(span, str):
                    span = span.strip('-')
                else:
                    span = - span
                sell_price = prices.groupby(level=code_index).shift(-lag).groupby([
                    pd.Grouper(level=date_index, freq=span, label='left'),
                    pd.Grouper(level=code_index)
                ]).last()
                buy_price = prices.groupby(level=code_index).shift(-lag).groupby([
                    pd.Grouper(level=date_index, freq=span, label='left'),
                    pd.Grouper(level=code_index)
                ]).first()
            else:
                sell_price = prices.groupby([
                    pd.Grouper(level=date_index, freq=span, label='right'),
                    pd.Grouper(level=code_index)
                ]).last()
                buy_price = prices.groupby([
                    pd.Grouper(level=date_index, freq=span, label='right'),
                    pd.Grouper(level=code_index)
                ]).first()
    
    elif isinstance(prices.index, pd.DatetimeIndex):
        if isinstance(span, int):
            if span > 0:
                sell_price = prices
                buy_price = prices.shift(span)
            else:
                sell_price = prices.shift(-span - lag)
                buy_price = prices.shift(-lag)
        
        else:
            if '-' in str(span):
                if isinstance(span, str):
                    span = span.strip('-')
                else:
                    span = - span
                sell_price = prices.shift(-lag).resample(span, label='left').last()
                buy_price = prices.shift(-lag).resample(span, label='left').first()
            else:
                sell_price = prices.resample(span, label='right').last()
                buy_price = prices.resample(span, label='right').first()
        
    else:
        raise ValueError('Can only convert time series data to return')

    if method == 'algret':
        return (sell_price - buy_price) / buy_price
    elif method == 'logret':
        return np.log(sell_price / buy_price)
 