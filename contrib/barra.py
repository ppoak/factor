"""
This script is used to compute barra factors:

- size: the circulation value in a share stock market
- momentum: the historical return on a stock in past n days
- volatility: the standard deviation of a stock's historical return
- ep: the ratio of net profit to the total circulate a share value
"""


# %% here is to import some packages
import numpy as np
from quool.table import PanelTable

# %% here is to create some interfaces
start = "20000101"
stop = None

date_level = 'date'
code_level = 'order_book_id'

qtd = PanelTable("/home/data/quotes-day", date_level=date_level, code_level=code_level)
fin = PanelTable("/home/data/financial", date_level=date_level, code_level=code_level)
fct = PanelTable("/home/data/barra", date_level=date_level, code_level=code_level)

# %% here is to read some basic data
qtddata = qtd.read('circulation_a, close, adjfactor', start=start, stop=stop)
findata = fin.read('net_profit', start=start, stop=stop).dropna()
findata = findata.reindex(qtddata.index).groupby(level=code_level).ffill()

# %% compute size factor
size = np.log(qtddata['circulation_a'] * qtddata['close'] * qtddata['adjfactor'])

# %% compute momentum factor
price = qtddata['close'] * qtddata['adjfactor']
momentum20 = np.log(price / price.groupby(level=code_level).shift(20))

# %% compute volatility factor
volatility20 = np.log(
    price / price.groupby(level=code_level).shift(1)
).groupby(level=code_level).rolling(20).std().droplevel(0)

# %% compute ep factor
profit = findata['net_profit'].reindex(qtddata.index).groupby(level=code_level).ffill()
ep = np.log(profit / size)

# %% save size factor
size.name = 'size'
fct.update(size) if size.name in fct.columns else fct.add(size)

# %% save momentum factor
momentum20.name = 'momentum20'
fct.update(momentum20) if momentum20.name in fct.columns else fct.add(momentum20)

# %% save volatility factor
volatility20.name = 'volatility20'
fct.update(volatility20) if volatility20.name in fct.columns else fct.add(volatility20)

# %% save ep factor
ep.name = 'ep'
fct.update(ep) if ep.name in fct.columns else fct.add(ep)
