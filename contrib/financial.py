"""
This script is used to compute financial factors:

- roa: the ratio of net profit to the total assets
- roe: the ratio of net profit to the equity
- catoasset: the ratio of current assets to total assets
"""

# %% here is to import some packages
from quool.table import PanelTable

# %% here is to create some interfaces
start = "20000101"
stop = None

date_level = 'date'
code_level = 'order_book_id'

qtd = PanelTable("/home/data/quotes-day", date_level=date_level, code_level=code_level)
fin = PanelTable("/home/data/financial", date_level=date_level, code_level=code_level)
fctdev = PanelTable("/home/data/factordev", date_level=date_level, code_level=code_level)

# %% here is to read some basic data
qtddata = qtd.read('close', start=start, stop=stop)
findata = fin.read(
    'current_assets, total_assets, total_equity, net_profit', 
    start=start, stop=stop
).dropna()
findata = findata.reindex(qtddata.index).groupby(level=code_level).ffill()

# %% compute the roa factor
roa = findata['net_profit'] / findata['total_assets']

# %% compute the roe factor
roe = findata['net_profit'] / findata['total_equity']

# %% compute current assets to total assets
catoasset = findata['current_assets'] / findata['total_assets']

# %% save size factor
roa.name = 'roa'
fctdev.add(roa) if 'roa' in fctdev.columns else fctdev.add(roa)

# %% save size factor
roe.name = 'roe'
fctdev.add(roe) if 'roe' in fctdev.columns else fctdev.add(roe)

# %% save size factor
catoasset.name = 'catoasset'
fctdev.add(catoasset) if 'catoasset' in fctdev.columns else fctdev.add(catoasset)
