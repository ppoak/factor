"""
This script is used to compute high frequency factors:

- tail volume: the total volume in tail 3-minutes
"""

# %% here is to import some packages
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from quool.table import PanelTable

# %% here is to create some interfaces
start = "20000101"
stop = None

qtm = PanelTable("/home/data/quotes-min")
qtd = PanelTable('/home/data/quotes-day')
fin = PanelTable("/home/data/financial")
fctdev = PanelTable("/home/data/factordev")

# %% get trading days for data is too large, can only be computed daily
trading_days = qtd.read('close', code='000001.XSHE').index.get_level_values('date')

# %% tail volume factor computation
@delayed
def _get_tailvolume(_cpd):
    _data = qtm.read('volume', start=_cpd, stop=_cpd + pd.Timedelta(days=1)).iloc[:, 0]
    _factor = _data.unstack(level='order_book_id').between_time("14:57", "15:00").sum(axis=0)
    return _factor

computing_days = trading_days[4359:]
tailvolume3min = Parallel(n_jobs=-1, backend='loky')(_get_tailvolume(cpd) for cpd in tqdm(computing_days))
tailvolume3min = pd.concat(tailvolume3min, axis=0, keys=computing_days).reorder_levels(['order_book_id', 'date'])

# %% 
tailvolume3min.name = 'tailvolume3min'
fctdev.update(tailvolume3min) if 'tailvolume3min' in fctdev.columns else fctdev.add(tailvolume3min)
