import pandas as pd
import factor as ft
from tqdm import tqdm
from joblib import Parallel, delayed


QTM_URI = '/home/data/quotes-min'
QTD_URI = '/home/data/quotes-day'

WEEK = 5
MONTH = 21
YEAR = 252
DATE_LEVEL = 'datetime'


def get_tail_volume_percent(start: str, stop: str) -> pd.DataFrame:
    def _get(_start, _stop):
        _data = ft.get_data(QTM_URI, "volume", start=_start, stop=_stop + pd.Timedelta(days=1))
        _tail_vol = _data.between_time("14:30", "15:00").resample('d').sum()
        _day_vol = _data.resample('d').sum()
        result = (_tail_vol / _day_vol).astype('float').mean()
        result.name = pd.to_datetime(_data.index[-1].strftime('%Y%m%d'))
        return result
        
    rollback = ft.get_trading_days_rollback(QTD_URI, start, WEEK)
    trading_days = ft.get_trading_days(QTD_URI, rollback, stop)
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
        (_start, _stop) for _start, _stop in  tqdm(list(
            zip(trading_days[:-WEEK + 1], trading_days[WEEK - 1:])
        ))), axis=1).T.loc[start:stop]
