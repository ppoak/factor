import numpy as np
import pandas as pd
import factor as ft
from tqdm import tqdm
from joblib import Parallel, delayed


QTM_URI = '/home/data/quotes-min'
QTD_URI = '/home/data/quotes-day'

MIN = 240
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

def get_realized_skew(start: str, stop: str) -> pd.DataFrame:
    def _get(_date):
        rollback = ft.get_trading_days_rollback(QTD_URI, _date, 1)
        _data = ft.get_data(QTM_URI, "close", start=rollback, stop=_date + pd.Timedelta(days=1))
        _return = _data.pct_change(fill_method=None).loc[_date.strftime(r'%Y%m%d')]
        _res = _return.skew()
        _res.name = _date
        return _res
    
    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
        (_date) for _date in tqdm(trading_days)), axis=1).T.loc[start:stop]
    
def get_realized_kurt(start: str, stop: str) -> pd.DataFrame:
    def _get(_start, _stop):
        _data = ft.get_data(QTM_URI, "close", start=_start, stop=_stop + pd.Timedelta(days=1))
        _return = _data.pct_change(fill_method=None)
        return _return.resample('d').apply(lambda x: (((x - x.mean()).pow(4).sum()) / (MIN -1) \
                                                        / (x.std().pow(4)))-3).dropna(axis=0, how='all')  
    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    tasks = [(trading_days[WEEK*i:WEEK*(i+1)][0], trading_days[WEEK*i:WEEK*(i+1)][-1]) for i in range((len(trading_days) // WEEK)+1)]
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
            (_start, _stop) for _start, _stop in tqdm(tasks)), axis=0)

def get_long_short_ratio(start: str, stop: str):
    def _get(_date):
        _price = ft.get_data(QTM_URI, "close", start=_date, stop=_date + pd.Timedelta(days=1))
        _vol = ft.get_data(QTM_URI, "volume", start=_date, stop=_date + + pd.Timedelta(days=1))
        _return = _price.pct_change(fill_method=None)
        _vol_per_unit = abs(0.0001/_return * _vol).replace([np.inf, -np.inf], np.nan)
        _1 = _vol_per_unit.mean()
        _day_return = (_price.iloc[-1] - _price.iloc[0]) / _price.iloc[0]
        res = (abs(_day_return / 0.0001) * _1) / _vol.sum()
        res.name = _date
        return res
        
    trading_days = ft.get_trading_days(QTD_URI, start, stop) 
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
        (_date) for _date in tqdm(trading_days)), axis=1).T.loc[start:stop]

if __name__ == "__main__":
    print(get_long_short_ratio('20230606','20230610'))