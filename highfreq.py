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
        _res = (_tail_vol / _day_vol).astype('float').mean()
        _res.name = pd.to_datetime(_data.index[-1].strftime('%Y%m%d'))
        return _res
        
    rollback = ft.get_trading_days_rollback(QTD_URI, start, WEEK)
    trading_days = ft.get_trading_days(QTD_URI, rollback, stop)
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
        (_start, _stop) for _start, _stop in  tqdm(list(
            zip(trading_days[:-WEEK + 1], trading_days[WEEK - 1:])
        ))), axis=1).T.loc[start:stop]

def get_interday_distribution(start: str, stop: str) -> pd.DataFrame:
    def _get(_date):
        _data = ft.get_data(QTM_URI, "close", start=_date, stop=_date + pd.Timedelta(days=1))
        _return = _data.pct_change(fill_method=None)
        res = pd.concat([_return.skew(), _return.kurt()], axis=1, keys=['skew', 'kurt'])
        res.index = pd.MultiIndex.from_product([
             res.index, [_date],
        ], names=["order_book_id", "date"])
        return res
    
    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    result = pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
        (date) for date in  tqdm(list(trading_days))), axis=0)
    result = result.sort_index().loc(axis=0)[:, start:stop]
    return result

def get_long_short_ratio(start: str, stop: str):
    def _get(_start, _stop):
        _price = ft.get_data(QTM_URI, "close", start=_start, stop=_stop + pd.Timedelta(days=1))
        _vol = ft.get_data(QTM_URI, "volume", start=_start, stop=_stop + pd.Timedelta(days=1))
        _return = _price.pct_change(fill_method=None)
        _vol_per_unit = abs(_vol / _return).replace([np.inf, -np.inf], np.nan)
        _1 = _vol_per_unit.mean()
        _tot_return = (_price.iloc[-1] / _price.iloc[0] - 1).abs()
        _res = (_tot_return * _1) / _vol.sum()
        _res.name = _stop
        return _res
        
    rollback = ft.get_trading_days_rollback(QTD_URI, start, WEEK)
    trading_days = ft.get_trading_days(QTD_URI, rollback, stop)
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
        (_start, _stop) for _start, _stop in  tqdm(list(
            zip(trading_days[:-WEEK + 1], trading_days[WEEK - 1:])
        ))), axis=1).T.loc[start:stop]

def get_price_volume_corr(start: str, stop: str) -> pd.DataFrame:
    def _get(_date):
        _price = ft.get_data(QTM_URI, "close", start=_date, stop=_date + pd.Timedelta(days=1))
        _volume = ft.get_data(QTM_URI, "volume", start=_date, stop=_date + pd.Timedelta(days=1))
        _res = _price.corrwith(_volume, axis=0).replace([np.inf, -np.inf], np.nan)
        _res.name = _date
        return _res

    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
        (date) for date in  tqdm(list(trading_days))), axis=1).T.loc[start:stop]

def get_down_trend_volatility(start: str, stop: str) -> pd.DataFrame:
    def _get(_date):
        _price = ft.get_data(QTM_URI, "close", start=_date, stop=_date + pd.Timedelta(days=1))
        _return = _price.pct_change(fill_method=None)
        res = _return.apply(lambda x: x[x < 0].pow(2).sum() / x.pow(2).sum())
        res.name = _date
        return res
    
    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
        (date) for date in  tqdm(list(trading_days))), axis=1).T.loc[start:stop]

def get_regret(start: str, stop: str) -> pd.DataFrame:
    def _get(_date):
        _price = ft.get_data(QTM_URI, "close", start=_date, stop=_date + pd.Timedelta(days=1))
        _vol = ft.get_data(QTM_URI, "volume", start=_date, stop=_date + pd.Timedelta(days=1))
        lc = _price <= _price.iloc[-1]
        hc = _price > _price.iloc[-1]
        lcvol = _vol.where(lc).sum() / _vol.sum()
        hcvol = _vol.where(hc).sum() / _vol.sum()
        lcp = (_price.where(lc) * _vol.where(lc)).sum() / _vol.where(lc).sum() / _price.iloc[-1] - 1
        hcp = (_price.where(hc) * _vol.where(hc)).sum() / _vol.where(hc).sum() / _price.iloc[-1] - 1
        res = pd.concat([lcvol, hcvol, lcp, hcp], axis=1, keys=["lcvol", "hcvol", "lcp", "hcp"])
        res.index = pd.MultiIndex.from_product([[_date], res.index], names=["date", "order_book_id"])
        return res

    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
        (date) for date in  tqdm(list(trading_days))), axis=1).T.loc[start:stop]
