import pandas as pd
import factor as ft
from tqdm import tqdm
from joblib import Parallel, delayed


QTM_URI = '/home/data/quotes-min'
QTD_URI = '/home/data/quotes-day'


def get_tail_volume_percent(
    start: str, stop: str,
) -> pd.DataFrame:

    def _get_tvp(td):
        _data = ft.get_data(QTM_URI, "volume", start=td, stop=td + pd.Timedelta(days=1))
        return _data.between_time("14:30", "15:00").sum(axis=0) / _data.sum(axis=0)
        
    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    return -pd.concat(Parallel(n_jobs=-1, backend='loky')
        (delayed(_get_tvp)(td) for td in tqdm(trading_days)),
        axis=0, keys=trading_days
    ).swaplevel()
