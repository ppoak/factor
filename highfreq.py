import pandas as pd
from tqdm import tqdm
from factor import get_data
from joblib import Parallel, delayed


def get_tail_volume_percent(
    start: str, stop: str,
) -> pd.DataFrame:

    def _get_tvp(td):
        _data = get_data(qtm_uri, "volume", start=td, stop=td + pd.Timedelta(days=1))
        return _data.between_time("14:30", "15:00").sum(axis=0) / _data.sum(axis=0)
        
    qtm_uri = '/home/data/quotes-min'
    qtd_uri = '/home/data/quotes-day'
    trading_days = get_data(qtd_uri, "close", start=start, stop=stop, pool='000001.XSHE').index
    return pd.concat(Parallel(n_jobs=-1, backend='loky')
        (delayed(_get_tvp)(td) for td in tqdm(trading_days)),
        axis=0, keys=trading_days
    ).swaplevel()
