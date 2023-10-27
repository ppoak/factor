import datetime
import pandas as pd


def parse_commastr(
    commastr: 'str | list',
) -> pd.Index:
    if isinstance(commastr, str):
        commastr = commastr.split(',')
        return list(map(lambda x: x.strip(), commastr))
    elif commastr is None:
        return None
    else:
        return commastr

def daily_kline(
    code: str | list = None,
    field: str | list | None = None,
    start: str = None,
    stop: str = None,
):
    code = parse_commastr(code)
    field = parse_commastr(field)
    start = start or "2000-01-04"
    stop = stop or datetime.datetime.today().strftime(r"%Y-%m-%d")

    filters = [
        ("date", ">=", pd.to_datetime(start)),
        ("date", "<=", pd.to_datetime(stop)),
    ]
    if code is not None:
        filters += [("code", "in", code)]
    data = pd.read_parquet(
        '/home/data/quotes-day',
        engine = 'pyarrow',
        columns = field,
        filters = filters
    )
    return data
