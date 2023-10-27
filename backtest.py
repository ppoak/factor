import pandas as pd
from factor.tools import daily_kline
from factor.definition import (
    ret, hist_percent
)


close = daily_kline(field='close')['close'].unstack(level=0)
ret_ = ret(close)
hist_percent_ = hist_percent(close)