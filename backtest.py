import quool
import datetime
import numpy as np
import factor as ft
from pathlib import Path


name = 'ep' # factorr name
factor_uri = '/home/data/factordev' # factor table uri
price_uri = '/home/data/quotes-day' # price table uri
pool_uri = '/home/data/index-weights' # pool table uri
benchmark_uri = '/home/data/index-quotes-day' # benchmark table uri
start = '20180101' # backtest start
price = "open" # the buy price to compute future return
stop = None # backtest stop
pool = '000985.XSHG' # backtest pool
delay = 1 # the delayed days to execute buy
rebalance = 5 # rebalance day
code_level = 'order_book_id' # code level in database
date_level = 'date' # date level name in database
result_path = './report' # output directory
crossdate = 0 # this should be a trading day
ngroup = 10 # how many groups to divide
topk = 10 # under development
commission = 0.005 # commission used in group test
n_jobs = -1 # how many cpus to use in layering test

today = datetime.datetime.today().strftime(r'%Y%m%d')
stop = stop or today
expname = f'{start}-{stop}-{pool}-{rebalance}'
result_path = (Path(result_path) / name / expname).expanduser().resolve()
result_path.mkdir(exist_ok=True, parents=True)
logger = quool.Logger("Tester")

logger.info("preparing data")
price = ft.get_price(price_uri, "open", pool, pool_uri, start, stop)
benchmark = ft.get_data(benchmark_uri, "close", start, stop, pool, None)
raw_factor = ft.get_data(factor_uri, name, start, stop, pool, pool_uri)

logger.info("preprocessing data")
factor = ft.replace(raw_factor, 0, np.nan)
factor = ft.log(factor)
factor = ft.madoutlier(factor, 5)
factor = ft.zscore(factor)

logger.info("performing cross section test")
ft.perform_crosssection(factor, price, 5, 
    image=result_path / 'cross-section.png')

logger.info("performing information coefficiency test")
ft.perform_inforcoef(factor, price, rebalance, 
    image=result_path / 'information-coefficient.png')

logger.info("performing backtest")
ft.perform_backtest(factor, price, longshort=-1, topk=100, 
    benchmark=benchmark, image=result_path / 'backtest.png')
