# %%
import quool
import datetime
import numpy as np
import factor as ft
import operators as op
from pathlib import Path


# %% factor analyzing parameters
name = 'tailvolume30min' # factorr name
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

# %% computing parameters from parameters
today = datetime.datetime.today().strftime(r'%Y%m%d')
stop = stop or today
expname = f'{start}-{stop}-{pool}-{rebalance}'
result_path = (Path(result_path) / name / expname).expanduser().resolve()
result_path.mkdir(exist_ok=True, parents=True)
logger = quool.Logger("Tester")

# %% data prepare
logger.info("preparing data")
benchmark = ft.get_benchmark(benchmark_uri, price, pool, start, stop, code_level)
pool_index = ft.get_pool(pool_uri, pool, start, stop)
raw_factor = ft.get_factor(factor_uri, name, pool_index, start, stop, code_level)
price = ft.get_price(price_uri, price, pool_index, start, stop, code_level)

# %% preprocess data for backtest
logger.info("preprocessing data")
factor = op.replace(raw_factor, 0, np.nan)
factor = op.log(factor)
factor = op.madoutlier(factor, 5)
factor = op.zscore(factor)

# %% cross section test
logger.info("performing cross section test")
ft.perform_crosssection(factor, price, 5, 
    image=result_path / 'cross-section.png')

# %% perform information coefficiency test
logger.info("performing information coefficiency test")
ft.perform_inforcoef(factor, price, rebalance, 
    image=result_path / 'information-coefficient.png')

# %% perform back test
logger.info("performing backtest")
ft.perform_backtest(factor, price, longshort=-1, topk=100, 
    benchmark=benchmark, image=result_path / 'backtest.png')
