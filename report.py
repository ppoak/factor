import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from quool.table import PanelTable
from joblib import Parallel, delayed


# %% factor analyzing parameters
name = 'momentum20' # factorr name
uri = '/home/data/barra' # factor database uri
start = '20230101' # backtest start
stop = None # backtest stop
pool = '399101.XSHE' # backtest pool
delay = 1 # the delayed days to execute buy
rebalance = 5 # rebalance day
buyon = "open" # the buy price to compute future return
sellon = "open" # the sell price to compute future return
preprocess = True # whether to preprocess the factor data
ndeviate = 5 # parameter used when processing outliers
code_level = 'order_book_id' # code level in database
date_level = 'date' # date level name in database
result_path = './report' # output directory
cross_section_test = True # whether to perform cross-section test
crossdate = 0 # this should be a trading day
infor_coef_test = True # whether to perform information coeffiency test
ngroup_test = True # whether to perform group test
ngroup = 10 # how many groups to divide
longshort_test = True # under development
topk_test = True # under development
topk = 10 # under development
commission = 0.005 # commission used in group test
n_jobs = 1 # how many cpus to use in layering test


# %% computing parameters from parameters
today = datetime.datetime.today().strftime(r'%Y%m%d')
stop = stop or today
directory = f'{name}'
expname = f'{name}-{start}-{stop}-{pool}-{rebalance}-{(1 - preprocess) * "no"}preprocess'
result_path = Path(result_path).joinpath(directory).expanduser().resolve()
result_path.mkdir(exist_ok=True, parents=True)

# %% data interfaces
qtd = PanelTable('/home/data/quotes-day')
idx = PanelTable('/home/data/index-weights/')
idxqtd = PanelTable('/home/data/index-quotes-day')

# %% useful functions
def reweight(
    weight: pd.DataFrame, 
    future_return_1d: pd.DataFrame, 
    delay: int,
    rebalance: int,
    comm: float,
):
    weight = weight.div(weight.sum(axis=1), axis=0)
    delta = weight.fillna(0) - weight.shift(1).fillna(0)
    turnover = delta.abs().sum(axis=1) / 2
    turnover = turnover.reindex(future_return_1d.index).shift(delay).fillna(0)
    weight = weight.fillna(0).reindex(future_return_1d.index).ffill()
    returns = future_return_1d * weight
    returns = returns.sum(axis=1).shift(delay + rebalance).fillna(0)
    returns -= comm * turnover
    return pd.concat([returns, turnover], axis=1, keys=['returns', 'turnover']).fillna(0)

# %% reading data
fct = PanelTable(uri)
code = None
if pool is not None:
    code = idx.read(pool, start=start, stop=stop).dropna().index.get_level_values(code_level).unique()
data = qtd.read("open, high, low, close, volume, st, suspended, adjfactor", code=code, start=start, stop=stop)
stsus = (data['st'] | data['suspended']).replace(np.nan, True).astype("bool")
price = data.iloc[:, :4].where(~stsus, axis=0)
price = price.mul(data['adjfactor'], axis=0)
factor = fct.read(name, code=code, start=start, stop=stop).iloc[:, 0]

# %% format data
future_return = price[sellon].unstack(level=code_level).shift(-rebalance - 1) / \
    price[buyon].unstack(level=code_level).shift(-delay) - 1
future_return_1d = price[sellon].unstack(level=code_level).shift(-2) / \
        price[buyon].unstack(level=code_level).shift(-1) - 1
factor = factor.unstack(level=code_level).iloc[::rebalance, :]

# %% preprocess data for backtest
if preprocess:
    median = factor.median(axis=1)
    ad = factor.sub(median, axis=0)
    mad = ad.abs().median(axis=1)
    thresh_up = median + ndeviate * mad
    thresh_down = median - ndeviate * mad
    factor = factor.clip(thresh_down, thresh_up, axis=0).where(~factor.isna())
    factor = factor.sub(factor.mean(axis=1), axis=0).div(factor.std(axis=1), axis=0)

# %% prepare images and data path
numfigs = cross_section_test * 2 + infor_coef_test + ngroup_test * 2 + \
    (pool is not None) + longshort_test + topk_test
fig, axes = plt.subplots(nrows=numfigs, ncols=1, figsize=(20, 10 * numfigs))
if numfigs == 1:
    axes = np.array([axes])
axeslist = axes.tolist()
writer = pd.ExcelWriter(str(result_path / expname) + '.xlsx')

# %% create a sheet for abstract
abstract = pd.Series(
    {
        "name": name,
        "start": start,
        "stop": stop,
        "pool": pool,
        "commission": commission,
        "ngroup": ngroup,
        "topk": topk,
    }
)
abstract.to_excel(writer, sheet_name=f'ABSTRACT')

# cross section test
if cross_section_test:
    if isinstance(crossdate, int):
        crossdate = factor.index[crossdate].strftime(r'%Y%m%d')
    crossdata = factor.loc[crossdate]
    crossreturn = future_return.loc[crossdate]
    crossdata = pd.concat([crossdata, crossreturn], axis=1, keys=[name, 'future return'])
    crossdata.iloc[:, 0].plot.hist(bins=300, ax=axeslist.pop(0), title=f'distribution')
    crossdata.plot.scatter(x=crossdata.columns[0], y=crossdata.columns[1], 
        ax=axeslist.pop(0), title=f"{crossdate}")
    crossdata.to_excel(writer, sheet_name=f'cross-section')

if infor_coef_test:
    inforcoef = factor.corrwith(future_return, axis=1).dropna()
    inforcoef.name = f"infocoef"
    ax = inforcoef.plot.bar(ax=axeslist.pop(0), title=inforcoef.name)
    ax.set_xticks([i for i in range(0, inforcoef.shape[0], inforcoef.shape[0] // 10)], 
        [inforcoef.index[i].strftime(r'%Y-%m-%d') for i in range(0, inforcoef.shape[0], inforcoef.shape[0] // 10)])
    inforcoef.to_excel(writer, sheet_name=inforcoef.name)

# %% perform ngroup test
if ngroup_test:
    benchmark = None
    if pool is not None:
        benchmark = idxqtd.read('close', code=pool, start=start, stop=stop)
        benchmark = benchmark.iloc[:, 0].unstack(level=code_level).pct_change().fillna(0)
    groups = factor.apply(lambda x: pd.qcut(x, q=ngroup, labels=False), axis=1) + 1
    result = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(reweight)(
            groups.where(groups == i), future_return_1d, delay, rebalance, commission
    ) for i in range(1, ngroup + 1))
    profit = pd.concat([res["returns"] for res in result], axis=1,
        keys=range(1, ngroup + 1)).add_prefix('group')
    turnover = pd.concat([res["turnover"] for res in result], axis=1,
        keys=range(1, ngroup + 1)).add_prefix('group')

    if benchmark is not None:
        exprofit = profit.sub(benchmark.iloc[:, 0], axis=0)
        profit = pd.concat([profit, benchmark], axis=1)
    
    cumprofit = (profit + 1).cumprod()
    if benchmark is not None:
        excumprofit = (exprofit + 1).cumprod()
    
    cumprofit.plot(ax=axeslist.pop(0), title=f'cumulative netvalue')
    if benchmark is not None:
        excumprofit.plot(ax=axeslist.pop(0), title=f'execess cumulative netvalue')
    turnover.plot(ax=axeslist.pop(0), title=f'turnover')
    
    profit.to_excel(writer, sheet_name=f'profit')
    cumprofit.to_excel(writer, sheet_name=f'cumprofit')
    if benchmark is not None:
        excumprofit.to_excel(writer, sheet_name=f'excumprofit')
    turnover.to_excel(writer, sheet_name=f'turnover')

# %% perform longshort test
if longshort_test:
    longshort_profit = (profit[f'group{ngroup}'] - profit['group1']) * np.sign(inforcoef.mean())
    longshort_cumprofit = (longshort_profit + 1).cumprod()
    longshort_profit.name = "longshort"
    longshort_cumprofit.plot(ax=axeslist.pop(0), title=f'longshort')
    longshort_profit.to_excel(writer, sheet_name=f'longshort')

# %% perform topk test
if topk_test:
    benchmark = None
    if pool is not None:
        benchmark = idxqtd.read('close', code=pool, start=start, stop=stop)
        benchmark = benchmark.iloc[:, 0].unstack(level=code_level).pct_change().fillna(0)
    
    selected = factor.rank(axis=1) * -np.sign(inforcoef.mean()) <= topk
    top = factor.mask(selected, 1).mask(~selected, 0)
    topk_result = reweight(top, future_return_1d, delay, rebalance, commission)
    if benchmark is not None:
        topk_result = pd.concat([topk_result, benchmark], axis=1)
    no_turnover_col = topk_result.columns[~topk_result.columns.str.contains('turnover')]
    topk_result.loc[:, no_turnover_col] = (topk_result.loc[:, no_turnover_col] + 1).cumprod()
    topk_result.plot(ax=axeslist.pop(0), title=f'top{topk}', secondary_y=['turnover'])
    topk_result.to_excel(writer, sheet_name=f'top{topk}')

# %% conclude the result
fig.tight_layout()
fig.savefig(str(result_path / expname) + '.png')

if cross_section_test:
    abstract["cross_section_skew"] = crossdata[name].skew()
    abstract["cross_section_kurtosis"] = crossdata[name].kurtosis()
if infor_coef_test:
    abstract["infor_coef_mean"] = inforcoef.mean()
    abstract["infor_coef_tvalue"] = inforcoef.mean() / inforcoef.std()
if ngroup_test:
    abstract = pd.concat([abstract, 100 * (cumprofit.iloc[-1, :] - 1).add_suffix("_return(%)")], axis=0)
    abstract = pd.concat([abstract, 100 * turnover.mean().add_suffix("_turnover(%)")], axis=0)
if longshort_test:
    abstract["longshort_return(%)"] = 100 * (longshort_cumprofit.iloc[-1] - 1)
if topk_test:
    abstract["topk_return(%)"] = 100 * (topk_result["returns"].iloc[-1] - 1)

abstract.index.name = "INDICATORS"
abstract.name = "ABSTRACT"
abstract.to_excel(writer, sheet_name=f'ABSTRACT')

writer.close()
