import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from quool.table import PanelTable
from joblib import Parallel, delayed


# factor analyzing parameters
name = 'size' # factorr name
uri = '/home/data/factordev' # factor database uri
start = '20230630' # backtest start
stop = '20231231' # backtest stop
pool = None # backtest pool
delay = 1 # the delayed days to execute buy
rebalance = 1 # rebalance day
buyon = "open" # the buy price to compute future return
sellon = "open" # the sell price to compute future return
preprocess = True # whether to preprocess the factor data
n = 5 # parameter used when processing outliers
code_level = 'order_book_id' # code level in database
date_level = 'date' # date level name in database
result_path = './report' # output directory
cross_section_test = True # whether to perform cross-section test
crossdate = 0 # this should be a trading day
infor_coef_test = True # whether to perform information coeffiency test
ngroup_test = True # whether to perform group test
ngroup = 10 # how many groups to divide
topk_test = True # under development
topk = 100 # under development
commission = 0.005 # commission used in group test
n_jobs = 1 # how many cpus to use in layering test

# computing parameters from parameters
directory = f'{name}'
expname = f'{name}-{start}-{stop}-{pool}-{rebalance}-{(1 - preprocess) * "no"}preprocess'
result_path = Path(result_path).joinpath(directory).expanduser().resolve()
result_path.mkdir(exist_ok=True, parents=True)

# data interfaces
qtd = PanelTable('/home/data/quotes-day')
idx = PanelTable('/home/data/index-weights/')
idxqtd = PanelTable('/home/data/index-quotes-day')

# reading data
fct = PanelTable(uri)
code = None
if pool is not None:
    code = idx.read(pool, start=start, stop=stop
        ).dropna().index.get_level_values(code_level).unique()
price = qtd.read("open, high, low, close, adjfactor", code=code, start=start, stop=stop).sort_index()
price = price.iloc[:, :4].mul(price.iloc[:, -1], axis=0)
factor = fct.read(name, code=code, start=start, stop=stop).iloc[:, 0]

# format data
future_return = price[sellon].unstack(level=code_level).shift(-rebalance - 1) / \
    price[buyon].unstack(level=code_level).shift(-delay) - 1
factor = factor.unstack(level=code_level).iloc[::rebalance, :]

# preprocess data for backtest
if preprocess:
    median = factor.median(axis=1)
    ad = factor.sub(median, axis=0)
    mad = ad.abs().median(axis=1)
    thresh_up = median + n * mad
    thresh_down = median - n * mad
    factor = factor.clip(thresh_down, thresh_up, axis=0).where(~factor.isna())
    factor = factor.sub(factor.mean(axis=1), axis=0).div(factor.std(axis=1), axis=0)

# prepare images and data path
numfigs = cross_section_test * 2 + infor_coef_test + ngroup_test * 2 + (pool is not None)
fig, axes = plt.subplots(nrows=numfigs, ncols=1, figsize=(20, 10 * numfigs))
if numfigs == 1:
    axes = np.array([axes])
axeslist = axes.tolist()
writer = pd.ExcelWriter(str(result_path / expname) + '.xlsx')

if cross_section_test:
    if isinstance(crossdate, int):
        crossdate = factor.index[crossdate].strftime(r'%Y%m%d')
    crossdata = factor.loc[crossdate]
    crossreturn = future_return.loc[crossdate]
    crossdata = pd.concat([crossdata, crossreturn], axis=1, keys=[name, 'future return'])
    crossdata.iloc[:, 0].plot.hist(bins=300, ax=axeslist.pop(0), title=f'{name} distribution')
    crossdata.plot.scatter(x=crossdata.columns[0], y=crossdata.columns[1], 
        ax=axeslist.pop(0), title=f"{crossdate}")
    crossdata.to_excel(writer, sheet_name=f'{name} cross-section')

if infor_coef_test:
    inforcoef = factor.corrwith(future_return, axis=1).dropna()
    inforcoef.name = f"infocoef {name}"
    ax = inforcoef.plot.bar(ax=axeslist.pop(0), title=inforcoef.name)
    ax.set_xticks([i for i in range(0, inforcoef.shape[0], inforcoef.shape[0] // 10)], 
        [inforcoef.index[i].strftime(r'%Y-%m-%d') for i in range(0, inforcoef.shape[0], inforcoef.shape[0] // 10)])
    inforcoef.to_excel(writer, sheet_name=inforcoef.name)

if ngroup_test:
    benchmark = None
    if pool is not None:
        benchmark = idxqtd.read('close', code=pool, start=start, stop=stop)
        benchmark = benchmark.iloc[:, 0].unstack(level=code_level).pct_change().fillna(0)
    groups = factor.apply(lambda x: pd.qcut(x, q=ngroup, labels=False), axis=1) + 1

    future_return = price[sellon].unstack(level=code_level).shift(-2) / \
        price[buyon].unstack(level=code_level).shift(-1) - 1
    def _backtest(n, comm):
        group = groups.where(groups == n) / n
        weight = group.div(group.sum(axis=1), axis=0)
        delta = weight.fillna(0) - weight.shift(1).fillna(0)
        turnover = delta.abs().sum(axis=1) / 2
        comm *= turnover
        weight = weight.fillna(0).reindex(future_return.index).ffill()
        returns = future_return * weight
        returns = returns.sum(axis=1) - comm.reindex(returns.index).fillna(0)
        returns = returns.shift(2).fillna(0)
        return pd.concat([returns, turnover], axis=1, keys=['returns', 'turnover']).fillna(0)

    result = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_backtest)(i, commission) for i in range(1, ngroup + 1)
    )
    profit = pd.concat([res["returns"] for res in result], axis=1,
        keys=range(1, ngroup + 1)).add_prefix('Group')
    turnover = pd.concat([res["turnover"] for res in result], axis=1,
        keys=range(1, ngroup + 1)).add_prefix('Group')

    if benchmark is not None:
        exprofit = profit.sub(benchmark.iloc[:, 0], axis=0)
        profit = pd.concat([profit, benchmark], axis=1)
    
    cumprofit = (profit + 1).cumprod()
    if benchmark is not None:
        excumprofit = (exprofit + 1).cumprod()
    
    cumprofit.plot(ax=axeslist.pop(0), title=f'cumulative netvalue {name}')
    if benchmark is not None:
        excumprofit.plot(ax=axeslist.pop(0), title=f'execess cumulative netvalue {name}')
    turnover.plot(ax=axeslist.pop(0), title=f'turnover {name}')
    
    profit.to_excel(writer, sheet_name=f'profit {name}')
    cumprofit.to_excel(writer, sheet_name=f'cumprofit {name}')
    if benchmark is not None:
        excumprofit.to_excel(writer, sheet_name=f'excumprofit {name}')
    turnover.to_excel(writer, sheet_name=f'turnover {name}')

fig.tight_layout()
fig.savefig(str(result_path / expname) + '.png')

writer.close()
