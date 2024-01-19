__version__ = "0.2.0"


import quool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def get_pool(
    uri: str,
    name: str,
    start: str | pd.Timestamp = None,
    stop: str | pd.Timestamp = None,
) -> pd.MultiIndex:
    pool_table = quool.PanelTable(uri)
    pool_data = pool_table.read(name, 
        start=start, stop=stop).dropna().index
    return pool_data

def get_factor(
    uri: str, 
    name: str,
    pool: pd.MultiIndex = None,
    start: str | pd.Timestamp = None,
    stop: str | pd.Timestamp = None,
    code_level: int | str = 1,
) -> pd.DataFrame:
    factor_table = quool.PanelTable(uri)
    code = None
    if pool is not None:
        code = pool.get_level_values(level=code_level).unique()
    factor_data = factor_table.read(name, 
        start=start, stop=stop, code=code).iloc[:, 0]
    if pool is not None:
        factor_data = factor_data.loc[
            factor_data.index.intersection(pool)]
    return factor_data.unstack(level=code_level)

def get_price(
    uri: str,
    name: str,
    pool: pd.MultiIndex = None,
    start: str | pd.Timestamp = None,
    stop: str | pd.Timestamp = None,
    code_level: int | str = 1,
) -> pd.DataFrame:
    price_table = quool.PanelTable(uri)
    code = None
    if pool is not None:
        code = pool.get_level_values(level=code_level).unique()
    price_data = price_table.read(f'{name}, adjfactor, st, suspended',
        start=start, stop=stop, code=code)
    stsus = (price_data['st'] | price_data['suspended']).replace(np.nan, True)
    price = price_data[name].where(~stsus) * price_data['adjfactor']
    if pool is not None:
        price = price.loc[price.index.intersection(pool)]
    return price.unstack(level=code_level)

def get_benchmark(
    uri: str,
    price: str,
    pool: str,
    start: str | pd.Timestamp = None,
    stop: str | pd.Timestamp = None,
    code_level: int | str = 1,
) -> pd.Series:
    benchmark_table = quool.PanelTable(uri, code_level=code_level)
    benchmark_data = benchmark_table.read(price, code=pool,
        start=start, stop=stop).iloc[:, 0].unstack(level=code_level)
    return benchmark_data.iloc[:, 0]

def perform_crosssection(
    factor: pd.DataFrame,
    price: pd.DataFrame = None,
    rebalance: int = 5,
    crossdate: str | int = -1,
    rank: bool = False,
    image: str | bool = True,
    result: str = None,
):
    crossdate = factor.index[min(crossdate, -rebalance)].strftime(r"%Y-%m-%d")
    factor = factor.loc[crossdate]
    if price is not None:
        future_returns = price.shift(-rebalance) / price - 1
        future_returns = future_returns.loc[crossdate]
    else:
        future_returns = pd.Series(index=factor.index)
    if rank:
        factor = factor.rank()
    data = pd.concat([factor, future_returns], 
        axis=1, keys=["factor", "future_returns"])
    if image is not None:
        fig, axes = plt.subplots(2, 1, figsize=(20, 20))
        data["factor"].plot.hist(bins=100, ax=axes[0])
        data.plot.scatter(ax=axes[1], title=crossdate,
            x="factor", y="future_returns")
        fig.tight_layout()
        if not isinstance(image, bool):
            fig.savefig(image)
        else:
            fig.show()
    if result is not None:
        data.to_excel(result)
    return data

def perform_inforcoef(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    rebalance: int = 5,
    method: str = 'pearson',
    image: str | bool = True,
    result: str = None,
):
    future_returns = price.shift(-rebalance) / price - 1
    inforcoef = factor.corrwith(future_returns, axis=1, method=method).dropna()
    inforcoef.name = f"infocoef"
    if image is not None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax = inforcoef.plot.bar(ax=ax, title=inforcoef.name)
        ax.set_xticks([i for i in range(0, inforcoef.shape[0], inforcoef.shape[0] // 10)], 
            [inforcoef.index[i].strftime(r'%Y-%m-%d') for i in range(0, inforcoef.shape[0], inforcoef.shape[0] // 10)])
        fig.tight_layout()
        if not isinstance(image, bool):
            fig.savefig(image)
        else:
            fig.show()
    if result is not None:
        inforcoef.to_excel(result)
    return inforcoef

def perform_backtest(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    longshort: int = 1,
    topk: int = 100,
    benchmark: pd.Series = None,
    delay: int = 1,
    ngroup: int = 5,
    commission: float = 0.002,
    n_jobs: int = -1,
    image: str | bool = True,
    result: str = None,
):
    # ngroup test
    groups = factor.apply(lambda x: pd.qcut(x, q=ngroup, labels=False), axis=1) + 1
    ngroup_result = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(quool.rebalance_strategy)(
            groups.where(groups == i), price, delay, 'both', commission, benchmark
    ) for i in range(1, ngroup + 1))
    ngroup_evaluation = pd.concat([res.evaluation for res in ngroup_result], 
        axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
    ngroup_returns = pd.concat([res.returns for res in ngroup_result], 
        axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
    ngroup_turnover = pd.concat([res.turnover for res in ngroup_result], 
        axis=1, keys=range(1, ngroup + 1)).add_prefix('group')

    # topk test
    topks = (factor.rank(axis=1) * longshort) <= topk
    topks = factor.mask(topks, 1).mask(~topks, 0)
    topk_result = quool.rebalance_strategy(topks, price, delay, 'both', commission, benchmark)
    topk_evaluation = topk_result.evaluation
    topk_returns = topk_result.returns
    topk_turnover = topk_result.turnover

    # longshort test
    longshort_returns = longshort * (ngroup_returns[f"group{ngroup}"] - ngroup_returns["group1"])
    longshort_value = (longshort_returns + 1).cumprod()
    longshort_value.name = f"group1 - group{ngroup}" if longshort > 0 else f"group{ngroup} - group1"
    
    # merge returns
    if benchmark is not None:
        benchmark = benchmark.pct_change().fillna(0)
        ngroup_exreturns = ngroup_returns.sub(benchmark, axis=0)
        ngroup_returns = pd.concat([ngroup_returns, benchmark], axis=1)
        topk_exreturns = topk_returns.sub(benchmark, axis=0)
        topk_returns = pd.concat([topk_returns, benchmark], axis=1)

    # compute value
    ngroup_value = (ngroup_returns + 1).cumprod()
    topk_value = (topk_returns + 1).cumprod()
    longshort_value = (longshort_returns + 1).cumprod()
    
    # compute exvalue
    if benchmark is not None:
        ngroup_exvalue = (ngroup_exreturns + 1).cumprod()
        topk_exvalue = (topk_exreturns + 1).cumprod()
    
    # naming
    ngroup_evaluation.name = "ngroup evaluation"
    ngroup_value.name = "ngroup value"
    topk_value.name = "topk value"
    topk_exvalue.name = "topk exvalue"
    longshort_value.name = "longshort value"
    ngroup_turnover.name = "ngroup turnover"
    topk_turnover.name = "topk turnover"
    if benchmark is not None:
        ngroup_exvalue.name = "ngroup exvalue"
        topk_exvalue.name = "topk exvalue"

    if image is not None:
        fignum = 5 + 2 * (benchmark is not None)
        fig, axes = plt.subplots(nrows=fignum, ncols=1, figsize=(20, 10 * fignum))
        ngroup_value.plot(ax=axes[0], title=ngroup_value.name)
        ngroup_turnover.plot(ax=axes[1], title=ngroup_turnover.name)
        topk_value.plot(ax=axes[2], title=topk_value.name)
        topk_turnover.plot(ax=axes[3], title=topk_turnover.name)
        longshort_value.plot(ax=axes[4], title=longshort_value.name)
        if benchmark is not None:
            ngroup_exvalue.plot(ax=axes[5], title=ngroup_exvalue.name)
            topk_exvalue.plot(ax=axes[6], title=topk_exvalue.name)
        fig.tight_layout()
        if not isinstance(image, bool):
            fig.savefig(image)
        else:
            fig.show()

    if result is not None:
        with pd.ExcelWriter(result) as writer:
            ngroup_evaluation.to_excel(writer, sheet_name=ngroup_evaluation.name)
            topk_evaluation.to_excel(writer, sheet_name=topk_evaluation.name)
            ngroup_value.to_excel(writer, sheet_name=ngroup_returns.name)
            ngroup_turnover.to_excel(writer, sheet_name=ngroup_turnover.name)
            topk_value.to_excel(writer, sheet_name=topk_returns.name)
            topk_turnover.to_excel(writer, sheet_name=topk_turnover.name)
            longshort_value.to_excel(writer, sheet_name=longshort_returns.name)
            
            if benchmark is not None:
                ngroup_exvalue.to_excel(writer, sheet_name=ngroup_exreturns.name)
                topk_exvalue.to_excel(writer, sheet_name=topk_exreturns.name)
