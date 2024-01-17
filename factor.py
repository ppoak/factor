import numpy as np
import pandas as pd
import quool.util as qu
import quool.core as qc
import quool.data as qd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed


def _get_future(
    price: pd.DataFrame | pd.Series,
    rebalance: int,
    buyon: str = "open",
    sellon: str = "open",
    delay: int = 1,
    code_level: str | int = 0,
):
    future = qd.Dim2Frame(qa.Return(price, buyon=buyon, sellon=sellon,
        code_level=code_level)(delay, rebalance))
    return future._data

def _get_factor(
    factor: pd.DataFrame | pd.Series,
    code_level: int | str = 0
):
    factor_formatter = qf.Factor(factor)
    return factor_formatter.data

def preprocess_factor(
    factor: pd.DataFrame,
    n: int = 5,
    code_level: int | str = 0,
    date_level: int | str = 1,
):
    factor = _get_factor(factor)
    factor = qa.RobustScaler(n=n, code_level=code_level, date_level=date_level).fit_transform(factor)
    factor = qa.StandardScaler(code_level=code_level, date_level=date_level).fit_transform(factor)
    return factor

def cross_section_test(
    factor: pd.DataFrame | pd.Series,
    price: pd.DataFrame | pd.Series,
    date: str,
    rebalance: int,
    buy: str = "open",
    sell: str = "close",
    code_level: int | str = 0,
    date_level: int | str = 1,
    image_path: str | Path = None,
    data_path: str | Path = None,
):
    factor = _get_factor(factor, code_level)
    future = _get_future(price, rebalance, buy, sell, 1, code_level, date_level)    
    d = factor.loc[date]
    fr = future.loc[date]
    d = pd.concat([d, fr], axis=1, keys=[name, 'future return'])
    if image_path:
        fig, axes = plt.subplots(2, 1, figsize=(20, 20))
        axeslist = axes.tolist()
        d.iloc[:, 0].plot.hist(bins=300, ax=axeslist.pop(0), title=f'{name} distribution')
        d.plot.scatter(x=d.columns[0], y=d.columns[1], ax=axeslist.pop(0), title=f"{date}")
        fig.tight_layout()
        fig.savefig(image_path)
    if data_path:
        d.to_excel(data_path, sheet_name=f'{name} cross-section')
    
    return d

def information_coef_test(
    factor: pd.DataFrame | pd.Series,
    price: pd.DataFrame | pd.Series,
    rebalance: int,
    buy: str = "open",
    sell: str = "close",
    code_level: int | str = 0,
    date_level: int | str = 1,
    image_path: str | Path = None,
    data_path: str | Path = None,
):
    factor = _get_factor(factor, code_level)
    future = _get_future(price, rebalance, buy, sell, 1, code_level, date_level)
    ic = qa.Corr(code_level=code_level, date_level=date_level).fit_transform(factor, future)
    ic.name = f"infocoef {name}"
    if image_path:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax = ic.plot.bar(ax=ax, title=ic.name)
        ax.set_xticks([i for i in range(0, ic.shape[0], rebalance)], 
            [ic.index[i].strftime(r'%Y-%m-%d') for i in range(0, ic.shape[0], rebalance)])
        fig.tight_layout()
        fig.savefig(image_path)
    if data_path:
        ic.to_excel(data_path, sheet_name=ic.name)
    return ic

def layering_test(
    factor: pd.DataFrame | pd.Series,
    price: pd.DataFrame | pd.Series,
    rebalance: int,
    benchmark: pd.Series = None,
    buy: str = "open",
    sell: str = "close",
    n_jobs: int = -1,
    code_level: int | str = 0,
    date_level: int | str = 1,
    image_path: str | Path = None,
    data_path: str | Path = None,
):
    pass



if __name__ == '__main__':
    # factor analyzing parameters
    name = 'ep' # factorr name
    uri = '/home/data/factordev' # factor database uri
    start = '20180101' # backtest start
    stop = '20231231' # backtest stop
    pool = '000985.XSHG' # backtest pool
    preprocess = True # whether to preprocess the factor data
    n = 5 # n is the parameter in RobustScaler
    rebalance = 20 # rebalance day, this will divide all cash into `rebalance` shares
    buy = "open" # buy price
    sell = "open" # sell price
    code_level = 'order_book_id' # code level in database
    date_level = 'date' # date level name in database
    result_path = './output' # output directory
    crosssection = True # whether to perform cross-section test
    crossdate = "20180102" # this should be a trading day
    information = True # whether to perform information coeffiency test
    layering = True # whether to perform layering test
    ngroup = 10 # how many groups to divide
    n_jobs = -1 # how many cpus to use in layering test

    expname = Path(f'{name}-{start}-{stop}-{pool}-{rebalance}-{(1 - preprocess) * "no"}preprocess')
    result_path = Path(result_path / expname).expanduser().resolve()
    result_path.mkdir(exist_ok=True, parents=True)
    qtd = qd.Dim3Table('/home/data/quotes-day')
    idx = qd.Dim3Table('/home/data/index-weights/')
    idxqtd = qd.Dim3Table('/home/data/index-quotes-day')
    fct = qd.Dim3Table(uri)
    code = idx.read(pool, start=start, stop=stop
        ).dropna().index.get_level_values(code_level).unique()

    price = qtd.read("open, high, low, close, adjfactor", code=code, start=start, stop=stop).sort_index()
    price = price.iloc[:, :-1].mul(price.iloc[:, -1], axis=0)
    factor = fct.read(name, code=code, start=start, stop=stop).iloc[:, 0]

    # backtesting ...
    if preprocess:
        factor = preprocess_factor(factor, n, code_level, date_level)
        
    if crosssection:
        cross_section_test(factor, price, crossdate, 
            rebalance, buy, sell, code_level, date_level, 
            result_path / 'cross_section.png',
            result_path / 'cross_section.xlsx',
        )

    if information:
        information_coef_test(
            factor, price, rebalance, buy, sell,
            code_level, date_level,
            result_path / 'information_coef.png',
            result_path / 'information_coef.xlsx',
        )

    if layering:
        benchmark = idxqtd.read('close', code=pool, 
            start=start, stop=stop).iloc[:, 0].droplevel(code_level)
        layering_test(
            factor, price, rebalance, benchmark, buy, sell,
            n_jobs, code_level, date_level,
            result_path / 'layering.png',
            result_path / 'layering.xlsx',
        )
