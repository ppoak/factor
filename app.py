import os
import time
import quool
import barra
import argparse
import datetime
import highfreq
import numpy as np
import pandas as pd
import factor as ft
from pathlib import Path


GROUP_NAME = "多因子模型"
CODE_LEVEL = 'order_book_id'
DATE_LEVEL = 'date'

QTD_URI = "/home/data/quotes-day"
IDXWGT_URI = "/home/data/index-weights"
BENCHMARK_URI = '/home/data/index-quotes-day'
FACTOR_URI = "/home/data/factor"

TODAY_STR = datetime.datetime.now().strftime(r"%Y%m%d")
YESTERDAY_STR = ft.get_trading_days_rollback(QTD_URI, TODAY_STR, 1).strftime(r"%Y%m%d")
FACTOR_INFO = {
    "industry": {"module": barra, "uri": barra.BARRA_URI}, 
    "logsize": {"module": barra, "uri": barra.BARRA_URI}, 
    "beta": {"module": barra, "uri": barra.BARRA_URI}, 
    "momentum": {"module": barra, "uri": barra.BARRA_URI}, 
    "volatility": {"module": barra, "uri": barra.BARRA_URI}, 
    "nonlinear_size": {"module": barra, "uri": barra.BARRA_URI},
    "volatility": {"module": barra, "uri": barra.BARRA_URI},
    "nonlinear_size": {"module": barra, "uri": barra.BARRA_URI},
    "bp": {"module": barra, "uri": barra.BARRA_URI},
    "liquidity": {"module": barra, "uri": barra.BARRA_URI},
    "leverage": {"module": barra, "uri": barra.BARRA_URI},
}

def factor_performance(
    factor_data: pd.DataFrame,
    ptype: str = 'open',
    rebperiod: int = 20,
    pool: str = None,
    pool_uri: str = None,
    ngroup: int = 10,
    topk: int = 100,
    result_path: str = "report",
):
    start = factor_data.index.min()
    stop = factor_data.index.max()
    logger = quool.Logger("FactorTester", display_name=True)
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)
    price = ft.get_price(QTD_URI, ptype, pool, pool_uri, 
        start, stop, code_level=CODE_LEVEL, date_level=DATE_LEVEL)
    benchmark = None
    if pool is not None:
        benchmark = ft.get_data(BENCHMARK_URI, "close", start, stop, pool, None)
    factor_data = factor_data.iloc[::rebperiod]

    logger.info("performing cross section test")
    ft.perform_crosssection(factor_data, price, rebperiod, 
        image=result_path / 'cross-section.png')

    logger.info("performing information coefficiency test")
    ft.perform_inforcoef(factor_data, price, rebperiod, 
        image=result_path / 'information-coefficient.png')

    logger.info("performing backtest")
    ft.perform_backtest(factor_data, price, topk=topk, ngroup=ngroup,
        benchmark=benchmark, image=result_path / 'backtest.png')

def dump(factor: str, start: str, stop: str):
    data = getattr(FACTOR_INFO[factor]["module"], f'get_{factor}')(start, stop)
    ft.save_data(data, factor, FACTOR_INFO[factor]["uri"])

def rebalance(factor: str, pool: str, date: str, topk: int = 100):
    logger = quool.Logger("Rebalnacer", stream=True, display_name=True, file="task.log")
    qtd = quool.PanelTable(QTD_URI)
    fct = quool.PanelTable(FACTOR_INFO[factor]['uri'])
    idxwgt = quool.PanelTable(IDXWGT_URI)
    
    if pool is not None:
        pool = idxwgt.read(pool, start=date, stop=date).dropna()
        pool = pool.index.get_level_values(CODE_LEVEL).unique()
    
    stsus = qtd.read('st, suspended', code=pool, start=date, stop=date)
    stsus = stsus['st'].replace(np.nan, True) | stsus['suspended'].replace(np.nan, True)
    stsus = stsus[stsus].droplevel(DATE_LEVEL).index.get_level_values(CODE_LEVEL)
    factor_data = fct.read(factor, start=date, stop=date, code=pool)
    factor_data = factor_data.droplevel(DATE_LEVEL).squeeze()
    factor_data = factor_data.loc[~factor_data.index.isin(stsus)]
    factor_data = factor_data.sort_values(ascending=False)
    factor_data = factor_data[factor_data <= topk]

    target = pd.Series(np.ones_like(target) / topk, 
        index=factor_data.index.str.slice(0, 6))
    logger.info(f"targe: {target};")
    
    snow = quool.SnowBall(os.environ["XQATOKEN"])
    groups = snow.group_list()
    for group in groups:
        if group["name"] == GROUP_NAME:
            gid = group["gid"]
    snow.order_target_percent(gid, target)

def parsearg():
    parser = argparse.ArgumentParser(description="Factor App")
    parser.add_argument("--dump", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=True)
    parser.add_argument("--factor", type=str, default=None)
    parser.add_argument("--factor_uri", type=str, default=None)
    parser.add_argument("--ptype", type=str, default="open")
    parser.add_argument("--rebperiod", type=int, default=5)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--pool", type=str, default=None)
    parser.add_argument("--pool_uri", type=str, default=None)
    parser.add_argument("--result_path", type=str, default="report")
    return parser.parse_args()


if __name__ == "__main__":
    args = parsearg()
    if args.dump:
        start = args.start or TODAY_STR
        stop = args.stop or TODAY_STR
        for factor in FACTOR_INFO.keys():
            dump(factor, start, stop)
    elif args.test:
        if args.factor is None or args.factor_uri is None:
            raise ValueError("you need to assign a factor and its database when testing")
        factor_data = ft.get_data(args.factor, args.factor_uri, args.start, args.stop, args.pool, args.pool_uri)
        factor_performance(factor_data, args.ptype, args.rebperiod, args.pool, args.pool_uri, args.topk, args.result_path)
    else:
        rebalance(args.factor, args.pool, YESTERDAY_STR, args.topk)
