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
BARRA_URI = "/home/data/barra"
FACTOR_URI = "/home/data/factor"

TODAY_STR = datetime.datetime.now().strftime(r"%Y%m%d")
YESTERDAY_STR = ft.get_trading_days_rollback(QTD_URI, TODAY_STR, 1).strftime(r"%Y%m%d")
FACTOR_INFO = {
    "logsize": {"module": barra, "uri": BARRA_URI},
    "momentum_20d": {"module": barra, "uri": BARRA_URI},
    "volatility_20d": {"module": barra, "uri": BARRA_URI},
    "ep": {"module": barra, "uri": BARRA_URI},
    "roa": {"module": barra, "uri": BARRA_URI},
    "roe": {"module": barra, "uri": BARRA_URI},
    "current_asset_ratio": {"module": barra, "uri": BARRA_URI},
    "tail_volume_percent": {"module": highfreq, "uri": FACTOR_URI}
}

def factor_performance(
    factor: str, 
    factor_uri: str,
    start: str, 
    stop: str,
    ptype: str = 'open',
    rebperiod: int = 20,
    pool: str = None,
    pool_uri: str = None,
    topk: int = 100,
    result_path: str = "report",
):
    logger = quool.Logger("FactorTester", display_name=True)
    result_path = Path(result_path) / factor
    result_path.mkdir(parents=True, exist_ok=True)
    price = ft.get_price(QTD_URI, ptype, pool, pool_uri, 
        start, stop, code_level=CODE_LEVEL, date_level=DATE_LEVEL)
    benchmark = None
    if pool is not None:
        benchmark = ft.get_data(BENCHMARK_URI, "close", start, stop, pool, None)
    raw_factor = ft.get_data(factor_uri, factor, start, stop, pool, pool_uri)
    raw_factor = raw_factor.iloc[::rebperiod]
    logger.info("preprocessing data")
    # factor = ft.replace(raw_factor, 0, np.nan)
    # factor = ft.log(raw_factor)
    factor = ft.madoutlier(raw_factor, 5)
    factor = ft.zscore(factor)

    logger.info("performing cross section test")
    ft.perform_crosssection(factor, price, rebperiod, 
        image=result_path / 'cross-section.png')

    logger.info("performing information coefficiency test")
    ft.perform_inforcoef(factor, price, rebperiod, 
        image=result_path / 'information-coefficient.png')

    logger.info("performing backtest")
    ft.perform_backtest(factor, price, topk=topk,
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
    parser.add_argument("--start", type=str, default="20230101")
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--pool", type=str, default=None)
    parser.add_argument("--pool_uri", type=str, default=None)
    parser.add_argument("--result_path", type=str, default="report")
    return parser.parse_args()


if __name__ == "__main__":
    args = parsearg()
    if args.dump:
        for factor in FACTOR_INFO.keys():
            dump(factor, YESTERDAY_STR, YESTERDAY_STR)
    elif args.test:
        if args.factor is None or args.factor_uri is None:
            raise ValueError("you need to assign a factor and its database when testing")
        factor_performance(args.factor, args.factor_uri, args.start, args.stop, 
            args.ptype, args.rebperiod, args.pool, args.pool_uri, args.topk, args.result_path)
    else:
        rebalance(args.factor, args.pool, YESTERDAY_STR, args.topk)
