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


GROUP_NAME = "多因子模型"
CODE_LEVEL = 'order_book_id'
DATE_LEVEL = 'date'

QTD_URI = "/home/data/quotes-day"
IDXWGT_URI = "/home/data/index-weights"
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
    parser.add_argument("--dump", action="store_true", default=True)
    parser.add_argument("--factor", type=str, default="logsize")
    parser.add_argument("--pool", type=str, default="000985.XSHG")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    dump_, factor, pool, topk = parsearg()
    if dump_:
        for ft in FACTOR_INFO.keys():
            dump(factor, YESTERDAY_STR, YESTERDAY_STR)
    else:
        rebalance(factor, pool, YESTERDAY_STR, topk)
