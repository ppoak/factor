import os
import time
import quool
import barra
import argparse
import datetime
import highfreq
import financial
import numpy as np
import pandas as pd
import factor as ft


CODELEVEL = 'order_book_id'
DATELEVEL = 'date'

QTDURI = "/home/data/quotes-day"
IDXWGTURI = "/home/data/index-weights"
BARRAURI = "/home/data/barra"
FACTORURI = "/home/data/factor"

TODAYSTR = datetime.datetime.now().strftime(r"%Y%m%d")
YESTERDAYSTR = ft.get_trading_days_rollback(QTDURI, TODAYSTR, 1).strftime(r"%Y%m%d")
FACTORINFO = {
    "logsize": {"module": barra, "uri": BARRAURI},
    "momentum_20d": {"module": barra, "uri": BARRAURI},
    "volatility_20d": {"module": barra, "uri": BARRAURI},
    "ep": {"module": barra, "uri": BARRAURI},
    "roa": {"module": financial, "uri": FACTORURI},
    "roe": {"module": financial, "uri": FACTORURI},
    "current_asset_ratio": {"module": financial, "uri": FACTORURI},
    "tail_volume_percent": {"module": highfreq, "uri": FACTORURI}
}


def dump(factor: str, start: str, stop: str):
    data = getattr(FACTORINFO[factor]["module"], f'get_{factor}')(start, stop)
    ft.save_data(data, factor, FACTORINFO[factor]["uri"])

def rebalance(factor: str, pool: str, date: str, topk: int = 100):
    logger = quool.Logger("Rebalnacer", stream=True, display_name=True, file="task.log")
    qtd = quool.PanelTable(QTDURI)
    fct = quool.PanelTable(FACTORINFO[factor]['uri'])
    idxwgt = quool.PanelTable(IDXWGTURI)
    
    if pool is not None:
        pool = idxwgt.read(pool, start=date, stop=date).dropna()
        pool = pool.index.get_level_values(CODELEVEL).unique()
    
    stsus = qtd.read('st, suspended', code=pool, start=date, stop=date)
    stsus = stsus['st'].replace(np.nan, True) | stsus['suspended'].replace(np.nan, True)
    stsus = stsus[stsus].droplevel(DATELEVEL).index.get_level_values(CODELEVEL)
    factor_data = fct.read(factor, start=date, stop=date, code=pool)
    factor_data = factor_data.droplevel(DATELEVEL).squeeze()
    factor_data = factor_data.loc[~factor_data.index.isin(stsus)]
    factor_data = factor_data.sort_values(ascending=False)
    factor_data = factor_data[factor_data <= topk]

    target = factor_data.index
    target = target.str.slice(0, 6)
    logger.info(f"targe: {target};")
    
    snow = quool.SnowBall(os.environ["XQATOKEN"])
    gid = snow.group_list()[1]["gid"]
    info = snow.performance(gid)["result_data"]["performances"]
    value = info[0]["assets"]
    cash = info[0]["cash"]
    if len(info) > 1:
        pos = pd.DataFrame(info[1]["list"]).set_index('symbol')
        pos.index = pos.index.str.slice(2)
    else:
        pos = pd.DataFrame()

    holds = pos.index.intersection(target)
    holds = pos.loc[holds, "current"] * pos.loc[holds, "shares"]
    adjust = value / topk - holds
    sells = pos.index.difference(target)
    buys = target.difference(pos.index)

    logger.info("-" * 10 + "SELLINGS" + "-" * 10)
    for code in sells:
        code = f'SZ{code}' if code.startswith('0') or code.startswith('3') else f'SH{code}'
        snow.transaction_add(gid, code, -pos.loc[code, "shares"], pos.loc[code, "current"])
        logger.info(f"{code} closed position {shares:.0f} shares")
        cash += pos.loc[code, "current"] * pos.loc[code, "shares"]
    
    logger.info("-" * 10 + "ADJUSTING" + "-" * 10)
    for code in adjust.index:
        code = f'SZ{code}' if code.startswith('0') or code.startswith('3') else f'SH{code}'
        if cash < adjust.loc[code]:
            logger.info(f"short in cash abort adjust {code}")
            continue
        shares = (adjust.loc[code] / pos.loc[code, "current"] // 100) * 100
        if 0 < shares < 100:
            logger.info(f"not enough share: {shares} to adjust")
            continue
        snow.transaction_add(gid, code, shares, pos.loc[code, "current"])
        logger.info(f"{code} adjusted position {shares:.0f} shares")
        cash -= shares * pos.loc[code, "current"]
    
    logger.info("-" * 10 + "BUYINGS" + "-" * 10)
    for code in buys:
        code = f'SZ{code}' if code.startswith('0') or code.startswith('3') else f'SH{code}'
        if cash < value / topk:
            logger.info(f"short in cash abort {code}")
            continue
        price = snow.quote(code).loc[code, 'current']
        shares = (value / topk / price // 100) * 100
        if shares < 100:
            logger.info(f"current cash {cash} is not enough for 100 share")
            continue
        snow.transaction_add(gid, code, shares, price)
        logger.info(f"{code} buy {shares} at {price}")
        cash -= price * shares
        time.sleep(2)

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
        for factor in FACTORINFO.keys():
            dump(factor, YESTERDAYSTR, YESTERDAYSTR)
    else:
        rebalance(factor, pool, YESTERDAYSTR, topk)
