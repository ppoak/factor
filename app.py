import quool
import barra
import datetime
import highfreq
import financial
import factor as ft
import akshare as ak
from flask import Flask, jsonify, request

app = Flask("FactorStrategy")

FACTOR_INFO = {
    "logsize": {"module": barra, "uri": "/home/data/barra"},
    "momentum_20d": {"module": barra, "uri": "/home/data/barra"},
    "volatility_20d": {"module": barra, "uri": "/home/data/barra"},
    "ep": {"module": barra, "uri": "/home/data/barra"},
    "roa": {"module": financial, "uri": "/home/data/factor"},
    "roe": {"module": financial, "uri": "/home/data/factor"},
    "current_asset_ratio": {"module": financial, "uri": "/home/data/factor"},
}


@app.route('/factors')
def factors():
    return jsonify({"factors": list(FACTOR_INFO.keys())})

def dump(factor: str, start: str, stop: str):
    data = getattr(FACTOR_INFO[factor]["module"], f'get_{factor}')(start, stop)
    ft.save_data(data, factor, FACTOR_INFO[factor]["uri"])

@app.route('/returns/<string:factor>')
def returns(factor: str):
    topk = request.args.get("topk", default=100, type=int)
    ndays = request.args.get("ndays", default=1, type=int)
    todaystr = datetime.datetime.now().strftime(r'%Y-%m-%d')
    table = quool.PanelTable(FACTOR_INFO[factor]["uri"])
    rollback_day = ft.get_trading_days_rollback('/home/data/quotes-day', todaystr, ndays)
    quotes = ak.stock_zh_a_spot_em().set_index("代码").drop("序号", axis=1)
    factor_data = ft.get_data(table.path, factor, rollback_day, rollback_day)
    factor_data = factor_data.squeeze()
    factor_rank = factor_data.rank(ascending=False)
    weight = factor_rank[factor_rank < topk]
    weight.iloc[:] = 1 / topk
    weight.index = weight.index.str.slice(0, 6)
    ret = (weight * quotes["涨跌幅"]).sum()
    return jsonify({factor: ret})

if __name__ == "__main__":
    todaystr = datetime.datetime.now().strftime(r'%Y-%m-%d')
    for factor in list(FACTOR_INFO.keys())[-1:]:
        dump(factor, todaystr, todaystr)
