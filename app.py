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
    "logsize": {"module": barra, "direction": -1, "uri": "/home/data/barra"},
    "momentum_20d": {"module": barra, "direction": -1, "uri": "/home/data/barra"},
    "volatility_20d": {"module": barra, "direction": -1, "uri": "/home/data/barra"},
    "ep": {"module": barra, "direction": 1, "uri": "/home/data/barra"},
    "roa": {"module": financial, "direction": 1, "uri": "/home/data/factor"},
    "roe": {"module": financial, "direction": 1, "uri": "/home/data/factor"},
    "current_asst_ratio": {"module": financial, "direction": 1, "uri": "/home/data/factor"},
}


@app.route('/factor')
def factor():
    return jsonify({"factors": list(FACTOR_INFO.keys())})

@app.route('/dump/<string:factor>')
def dump(factor: str):
    today = datetime.datetime.now()
    if today.hour < 18:
        return jsonify({"error": "data is not dump yet!"})
    todaystr = today.strftime('%Y-%m-%d')
    try:
        data = getattr(FACTOR_INFO[factor]["module"], f'get_{factor}')(todaystr, todaystr)
        ft.save_data(data, factor, FACTOR_INFO[factor]["uri"])
        return jsonify({"message": "success"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/returns/<string:factor>')
def returns(factor: str):
    topk = request.args.get("topk", default=100, type=int)
    todaystr = datetime.datetime.now().strftime(r'%Y-%m-%d')
    table = quool.PanelTable(FACTOR_INFO[factor]["uri"])
    rollback_day = ft.get_trading_days_rollback('/home/data/quotes-day', todaystr, 1)
    quotes = ak.stock_zh_a_spot_em().set_index("代码").drop("序号", axis=1)
    factor_data = ft.get_data(table.path, factor, rollback_day)
    factor_data = factor_data.iloc[-1]
    if factor_data.name.strftime(r'%Y-%m-%d') != todaystr:
        return jsonify({"error": "data is not dump yet!"})
    factor_rank = -FACTOR_INFO[factor]["direction"] * factor_data.rank()
    weight = factor_rank[factor_rank < topk]
    weight.iloc[:] = 1 / topk
    weight.index = weight.index.str.slice(0, 6)
    ret = (weight * quotes["涨跌幅"]).mean()
    return jsonify({factor: ret})

if __name__ == "__main__":
    app.run(ssl_context=(
        '/home/kali/script/proxies/mycert.crt',
        '/home/kali/script/proxies/mykey.key'
    ), port=5000, host='0.0.0.0')