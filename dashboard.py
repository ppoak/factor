import quool
import factor
import random
import requests
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from IPython.core.magic import (magics_class, line_magic, Magics)
from IPython.terminal.prompts import Prompts, Token


@magics_class
class FactorMagics(Magics):

    def __init__(self, shell=None, **kwargs):
        super().__init__(shell, **kwargs)
        self.logger = quool.Logger("dashboard")
        self._quotes_path = "./data/quotes-day"
        self._factor_path = "./data/factor"

    @property
    def factor_table(self):
        return quool.PanelTable(self._factor_path)
    
    @property
    def quotes_table(self):
        return quool.PanelTable(self._quotes_path)
    
    @line_magic
    def set_quotes_path(self, line):
        self._quotes_path = line

    @line_magic
    def set_factor_path(self, line):
        self._factor_path = line
    
    @line_magic
    def get_factor(self, line):
        return self.factor_table.columns
    
    @line_magic
    def load_factor(self, line):
        """Usage: %load_factor [ptype] [delay] [start] [stop] name
        
        name: factor name, default,
        ptype: price type, default to open price,
        delay: delay in computing future return, default to 1,
        start: start date, default to 20100101,
        stop: stop date, default to today
        """
        opts, name = self.parse_options(line, "", "name=ptype=delay=start=stop=")
        ptype = opts.get("ptype", "open")
        delay = opts.get("delay", 1)
        start = opts.get("start", "20100101")
        stop = opts.get("stop", datetime.date.today().strftime("%Y%m%d"))

        self.logger.info("Loading factor data, please wait")
        data = self.get_data(name, start=start, stop=stop)
        self.logger.info("Loading price data, please wait")
        price = self.get_price(ptype, start, stop)
        self.logger.info("Computing future return")
        price_ = price.shift(-delay).loc[data.index]
        future = price_.shift(-1) / price_ - 1

        self.shell.user_ns['name'] = name
        self.shell.user_ns['price'] = price
        self.shell.user_ns['data'] = data
        self.shell.user_ns['future'] = future

    def get_price(self, ptype: str = "open", start: str = None, stop: str = None):
        names = [ptype] + ["st", "suspended", "adjfactor"]
        data = self.quotes_table.read(names, start=start, stop=stop)
        stsus = (data['st'] | data['suspended']).fillna(True)
        price = data.loc[:, ptype].where(~stsus)
        price = price.mul(data['adjfactor'], axis=0).unstack(self.quotes_table._code_level)
        return price
    
    def get_data(self, name: str, start: str = None, stop: str = None):
        return self.factor_table.read(name, 
            start=start, stop=stop
        )[name].unstack(self.factor_table._code_level)
    
    @line_magic
    def cross_section(self, line):
        """Usage: %cross_section [delay] [rank] [image] [result] date
        
        date: which date to inspect
        delay: delay in computing future return, default to 1,
        rank: whether to rank the factor, default to False,
        image: path to save the image, default to True
        result: path to save the result data, default to None
        """
        opts, date = self.parse_options(line, "", "delay=date=rank=image=result=")
        return factor.perform_crosssection(
            self.shell.user_ns['data'],
            self.shell.user_ns['price'],
            opts.get('delay', 1), int(date) if date.isdigit() else date,
            opts.get('rank', False), opts.get('image', True), opts.get('result')
        )
    
    @line_magic
    def infor_coef(self, line):
        """Usage: %cross_section [delay] [method] [image] [result]
        
        delay: delay in computing future return, default to 1,
        method: 'pearson' or 'spearman', default to 'pearson',
        image: path to save the image, default to True
        result: path to save the result data, default to None
        """
        opts, _ = self.parse_options(line, "", "delay=date=rank=image=result=")
        return factor.perform_inforcoef(
            self.shell.user_ns['data'],
            self.shell.user_ns['price'],
            opts.get('delay', 1), opts.get('method', 'pearson'),
            opts.get('image', True), opts.get('result')
        )

    @line_magic
    def backtest(self, line):
        """Usage: %cross_section [delay] [ngroup] [topk] [benchmark] [commission] [n_jobs] [result]
        
        delay: delay in computing future return, default to 1,
        ngroup: how many groups we divide into, default to 10,
        topk: pure long group select topk stock, default to 100,
        benchmark: to which we compare, default to None,
        commission: trasaction commission rate, default to 0.002,
        n_jobs: how many worker we use, default -1 to all,
        image: path to save the image, default to True
        result: path to save the result data, default to None
        """
        opts, _ = self.parse_options(line, "", "delay=date=rank=image=result=")
        return factor.perform_backtest(
            self.shell.user_ns['data'],
            self.shell.user_ns['price'],
            opts.get('delay', 1), opts.get('topk', 100), opts.get('benchmark'),
            opts.get('ngroup', 10), opts.get('commission', 0.002), opts.get('n_jobs', -1),
            opts.get('image', True), opts.get('result')
        )


@magics_class
class FetcherMagics(Magics):

    def __init__(self, shell=None, **kwargs):
        super().__init__(shell, **kwargs)
        self._proxy_path = "./data/proxy"
        self.proxies = self.proxy_table.read()

    @property
    def proxy_table(self):
        return quool.FrameTable(self._proxy_path)

    @line_magic
    def spot(self, line):
        self.shell.user_ns['spot'] = get_spot_data(
            self.proxies[['http']].to_dict(orient='records'))
        return self.shell.user_ns['spot']


class FactorPrompt(Prompts):
    def in_prompt_tokens(self):
        name = self.shell.user_ns.get('name', 'Not Defined')
        count = self.shell.execution_count
        return [(Token.Prompt, f'[{count}]{name}<<< ')]

    def out_prompt_tokens(self):
        name = self.shell.user_ns.get('name', 'Not Defined')
        count = self.shell.execution_count
        return [(Token.OutPrompt, f'[{count}]{name}>>> ')]


def load_ipython_extension(ipython):
    ipython.register_magics(FactorMagics)
    ipython.register_magics(FetcherMagics)

def proxy_request(url: str, proxies: list = None, **kwargs):
    if not isinstance(proxies, list):
        proxies = [proxies]
    proxies = deepcopy(proxies)
    while len(proxies):
        proxy = proxies.pop(random.randint(0, len(proxies) - 1))
        try:
            return requests.get(url, proxies=proxy, **kwargs)
        except:
            continue
    raise ConnectionError("request failed")

def format_code(code, format="{code}.{market}", style: str = "wind", upper: bool = False):
    if code.startswith("6"):
        if upper:
            return format.format(code=code, market="SH" if style == "wind" else "XSHG")
        else:
            return format.format(code=code, market="sh" if style == "wind" else "xshg")
    elif code.startswith("3") or code.startswith("0"):
        if upper:
            return format.format(code=code, market="SZ" if style == "wind" else "XSHE")
        else:
            return format.format(code=code, market="sz" if style == "wind" else "xshe")
    else:
        return np.nan

def get_spot_data(proxies: list = None) -> pd.DataFrame:
    url = "http://82.push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1",
        "pz": "50000",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152",
        "_": "1623833739532",
    }
    r = proxy_request(url, proxies, params=params)
    data_json = r.json()
    if not data_json["data"]["diff"]:
        return pd.DataFrame()
    temp_df = pd.DataFrame(data_json["data"]["diff"])
    temp_df.columns = [
        "_",
        "latest_price",
        "change_rate",
        "change_amount",
        "volume",
        "turnover",
        "amplitude",
        "turnover_rate",
        "pe_ratio_dynamic",
        "volume_ratio",
        "five_minute_change",
        "code",
        "_",
        "name",
        "highest",
        "lowest",
        "open",
        "previous_close",
        "market_cap",
        "circulating_market_cap",
        "speed_of_increase",
        "pb_ratio",
        "sixty_day_change_rate",
        "year_to_date_change_rate",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    
    temp_df["code"] = temp_df["code"].map(lambda x: format_code(x, format='{code}.{market}', style='ricequant', upper=True))
    temp_df = temp_df.dropna(subset=["code"]).set_index("code")
    temp_df = temp_df.drop(["-", "_"], axis=1)
    for col in temp_df.columns:
        if col != 'name':
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
    return temp_df
