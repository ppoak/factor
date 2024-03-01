import quool
import factor
import datetime
import pandas as pd
from pathlib import Path
from IPython.core.magic import (magics_class, line_magic, Magics)
from IPython.terminal.prompts import Prompts, Token


@magics_class
class DashboardMagics(Magics):

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
    ipython.register_magics(DashboardMagics)
