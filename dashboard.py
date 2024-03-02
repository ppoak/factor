import quool
import factor
import datetime
import squarify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.magic import magics_class, line_magic, Magics
from IPython.terminal.prompts import Prompts, Token


@magics_class
class FactorMagics(Magics):

    def __init__(self, shell=None, **kwargs):
        super().__init__(shell, **kwargs)
        self.logger = quool.Logger("dashboard")
        self._quotes_path = "./data/quotes-day"
        self._factor_path = "./data/factor"
        self._proxy_path = "./data/proxy"
        self.proxies = self.proxy_table.read()

    @property
    def proxy_table(self):
        return quool.FrameTable(self._proxy_path)

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
    
    @line_magic
    def spot(self, line):
        self.shell.user_ns['spot'] = quool.get_spot_data(
            self.proxies[['http']].to_dict(orient='records'))
        return self.shell.user_ns['spot']
    
    @line_magic
    def heatmap(self, line):
        """Usage: %heatmap [sort_by] [days] [figsize] [character] [image]

        sort_by: select which field to sort, default to circulating_market_cap;
        days: compute n-day return, default None to return to yesterday-close;
        figsize: the size of the rectangle, default to 20;
        character: how many names will be labelled on the heatmap, default to 30;
        image: path to save the image, default to None;
        """
        opts, _ = self.parse_options(line, "", "sort_by=", "days=", "figsize=", "character=", "image=")
        df = self.shell.user_ns.get('spot', self.spot(None))
        df = df[["change_rate", "circulating_market_cap", "name"]].copy()
        data = self.shell.user_ns.get('data')
        if data is not None:
            data = data.iloc[-1]
            data.index = data.index.str.slice(0, 6)
            data = data[~data.index.duplicated(keep='first')]
            data.name = self.shell.user_ns['name']
            data += abs(data.min())
            df = pd.concat([df, data], axis=1, join='inner')
            df = df[df != 0].dropna(subset=data.name)
        df = df.loc[df.index.str.startswith("0") | df.index.str.startswith("3") | df.index.str.startswith("6")]
        df = df[df["change_rate"] <= 20]
        df = df.sort_values(by=opts.get('sort_by', 'circulating_market_cap'), ascending=False)
        df.iloc[int(opts.get('character', 30)):, df.columns.get_indexer_for(['name'])[0]] = ''
        colors = pd.Series(dict([
                (-20, "#00FF00"), 
                (-15, "#00E000"), 
                (-10, "#00D000"), (-9, "#00C000"), 
                (-7, "#00B000"), (-6, "#00A000"), (-5, "#009C00"), (-4, "#007C00"), 
                (-3, "#005C00"), (-2, "#003C00"), (-1, "#002C00"), 
                (0, "#000000"),
                (1, "#2C0000"), (2, "#3C0000"), (3, "#5C0000"), (4, "#7C0000"), 
                (5, "#9C0000"), (6, "#A00000"), (7, "#B00000"), 
                (9, "#C00000"), (10, "#D00000"), 
                (15, "#E00000"), 
                (20, "#FF0000")
            ]))
        colors = df["change_rate"].map(lambda x: colors[colors.index <= x].iloc[-1])
        figsize = int(opts.get('figsize', 20))
        plt.figure(figsize=(figsize, figsize))
        squarify.plot(sizes=df[opts.get('sort_by', 'circulating_market_cap')], label=df["name"], 
            color=colors, alpha=0.7, text_kwargs=dict(fontsize=int(figsize / 2)))
        plt.axis('off')
        plt.show()
        if opts.get('image', False):
            plt.savefig(opts['image'])
        plt.close()
        return df


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
