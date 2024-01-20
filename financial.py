import pandas as pd
from factor import get_data

def get_roa(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    fin_uri = '/home/data/financial'
    price = get_data(qtd_uri, "close", start=start, stop=stop)
    net_profit = get_data(fin_uri, 'net_profit', start=start, stop=stop)
    net_profit = net_profit.reindex(price.index).ffill()
    total_asset = get_data(fin_uri, 'total_asset', start=start, stop=stop)
    total_asset = total_asset.reindex(price.index).ffill()
    return net_profit / total_asset

def get_roe(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    fin_uri = '/home/data/financial'
    price = get_data(qtd_uri, "close", start=start, stop=stop)
    net_profit = get_data(fin_uri, 'net_profit', start=start, stop=stop)
    net_profit = net_profit.reindex(price.index).ffill()
    total_equity = get_data(fin_uri, 'total_equity', start=start, stop=stop)
    total_equity = total_equity.reindex(price.index).ffill()
    return net_profit / total_equity

def get_current_asset_ratio(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    fin_uri = '/home/data/financial'
    price = get_data(qtd_uri, "close", start=start, stop=stop)
    current_assets = get_data(fin_uri, 'current_assets', start=start, stop=stop)
    current_assets = current_assets.reindex(price.index).ffill()
    total_asset = get_data(fin_uri, 'total_asset', start=start, stop=stop)
    total_asset = total_asset.reindex(price.index).ffill()
    return current_assets / total_asset
