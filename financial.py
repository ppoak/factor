import factor as ft
import pandas as pd

def get_roa(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    fin_uri = '/home/data/financial'
    rollback = ft.get_trading_days_rollback(qtd_uri, start, 250)
    trading_days = ft.get_trading_days(qtd_uri, rollback, stop)
    net_profit = ft.get_data(fin_uri, 'net_profit', start=rollback, stop=stop)
    net_profit = net_profit.reindex(trading_days).ffill()
    total_asset = ft.get_data(fin_uri, 'total_assets', start=rollback, stop=stop)
    total_asset = total_asset.reindex(trading_days).ffill()
    return (net_profit / total_asset).loc[start:stop]

def get_roe(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    fin_uri = '/home/data/financial'
    rollback = ft.get_trading_days_rollback(qtd_uri, start, 250)
    trading_days = ft.get_trading_days(qtd_uri, rollback, stop)
    net_profit = ft.get_data(fin_uri, 'net_profit', start=rollback, stop=stop)
    net_profit = net_profit.reindex(trading_days).ffill()
    total_equity = ft.get_data(fin_uri, 'total_equity', start=rollback, stop=stop)
    total_equity = total_equity.reindex(trading_days).ffill()
    return (net_profit / total_equity).loc[start:stop]

def get_current_asset_ratio(
    start: str, stop: str,
) -> pd.DataFrame:
    qtd_uri = '/home/data/quotes-day'
    fin_uri = '/home/data/financial'
    rollback = ft.get_trading_days_rollback(qtd_uri, start, 250)
    trading_days = ft.get_trading_days(qtd_uri, rollback, stop)
    current_assets = ft.get_data(fin_uri, 'current_assets', start=rollback, stop=stop)
    current_assets = current_assets.reindex(trading_days).ffill()
    total_asset = ft.get_data(fin_uri, 'total_assets', start=rollback, stop=stop)
    total_asset = total_asset.reindex(trading_days).ffill()
    return (current_assets / total_asset).loc[start:stop]
