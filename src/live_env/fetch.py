from constants import *
import pandas as pd
from functools import reduce


def add_friday_long_indicator(combined_data):
    combined_data['friday_long'] = combined_data['kline_open_time'].apply(
        lambda x: pd.to_datetime(x, unit="ms")).apply(
        lambda x: x.weekday() == 4 and x.hour == 6)

    return combined_data


def fetch_data(trade_engine):
    trading_pairs = trade_engine.required_symbols

    data = []

    for trade_pair in trading_pairs:
        historical_candles = trade_engine.get_historical_klines(
            trade_pair.symbol, trade_engine.KLINE_INTERVAL_1HOUR, "42 day ago UTC")
        

        required_candles = historical_candles[-(trade_engine.required_candles + 1):]
        required_candles.pop()
       
        df = pd.DataFrame(required_candles, columns=COLUMN_NAMES)
        df.drop(['kline_close_time', 'ignore'], axis=1, inplace=True)
        cols_to_drop = [
            col for col in trade_pair.cols_to_drop if col in df.columns]
        df.drop(cols_to_drop, axis=1, inplace=True)
        df = df.add_prefix(trade_pair.prefix)
        df.rename(
            columns={trade_pair.prefix + 'kline_open_time': 'kline_open_time'}, inplace=True)
        data.append(df)
    df_final = reduce(lambda left, right: pd.merge(
        left, right, on='kline_open_time'), data)
    df_final = add_friday_long_indicator(df_final)
    df_final.sort_values(by=['kline_open_time'], inplace=True)
    df_final.drop(['kline_open_time'], axis=1, inplace=True)
    df_final['friday_long'] = df_final['friday_long'].astype(int)
    return df_final
