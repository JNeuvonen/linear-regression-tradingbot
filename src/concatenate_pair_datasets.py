import os
import pandas as pd
import sys


def load_and_combine_data(path_to_data, column_names):
    combined_data = []

    for filename in os.listdir(path_to_data):
        file_path = os.path.join(path_to_data, filename)
        if "merged" in filename:
            continue

        if os.path.isfile(file_path):
            pd_df = pd.read_csv(file_path)
            pd_df.columns = column_names
            combined_data.append(pd_df)

    return pd.concat(combined_data)


def add_friday_long_indicator(combined_data):
    combined_data['friday_long'] = combined_data['kline_open_time'].apply(
        lambda x: pd.to_datetime(x, unit="ms")).apply(
        lambda x: x.weekday() == 4 and x.hour == 6)

    combined_data['return'] = combined_data['open_price'].shift(
        -24) / combined_data['open_price']

    return combined_data


def main():
    # GENERAL SETTINGS

    args = sys.argv[1:]
    path = args[0]
    pair = args[1]
    market_type = args[2]
    candle_interval = args[3]
    strategy_name = args[4]
    COLUMN_NAMES = ["kline_open_time", "open_price", "high_price", "low_price", "close_price", "volume", "kline_close_time",
                    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]

    COL_PREFIX = f"{pair.lower()}_{market_type.lower()}_{candle_interval.lower()}_"
    combined_data = load_and_combine_data(path, COLUMN_NAMES)
    combined_data.drop('ignore', axis=1, inplace=True)
    combined_data = add_friday_long_indicator(combined_data)
    combined_data = combined_data.add_prefix(COL_PREFIX)

    combined_data = combined_data.rename(
        columns={COL_PREFIX + "kline_open_time": "kline_open_time"})

    combined_data = combined_data.rename(
        columns={COL_PREFIX + "friday_long": "friday_long"})

    if not os.path.exists(path + "/linear_regr_v2"):
        os.makedirs(path + "/linear_regr_v2")
    combined_data.to_csv(
        path + f"/{strategy_name}/{pair}_{market_type}_{candle_interval}_combined_data.csv", index=False)


if __name__ == '__main__':
    main()
