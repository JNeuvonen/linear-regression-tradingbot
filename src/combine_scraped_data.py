import sys
import pandas as pd
import numpy as np
import os


def parse_details_from_path(path):
    path_parts = path.split("/")
    candles = path_parts[-1]
    pair = path_parts[-2]
    type = "futures" if "futures" in path else "spot"
    return candles, pair, type


def add_volume_driven_cols(df):

    df["taker_sell_quote_asset_volume"] = df["quote_asset_volume"] - \
        df["taker_buy_quote_asset_volume"]

    df["taker_sell_base_asset_volume"] = df["volume"] - \
        df["taker_buy_base_asset_volume"]

    df["base_taker_buy_div_by_vol"] = np.where(
        df["volume"] != 0, df["taker_buy_base_asset_volume"] / df["volume"], 0)
    df["base_taker_sell_vol_perc"] = np.where(
        df["volume"] != 0, 1 - df["base_taker_buy_div_by_vol"], 0)

    df["base_taker_sell_vol"] = df["taker_buy_base_asset_volume"] - df["volume"]

    df["base_taker_sell_div_by_vol"] = np.where(
        df["volume"] != 0, df["base_taker_sell_vol"] / df["volume"], 0)
    df["base_taker_buy_vol_perc"] = np.where(
        df["volume"] != 0, 1 - df["base_taker_sell_div_by_vol"], 0)

    df["taker_sell_quote_asset_vol_div_by_trades"] = np.where(
        df["number_of_trades"] != 0, df["taker_sell_quote_asset_volume"] / df["number_of_trades"], 0)
    df["taker_buy_quote_asset_vol_div_by_trades"] = np.where(
        df["number_of_trades"] != 0, df["taker_buy_quote_asset_volume"] / df["number_of_trades"], 0)

    df["taker_sell_quote_asset_vol_div_by_trades_minus_taker_buy_quote_asset_vol_div_by_trades"] = df[
        "taker_sell_quote_asset_vol_div_by_trades"] - df["taker_buy_quote_asset_vol_div_by_trades"]

    df["quote_asset_vol_div_by_trades"] = np.where(
        df["number_of_trades"] != 0, df["quote_asset_volume"] / df["number_of_trades"], 0)

    df["base_asset_vol_div_by_trades"] = np.where(
        df["number_of_trades"] != 0, df["volume"] / df["number_of_trades"], 0)


def add_price_driven_cols(df):
    df["high_price_minus_low_price"] = df["high_price"] - df["low_price"]
    df["high_price_minus_open_price"] = df["high_price"] - df["open_price"]
    df["high_price_minus_close_price"] = df["high_price"] - df["close_price"]
    df["low_price_minus_open_price"] = df["low_price"] - df["open_price"]
    df["low_price_minus_close_price"] = df["low_price"] - df["close_price"]


def main():
    try:
        DF_COLS = ["kline_open_time", "open_price", "high_price", "low_price", "close_price", "volume", "kline_close_time",
                   "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
        args = sys.argv[1:]
        path = args[0]
        dfs = []
        candles, pair, spot = parse_details_from_path(path)
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path, names=DF_COLS, skiprows=1)
                    df.drop(columns=["kline_close_time",
                            "ignore"], inplace=True)
                    add_volume_driven_cols(df)
                    PREFIX = pair + "_" + spot + "_"
                    df = df.add_prefix(PREFIX)
                    df = df.rename(
                        columns={PREFIX + "kline_open_time": "kline_open_time"})
                    df.columns = df.columns.str.lower()
                    dfs.append(df)
                except pd.errors.EmptyDataError:
                    print(f"Empty file found: {file_path}")
                except pd.errors.ParserError:
                    print(f"Error parsing file: {file_path}")

        combined_dir = os.path.join(path, "merged")
        if not os.path.exists(combined_dir):
            os.makedirs(combined_dir)

        result = pd.concat(dfs, axis=0)
        result = result.sort_values(by=["kline_open_time"])
        result.to_csv(combined_dir + "/combined.csv", index=False)
        print("Subprocess finished")

    except IndexError:
        print("Please provide a valid path argument.")
    except FileNotFoundError:
        print("The specified path does not exist.")


if __name__ == "__main__":
    main()
