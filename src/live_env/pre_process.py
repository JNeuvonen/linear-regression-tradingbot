import pandas as pd
from constants import IGNORED_COLS
import pickle
import numpy as np
import torch


def turn_col_into_sma(df, col, delete_col, sma_windows):

    df[col] = df[col].astype(float)
    for sma in sma_windows:
        sma = str(sma)
        df[col + "_" + sma] = df[col].rolling(window=int(sma)).mean()
        df[col + "_diff_to_" + sma] = df[col] / df[col + "_" + sma] - 1

    if delete_col:
        df.drop(col, axis=1, inplace=True)
        for sma in sma_windows:
            sma = str(sma)
            df.drop(col + "_" + sma, axis=1, inplace=True)


def arrange_columns(data, column_order_csv_path):
    column_order = pd.read_csv(column_order_csv_path)["Column Names"]

    reordered_data = data[column_order]

    return reordered_data


def pre_process(data, sma_windows, glassnode_windows):
    DATA_COLS = data.columns
    data = data.dropna()

    for col in DATA_COLS:
        ignore_col = False
        for elem in IGNORED_COLS:
            if elem in col:
                ignore_col = True
                break
        if ignore_col:
            continue
        if col not in IGNORED_COLS:
            if 'glassnode' in col:
                windows = glassnode_windows
            else:
                windows = sma_windows
            turn_col_into_sma(data, col, delete_col=True,
                              sma_windows=windows)
    cols_to_normalize = data.columns.difference(
        ['friday_long'])

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    df_normalized = pd.DataFrame(scaler.transform(
        data[cols_to_normalize]), columns=cols_to_normalize)

    df_normalized.index = data.index
    df_normalized['friday_long'] = data['friday_long']
    df_normalized = arrange_columns(
        df_normalized, "column_order_for_live_env.csv")
    last_row = df_normalized.tail(1)
    inputs = torch.Tensor(last_row.values.astype(np.float32))
    return inputs
