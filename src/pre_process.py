import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from functools import reduce
import pickle


def turn_col_into_sma(df, col, delete_col, sma_windows):

    for sma in sma_windows:
        sma = str(sma)
        df[col + "_" + sma] = df[col].rolling(window=int(sma)).mean()
        df[col + "_diff_to_" + sma] = df[col] / df[col + "_" + sma] - 1

    if delete_col:
        df.drop(col, axis=1, inplace=True)
        for sma in sma_windows:
            sma = str(sma)
            df.drop(col + "_" + sma, axis=1, inplace=True)


def parse_prefix_from_path(path):

    parts = path.split('/')

    filename = parts[-1]

    filename_parts = filename.split('_')

    desired_string = "_".join(filename_parts[:3])

    return desired_string.lower() + "_"


def add_glassnode_cols(data, glassnode_data_paths):

    for glassnode_data_path in glassnode_data_paths:
        if glassnode_data_path == "":
            continue
        df = pd.read_csv(glassnode_data_path, header=0)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['glassnode_' +
            glassnode_data_path.split("/")[-1]] = df['value'].astype(float)
        df['kline_open_time'] = df['timestamp'].apply(
            lambda x: int(x.timestamp() * 1000))

        data.sort_values(by=['kline_open_time'], inplace=True)
        df.sort_values(by=['kline_open_time'], inplace=True)
        df.drop(['timestamp', 'value'], axis=1, inplace=True)

        data = pd.merge_asof(
            data, df, on='kline_open_time', direction='backward')

        added_val = data['glassnode_' + glassnode_data_path.split("/")[-1]]
        added_val.to_csv("glassnode_inspect.csv", index=False)

    return data



def main(data_paths, target_pair, glassnode_data_paths=[]):

    datasets = []

    for data_path in data_paths:
        data = pd.read_csv(data_path, header=0)
        prefix = parse_prefix_from_path(data_path)

        if target_pair not in data_path.lower():
            cols_to_drop = [prefix + "return",
                            prefix + "open_price", "friday_long"]
            cols_to_drop = [col for col in cols_to_drop if col in data.columns]
            data.drop(cols_to_drop, axis=1, inplace=True)
        data.drop([prefix + "kline_close_time"], axis=1, inplace=True)
        datasets.append(data)

    data = reduce(lambda left, right: pd.merge(
        left, right, on='kline_open_time', how='outer'), datasets)

    data = add_glassnode_cols(data, glassnode_data_paths)

    data.drop(['kline_open_time'],
              axis=1, inplace=True)

    IGNORED_COLS = {
        'friday_long', 'return'}

    DATA_COLS = data.columns
    data = data.dropna()
    open_price = data[target_pair + 'open_price']
    data['friday_long'] = data['friday_long'].astype(int)

    sma_windows = [24, 100, 1000]
    glassnode_windows = [3, 7, 30]

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
        ['friday_long', target_pair + 'return'])

    test_size = 0.05
    test_start_index = int(len(data) * (1 - test_size))

    data_train = data.iloc[:test_start_index]
    data_valid = data.iloc[test_start_index:]

    scaler = MinMaxScaler()
    data_train_normalized = pd.DataFrame(scaler.fit_transform(
        data_train[cols_to_normalize]), columns=cols_to_normalize, index=data_train.index)
    data_valid_normalized = pd.DataFrame(scaler.transform(
        data_valid[cols_to_normalize]), columns=cols_to_normalize, index=data_valid.index)

    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    for df in [data_train_normalized, data_valid_normalized]:
        df['return'] = data[target_pair + 'return'] * 100
        df['friday_long'] = data['friday_long']
        df['open_price'] = open_price
        df['open_price'] = df['open_price'].shift(1)

    column_names = data_train_normalized.columns.tolist()  # Convert column names to a list
    pd.DataFrame({'Column Names': column_names}).to_csv(
        'column_names.csv', index=False)

    data_train_normalized.to_csv(
        "regression_model.csv", index=False)

    df_backtest_valid = data_valid_normalized.tail(720 * 4)
    df_backtest_valid.to_csv(
        "backtest_valid.csv", index=False)
    data_valid_normalized.to_csv(
        "regression_model_valid.csv", index=False)
    print(f"DF of a shape {data_train_normalized.shape} processed")



def parse_arguments():

    path_arg = sys.argv[1]
    target_pair_arg = sys.argv[2]
    strat_name_arg = sys.argv[3]
    glassnode_paths_arg = sys.argv[4]

    path_parts = path_arg.split(" ")
    paths = []

    for path in path_parts:
        paths.append(path)

    glassnode_parts = glassnode_paths_arg.split(" ")
    glassnode_paths = []

    for path in glassnode_parts:
        glassnode_paths.append(path)

    return paths, target_pair_arg.lower(), strat_name_arg, glassnode_paths


if __name__ == '__main__':
    paths, target_param, strat_name, glassnode_paths = parse_arguments()

    main(paths, target_param, glassnode_data_paths=glassnode_paths)
