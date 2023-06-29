import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import shutil
from backtest import test_model


class TransformerModel(nn.Module):
    def __init__(self, input_params, nhead, num_layers, d_model, dim_feedforward):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_params, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x.unsqueeze(1))
        x = self.linear(x.squeeze(1))
        return x


class LinearModel(nn.Module):
    def __init__(self, input_params):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_params, 1)

    def forward(self, x):
        return self.linear(x)


def load_data(data_path):
    data = pd.read_csv(data_path, index_col=0, header=0)
    data.drop(["open_price"], axis=1, inplace=True)
    data = data.dropna(how='any')
    target = data.pop('return')
    pd.DataFrame({'Column Names': data.columns}).to_csv(
        'column_order_for_live_env.csv', index=False)
    X_train = torch.Tensor(data.values.astype(np.float32))
    y_train = torch.Tensor(target.values.reshape(-1, 1).astype(np.float64))
    return X_train, y_train


def get_long_short_thresholds():

    ret = []
    long_threshold = 1.0025
    short_threshold = 0.9975

    for i in range(1, 30):
        long_threshold = 1.0025 + (i * 0.001)
        short_threshold = 0.9975 - (i * 0.001)
        thresholds = {
            "strategy": "long_short",
            "long_threshold": long_threshold,
            "short_threshold": short_threshold
        }
        ret.append(thresholds)

    for i in range(1, 30):
        long_threshold = 1.0025 + (i * 0.001)
        short_threshold = 0
        thresholds = {
            "strategy": "long_only",
            "long_threshold": long_threshold,
            "short_threshold": short_threshold
        }
        ret.append(thresholds)

    for i in range(1, 30):
        long_threshold = 100
        short_threshold = 0.9975 - (i * 0.001)
        thresholds = {
            "strategy": "short_only",
            "long_threshold": long_threshold,
            "short_threshold": short_threshold
        }
        ret.append(thresholds)

    return ret


def train_model(model, train_loader, criterion, optimizer, device, model_name, num_epochs=1000, save_best_models=True):
    long_short_strats_arr = []
    long_only_strats_arr = []
    short_only_strats_arr = []

    strategies = get_long_short_thresholds()

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            l1_lambda = 0.01  # Again, this value can be tuned to your problem.

            # inside your training loop:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = criterion(outputs, labels) + l1_lambda * l1_norm
            #loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

        model_path = model_name + "/epoch_" + str(epoch) + ".pt"
        torch.save(model.state_dict(), model_path)

        if save_best_models:
            for strategy in strategies:
                result_vs_buy_and_hold = test_model(visualize_trades=False, model_epoch=False,
                                                    use_model_path=True, passed_model_path=model_path, long_threshold=strategy["long_threshold"], short_threshold=strategy["short_threshold"])

                if strategy['strategy'] == "long_short":
                    long_short_strats_arr.append({
                        "epoch": epoch,
                        "long_threshold": strategy["long_threshold"],
                        "short_threshold": strategy["short_threshold"],
                        "result_vs_buy_and_hold": result_vs_buy_and_hold,
                        "model_weights": model.state_dict()
                    })

                if strategy['strategy'] == "long_only":
                    long_only_strats_arr.append({
                        "epoch": epoch,
                        "long_threshold": strategy["long_threshold"],
                        "short_threshold": strategy["short_threshold"],
                        "result_vs_buy_and_hold": result_vs_buy_and_hold,
                        "model_weights": model.state_dict()
                    })

                if strategy['strategy'] == "short_only":
                    short_only_strats_arr.append({
                        "epoch": epoch,
                        "long_threshold": strategy["long_threshold"],
                        "short_threshold": strategy["short_threshold"],
                        "result_vs_buy_and_hold": result_vs_buy_and_hold,
                        "model_weights": model.state_dict()
                    })

            directories = ["best_long_short_models",
                           "best_long_only_models", "best_short_only_models"]
            for directory in directories:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                if not os.path.exists(directory):
                    os.makedirs(directory)
            strategies_with_paths = [{"strategies": long_short_strats_arr, "directory": "best_long_short_models"},
                                     {"strategies": long_only_strats_arr,
                                      "directory": "best_long_only_models"},
                                     {"strategies": short_only_strats_arr, "directory": "best_short_only_models"}]

            for strategy in strategies_with_paths:
                weights_dicts_sorted = sorted(
                    strategy['strategies'], key=lambda k: k['result_vs_buy_and_hold'], reverse=True)
                best_ten_perc = len(weights_dicts_sorted) * 0.1
                index = 0
                for item in weights_dicts_sorted:
                    if index < best_ten_perc:
                        model_path = strategy['directory'] + \
                            f"/epoch_{item['epoch']}_long_{item['long_threshold']}_short_{item['short_threshold']}.pt"
                        torch.save(item['model_weights'], model_path)
                        index += 1


def main():
    DATA_PATH = "regression_model.csv"
    X_train, y_train = load_data(DATA_PATH)

    model_dir = "all_models"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LinearModel(X_train.shape[1])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    device = torch.device("cuda" if torch.cuda.is_available()  # pylint: disable=no-member
                          else "cpu")  # pylint: disable=no-member
    model.to(device)

    train_model(model, train_loader, criterion, optimizer,
                device, model_dir, num_epochs=15000, save_best_models=False)


if __name__ == '__main__':
    main()
