import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse

START_BALANCE = 1000


def plot_strategies(strategies, visualize_mode):
    plt.figure(figsize=(12, 6))

    for strat in strategies:
        if strat.strat_type == "buy_and_hold":
            label = "Buy and Hold Strategy"
            balance_history = strat.balance_history
        elif strat.strat_type == "long":
            label = "Long Strategy"
            balance_history = strat.balance_history
        else:
            continue

        if strat.strat_type == "long":
            try:
                profit_factor = strat.cumulative_profits / strat.cumulative_losses
            except ZeroDivisionError:
                profit_factor = 0
            trade_count = max(strat.trades_closed_count, 1)
            if visualize_mode:
                plt.plot(strat.benchmark_history, label=str(strat.long_threshold) + " " + str(strat.short_threshold) + "\n" +
                         f"Max drawdown: {(strat.max_drawdown - 1) * 100}% +\n" + f"Win rate: {strat.win_rate / trade_count * 100}%\n" + f"Profit factor: {profit_factor}")
                plt.scatter(strat.long_trades_entry_times, strat.long_trades,
                            color='lime', marker='o', label='Long Trades', s=5, zorder=3)
            else:
                plt.plot(strat.balance_history, label=str(strat.long_threshold) + " " + str(strat.short_threshold) + "\n" +
                         f"Max drawdown: {(strat.max_drawdown - 1) * 100}% +\n" + f"Win rate: {strat.win_rate / trade_count * 100}%\n" + f"Profit factor: {profit_factor}")

        else:
            if not visualize_mode:
                plt.plot(balance_history, label=label)

    plt.xlabel("Time")
    plt.ylabel("Balance (k)")
    plt.title("Comparison of Strategies")
    plt.legend()
    plt.grid()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, loc: "{:,}".format(int(x / 1000)) + 'k'))
    plt.show()

    def __str__(self):
        return f"Start price: {self.start_price}\nStart time: {self.start_time}\nEnd price: {self.end_price}\nEnd time: {self.end_time}\nResult: {self.result}"


class BuyAndHoldStrategy:
    def __init__(self):
        self.balance = START_BALANCE
        self.balance_history = []
        self.prev_price = -1
        self.strat_type = "buy_and_hold"

    def simulate_trade(self, price_history):

        for item in price_history:
            if self.prev_price == -1:
                self.prev_price = item
                continue
            self.balance *= item / self.prev_price
            self.balance_history.append(self.balance)
            self.prev_price = item


class LongStrategy:
    def __init__(self, long_threshold, short_threshold=0):
        self.balance = START_BALANCE
        self.balance_ath = self.balance
        self.debug_flag = False
        self.benchmark_balance = self.balance
        self.benchmark_balance_ath = self.balance
        self.long_trades = []
        self.long_trades_entry_times = []
        self.short_trades = []
        self.cumulative_profits = 0
        self.cumulative_losses = 0
        self.short_trades_entry_times = []
        self.balance_before_trade = 0
        self.close_trades = []
        self.close_trades_entry_times = []
        self.max_drawdown = 10000
        self.max_drawdown_printable = ""
        self.benchmark_drawdown = 10000
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.trade_result = 0
        self.start_position_price = 0
        self.trade_lock_counter = 0
        self.average_return_per_trade = 0
        self.position_type = "none"
        self.strat_type = "long"
        self.max_drawdown = 0
        self.margin_loan_hourly_rate = 1 - (0.00244 / 100)
        self.total_margin_loan_fees = 1
        self.win_rate = 0
        self.benchmark_history = []
        self.prev_price = -1
        self.balance_history = []
        self.position_held_count = 0
        self.trades_closed_count = 0
        self.trades_opened_count = 0

    def update_strategy_metrics(self, balance_before_trade, balance_after_close):
        if balance_before_trade < balance_after_close:
            self.cumulative_profits += balance_after_close - balance_before_trade

        if balance_before_trade > balance_after_close:
            self.cumulative_losses += balance_before_trade - balance_after_close

    def close_trade(self, result):
        TRADING_FEE = 0.999
        SLIPPAGE = 0.999
        if self.position_type == "long":
            balance_before_close = self.balance
            self.balance *= result
            self.balance *= TRADING_FEE * TRADING_FEE * SLIPPAGE * SLIPPAGE

            if balance_before_close < self.balance:
                self.win_rate += 1

        elif self.position_type == "short":
            balance_before_close = self.balance
            self.balance *= result
            self.balance *= TRADING_FEE * TRADING_FEE * \
                SLIPPAGE * SLIPPAGE * self.total_margin_loan_fees

            if balance_before_close < self.balance:
                self.win_rate += 1

        self.update_strategy_metrics(self.balance_before_trade, self.balance)
        self.position_type = "none"
        self.trades_closed_count += 1
        self.trade_lock_counter = 0

    def initiate_pos_long_side(self, pred_value, label_value, price):
        if pred_value > self.long_threshold and self.position_type == "none":
            self.balance_before_trade = self.balance
            self.trade_lock_counter = 25
            self.trade_result = label_value
            self.position_type = "long"
            self.start_position_price = price
            self.total_margin_loan_fees = 1
            self.long_trades.append(self.benchmark_balance)
            self.long_trades_entry_times.append(len(self.benchmark_history))
            self.trades_opened_count += 1

    def initiate_pos_short_side(self, pred_value, label_value, price):
        if pred_value < self.short_threshold and self.position_type == "none":
            self.balance_before_trade = self.balance
            self.trade_lock_counter = 25
            self.trade_result = label_value
            self.position_type = "short"
            self.start_position_price = price
            self.total_margin_loan_fees = 1
            self.short_trades.append(self.benchmark_balance)
            self.short_trades_entry_times.append(len(self.benchmark_history))
            self.trades_opened_count += 1

    def check_early_cancel_trade(self, pred_value, price):
        if self.position_type == "long" and pred_value < self.long_threshold:
            self.close_trade(price / self.start_position_price)

        if self.position_type == "short" and pred_value > self.short_threshold:
            move = self.start_position_price - price

            if move == 0:
                self.close_trade(1)
            elif move > 0:
                self.close_trade(1 + move / self.start_position_price)
            else:
                self.close_trade(1 - abs(move) / self.start_position_price)

    def update_trade_counter(self):
        if self.position_type != "none":
            self.position_held_count += 1
        if self.trade_lock_counter > 0:
            self.trade_lock_counter -= 1

    def update_stategy_max_drawdown(self):
        if self.balance < self.balance_ath:

            self.max_drawdown = min(
                self.balance / self.balance_ath, self.max_drawdown)

            if self.max_drawdown == 0:
                self.max_drawdown = 1

    def update_benchmark_max_drawdown(self):
        if self.benchmark_balance < self.balance_ath:
            self.benchmark_drawdown = min(
                self.benchmark_balance / self.benchmark_balance_ath, self.benchmark_drawdown)

    def update_utils(self, price):

        self.balance_history.append(self.balance)

        if self.balance > self.balance_ath:
            self.balance_ath = self.balance

        self.update_stategy_max_drawdown()
        if self.prev_price == -1:
            self.benchmark_balance *= 1
        else:
            self.benchmark_balance *= price / self.prev_price
        self.prev_price = price
        self.benchmark_history.append(self.benchmark_balance)
        self.update_benchmark_max_drawdown()

        if self.position_type == "short":
            self.total_margin_loan_fees *= self.margin_loan_hourly_rate

    def simulate_trade(self, preds, labels, prices):
        for pred, label, price in zip(preds, labels, prices):
            pred_value = pred.item()
            label_value = label.item()
            price = price.item()

            self.check_early_cancel_trade(pred_value, price)
            self.initiate_pos_long_side(pred_value, label_value, price)
            self.initiate_pos_short_side(pred_value, label_value, price)
            self.update_trade_counter()
            self.update_utils(price)


class LinearModel(nn.Module):
    def __init__(self, input_params):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_params, 1)

    def forward(self, x):
        return self.linear(x)


def load_test_data(data_path):
    data = pd.read_csv(data_path, index_col=0)
    data = data.dropna(how='any')
    target = data.pop('return')
    price_data = data.pop('open_price')
    X_test = torch.Tensor(data.values.astype(np.float32))
    y_test = torch.Tensor(target.values.reshape(-1, 1).astype(np.float64))
    price_data = torch.Tensor(price_data.values.astype(
        np.float32))  # Convert price_data to a tensor
    return X_test, y_test, price_data


def evaluate_model(model, test_loader, device, strategy_arr):
    model.eval()
    with torch.no_grad():
        for inputs, labels, prices in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            labels_fmt = (
                labels / 100).squeeze().cpu().detach().numpy().flatten()
            outputs_fmt = (
                outputs / 100).squeeze().cpu().detach().numpy().flatten()

            prices_fmt = prices.squeeze().cpu().detach().numpy().flatten()

            for strategy in strategy_arr:
                if strategy.strat_type == "long":
                    strategy.simulate_trade(
                        outputs_fmt, labels_fmt, prices_fmt)
                else:
                    strategy.simulate_trade(prices_fmt)

    strat = None
    buy_and_hold = None

    for strategy in strategy_arr:
        if strategy.strat_type == "long":
            strat = strategy
        else:
            buy_and_hold = strategy

    return strat.balance / buy_and_hold.balance


def test_model(visualize_trades=False, model_epoch=100,
               use_model_path=False, passed_model_path="",
               long_threshold=1.01, short_threshold=0.99):

    if use_model_path:
        model_path = passed_model_path
        TEST_DATA_PATH = "backtest_valid.csv"
    else:
        model_path = f"all_models/epoch_{model_epoch}.pt"
        TEST_DATA_PATH = "backtest_valid.csv"
    X_test, y_test, price_history = load_test_data(TEST_DATA_PATH)

    dataset = TensorDataset(X_test, y_test, price_history)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("cpu")  # Force CPU usage

    model = LinearModel(X_test.shape[1])
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    model.to(device)

    LONG_THRESHOLD = long_threshold
    SHORT_THRESHOLD = short_threshold

    strat_arr = [LongStrategy(long_threshold=LONG_THRESHOLD, short_threshold=SHORT_THRESHOLD),
                 BuyAndHoldStrategy()]

    result_vs_buy_and_hold = evaluate_model(
        model, test_loader, device, strat_arr)

    if use_model_path:
        return result_vs_buy_and_hold

    for strat in strat_arr:
        if strat.strat_type == "long":
            print(strat.position_held_count / len(strat.balance_history))
            print(strat.trades_opened_count, strat.trades_closed_count)

    plot_strategies(strat_arr, visualize_trades)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('epoch', type=int, help='Model epoch number')
    parser.add_argument('--long', type=float, default=1.0145,
                        help='Threshold for long strategy')
    parser.add_argument('--short', type=float, default=0.9925,
                        help='Threshold for short strategy')
    args = parser.parse_args()

    print(
        f"Epoch {args.epoch} with long threshold {args.long} and short threshold {args.short}")

    test_model(False, model_epoch=args.epoch,
               long_threshold=args.long, short_threshold=args.short)
