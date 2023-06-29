import os
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from dotenv import load_dotenv
from enum import Enum
import time
import pandas as pd
import threading
from datetime import datetime, timedelta
from server import launch_server
from fetch import fetch_data
from pre_process import pre_process
from data_structures import MarginAccountAsset, Trade, Fill
from model import initiate_model
from math_util import get_qty, get_trade_details, convert_to_quote, convert_to_decimal_places, convert_to_base
from enum import Enum

load_dotenv()
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')


class Direction(Enum):
    LONG = 1
    SHORT = 2
    NEUTRAL = 3


class TradeEngine(Client):
    def __init__(self, api_key, api_secret,
                 long_threshold, short_threshold,
                 kline_interval, required_symbols,
                 traded_pair, required_candles, sma_windows,
                 glassnode_windows, strategy_id, max_betsize, trade_asset, decimal_places, debug_mode_on=False):
        self.lock = threading.Lock()
        self.session_start_time = datetime.now()
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.trade_interval = kline_interval
        self.required_symbols = required_symbols
        self.traded_pair = traded_pair
        self.required_candles = required_candles
        self.sma_windows = sma_windows
        self.glassnode_windows = glassnode_windows
        self.model = None
        self.prediction = None
        self.strategy_id = strategy_id
        self.max_betsize = max_betsize
        self.max_betsize_in_quote = None
        self.portfolio = None
        self.trade_asset = trade_asset
        self.latest_log_message = None
        self.traded_asset_price = None
        self.position_direction = None
        self.decimal_places = decimal_places
        self.trades = []
        self.trade_invested_quote_amount = 0
        self.trade_asset_balance = None
        self.debug_mode_on = False
        self.init_strat_fetch_done = False
        self.prev_pred_inputs = []
        super().__init__(api_key, api_secret)

    def pay_margin_loan_with_free_balance(self, free_balance, borrowed_balance):
        try:
            if free_balance > 0 and borrowed_balance > 0:
                if free_balance >= borrowed_balance and convert_to_decimal_places(borrowed_balance, self.decimal_places) != 0.0:
                    self.repay_margin_loan(asset=self.trade_asset, amount=convert_to_decimal_places(
                        borrowed_balance, self.decimal_places))
                elif convert_to_decimal_places(free_balance, self.decimal_places) != 0.0:
                    self.repay_margin_loan(asset=self.trade_asset, amount=convert_to_decimal_places(
                        free_balance, self.decimal_places))
        except Exception as e:
            print(f"Error paying margin loan: {str(e)}")
            self.latest_log_message = f"Error paying margin loan: {str(e)}"

    def get_trade_engine_state(self):
        return {
            "prediction": self.prediction,
            "latest_log_message": self.latest_log_message,
            "traded_asset_price": self.traded_asset_price,
            "position_direction": str(self.position_direction),
            "usdt_value": self.total_net_asset_of_usdt,
            "trade_asset_balance": self.trade_asset_balance,
            "prev_pred_inputs": self.prev_pred_inputs.tolist(),
            "trades": [trade.__dict__ for trade in self.trades],
            "decimal_places": self.decimal_places,
            "short_threshold": self.short_threshold,
            "long_threshold": self.long_threshold,
            "debug_mode_on": self.debug_mode_on,
        }

    def fetch_strategy_state(self):
        self.fetch_price()
        self.update_portfolio()

        trade_asset_balance = self.get_margin_account_asset(self.trade_asset)
        self.trade_asset_balance = trade_asset_balance
        net_trade_asset_balance = float(trade_asset_balance.netAsset)

        if convert_to_quote(net_trade_asset_balance, self.traded_asset_price) > 15.0:
            self.position_direction = Direction.LONG

        elif convert_to_quote(net_trade_asset_balance, self.traded_asset_price) < -15.0:
            self.position_direction = Direction.SHORT

        else:
            if self.position_direction != Direction.NEUTRAL:
                self.trade_invested_quote_amount = 0
            self.position_direction = Direction.NEUTRAL

        self.pay_margin_loan_with_free_balance(free_balance=float(
            trade_asset_balance.free), borrowed_balance=float(trade_asset_balance.borrowed))
        if self.init_strat_fetch_done == False:
            self.init_strat_fetch_done = True
            self.fetch_strategy_state()

    def get_margin_account_asset(self, asset):
        margin_assets = self.portfolio['userAssets']
        for margin_asset in margin_assets:
            if margin_asset['asset'] == asset:
                return MarginAccountAsset(**margin_asset)
        return None

    def update_portfolio(self):
        self.portfolio = self.get_margin_account()
        self.total_net_asset_of_btc = float(
            self.portfolio['totalNetAssetOfBtc'])
        self.total_net_asset_of_usdt = self.total_net_asset_of_btc * self.btc_price
        self.max_betsize_in_quote = self.total_net_asset_of_usdt * self.max_betsize

    def fetch_price(self):
        self.traded_asset_price = float(
            self.get_symbol_ticker(symbol=self.traded_pair)['price'])
        self.btc_price = float(
            self.get_symbol_ticker(symbol='BTCUSDT')['price'])

    def execute_market_order(self, symbol, side, qty, quote_amount):
        try:
            if float(quote_amount) < 15:
                return "order size too small"

            trade_req_res = self.create_margin_order(
                symbol=symbol, side=side, type="MARKET", quantity=qty)

            trade = Trade(trade_req_res)
            self.trades.append(trade)
            if side == "SELL":
                self.update_strategy_position_size(
                    -float(trade.cummulativeQuoteQty))
            else:
                self.update_strategy_position_size(
                    float(trade.cummulativeQuoteQty))

            if len(self.trades) > 10:
                self.trades.pop(0)

            self.latest_log_message = f"Executed {side} market order of {qty} {symbol} at {self.traded_asset_price} {self.trade_asset} per {self.traded_asset_price} {self.trade_asset}"
            self.fetch_strategy_state()

            return trade
        except Exception as e:
            print(f"Error executing market order: {str(e)}")
            self.latest_log_message = f"Error executing market order: {str(e)}"
            return None

    def send_log_message_to_db(self, message):
        pass

    def update_strategy_position_size(self, delta):
        pass

    def update_position_direction(self, new_direction):
        pass

    def close_long_position(self):
        trade_asset_balance = self.get_margin_account_asset(self.trade_asset)
        free_trade_asset_balance = float(trade_asset_balance.free)

        if float(trade_asset_balance.free) > 0:
            self.execute_market_order(symbol=self.traded_pair, side="SELL",
                                      qty=convert_to_decimal_places(
                                          free_trade_asset_balance, self.decimal_places),
                                      quote_amount=convert_to_quote(free_trade_asset_balance, self.traded_asset_price))

    def initiate_long_position(self):
        usdt_balance = self.get_margin_account_asset('USDT')
        free_usdt = float(usdt_balance.free)

        if free_usdt > 0:
            available_usdt = min(free_usdt, self.max_betsize_in_quote)

            remaining_bet_size_in_quote = available_usdt - self.trade_invested_quote_amount
            remaining_bet_size_in_base = convert_to_base(
                remaining_bet_size_in_quote, self.traded_asset_price, self.decimal_places)

            trade = self.execute_market_order(
                symbol=self.traded_pair, side="BUY", qty=remaining_bet_size_in_base, quote_amount=remaining_bet_size_in_quote)

            if trade is not None and trade != "order size too small":
                self.trade_invested_quote_amount += float(
                    trade.cummulativeQuoteQty)

    def initiate_short_position(self):

        max_betsize_in_base = self.max_betsize_in_quote / self.traded_asset_price
        remaining_betsize = max_betsize_in_base - \
            self.trade_invested_quote_amount * self.traded_asset_price
        remaining_betsize = convert_to_decimal_places(
            remaining_betsize, self.decimal_places)

        self.create_margin_loan(asset=self.trade_asset, amount=convert_to_decimal_places(
            remaining_betsize, self.decimal_places))

        trade = self.execute_market_order(symbol=self.traded_pair, side="SELL",
                                          qty=remaining_betsize, quote_amount=remaining_betsize * self.traded_asset_price)

        if trade is not None and trade != "order size too small":
            self.trade_invested_quote_amount += float(
                trade.cummulativeQuoteQty)
            print("short position initiated")

    def close_short_position(self):

        borrowed_amount_in_quote = float(self.get_margin_account_asset(
            self.trade_asset).borrowed) * self.traded_asset_price
        free_usdt = float(self.get_margin_account_asset('USDT').free)

        if borrowed_amount_in_quote > free_usdt:
            order_size = free_usdt / self.traded_asset_price
        else:
            order_size = float(self.get_margin_account_asset(
                self.trade_asset).borrowed)

        self.execute_market_order(symbol=self.traded_pair, side="BUY",
                                  qty=convert_to_decimal_places(
                                      order_size, self.decimal_places),
                                  quote_amount=order_size * self.traded_asset_price)

    def execute_long_logic(self, prediction):
        if prediction >= self.long_threshold and self.position_direction == Direction.NEUTRAL:
            self.initiate_long_position()

        if prediction < self.long_threshold and self.position_direction == Direction.LONG:
            self.close_long_position()

    def execute_short_logic(self, prediction):
        if prediction <= self.short_threshold and self.position_direction == Direction.NEUTRAL:
            self.initiate_short_position()

        if prediction > self.short_threshold and self.position_direction == Direction.SHORT:
            self.close_short_position()

    def trade(self):
        self.fetch_strategy_state()

        data = fetch_data(self)
        inputs = pre_process(data, self.sma_windows, self.glassnode_windows)
        if self.model is None:
            self.model = initiate_model(inputs.shape[1])
        prediction = self.model(inputs).item() / 100
        self.prediction = prediction
        self.prev_pred_inputs = inputs

        if self.debug_mode_on == False:
            print("Executing trading logic...")
            self.execute_long_logic(prediction)
            self.execute_short_logic(prediction)
        print(f"Trade loop ran at {datetime.now()}")
        print(
            f"Trade direction: {self.position_direction}, prediction: {prediction}")


class TradingPair:
    def __init__(self, symbol, market_type, interval, cols_to_drop):
        self.symbol = symbol
        self.market_type = market_type
        self.interval = interval
        self.cols_to_drop = cols_to_drop
        self.prefix = symbol.lower() + "_" + market_type.lower() + \
            "_" + interval.lower() + "_"


def main() -> None:

    LONG_THRESHOLD = 1.0145
    SHORT_THRESHOLD = 0.9925
    DECIMAL_PLACES = 0
    TRADE_ASSET = "IOTA"
    TRADED_PAIR = "IOTAUSDT"
    STRATEGY_ID = "iota_bot"
    MAX_BETSIZE = 1
    REQUIRED_CANDLES = 1000
    DEBUG_MODE_ON = False

    trade_engine = TradeEngine(
        api_key=API_KEY, api_secret=API_SECRET, long_threshold=LONG_THRESHOLD,
        short_threshold=SHORT_THRESHOLD, kline_interval=Client.KLINE_INTERVAL_1HOUR,
        required_symbols=[TradingPair(TRADED_PAIR, "spot", "1h", cols_to_drop=[]),
                          TradingPair("BTCUSDT", "spot", "1h", cols_to_drop=["open_price", "high_price",
                                                                             "low_price", "close_price", "volume",
                                                                             "quote_asset_volume", "number_of_trades", "ignore"]),
                          ], traded_pair=TRADED_PAIR, required_candles=REQUIRED_CANDLES,
        sma_windows=[24, 100, 1000],
        glassnode_windows=[], strategy_id=STRATEGY_ID, max_betsize=MAX_BETSIZE, trade_asset=TRADE_ASSET, decimal_places=DECIMAL_PLACES, debug_mode_on=DEBUG_MODE_ON)

    server_thread = threading.Thread(
        target=launch_server, args=(trade_engine,))

    if DEBUG_MODE_ON == False:
        server_thread.start()

    while True:
        trade_engine.trade()
        time.sleep(5)


if __name__ == "__main__":
    main()
