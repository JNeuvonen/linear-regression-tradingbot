
import math




def convert_to_decimal_places(number, decimal_places):
    return math.floor(float(number) * 10 ** decimal_places) / 10 ** decimal_places


def get_qty(bet_size, price, decimal_places):
    return convert_to_decimal_places((float(bet_size) - float(bet_size) * 0.001) / price, decimal_places)


def convert_to_quote(qty, price):
    return qty * price

def convert_to_base(qty, price, decimal_places):
    return convert_to_decimal_places(qty / price, decimal_places)


def get_trade_details(trade):
    total_price = 0.0
    total_qty = 0.0

    for fill in trade['fills']:
        total_price += float(fill['price']) * float(fill['qty'])
        total_qty += float(fill['qty'])

    if total_qty == 0:
        return 0
    else:
        return total_price / total_qty, trade.cummulativeQuoteQty, trade.executedQty
