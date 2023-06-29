import os
import math
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

client = Client(API_KEY, API_SECRET)


def get_margin_account_asset(client, asset):
    margin_assets = client.get_margin_account()['userAssets']
    for margin_asset in margin_assets:
        if margin_asset['asset'] == asset:
            return margin_asset
    return None


def convert_to_decimal_places(number, decimal_places):
    return math.floor(float(number) * 10 ** decimal_places) / 10 ** decimal_places


def get_balances(positions):
    net_asset_balance = float(positions['netAsset'])
    borrowed_balance = float(positions['borrowed'])
    free_balance = float(positions['free'])

    return {"net": net_asset_balance, "borrowed": borrowed_balance, "free": free_balance}


def close_all_positions(asset, decimal_places=1):

    asset_balance = get_margin_account_asset(client, asset)

    balances = get_balances(asset_balance)

    if balances['net'] > 0:

        if balances['borrowed'] > 0:
            try:
                client.repay_margin_loan(asset=asset,
                                         amount=convert_to_decimal_places(balances['borrowed'], decimal_places))
                print(
                    f"Repaying {balances['borrowed']} {asset} borrowed balance")
            except Exception as e:
                pass
        asset_balance = get_margin_account_asset(client, asset)
        balances = get_balances(asset_balance)

        client.create_margin_order(
            symbol=f"{asset}USDT",
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=convert_to_decimal_places(
                balances['free'], decimal_places)
        )
        print(f"Selling {balances['free']} {asset} free balance")

    elif balances['net'] < 0:

        if balances['free'] > 0:
            try:
                client.repay_margin_loan(asset=asset,
                                         amount=convert_to_decimal_places(balances['free'], decimal_places))
                print(f"Repaying {balances['free']} {asset} free balance")
                asset_balance = get_margin_account_asset(client, asset)
                balances = get_balances(asset_balance)
            except Exception as e:
                pass

        try:
            client.create_margin_order(
                symbol=f"{asset}USDT",
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=convert_to_decimal_places(
                    balances['borrowed'] * 1.01, decimal_places)
            )
            asset_balance = get_margin_account_asset(client, asset)
            balances = get_balances(asset_balance)

            client.repay_margin_loan(asset=asset,
                                     amount=convert_to_decimal_places(balances['free'], decimal_places))

            print(
                f"Buying and closing {balances['borrowed']} {asset} borrowed balance")

        except Exception as e:
            print(e)
        pass


if __name__ == "__main__":
    asset = "IOTA"
    decimal_places = 0
    close_all_positions(asset, decimal_places)
