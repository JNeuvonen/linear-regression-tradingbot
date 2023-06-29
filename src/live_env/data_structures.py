from dataclasses import dataclass
from typing import List

@dataclass
class MarginAccountAsset:
    asset: str
    free: float
    locked: float
    borrowed: float
    interest: float
    netAsset: float


@dataclass
class Fill:
    price: str
    qty: str
    commission: str
    commissionAsset: str


@dataclass
class FilledOrder:
    symbol: str
    orderId: int
    clientOrderId: str
    transactTime: int
    price: str
    origQty: str
    executedQty: str
    cummulativeQuoteQty: str
    status: str
    timeInForce: str
    type: str
    side: str
    fills: List[Fill]
    isIsolated: bool


class Trade(FilledOrder):
    def __init__(self, trade_req_res):
        super().__init__(**trade_req_res)
