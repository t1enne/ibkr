"""Shared order domain — pure types and rules used by backtest and live.

This package holds the ONE order vocabulary both sides speak: order types and
time-in-force, the ``OrderRequest``/``Fill`` records, the order-ref scheme, the
candle matcher, and the friction helpers. It is pure (no I/O, no engine state),
so ``SimExchange`` and a real broker adapter can share it without either owning
the other.
"""

from src.exec.friction import (
    apply_friction,
    commission_for_fill,
)
from src.exec.matching import match_bar, match_price
from src.exec.ports import Broker, Exchange
from src.exec.refs import order_ref
from src.exec.types import (
    CommissionModel,
    Fill,
    FixedCommission,
    FrictionResult,
    OrderAck,
    OrderRequest,
    OrderSide,
    OrderState,
    OrderType,
    PerShareCommission,
    RejectReason,
    TimeInForce,
)

__all__ = [
    "Broker",
    "CommissionModel",
    "Exchange",
    "Fill",
    "FixedCommission",
    "FrictionResult",
    "OrderAck",
    "OrderRequest",
    "OrderSide",
    "OrderState",
    "OrderType",
    "PerShareCommission",
    "RejectReason",
    "TimeInForce",
    "apply_friction",
    "commission_for_fill",
    "match_bar",
    "match_price",
    "order_ref",
]
