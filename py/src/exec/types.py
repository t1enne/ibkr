"""Shared order vocabulary: order records and their enums.

These types are the contract between ``SimExchange`` and a real broker adapter.
They are deliberately broker-agnostic: an ``OrderRequest`` is what a strategy
*wants*, an ``OrderAck`` is what the broker *accepted*, and a ``Fill`` is what
actually traded. Failure is a value (``OrderAck.accepted``/``RejectReason``),
never an exception thrown out of a matching decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class OrderType(Enum):
    """How an order is priced: at the market, or at a limit."""

    MKT = "MKT"
    LMT = "LMT"


class TimeInForce(Enum):
    """How long an unfilled order rests. ``DAY`` drops it at its bar."""

    DAY = "DAY"
    IOC = "IOC"


class OrderSide(Enum):
    """The direction of an order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderState(Enum):
    """Lifecycle state of an order as the broker sees it."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    UNFILLED = "UNFILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class RejectReason(Enum):
    """Why an order was not accepted or did not fill."""

    NONE = "NONE"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    UNKNOWN_ORDER = "UNKNOWN_ORDER"
    INVALID = "INVALID"


@dataclass(frozen=True)
class OrderRequest:
    """A single order: what to trade, how to price it, and its stable identity.

    ``order_ref`` is minted by ``refs.order_ref`` and is the dedupe key a broker
    uses across re-sends. ``limit_price`` is required for ``LMT`` and unused for
    ``MKT``.
    """

    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    order_ref: str
    limit_price: Optional[float] = None
    tif: TimeInForce = TimeInForce.DAY
    tag: str = ""


@dataclass(frozen=True)
class OrderAck:
    """The broker's response to a submit/cancel: accepted or not, and why not."""

    order_ref: str
    accepted: bool
    state: OrderState
    reason: RejectReason = RejectReason.NONE
    message: str = ""


@dataclass(frozen=True)
class Fill:
    """An executed quantity of an order at a single price.

    ``commission``/``spread``/``slippage`` are qty-scaled dollar costs when the
    adapter models them, else zero. ``timestamp`` is optional because a pure
    candle matcher may not carry a clock.
    """

    order_ref: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0
    timestamp: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class FixedCommission:
    """Flat $ charge per fill; independent of qty and price."""

    amount: float


@dataclass(frozen=True)
class PerShareCommission:
    """IBKR-style per-share charge with a per-fill floor and optional cap.

    ``max_pct_of_value`` is a percent (not fraction) of the fill's traded
    value; ``None`` disables the cap.
    """

    per_share: float
    min_per_fill: float = 0.0
    max_pct_of_value: float | None = None


CommissionModel = FixedCommission | PerShareCommission


@dataclass(frozen=True)
class FrictionResult:
    """Executed price plus the qty-scaled dollar cost of each friction."""

    executed_price: float
    spread_cost: float  # qty-scaled $ (never a per-share fraction)
    slippage_cost: float  # qty-scaled $
