"""The pure candle matcher: decisions a bar can honestly make about an order.

This is the ONLY place a candle-based decision may differ from a real matching
engine, so it is pure and table-tested. ``match_price`` is the price core (and
what ``SimExchange`` reuses for its MKT base price); ``match_bar`` wraps it into
a ``Fill``.

Rules:
  - MKT: fills at the bar's open.
  - LMT buy:  fills only if ``low <= limit``, at ``min(limit, open)`` — a bar
    that gapped below the limit is not improved on.
  - LMT sell: fills only if ``high >= limit``, at ``max(limit, open)``.

A bar the rule does not touch yields ``None``: an unfilled order, not an
invented fill. ``DAY``/``IOC`` both resolve within the single bar given;
carry-over across bars is a later concern.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.exec.types import Fill, OrderRequest, OrderSide, OrderType


def match_price(
    order: OrderRequest,
    *,
    open_: float,
    high: float,
    low: float,
) -> Optional[float]:
    """Base fill price for one bar, or ``None`` when the bar does not fill it.

    MKT is unconditional (the bar's open is the fill). LMT requires the bar's
    low (buy) / high (sell) to reach ``limit_price``; a limit with no price set,
    or an unknown order type, never fills.
    """
    if order.order_type is OrderType.MKT:
        return open_
    if order.order_type is OrderType.LMT:
        limit = order.limit_price
        if limit is None:
            return None
        if order.side is OrderSide.BUY:
            return min(limit, open_) if low <= limit else None
        return max(limit, open_) if high >= limit else None
    return None


def match_bar(order: OrderRequest, bars: pd.DataFrame) -> Optional[Fill]:
    """Fill ``order`` against the first bar of ``bars``, or ``None`` if unfilled.

    Caller contract: ``bars`` is the FILL bar — the bar the order trades against
    (a one-row OHLC frame; extra rows are ignored). For a MKT order that means
    this bar's open, i.e. the same bar a next-open fill uses
    (``fill_at_next_open``). The returned ``Fill.price`` is the FRICTIONLESS
    base price; applying spread/slippage/commission is the adapter's job (see
    ``SimExchange.match_bar``). The ``Fill`` carries the order's own
    ref/symbol/side/qty and the matched price; costs are left zero here.
    """
    if bars is None or bars.empty:
        return None
    row = bars.iloc[0]
    price = match_price(order, open_=row["open"], high=row["high"], low=row["low"])
    if price is None:
        return None
    ts = bars.index[0]
    return Fill(
        order_ref=order.order_ref,
        symbol=order.symbol,
        side=order.side,
        qty=order.qty,
        price=float(price),
        timestamp=ts if isinstance(ts, pd.Timestamp) else None,
    )
