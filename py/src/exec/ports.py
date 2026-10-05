"""The two broker-agnostic seams shared by backtest and live: ``Broker``/``Exchange``.

``Broker`` is the order edge (submit/cancel/fills/close). ``Exchange`` is the
matching decision: given an order and the bars it may trade against, either a
fill or ``None``. Both are structural Protocols so a candle simulator and a real
broker adapter satisfy the same contracts without inheritance.

``Exchange`` is satisfied by the pure matcher (``src.exec.matching.match_bar``)
and is deliberately frictionless: pricing friction/costs is the adapter's job,
so ``SimExchange.match_bar`` takes extra keyword-only friction args and is NOT
structurally an ``Exchange`` (see the conformance test in
``src/bt/exchange/tests``). These two Protocols stay broker-agnostic; the
backtest engine's bt-shaped fill surface lives in ``src/bt/exchange/ports.py``.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from src.exec.types import Fill, OrderAck, OrderRequest


class Exchange(Protocol):
    """Decides whether (and at what price) an order fills against candles.

    The returned ``Fill.price`` is the frictionless base price; the caller
    (adapter) applies spread/slippage/commission on top. Satisfied by the pure
    matcher; ``SimExchange.match_bar`` adds required friction args and so is a
    different, adapter-shaped contract.
    """

    def match_bar(self, order: OrderRequest, bars: pd.DataFrame) -> Fill | None: ...


class Broker(Protocol):
    """The order edge: submit, cancel, read fills, close the session.

    Fails as a value (``OrderAck.accepted``), never by raising out of a routing
    decision. ``close`` tears the session down; it does not flatten positions.
    """

    def submit(self, order: OrderRequest) -> OrderAck: ...

    def cancel(self, order_ref: str) -> OrderAck: ...

    def fills(self) -> tuple[Fill, ...]: ...

    def close(self) -> None: ...
