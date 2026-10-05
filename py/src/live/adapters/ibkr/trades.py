"""Execution replay — lots come from trade history, not from a local ledger.

``GET /iserver/account/trades`` returns one record per execution: the ``order_ref``
we set as the client order id (``cOID``), the broker's ``order_id``, ``size``,
``price``, ``commission`` and ``trade_time_r``. A *lot* is therefore something we
can derive, one ``order_id`` at a time, instead of something a ledger has to
guess and drift-check.

Rules (plan §3):

- **Ours** = ``order_ref.startswith(ref_prefix)``. A foreign order is excluded
  from the strategy book and *reported* — never traded, never silently absorbed.
- **One lot per ``order_id``**: qty is the net of its executions, ``entry_price``
  is the VWAP of the **opening** executions, open/closed follows the net sign.
- Commission is summed per order (broker-reported, exact — plan §7.3).

Pure: no I/O, no clock. Order of input executions does not matter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.exec.types import OrderSide

LotStatus = Literal["open", "closed"]


@dataclass(frozen=True)
class Execution:
    """One broker execution: a single fill of a single order."""

    execution_id: str
    order_id: str
    order_ref: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    commission: float
    ts: pd.Timestamp


@dataclass(frozen=True)
class ReplayedLot:
    """One open/closed lot derived from one ``order_id``'s executions.

    ``position_id`` IS the broker's ``order_id`` — the canonical handle a close
    targets later. ``qty`` is absolute; the side lives on ``side``.
    """

    order_id: str
    order_ref: str
    symbol: str
    side: OrderSide
    qty: float
    entry_price: float
    commission: float
    status: LotStatus
    ts: pd.Timestamp

    @property
    def position_id(self) -> str:
        """The broker lot handle (the order id) a close intent targets."""
        return self.order_id


@dataclass(frozen=True)
class BrokerSnapshot:
    """The replayed strategy book: our lots, the foreign orders, and warnings."""

    lots: tuple[ReplayedLot, ...]
    foreign: tuple[Execution, ...]
    warnings: tuple[str, ...]

    def open_lots(self) -> tuple[ReplayedLot, ...]:
        """Only the lots still open (the tradable book)."""
        return tuple(lot for lot in self.lots if lot.status == "open")


def _signed(execution: Execution) -> float:
    """Signed quantity: a BUY adds, a SELL subtracts."""
    return execution.qty if execution.side is OrderSide.BUY else -execution.qty


def _vwap(executions: Iterable[Execution]) -> float:
    """Volume-weighted average price; ``0.0`` when there is no volume."""
    total_qty = sum(e.qty for e in executions)
    if total_qty <= 0:
        return 0.0
    return sum(e.price * e.qty for e in executions) / total_qty


def _is_ours(execution: Execution, ref_prefix: str) -> bool:
    """Ownership: the client order ref we minted starts with our strategy prefix."""
    return execution.order_ref.startswith(ref_prefix)


def replay(executions: tuple[Execution, ...], *, ref_prefix: str) -> BrokerSnapshot:
    """Replay *executions* into the strategy book (pure).

    Groups ours by ``order_id``; a group's net sign decides open/closed and the
    opening side. ``entry_price`` is the VWAP of the executions on that opening
    side (the ones that built the lot, not the ones that unwound it). Foreign
    orders are returned in ``foreign`` untouched. An empty ``ref_prefix`` would
    claim every order, so it is rejected.
    """
    if not ref_prefix:
        raise ValueError("ref_prefix must be non-empty (ownership scope)")
    ours: dict[str, list[Execution]] = {}
    foreign: list[Execution] = []
    for execution in executions:
        if _is_ours(execution, ref_prefix):
            ours.setdefault(execution.order_id, []).append(execution)
        else:
            foreign.append(execution)

    lots: list[ReplayedLot] = []
    warnings: list[str] = []
    for order_id, group in ours.items():
        ordered = sorted(group, key=lambda e: e.ts)
        net = sum(_signed(e) for e in ordered)
        opening_side = (
            OrderSide.BUY if net > 0 else OrderSide.SELL if net < 0 else ordered[0].side
        )
        opening = tuple(e for e in ordered if e.side is opening_side)
        qty = abs(net)
        if qty == 0:
            warnings.append(
                f"order {order_id} nets to zero from {len(ordered)} execution(s)"
            )
        lots.append(
            ReplayedLot(
                order_id=order_id,
                order_ref=ordered[0].order_ref,
                symbol=ordered[0].symbol,
                side=opening_side,
                qty=qty,
                entry_price=_vwap(opening),
                commission=sum(e.commission for e in ordered),
                status="open" if net != 0 else "closed",
                ts=ordered[-1].ts,
            )
        )
    return BrokerSnapshot(
        lots=tuple(lots),
        foreign=tuple(foreign),
        warnings=tuple(warnings),
    )
