"""The sim BROKER's bracket side — the sim store IS the broker here.

The sim has no live tape between cycles, so it cannot represent the resting-stop
CLOCK behaviour (a stop fires only when a cycle observes a mark past the level —
the observed-price proxy; see below). What it CAN model, keeping parity with the
shared rules rather than forking them:

- the pair rests: both legs are written into the sim book's ``orders`` (the
  broker's working-orders surface) with the spec's field names, refs minted by
  ``src.live.protect.child_ref`` and attribution read back through the shared
  ``parse_working_order``/``resting_bracket``;
- the stop triggers on PRICE: a resting stop whose level the lot's mark has
  passed fills at the observed mark (a gap open is already worse than the level),
  booked through the SAME ``execute_signal`` pricing the entries use, and the OCA
  sibling is cancelled;
- REPLACE on tighten: submitting a new pair for a conid that already rests ours
  cancels BOTH old legs first, then stores the new pair — never one leg.

The entry fill -> pair placement happens in the adapter (``place_cohort``)
because the pair must be placed in the SAME cycle as the confirmed entry.
``resync`` fires the stops, so a breach surfaces at cycle start and the fired
fill is booked through the pipeline like any placement.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import cast

import pandas as pd

from src.bt.exchange import SimExchange
from src.bt.state import ActionType, ExecutionParams, FillEvent
from src.exec.types import OrderSide, OrderType
from src.live.adapters.ibkr.orders import parse_working_order
from src.live.adapters.sim.store import SimBook, synthetic_conid
from src.live.identity import OrderOutcome, WorkingOrder
from src.live.pure import OrderResult, trade_signal
from src.live.pure_plan import _cash_delta
from src.live.protect import child_ref, kind_of, resting_bracket
from src.live.types import OrderIntent

#: The flat-net epsilon for the breach check (mirrors the close guard's).
_FLAT_EPS = 1e-9


def wire_side(reduce_action: ActionType) -> str:
    """The wire BUY/SELL of a reduce leg: reducing a long sells, a short buys."""
    return "BUY" if reduce_action is ActionType.long else "SELL"


def _leg_level(child: OrderIntent) -> float:
    """The resting trigger/limit a protective child carries."""
    level = child.stop_loss if child.order_type is OrderType.STP else child.take_profit
    assert level is not None
    return level


def leg_row(
    child: OrderIntent,
    *,
    scope_tag: str,
    conid: int,
    anchor: pd.Timestamp,
    order_id: str,
    status: str = "Submitted",
) -> dict[str, object]:
    """The IBKR-shaped working row for ONE protective *child* (ours, resting).

    The row mirrors the captured bracket fixture (spec §0): ``order_ref`` the
    minted ref, ``orderType`` the kind, ``stop_price``/``price`` the level,
    ``status`` ``Submitted`` means resting, ``timeInForce`` ``GTC``. Field names
    are the spec's, never invented.
    """
    level = _leg_level(child)
    return {
        "acct": "SIM",
        "account": "SIM",
        "orderId": order_id,
        "conid": conid,
        "conidex": str(conid),
        "ticker": child.symbol,
        "side": wire_side(cast("ActionType", child.position_side)),
        "remainingQuantity": child.qty,
        "filledQuantity": 0.0,
        "totalSize": child.qty,
        "status": status,
        "order_ccp_status": status,
        "orderType": child.order_type.value,
        "origOrderType": "STOP" if child.order_type is OrderType.STP else "LIMIT",
        "timeInForce": "GTC",
        "order_ref": child_ref(scope_tag, child, conid=conid, anchor=anchor),
        "order_id": order_id,
        **(
            {"stop_price": f"{level:g}"}
            if child.order_type is OrderType.STP
            else {"price": f"{level:g}"}
        ),
    }


def resting_rows(
    book: SimBook, *, scope_tag: str, conid: int
) -> tuple[WorkingOrder, WorkingOrder] | None:
    """Our resting pair in the SIM broker's own working-orders read, or ``None``.

    Reads the store through the SAME shared ``resting_bracket`` rule the IBKR
    edge uses (parity, never a forked read): the broker is the single source of
    truth for what rests.
    """
    parsed = tuple(
        w for w in (_to_working(r) for r in book.orders) if w is not None
    )
    return resting_bracket(parsed, scope_tag=scope_tag, conid=conid)


def _to_working(row: object) -> WorkingOrder | None:
    """The shared parser over a sim-order row (they ARE the gateway's shape)."""
    return parse_working_order(row)


def apply_bracket(
    book: SimBook,
    children: Sequence[OrderIntent],
    *,
    scope_tag: str,
    conid: int,
    anchor: pd.Timestamp,
    log: Callable[[str], None],
) -> tuple[tuple[OrderResult, ...], SimBook]:
    """Place (or REPLACE) a protective pair in the sim broker.

    REPLACE = cancel BOTH old legs first, then store the new pair — never one
    leg; an unconfirmable cancel would refuse (the sim store confirms by
    construction, so a replace always proceeds).
    """
    rows: list[dict[str, object]] = [dict(r) for r in book.orders]
    old = resting_rows(book, scope_tag=scope_tag, conid=conid)
    if old is not None:
        for leg in old:
            log(
                f"replacing {leg.order_ref} (order {leg.order_id}): "
                f"cancelled, absent from the working read"
            )
        rows = [r for r in rows if not _ours_on(r, scope_tag, conid)]
    new_rows = tuple(
        leg_row(
            child,
            scope_tag=scope_tag,
            conid=conid,
            anchor=anchor,
            order_id=f"SIM-{_kind_child(child)}-{i}",
        )
        for i, child in enumerate(children)
    )
    results = tuple(
        OrderResult(
            intent=child,
            fill=None,
            ok=True,
            message=(
                f"protect {child.symbol} {_kind_child(child)} level="
                f"{_leg_level(child):g} resting GTC cOID={row['order_ref']}"
            ),
            position_id=children[0].position_id,
            outcome=OrderOutcome.PLACED,
            filled_qty=0.0,
        )
        for child, row in zip(children, new_rows, strict=True)
    )
    return results, replace(book, orders=tuple(rows + list(new_rows)))


def _kind_child(child: OrderIntent) -> str:
    """``stop``/``take_profit`` for a protective child (from its ticket kind)."""
    return "stop" if child.order_type is OrderType.STP else "take_profit"


def _ours_on(row: Mapping[str, object], scope_tag: str, conid: int) -> bool:
    """Whether a stored order row is OUR protective leg on *conid*."""
    w = _to_working(row)
    if w is None or w.conid != conid:
        return False
    return w.order_ref.startswith(f"{scope_tag}-")


def fire_stops(
    book: SimBook,
    *,
    scope_tag: str,
    params: ExecutionParams,
    exchange: SimExchange,
    seed_cash: float,
    log: Callable[[str], None],
) -> tuple[tuple[OrderResult, ...], SimBook]:
    """The sim broker's own trigger: fire every RESTING stop the mark has breached.

    The sim's (coarse) approximation of the resting-stop clock: a stop fires when
    a cycle observes a mark past its level; the fill is booked at the observed
    mark through the shared ``execute_signal`` pricing (parity with every other
    sim fill), the OCA sibling is cancelled and the fired leg is marked
    Cancelled. Returns the fired-stop results and the updated account book —
    positions reduced, cash moved by the ONE signed-cash rule, trades appended —
    exactly like a placement's ``_write_back``.
    """
    positions = [dict(r) for r in book.positions]
    trades = [dict(r) for r in book.trades]
    orders = [dict(r) for r in book.orders]
    cash_delta = 0.0
    results: list[OrderResult] = []
    for row in orders:
        w = _to_working(row)
        if w is None or not w.order_ref.startswith(f"{scope_tag}-"):
            continue
        if kind_of(w) != "stop" or w.status.strip().upper() != "SUBMITTED":
            continue
        level = w.level
        if level is None or level <= 0:
            continue
        lot = _lot_row(positions, w)
        if lot is None:
            continue
        if not _breached(lot["mark"], level, lot["long"]):
            continue
        intent = _stop_intent(w, lot)
        at = pd.Timestamp.now(tz="UTC")
        fill = exchange.execute_signal(
            trade_signal(
                symbol=intent.symbol,
                action=ActionType.close,
                price=lot["mark"],
                qty=intent.qty,
                ts=at,
                reason="stop fired",
                position_id=lot["position_id"],
                stop_loss=level,
                position_side=(
                    ActionType.long if lot["long"] else ActionType.short
                ),
            ),
            ref_candle(lot["mark"], intent.symbol, at),
            params,
        )
        # The OCA sibling is cancelled FIRST (invariant 6: nothing races the
        # fired leg).
        orders = [r for r in orders if not _is_sibling(r, w, scope_tag)]
        row["status"] = "Cancelled"
        row["order_ccp_status"] = "Cancelled"
        row["filledQuantity"] = lot["qty"]
        row["remainingQuantity"] = 0.0
        orders = [row if r.get("orderId") == w.order_id else r for r in orders]
        _apply_stop_fill(positions, trades, lot, fill)
        side = OrderSide.SELL if lot["long"] else OrderSide.BUY
        cash_delta += _cash_delta(
            side, fill.filled_qty, fill.executed_price, fill.commission
        )
        message = (
            f"stop {intent.symbol} FIRED qty={fill.filled_qty:g} "
            f"@ {lot['mark']:g} (level {level:g}); sibling cancelled"
        )
        log(message)
        results.append(
            OrderResult(
                intent=intent,
                fill=fill,
                ok=True,
                message=message,
                position_id=lot["position_id"],
                filled_qty=fill.filled_qty,
            )
        )
    if not results:
        return (), book
    summary = dict(book.summary)
    prior = _summary_cash(summary)
    summary["totalcashvalue"] = {
        "amount": (seed_cash if prior is None else prior) + cash_delta,
        "currency": "USD",
    }
    return tuple(results), replace(
        book,
        summary=summary,
        positions=tuple(positions),
        trades=tuple(trades),
        orders=tuple(orders),
    )


def _is_sibling(
    row: Mapping[str, object], fired: WorkingOrder, scope_tag: str
) -> bool:
    """Whether *row* is the OTHER leg of *fired*'s bracket (same conid, ours)."""
    w = _to_working(row)
    if w is None or w.order_id == fired.order_id:
        return False
    return (
        w.order_ref.startswith(f"{scope_tag}-")
        and w.conid == fired.conid
        and w.status.strip().upper() == "SUBMITTED"
    )


def _lot_row(
    positions: list[dict[str, object]], w: WorkingOrder
) -> dict[str, object] | None:
    """The position row for *w*'s conid, or ``None`` when the lot is gone/flat."""
    for row in positions:
        if _num(row.get("conid")) != w.conid:
            continue
        pos = _num(row.get("position"))
        if pos == 0.0:
            continue
        return {
            "position_id": _str(row.get("position_id")),
            "qty": abs(pos),
            "mark": _num(row.get("mktPrice")) or _num(row.get("avgCost")),
            "long": pos > 0,
        }
    return None


def _breached(mark: float, level: float, is_long: bool) -> bool:
    """Whether a stop's level is crossed: a long exits below the level, a short above."""
    return mark <= level + _FLAT_EPS if is_long else mark >= level - _FLAT_EPS


def _stop_intent(w: WorkingOrder, lot: dict[str, object]) -> OrderIntent:
    return OrderIntent(
        symbol=w.symbol,
        action=ActionType.close,
        qty=cast("float", lot["qty"]),
        ref_price=w.level if w.level is not None else cast("float", lot["mark"]),
        reason="stop fired",
        position_id=cast("str", lot["position_id"]),
        stop_loss=w.level,
        order_type=OrderType.STP,
        protective=True,
        position_side=(
            ActionType.long if cast("bool", lot["long"]) else ActionType.short
        ),
    )


def _apply_stop_fill(
    positions: list[dict[str, object]],
    trades: list[dict[str, object]],
    lot: dict[str, object],
    fill: FillEvent,
) -> None:
    """Apply the fired stop to the position/trade rows (a full reduce)."""
    pid = cast("str", lot["position_id"])
    for row in positions:
        if _str(row.get("position_id")) != pid:
            continue
        row["position"] = 0.0
        trades.append(
            {
                "execution_id": f"{pid}:stop",
                "order_ref": "",
                "symbol": fill.signal.symbol,
                "conid": _num(row.get("conid")) or synthetic_conid(
                    fill.signal.symbol
                ),
                "side": "S" if cast("bool", lot["long"]) else "B",
                "size": fill.filled_qty,
                "price": fill.executed_price,
                "commission": fill.commission,
                "trade_time_r": int(fill.timestamp.timestamp() * 1000),
                "account": "SIM",
            }
        )
        return


def _summary_cash(summary: object) -> float | None:
    """``totalcashvalue.amount`` when present, else ``None`` (no cash row to use)."""
    if not isinstance(summary, dict):
        return None
    total = summary.get("totalcashvalue")
    if not isinstance(total, dict):
        return None
    amount = total.get("amount")
    return float(amount) if isinstance(amount, (int, float)) else None


def _num(value: object, default: float = 0.0) -> float:
    """Best-effort float from a JSON value; garbage -> *default* (shared shape)."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", ""))
        except ValueError:
            return default
    return default


def _str(value: object, default: str = "") -> str:
    """Best-effort string from a JSON value; absent -> *default*."""
    return (
        default if value is None else (value if isinstance(value, str) else str(value))
    )


__all__ = ["apply_bracket", "fire_stops", "leg_row", "resting_rows", "wire_side"]