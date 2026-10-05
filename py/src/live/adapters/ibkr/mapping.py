"""Raw IBKR JSON -> typed records. Pure, total, and warning-carrying.

The gateway's JSON is loosely typed (numbers arrive as strings, fields are
routinely absent, timestamps are ms-epoch or a string depending on the endpoint).
Every parser here is *total*: each returns ``(records, warnings)`` — a malformed
row is skipped with a warning naming the row, never an exception that would kill
the whole book read. That is the right posture for a read path: one unreadable
position must not hide the nine readable ones, and the warning list is what makes
the skip visible instead of silent.

Nothing here decides ownership or trades — it only turns JSON into the typed
records :mod:`~src.live.adapters.ibkr.trades` and the portfolio source consume.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.exec.types import OrderSide
from src.live.adapters.ibkr.trades import Execution


@dataclass(frozen=True)
class IbkrPosition:
    """One net-per-instrument position: ``qty`` is signed (IBKR nets the book)."""

    account: str
    conid: int
    symbol: str
    qty: float
    avg_cost: float
    currency: str = ""
    mkt_price: float = 0.0


@dataclass(frozen=True)
class IbkrSummary:
    """The account summary fields this phase reads: cash and net liquidation."""

    account: str
    net_liquidation: float
    total_cash: float
    currency: str = ""


def _mapping(value: object) -> Mapping[str, object]:
    """Narrow a JSON value to a mapping (JSON objects are the only shape we accept)."""
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else {}


def num(value: object, default: float = 0.0) -> float:
    """Best-effort float: numbers and numeric strings parse, garbage -> *default*."""
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


def opt_str(value: object) -> str:
    """Best-effort string: whatever is there, stringified; absent -> ``""``."""
    return "" if value is None else (value if isinstance(value, str) else str(value))


def _canonical_order_id(value: object) -> str:
    """Canonical ``order_id``: a numeric id becomes ``str(int)``, else the string.

    IBKR may serve the same id as ``"97932"`` or ``97932.0``; both must mint the
    SAME lot handle, so a float-shaped id is normalised (``"97932.0"`` ->
    ``"97932"``) to match a ledger-shaped one. A non-numeric id is kept verbatim.
    Phase 3 needs this: the lot handle must be stable across endpoints.
    """
    if isinstance(value, bool) or value is None:
        return opt_str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value))
    if isinstance(value, str):
        token = value.strip()
        try:
            return str(int(float(token)))
        except ValueError:
            return token
    return opt_str(value)


def _amount(body: Mapping[str, object], key: str) -> tuple[float, bool]:
    """Read an IBKR ``{"amount": x, "currency": y}`` field: (value, present)."""
    field = body.get(key)
    if isinstance(field, Mapping):
        inner = _mapping(field)
        if "amount" in inner:
            return num(inner.get("amount")), True
        return 0.0, False
    if field is None:
        return 0.0, False
    return num(field), True


def parse_positions(
    raw: Iterable[object],
) -> tuple[tuple[IbkrPosition, ...], tuple[str, ...]]:
    """Parse ``/portfolio/{acct}/positions`` rows.

    A row with no symbol *and* no conid, or a net-zero quantity, is skipped with a
    warning: a flat line is noise and an unidentifiable one is unusable.
    """
    records: list[IbkrPosition] = []
    warnings: list[str] = []
    for index, entry in enumerate(raw):
        body = _mapping(entry)
        conid = int(num(body.get("conid")))
        symbol = opt_str(body.get("contractDesc")) or opt_str(body.get("ticker"))
        if conid == 0 and not symbol:
            warnings.append(f"position[{index}]: no conid or symbol; skipped")
            continue
        qty = num(body.get("position"))
        if qty == 0:
            warnings.append(f"position[{index}] {symbol or conid}: net-zero; skipped")
            continue
        records.append(
            IbkrPosition(
                account=opt_str(body.get("acctId")),
                conid=conid,
                symbol=symbol,
                qty=qty,
                avg_cost=num(body.get("avgCost")),
                currency=opt_str(body.get("currency")),
                mkt_price=num(body.get("mktPrice")),
            )
        )
    return tuple(records), tuple(warnings)


def parse_summary(
    raw: object, account: str = ""
) -> tuple[IbkrSummary, tuple[str, ...]]:
    """Parse ``/portfolio/{acct}/summary`` into the two amounts the book needs.

    Cash comes from ``totalcashvalue`` (falling back to ``availablefunds``), net
    liquidation from ``netliquidation``; a missing field is a warning and a zero,
    not a failure — the replay, not the summary, decides the lots.
    """
    body = _mapping(raw)
    warnings: list[str] = []
    net_liq, has_net = _amount(body, "netliquidation")
    cash, has_cash = _amount(body, "totalcashvalue")
    if not has_cash:
        cash, has_cash = _amount(body, "availablefunds")
    if not has_net:
        warnings.append("summary: netliquidation missing; using 0.0")
    if not has_cash:
        warnings.append("summary: totalcashvalue/availablefunds missing; using 0.0")
    currency = opt_str(_mapping(body.get("netliquidation")).get("currency"))
    return (
        IbkrSummary(
            account=account,
            net_liquidation=net_liq,
            total_cash=cash,
            currency=currency,
        ),
        tuple(warnings),
    )


def _side(value: object) -> OrderSide | None:
    """Map IBKR's buy/sell spelling (``B``/``S``/``BUY``/``SELL``) to an OrderSide."""
    token = opt_str(value).strip().upper()
    if token in ("B", "BUY", "BOT"):
        return OrderSide.BUY
    if token in ("S", "SELL", "SLD"):
        return OrderSide.SELL
    return None


def _timestamp_ms(value: object) -> pd.Timestamp | None:
    """``trade_time_r`` (ms epoch) -> UTC ``Timestamp``; unparseable -> ``None``."""
    ms = num(value, default=-1.0)
    if ms <= 0:
        return None
    return cast("pd.Timestamp", pd.Timestamp(int(ms), unit="ms", tz="UTC"))


def parse_executions(
    raw: Iterable[object],
) -> tuple[tuple[Execution, ...], tuple[str, ...]]:
    """Parse ``/iserver/account/trades`` rows into :class:`Execution`s.

    A row missing an order id, an interpretable side, a non-zero size or a usable
    ``trade_time_r`` is skipped with a warning — ordering by time is what makes
    the replay deterministic, so a timeless execution cannot be trusted.
    """
    records: list[Execution] = []
    warnings: list[str] = []
    for index, entry in enumerate(raw):
        body = _mapping(entry)
        execution_id = opt_str(body.get("execution_id"))
        order_id = _canonical_order_id(body.get("order_id"))
        if not order_id:
            warnings.append(f"trade[{index}] {execution_id}: no order_id; skipped")
            continue
        side = _side(body.get("side"))
        if side is None:
            warnings.append(f"trade[{index}] {execution_id}: bad side; skipped")
            continue
        qty = abs(num(body.get("size")))
        if qty == 0:
            warnings.append(f"trade[{index}] {execution_id}: zero size; skipped")
            continue
        ts = _timestamp_ms(body.get("trade_time_r"))
        if ts is None:
            warnings.append(f"trade[{index}] {execution_id}: bad timestamp; skipped")
            continue
        symbol = opt_str(body.get("symbol"))
        records.append(
            Execution(
                execution_id=execution_id,
                order_id=order_id,
                order_ref=opt_str(body.get("order_ref")),
                symbol=symbol,
                side=side,
                qty=qty,
                price=num(body.get("price")),
                commission=num(body.get("commission")),
                ts=ts,
            )
        )
    return tuple(records), tuple(warnings)


__all__ = [
    "IbkrPosition",
    "IbkrSummary",
    "parse_executions",
    "parse_positions",
    "parse_summary",
    "num",
    "opt_str",
]
