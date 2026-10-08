"""Portfolio sources — normalise a broker read into the shared ``PortfolioState``.

Only the edges touch I/O. ``load_mock_portfolio``/``write_mock_portfolio`` are
pure (fixture dict <-> book); ``MockPortfolioSource`` is the async wrapper that
reads a JSON file. No network for v1.

A managed fixture (one WE minted, marked ``MANAGED_KEY``) is the sim broker's
record of the cycle: the cycle reads its book from the file and writes the
settled book back, so a mock book persists across cycles instead of resetting
flat. A hand-authored fixture carries no marker and stays read-only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from src.bt.state import ActionType, PortfolioState, Position
from src.live.result import Err, Ok, Result
from src.live.types import FeedError, PortfolioSnapshot


class PortfolioSource(Protocol):
    """Async read of the current book. Failure is a value, never a raise.

    ``owns_book`` tells ``run_cycle`` how to scope close intents: ``True`` (the
    mock ledger-backed book) means closes must be filtered to lots the strategy
    is recorded as owning; ``False`` means the book ALREADY contains only this
    strategy's lots (the IBKR execution replay excludes foreign orders in
    ``trades.replay``), so every lot present is closable and the (possibly empty
    on a dry run) ledger must not blank them out.
    """

    owns_book: bool

    async def fetch(self) -> Result[PortfolioSnapshot, FeedError]: ...


def load_mock_portfolio(
    raw: Mapping[str, object],
    as_of: pd.Timestamp,
) -> PortfolioSnapshot:
    """Build a real ``PortfolioState`` from a fixture dict (pure).

    Fixture shape::

        {"cash": float,
         "initial_capital": float,          # optional, default = cash
         "positions": [{"symbol": str, "qty": float, "type": "long"|"short",
                        "entry_price": float, "position_id": str,
                        "stop_loss": float|null, "take_profit": float|null,
                        "tag": str, "last_price": float, "entry_time": str|null}]}

    ``qty`` is stored positive; the side lives on ``Position.type``. Lots are
    grouped per symbol into tuples in input order. An unknown key (e.g. the
    :data:`MANAGED_KEY` marker ``write_mock_portfolio`` emits) is ignored.
    Raises ``ValueError`` on malformed input (bad cash/symbol/type/qty/timestamp).
    """
    cash = _number(raw.get("cash"), "cash")
    if cash < 0:
        raise ValueError(f"cash must be non-negative, got {cash}")
    initial = raw.get("initial_capital")
    initial_capital = cash if initial is None else _number(initial, "initial_capital")

    grouped: dict[str, list[Position]] = {}
    for entry in _position_entries(raw.get("positions")):
        pos = _to_position(entry, as_of)
        grouped.setdefault(pos.symbol, []).append(pos)

    portfolio = PortfolioState(
        cash=cash,
        positions={sym: tuple(lots) for sym, lots in grouped.items()},
        trades=(),
        equity_curve=(),
        initial_capital=initial_capital,
    )
    return PortfolioSnapshot(portfolio=portfolio, as_of=as_of)


#: Fixture key marking a file as OURS (minted by the sim path, writable).
MANAGED_KEY = "managed"


def fixture_is_managed(path: str | Path) -> bool:
    """Whether *path* is a fixture we minted and may write the book back into.

    Unreadable or malformed files read as NOT ours (fail closed: never overwrite
    a file we cannot prove we own).
    """
    try:
        raw = json.loads(Path(path).read_text())
    except OSError, ValueError:
        return False
    return isinstance(raw, Mapping) and raw.get(MANAGED_KEY) is True


def write_mock_portfolio(path: str | Path, portfolio: PortfolioState) -> None:
    """Persist a settled book into a mock fixture — the inverse of ``load``.

    Emits :data:`MANAGED_KEY`, so the file this writes is one a later cycle may
    write again. Lots keep their ``position_id``, so the ledger's sim ownership
    (``live_sim_lot``) still names them on the next read.
    """
    doc: dict[str, object] = {
        MANAGED_KEY: True,
        "cash": float(portfolio.cash),
        "initial_capital": float(portfolio.initial_capital),
        "positions": [
            {
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "type": pos.type.value,
                "entry_price": float(pos.entry_price),
                "position_id": pos.position_id,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "tag": pos.tag,
                "last_price": float(pos.last_price),
                "entry_time": pos.entry_time.isoformat(),
            }
            for symbol in sorted(portfolio.positions)
            for pos in portfolio.positions[symbol]
        ],
    }
    Path(path).write_text(json.dumps(doc, indent=2))


def _position_entries(value: object) -> list[Mapping[str, object]]:
    """Validate the ``positions`` list into a list of mappings."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("positions must be a list")
    out: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each position must be a mapping")
        out.append(cast("Mapping[str, object]", item))
    return out


def _to_position(entry: Mapping[str, object], as_of: pd.Timestamp) -> Position:
    """Map one fixture position to a concrete ``Position`` (validated)."""
    symbol = entry.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError(f"bad symbol: {symbol!r}")
    qty = _number(entry.get("qty"), "qty")
    if qty <= 0:
        raise ValueError(f"qty must be > 0 for {symbol}, got {qty}")
    side = entry.get("type")
    if side not in ("long", "short"):
        raise ValueError(f"type must be 'long' or 'short' for {symbol}, got {side!r}")
    entry_price = _number(entry.get("entry_price"), "entry_price")
    last_raw = entry.get("last_price")
    last_price = entry_price if last_raw is None else _number(last_raw, "last_price")

    pos_id = entry.get("position_id")
    tag = entry.get("tag")
    stop_loss = _opt_number(entry.get("stop_loss"), "stop_loss")
    take_profit = _opt_number(entry.get("take_profit"), "take_profit")
    return Position(
        symbol=symbol,
        qty=qty,
        entry_price=entry_price,
        entry_time=_timestamp(entry.get("entry_time"), as_of),
        stop_loss=stop_loss,
        take_profit=take_profit,
        last_price=last_price,
        type=ActionType(side),
        position_id=pos_id if isinstance(pos_id, str) else "",
        tag=tag if isinstance(tag, str) else "",
    )


def _number(value: object, field: str) -> float:
    """Coerce a required numeric field, raising ``ValueError`` when absent/bad."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number, got {value!r}")
    return float(value)


def _opt_number(value: object, field: str) -> float | None:
    """Coerce an optional numeric field; ``None`` stays ``None``."""
    if value is None:
        return None
    return _number(value, field)


def _timestamp(value: object, as_of: pd.Timestamp) -> pd.Timestamp:
    """Parse an optional timestamp string, defaulting to ``as_of``."""
    if value is None:
        return as_of
    if isinstance(value, str):
        try:
            return cast(pd.Timestamp, pd.Timestamp(value))
        except ValueError as exc:  # pragma: no cover - message asserted in tests
            raise ValueError(f"bad entry_time {value!r}: {exc}") from exc
    raise ValueError(f"entry_time must be a string or null, got {value!r}")


class MockPortfolioSource:
    """Reads a JSON fixture. Async to match the Protocol; no network.

    The mock book may hold exogenous lots, so closes are ledger-scoped.
    """

    owns_book = True

    def __init__(self, path: str) -> None:
        self._path = path

    async def fetch(self) -> Result[PortfolioSnapshot, FeedError]:
        """Read and validate the fixture; any failure becomes an ``Err``."""
        try:
            raw = json.loads(Path(self._path).read_text())
            return Ok(load_mock_portfolio(raw, pd.Timestamp.now()))
        except (OSError, ValueError) as exc:
            return Err(FeedError(kind="bad_fixture", message=str(exc)))
