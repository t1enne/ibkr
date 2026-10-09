"""Stateless sim adapter — the simulated backend behind the ``LiveAdapter`` seam.

Two things make this a real adapter rather than a fixture reader:

1. **The book is a JSON file.** ``read_book`` builds the account book from the
   scope's ``sim_broker.json`` body (an operator-editable IBKR-shaped report), so
   a sim book persists across cycles and an operator can hand-edit the file to
   create a divergence. Nothing is held on the object; the ENGINE records our own
   fills into sqlite and this adapter writes the account side back.
2. **Fills go through the backtest execution core.** ``place_cohort`` runs the
   shared :func:`src.live.pure.sim_place_cohort` — per-order ``execute_signal``
   pricing, then ONE ``SimExchange.settle_cohort`` over the book it was given —
   with friction from ``exec_params_of(cfg)`` and the cohort scale from the shared
   ``apply_fills`` rule. The path is shared with every sim caller, never forked.

The returned ``OrderResult``s carry the settled ``position_id``/``filled_qty``;
``place_cohort`` then writes the account book back to the JSON file
(:func:`_write_back`), so the file is the durable account book and our sqlite
fill fold stays the single durable record of our own fills.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import cast

import pandas as pd

from src.bt.exchange import SimExchange, default_exchange
from src.bt.state import ActionType, PortfolioState, Position
from src.exec.types import OrderSide
from src.live.adapters.sim.store import SimBook, SimBookStore, synthetic_conid
from src.live.pure import OrderResult, sim_place_cohort
from src.live.pure_plan import _cash_delta
from src.live.ledger import LedgerReadError, SqliteLedger
from src.live.result import Err, Ok, Result
from src.live.types import (
    FeedError,
    LiveConfig,
    OrderIntent,
    PortfolioSnapshot,
    exec_params_of,
)

#: One open lot's worth of a book row: ``(position_id, symbol, side, qty, entry)``.
_BareLot = tuple[str, str, str, float, float]


@dataclass(frozen=True)
class SimAdapter:
    """The sim backend: JSON book, immediate deterministic fills, no gateway.

    A frozen dataclass of immutable deps — no mutable state. ``owns_book`` is
    ``True``: the account book may hold lots the strategy never opened, so a
    close is scoped to what the ledger says the scope owns.
    """

    scope: str
    config: LiveConfig
    ledger: SqliteLedger
    dry_run: bool
    log: Callable[[str], None]
    exchange: SimExchange
    store: SimBookStore
    owns_book: bool = True

    async def read_book(self) -> Result[PortfolioSnapshot, FeedError]:
        """The scope's account book from JSON: its lots + its derived cash.

        An unreadable store is an ``Err`` (refuse rather than trade a book we
        cannot read); a never-written scope reads as a flat funded book.
        """
        try:
            portfolio = build_account_book(
                self.store, self.ledger, self.scope, self.config
            )
        except LedgerReadError as exc:
            return Err(FeedError(kind="bad_fixture", message=str(exc)))
        return Ok(
            PortfolioSnapshot(portfolio=portfolio, as_of=pd.Timestamp.now(tz="UTC"))
        )

    async def resync(self) -> Result[tuple[OrderResult, ...], FeedError]:
        """Nothing to reconcile: a sim fill is settled in the cycle that placed it."""
        return Ok(())

    async def place(
        self, book: PortfolioState, intent: OrderIntent
    ) -> Result[OrderResult, FeedError]:
        """Route ONE order as a cohort of one (a lone open fills or rejects as ever)."""
        placed: Result[tuple[OrderResult, ...], FeedError] = await self.place_cohort(
            book, (intent,)
        )
        if isinstance(placed, Err):
            return Err(cast("FeedError", placed.error))
        return Ok(placed.value[0])

    async def place_cohort(
        self, book: PortfolioState, intents: tuple[OrderIntent, ...]
    ) -> Result[tuple[OrderResult, ...], FeedError]:
        """Price + settle the cohort over *book*, write the account book back.

        ``dry_run`` refuses at the edge (defence in depth): a sim adapter built
        for a read-only cycle places nothing even if the caller drops the guard.
        """
        if self.dry_run:
            return Err(
                FeedError(
                    kind="auth",
                    message="sim adapter built for a dry run: refusing to place",
                )
            )
        if not intents:
            return Ok(())
        _settled, results = sim_place_cohort(
            book,
            intents,
            exec_params_of(self.config),
            self.exchange,
            self.log,
        )
        skipped = _write_back(
            self.store,
            self.scope,
            results,
            self.ledger.cash_of(self.scope, self.config.initial_capital),
        )
        for pid in skipped:
            self.log(
                f"close {pid}: no account row to reduce; cash and trade not booked"
            )
        return Ok(results)

    async def close(self) -> Result[None, FeedError]:
        """Nothing to tear down: no session, no held book."""
        return Ok(None)


def build_account_book(
    store: SimBookStore, ledger: SqliteLedger, scope: str, config: LiveConfig
) -> PortfolioState:
    """The sim account book for *scope*: the JSON lots + their derived cash.

    Cash is the JSON summary's ``totalcashvalue`` when present, else the scope's
    own derived number (the store side is the authority on per-scope cash; the
    summary row is what the operator edits). ``initial_capital`` falls back to the
    config's, so a never-written scope reads as a funded flat book.
    """
    as_of = pd.Timestamp.now(tz="UTC")
    book = store.read(scope)
    lots = tuple(_bare_lot(row) for row in book.positions)
    grouped: dict[str, list[Position]] = {}
    for position_id, symbol, side, qty, entry in lots:
        if not symbol:
            continue  # ownership-only: no book position to carry
        grouped.setdefault(symbol, []).append(
            Position(
                symbol=symbol,
                qty=qty,
                entry_price=entry,
                entry_time=as_of,
                stop_loss=None,
                take_profit=None,
                last_price=entry,
                type=ActionType.long if side != "short" else ActionType.short,
                position_id=position_id,
            )
        )
    cash = _summary_cash(book.summary)
    if cash is None:
        cash = ledger.cash_of(scope, config.initial_capital)
    return PortfolioState(
        cash=cash,
        positions={sym: tuple(items) for sym, items in grouped.items()},
        trades=(),
        equity_curve=(),
        initial_capital=(ledger.initial_capital_of(scope) or config.initial_capital),
    )


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
    """Best-effort float from a JSON value; garbage -> *default*."""
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


def _bare_lot(row: object) -> _BareLot:
    """A JSON position row reduced to what the book needs.

    A row with no symbol/size is skipped: it names a lot we cannot describe, and
    inventing a size would put a phantom position in the book the cycle sizes
    against.
    """
    if not isinstance(row, Mapping):
        return ("", "", "", 0.0, 0.0)
    symbol = _str(row.get("contractDesc") or row.get("symbol") or "")
    qty = _num(row.get("position"))
    price = _num(row.get("avgCost"))
    pid = _str(row.get("position_id"))
    if not symbol or qty == 0.0:
        return (pid, "", "", 0.0, 0.0)
    side = _str(row.get("side"))
    if not side:
        side = "long" if qty > 0 else "short"
    return (pid, symbol, side, abs(qty), price)


def _write_back(
    store: SimBookStore,
    scope: str,
    results: tuple[OrderResult, ...],
    seed_cash: float,
) -> tuple[str, ...]:
    """Fold the cycle's fills into the JSON account book and write it back.

    Re-reads the file (so a concurrent operator edit is not clobbered wholesale),
    then for each ``ok`` result with a fill appends a trade and upserts (open) or
    reduces (close) the matching position row, moving ``totalcashvalue`` by the
    signed cash delta of the RESOLVED leg side — ``pure_plan._cash_delta``'s ONE
    rule: a SELL credits, a BUY debits. A close that reduced no account row is
    SKIPPED (no cash, no trade; its id is returned for the caller to report), so
    a phantom close can never move cash while the lot stays on the book. A scope
    whose summary carries no cash row is FUNDED first (``seed_cash``), so the
    broker's cash starts at the config's initial capital rather than at zero.
    """
    book = store.read(scope)
    positions = [dict(row) for row in book.positions]
    trades = [dict(row) for row in book.trades]
    cash_delta = 0.0
    skipped: list[str] = []
    for result in results:
        fill = result.fill
        if not result.ok or fill is None:
            continue
        pid = result.position_id or result.intent.position_id
        if not pid:
            continue
        sym = result.intent.symbol
        qty = fill.filled_qty
        price = fill.executed_price
        comm = fill.commission or 0.0
        opens = result.intent.action is not ActionType.close
        # A close's direction is the lot's MIRROR (long->Sell, short->Buy), which
        # the intent action alone cannot tell; the account row carries the side.
        side = _side_of(result.intent.action) if opens else _close_leg(positions, pid)
        if not _apply_fill(positions, pid, sym, qty, price, side, opens):
            skipped.append(pid)
            continue
        trades.append(
            _trade_row(pid, sym, qty, price, comm, side, opens, fill.timestamp)
        )
        cash_delta += _cash_delta(_order_side(side), qty, price, comm)
    store.write(scope, _stamped_book(book, positions, trades, cash_delta, seed_cash))
    return tuple(skipped)


def _stamped_book(
    book: SimBook,
    positions: list[dict[str, object]],
    trades: list[dict[str, object]],
    cash_delta: float,
    seed_cash: float,
) -> SimBook:
    """The book to write back: summary cash advanced by *cash_delta*.

    A scope whose summary carries no cash row is FUNDED first (``seed_cash``),
    so the broker's cash starts at the config's initial capital, never at zero.
    """
    summary = dict(book.summary)
    prior = _summary_cash(summary)
    prior_cash = seed_cash if prior is None else prior
    summary["totalcashvalue"] = {"amount": prior_cash + cash_delta, "currency": "USD"}
    return replace(
        book,
        account=book.account or "SIM",
        summary=summary,
        positions=tuple(positions),
        trades=tuple(trades),
    )


def _apply_fill(
    positions: list[dict[str, object]],
    pid: str,
    sym: str,
    qty: float,
    price: float,
    side: str,
    opens: bool,
) -> bool:
    """Apply one fill's position-row change; ``False`` when a close reduced nothing.

    An open upserts the lot; a close reduces it and reports whether a row was
    actually hit (a close on an already-missing account row must not book cash).
    """
    if opens:
        _upsert_position(positions, pid, sym, qty, price, side)
        return True
    return _reduce_position(positions, pid, qty, side)


def _trade_row(
    pid: str,
    sym: str,
    qty: float,
    price: float,
    comm: float,
    side: str,
    opens: bool,
    ts: pd.Timestamp,
) -> dict[str, object]:
    """The trade row a booked fill appends (``B``/``S`` mirror of the leg side)."""
    return {
        "execution_id": f"{pid}:{'open' if opens else 'close'}",
        #: ``OrderIntent`` carries no ``order_ref`` (that is minted at the
        #: identity layer per attempt), so the trade row leaves it blank.
        "order_ref": "",
        "symbol": sym,
        "conid": synthetic_conid(sym),
        "side": "B" if side == "BUY" else "S",
        "size": qty,
        "price": price,
        "commission": comm,
        "trade_time_r": int(ts.timestamp() * 1000),
        "account": "SIM",
    }


def _order_side(side: str) -> OrderSide:
    """The resolved leg side (``BUY``/``SELL``) as the shared cash rule's enum."""
    return OrderSide.SELL if side == "SELL" else OrderSide.BUY


def _side_of(action: ActionType) -> str:
    """The side an OPEN leg trades: a long BUYS, a short SELLS."""
    return "BUY" if action is ActionType.long else "SELL"


def _close_leg(positions: list[dict[str, object]], pid: str) -> str:
    """The side a close trades, read from the account row it reduces.

    The intent action for a close is just ``close`` (no direction), so the leg is
    the lot's MIRROR: a long lot is closed by a SELL, a short by a BUY. An unknown
    row side falls back to SELL (a long-closing default), never a guess that could
    invert a position.
    """
    for row in positions:
        if row.get("position_id") == pid:
            stored = _str(row.get("side"))
            if stored == "short":
                return "BUY"
            # ``long`` or an unstated side both reduce a positive lot as a SELL.
            return "SELL"
    return "SELL"


def _upsert_position(
    positions: list[dict[str, object]],
    pid: str,
    sym: str,
    qty: float,
    price: float,
    leg_side: str,
) -> None:
    """Upsert the open position row by ``position_id`` (side, marks, cash price).

    ``qty`` is the fill's ABSOLUTE size and ``leg_side`` its BUY/SELL direction,
    so a short open is stored as a NEGATIVE ``position`` with ``side='short'``
    (IBKR's sign convention) — a positive one would read back as a long and make
    an unedited short book report a false divergence.
    """
    signed = qty if leg_side == "BUY" else -qty
    fields: dict[str, object] = {
        "conid": synthetic_conid(sym),
        "contractDesc": sym,
        "position": signed,
        "side": "long" if signed > 0 else "short",
        "avgCost": price,
        "mktPrice": price,
        "currency": "USD",
    }
    for row in positions:
        if row.get("position_id") == pid:
            row.update(fields)
            return
    positions.append({"position_id": pid, **fields})


def _reduce_position(
    positions: list[dict[str, object]], pid: str, qty: float, leg_side: str
) -> bool:
    """Apply a close's fill to its row: SELL reduces, BUY raises the signed size.

    A PARTIAL close therefore leaves the remainder on the book, exactly as our
    fill fold keeps the lot partially open, so the two books still agree.
    Returns ``False`` when no row carries ``pid`` (the close reduced nothing).
    """
    for row in positions:
        if row.get("position_id") != pid:
            continue
        remaining = _num(row.get("position")) - (qty if leg_side == "SELL" else -qty)
        row["position"] = remaining
        if remaining != 0.0:
            row["side"] = "long" if remaining > 0 else "short"
        return True
    return False


def build_sim_adapter(
    cfg: LiveConfig,
    scope: str,
    ledger: SqliteLedger,
    dry_run: bool,
    log: Callable[[str], None],
    store: SimBookStore | None = None,
) -> SimAdapter:
    """The sim adapter factory (see ``src.live.adapter.resolve_adapter``)."""
    from src.live.adapters.sim.store import resolve_sim_book_path

    return SimAdapter(
        scope=scope,
        config=cfg,
        ledger=ledger,
        dry_run=dry_run,
        log=log,
        exchange=default_exchange(),
        store=store if store is not None else SimBookStore(resolve_sim_book_path()),
    )


__all__ = ["SimAdapter", "build_account_book", "build_sim_adapter"]
