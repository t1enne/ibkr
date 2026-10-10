"""Side-by-side portfolio view: what the BROKER holds vs what OUR store records.

``ibkr live pf`` answers one operator question: do the broker's book and our
durable sqlite store agree? The two sides are read independently and rendered
together, so a divergence (a lot we own that the broker does not show, or a lot
the broker shows that our store does not claim) is visible rather than assumed.

The shapes here are DISPLAY-ONLY: :class:`BrokerSide` for ``ibkr`` carries the
account summary numbers (net liquidation, cash) purely for the reader — the store
side is the authority on per-scope cash, and the summary is the whole account's,
shared by every strategy. The report's own view of a scope is the MERGED
:class:`PositionRow` (one row per symbol spanning the open lot, the newest order
intent and the stored fills) plus the :class:`ScopeStats` headline numbers, so
the raw lots/orders/trades reads are never rendered as three parallel tables.
Building a report reads; it never writes a ledger row or places an order. The
only I/O is the async broker read at the edge (``read_ibkr_broker``); everything
else is pure.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import cast

import pandas as pd

from src.bt.cmds._shared import _json_default
from src.bt.state import ActionType
from src.shared.style import PLAIN, Role, Styler
from src.bt.table import Col, Table, render
from src.data.ibkr.client import IbkrClient, IbkrError
from src.live.identity import ref_is_ours
from src.live.adapters.ibkr.mapping import (
    IbkrPosition,
    num,
    opt_str,
    parse_positions,
    parse_summary,
)
from src.live.adapters.ibkr.orders import parse_working_order
from src.live.identity import IntentRecord, WorkingOrder
from src.live.adapters.sim.store import SimBookStore
from src.live.divergence import book_from_executions
from src.live.ledger import ExecutionRecord, SqliteLedger, StrategyAudit

#: The side label a signed quantity reduces to; the broker side and the store
#: side both use it so a long/short lot compares directly.
_LONG = "long"
_SHORT = "short"


@dataclass(frozen=True)
class BrokerLot:
    """One broker position, keyed by the id our store would use for it.

    ``id`` is the ``conid`` for ``ibkr`` and the ``position_id`` for ``sim``, so
    ``owned`` (our store holds an OPEN lot with that id) is a direct comparison.
    ``side`` reduces the broker's signed quantity; ``qty`` is the ABSOLUTE size.
    """

    symbol: str
    id: str
    qty: float
    side: str
    avg_cost: float
    last_price: float
    market_value: float
    owned: bool


@dataclass(frozen=True)
class BrokerSide:
    """The broker's book as read for the report (display-only numbers).

    For ``ibkr`` the ``net_liquidation``/``cash`` figures come from the account
    summary and are DISPLAY-ONLY: they describe the WHOLE account (every strategy
    sharing it), never this scope's cash — the store side owns that. They are
    ``None`` when the adapter cannot truthfully provide them. ``working_orders``
    is every order the broker reports; ``ours_orders`` is the subset whose
    ``order_ref`` carries this scope's tag (``identity.ref_is_ours``).
    """

    adapter: str
    source: str
    account: str
    net_liquidation: float | None
    cash: float | None
    positions: tuple[BrokerLot, ...]
    working_orders: tuple[WorkingOrder, ...]
    ours_orders: tuple[WorkingOrder, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class StoreLot:
    """One OPEN lot from our durable book (``BookRow`` projected for display)."""

    id: str
    symbol: str
    side: str
    qty: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    tag: str
    order_ref: str


@dataclass(frozen=True)
class StoreSide:
    """Our durable store for one scope: audit rows, cash, lots and orders/trades.

    ``db_path`` is the resolved sqlite file; ``strategy_rows`` is the audit trail
    (oldest first) so the report can name the latest revision and its cycle.
    ``orders`` is EVERY order intent for the scope (open and closed, newest
    first); ``trades`` is every stored fill (oldest first). ``lots`` are the OPEN
    lots from the fill fold (``book_from_executions``), so both the IBKR reconcile
    book and the sim path report the same way.
    """

    scope: str
    db_path: str
    strategy_rows: tuple[StrategyAudit, ...]
    cash: float
    initial_capital: float
    lots: tuple[StoreLot, ...]
    orders: tuple[IntentRecord, ...]
    trades: tuple[ExecutionRecord, ...]


@dataclass(frozen=True)
class PfReport:
    """One read: the store scopes asked for, plus the broker side when one ran.

    ``broker`` is ``None`` when no config was given — there is no adapter to
    resolve and no account to read, so the report covers the local store alone.
    ``stores`` holds one entry per scope: the config's scope, or every scope the
    store knows about.
    """

    as_of: pd.Timestamp
    stores: tuple[StoreSide, ...]
    broker: BrokerSide | None = None


def read_store(
    ledger: SqliteLedger, scope: str, *, initial_capital: float = 0.0
) -> StoreSide:
    """Read our durable store for *scope* through the ledger's PUBLIC reads only.

    Every figure comes from a public read method (``cash_of``,
    ``initial_capital_of``, ``executions_of``, ``strategies_of``); this never
    writes and never touches a model directly. The stored ``initial_capital``
    wins for cash's seed; a scope that has never written falls back to
    *initial_capital* (``0.0`` without a config, so a scope read outside its own
    config invents no number).
    """
    executions = ledger.executions_of(scope)
    book = book_from_executions(executions)
    open_lots = tuple(
        StoreLot(
            id=position.position_id,
            symbol=position.symbol,
            side=position.type.value,
            qty=position.qty,
            entry_price=position.entry_price,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            tag=position.tag,
            order_ref="",
        )
        for lots in book.positions.values()
        for position in lots
    )
    return StoreSide(
        scope=scope,
        db_path=ledger.db_path,
        strategy_rows=ledger.strategies_of(scope),
        cash=ledger.cash_of(scope, initial_capital),
        initial_capital=ledger.initial_capital_of(scope) or initial_capital,
        lots=open_lots,
        orders=ledger.intents_of(scope),
        trades=executions,
    )


def read_sim_broker(
    store: SimBookStore, scope: str, owned: frozenset[str]
) -> BrokerSide:
    """Read the sim ACCOUNT book from the JSON store (no fixture, no network).

    The JSON book is the human-editable side of the sim account; a lot is
    ``owned`` when its ``position_id`` is in the fill fold's open ids. ``source``
    names the JSON file path. The summary's cash/net-liquidation are display-only
    (the operator edits them to create a divergence).
    """
    book = store.read(scope)
    summary, _ = parse_summary(book.summary)
    return BrokerSide(
        adapter="sim",
        source=str(store.path),
        account=book.account,
        net_liquidation=summary.net_liquidation or None,
        cash=summary.total_cash,
        positions=_broker_lots(book.positions, owned),
        working_orders=(),
        ours_orders=(),
        warnings=(),
    )


def _broker_lots(
    rows: tuple[Mapping[str, object], ...], owned: frozenset[str]
) -> tuple[BrokerLot, ...]:
    """One broker lot per JSON position row, keyed by its ``position_id``.

    The rows are read DIRECTLY rather than through ``parse_positions``: the sim
    book is per-lot, and two lots of one symbol share the synthetic ``conid``, so
    a conid-keyed map would silently collapse them into one.
    """
    lots: list[BrokerLot] = []
    for row in rows:
        symbol = opt_str(row.get("contractDesc") or row.get("symbol"))
        pid = opt_str(row.get("position_id"))
        qty = num(row.get("position"))
        if not symbol or qty == 0.0:
            continue
        side = opt_str(row.get("side")) or (_LONG if qty > 0 else _SHORT)
        avg_cost = num(row.get("avgCost"))
        lots.append(
            BrokerLot(
                symbol=symbol,
                id=pid,
                qty=abs(qty),
                side=side,
                avg_cost=avg_cost,
                last_price=num(row.get("mktPrice")) or avg_cost,
                market_value=qty * avg_cost,
                owned=pid in owned,
            )
        )
    return tuple(lots)


async def read_ibkr_broker(
    client: IbkrClient,
    account: str,
    scopes: tuple[str, ...],
    owned_conids: frozenset[str],
) -> BrokerSide:
    """Read the IBKR account's summary, positions and working orders (read-only).

    The summary numbers are DISPLAY-ONLY (the whole account, shared by every
    strategy). Parser warnings — a skipped position, a missing summary field, a
    row with no ``order_ref`` — are carried, never raised: one unreadable row
    must not blank the report. ``ours_orders`` is the working orders minted by
    ANY of *scopes* (``identity.ref_is_ours``, the same predicate the cycle's
    adoption uses); ``working_orders`` holds only the rows that carry a ref.

    The gateway serves positions from a snapshot it regenerates on its own
    cadence, which leaves ``mktPrice`` frozen however often this is called — so
    the cache is discarded FIRST (``positions_invalidate``) and the read that
    follows is freshly obtained. The discard never places, amends or cancels
    anything; when it fails the read still proceeds, cached, and says so in
    ``warnings``.
    """
    summary, summary_warnings = parse_summary(
        await client.portfolio_summary(account), account
    )
    refresh_warnings = await _discard_position_cache(client, account)
    positions, position_warnings = parse_positions(await client.positions_all(account))
    working = _working_orders(await client.open_orders())
    return BrokerSide(
        adapter="ibkr",
        source=client.base_url,
        account=account,
        net_liquidation=summary.net_liquidation,
        cash=summary.total_cash,
        positions=tuple(_ibkr_lot(pos, owned_conids) for pos in positions),
        working_orders=working,
        ours_orders=tuple(
            o for o in working if any(ref_is_ours(s, o.order_ref) for s in scopes)
        ),
        warnings=summary_warnings + refresh_warnings + position_warnings,
    )


async def _discard_position_cache(client: IbkrClient, account: str) -> tuple[str, ...]:
    """Force a fresh positions snapshot; a failure is a warning, not a blank report."""
    try:
        await client.positions_invalidate(account)
    except IbkrError as exc:
        return (f"positions refresh: {exc}; marks may be cached",)
    return ()


def _working_orders(entries: list[dict[str, object]]) -> tuple[WorkingOrder, ...]:
    """Parse the open-orders rows, dropping the ones with no ``order_ref``.

    A foreign order (another client, or the UI) carries no ``order_ref`` at all —
    measured against the captured endpoint — so it can never be mistaken for ours.
    """
    return tuple(
        order for entry in entries if (order := parse_working_order(entry)) is not None
    )


def _ibkr_lot(pos: IbkrPosition, owned_conids: frozenset[str]) -> BrokerLot:
    """One IBKR position as a ``BrokerLot``; the summary net is never the lot."""
    return BrokerLot(
        symbol=pos.symbol,
        id=str(pos.conid),
        qty=abs(pos.qty),
        side=_LONG if pos.qty > 0 else _SHORT,
        avg_cost=pos.avg_cost,
        last_price=pos.mkt_price,
        market_value=pos.qty * pos.mkt_price,
        owned=str(pos.conid) in owned_conids,
    )


@dataclass(frozen=True)
class _LotView:
    """One symbol's OPEN exposure collapsed from its lots (display-only)."""

    qty: float
    entry: float
    side: str
    stop_loss: float | None
    take_profit: float | None
    order_ref: str


@dataclass(frozen=True)
class _Flows:
    """One symbol's stored fills: net size, the two legs' averages and cash.

    ``cash`` is the summed ``cash_delta`` — every commission already deducted — so
    the entry commission of a lot that is still OPEN must be added back out of
    ``realized`` by the proportional share of the entry leg (``buy_comm`` for a
    long, ``sell_comm`` for a short).
    """

    net: float
    buy_qty: float
    buy_notional: float
    buy_comm: float
    sell_qty: float
    sell_notional: float
    sell_comm: float
    cash: float


_EMPTY_FLOWS = _Flows(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PositionRow:
    """One symbol's whole lifecycle in a scope: its lot, its order and its P&L.

    The merged view that replaces the separate lots/orders/trades tables: the open
    position (``qty``/``entry``/``stop_loss``/``take_profit``), the newest order
    intent for the symbol (``status``/``order_id``/``order_ref``) and the P&L the
    stored fills imply. A symbol that never filled (a rejected or unfilled intent)
    or whose lot is already gone (closed) is still a row, so nothing is dropped.

    ``unrealized`` is BROKER-MARKED: it is ``None`` when no mark for that symbol
    was read, never a guess re-derived from our own fills. ``realized`` is the
    stored cash flow plus the cost basis still open, so it covers everything
    already round-tripped.
    """

    scope: str
    symbol: str
    side: str
    status: str
    qty: float
    entry: float | None
    last: float | None
    unrealized: float | None
    realized: float
    stop_loss: float | None
    take_profit: float | None


@dataclass(frozen=True)
class ScopeStats:
    """A scope's headline numbers: cash, cost basis and realized/unrealized P&L.

    ``realized_pnl`` is the stored cash flows plus the cost basis of what is still
    open, so it is the P&L of everything already round-tripped. ``unrealized_pnl``
    is ``None`` unless EVERY open position carries a broker mark (never a partial
    sum); ``total_pnl`` and ``total_return`` follow it, and a scope with nothing
    open totals its realized P&L without any broker read.
    """

    scope: str
    initial_capital: float
    cash: float
    open_cost: float
    realized_pnl: float
    unrealized_pnl: float | None
    total_pnl: float | None
    total_return: float | None
    commission: float
    open_lots: int
    trades: int
    wins: int
    losses: int


def position_rows(
    store: StoreSide, broker: BrokerSide | None = None
) -> tuple[PositionRow, ...]:
    """Every symbol *store* touched, merged into one row (broker marks optional).

    The symbol set is the UNION of the open lots, the order intents and the
    stored fills, so a symbol only one of the three names is not lost. *broker*
    is read for last prices only.
    """
    views, symbols = _open_views(store, broker)
    intents = _intents_by_symbol(store.orders)
    flows = _flows(store.trades)
    marks = _marks(broker)
    return tuple(
        _position_row(
            store.scope,
            symbol,
            views.get(symbol),
            intents.get(symbol),
            flows.get(symbol, _EMPTY_FLOWS),
            marks.get(symbol),
        )
        for symbol in symbols
    )


def scope_stats(store: StoreSide, broker: BrokerSide | None = None) -> ScopeStats:
    """The scope's headline numbers, from its own rows plus the broker's marks.

    Every figure is summed from :func:`position_rows`, so the stats table and the
    positions table can never disagree about what is open or what was made.
    """
    rows = position_rows(store, broker)
    realized = sum(row.realized for row in rows)
    unrealized = _aggregate_unrealized(rows)
    total = None if unrealized is None else realized + unrealized
    closed = tuple(row for row in rows if row.qty == 0.0)
    return ScopeStats(
        scope=store.scope,
        initial_capital=store.initial_capital,
        cash=store.cash,
        open_cost=sum(row.qty * (row.entry or 0.0) for row in rows),
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        total_pnl=total,
        total_return=(
            total / store.initial_capital
            if total is not None and store.initial_capital
            else None
        ),
        commission=sum(abs(trade.commission) for trade in store.trades),
        open_lots=len(store.lots),
        trades=len(store.trades),
        wins=sum(1 for row in closed if row.realized > 0.0),
        losses=sum(1 for row in closed if row.realized < 0.0),
    )


def _open_views(
    store: StoreSide, broker: BrokerSide | None
) -> tuple[dict[str, _LotView], list[str]]:
    """Open exposure per symbol, plus the deterministic list of symbols to show.

    Our book rows become views (collapsed per symbol). A symbol our book already
    covers is never re-added from the broker (the two adapters key lots
    differently).
    """
    views = {
        symbol: _lot_view(lots) for symbol, lots in _lots_by_symbol(store.lots).items()
    }
    symbols = sorted(
        set(views)
        | {trade.symbol or "-" for trade in store.trades}
        | {record.key.symbol for record in store.orders}
    )
    return views, symbols


def _owned_lots(store: StoreSide) -> tuple[StoreLot, ...]:
    """The lots we own: the fill fold's open lots (``store.lots``)."""
    return store.lots


def _lots_by_symbol(lots: tuple[StoreLot, ...]) -> dict[str, tuple[StoreLot, ...]]:
    """Group our open book rows by symbol (one symbol may hold several lots)."""
    grouped: dict[str, list[StoreLot]] = {}
    for lot in lots:
        grouped.setdefault(lot.symbol, []).append(lot)
    return {symbol: tuple(rows) for symbol, rows in grouped.items()}


def _lot_view(lots: tuple[StoreLot, ...]) -> _LotView:
    """One symbol's lots collapsed: summed size, weighted entry, first stops set."""
    qty = sum(lot.qty for lot in lots)
    cost = sum(lot.qty * lot.entry_price for lot in lots)
    return _LotView(
        qty=qty,
        entry=cost / qty if qty else lots[0].entry_price,
        side=lots[0].side,
        stop_loss=_first(lot.stop_loss for lot in lots),
        take_profit=_first(lot.take_profit for lot in lots),
        order_ref=_first(lot.order_ref or None for lot in lots) or "",
    )


def _first[T](values: Iterable[T | None]) -> T | None:
    """The first value that is not ``None`` (stops, refs — never a guess)."""
    return next((value for value in values if value is not None), None)


def _intents_by_symbol(orders: tuple[IntentRecord, ...]) -> dict[str, IntentRecord]:
    """The NEWEST intent per symbol (``intents_of`` already returns newest first)."""
    newest: dict[str, IntentRecord] = {}
    for record in orders:
        newest.setdefault(record.key.symbol, record)
    return newest


def _flows(trades: tuple[ExecutionRecord, ...]) -> dict[str, _Flows]:
    """Aggregate one scope's stored fills by symbol, buys and sells kept apart."""
    flows: dict[str, _Flows] = {}
    for trade in trades:
        symbol = trade.symbol or "-"
        prev = flows.get(symbol, _EMPTY_FLOWS)
        buy = trade.side.upper().startswith("B")
        flows[symbol] = _Flows(
            net=prev.net + (trade.qty if buy else -trade.qty),
            buy_qty=prev.buy_qty + (trade.qty if buy else 0.0),
            buy_notional=prev.buy_notional + (trade.qty * trade.price if buy else 0.0),
            buy_comm=prev.buy_comm + (abs(trade.commission) if buy else 0.0),
            sell_qty=prev.sell_qty + (0.0 if buy else trade.qty),
            sell_notional=prev.sell_notional
            + (0.0 if buy else trade.qty * trade.price),
            sell_comm=prev.sell_comm + (0.0 if buy else abs(trade.commission)),
            cash=prev.cash + trade.cash_delta,
        )
    return flows


def _marks(broker: BrokerSide | None) -> dict[str, float]:
    """The broker's last price per symbol (empty without a broker read)."""
    if broker is None:
        return {}
    return {lot.symbol: lot.last_price for lot in broker.positions if lot.last_price}


def _position_row(
    scope: str,
    symbol: str,
    view: _LotView | None,
    intent: IntentRecord | None,
    flow: _Flows,
    mark: float | None,
) -> PositionRow:
    """One merged row: the lot view when open, else the fill/order trail."""
    qty = view.qty if view is not None else abs(flow.net)
    side = _side(view, flow, intent)
    entry = view.entry if view is not None else _avg_entry(flow)
    return PositionRow(
        scope=scope,
        symbol=symbol,
        side=side,
        status="open" if view is not None else _status(intent, flow),
        qty=qty,
        entry=entry,
        last=mark if mark is not None else (None if qty else _avg_exit(flow)),
        unrealized=_unrealized(view, mark),
        realized=_realized(flow, side, qty, entry),
        stop_loss=view.stop_loss if view is not None else None,
        take_profit=view.take_profit if view is not None else None,
    )


def _realized(flow: _Flows, side: str, qty: float, entry: float | None) -> float:
    """The P&L the fills already round-tripped, closed size only.

    ``flow.cash`` holds every cash delta (commissions included), so the cost basis
    of what is still OPEN and that entry commission's proportional share are added
    back: the open lot's entry is not a result yet. The basis is SIGNED by the
    side — a long's basis is cash already spent, a short's is cash already taken —
    so an open short does not read as profit.
    """
    if qty == 0.0 or entry is None:
        return flow.cash
    long_side = side == _LONG
    entry_qty = flow.buy_qty if long_side else flow.sell_qty
    entry_comm = flow.buy_comm if long_side else flow.sell_comm
    share = qty / entry_qty if entry_qty else 0.0
    direction = 1.0 if long_side else -1.0
    return flow.cash + direction * qty * entry + entry_comm * share


def _side(view: _LotView | None, flow: _Flows, intent: IntentRecord | None) -> str:
    """``long``/``short``: the open lot's own side, else the fills', else the ORDER's.

    With no lot and no fill there is no exposure to read a side off, so the newest
    intent's action names it — a pending long OPEN is LONG, not "not a buy". A
    close (or no intent at all) leaves the side BLANK: an unprovable side is never
    defaulted to a short.
    """
    if view is not None:
        return view.side
    if flow.net != 0.0:
        return _LONG if flow.net > 0 else _SHORT
    if flow.buy_qty or flow.sell_qty:
        return _LONG if flow.buy_qty else _SHORT
    if intent is not None:
        if intent.key.action is ActionType.long:
            return _LONG
        if intent.key.action is ActionType.short:
            return _SHORT
    return ""


def _status(intent: IntentRecord | None, flow: _Flows) -> str:
    """A row with no lot: the newest intent's state, else open/closed on the fills.

    A ``closed`` row means our fills net to zero and the book holds no lot; a
    ``filled`` row with a non-zero ``qty`` is an open position our book lost — the
    state names the ORDER, the ``qty`` column names the exposure.
    """
    if intent is not None:
        return intent.state.value
    return "closed" if flow.net == 0.0 else "open"


def _avg_entry(flow: _Flows) -> float | None:
    """The entry leg's average fill price (the buys, else the sells' average)."""
    qty = flow.buy_qty or flow.sell_qty
    notional = flow.buy_notional or flow.sell_notional
    return notional / qty if qty else None


def _avg_exit(flow: _Flows) -> float | None:
    """The closing leg's average fill price (``None`` when nothing was closed)."""
    if flow.buy_qty and flow.sell_qty:
        return flow.sell_notional / flow.sell_qty
    return None


def _unrealized(view: _LotView | None, mark: float | None) -> float | None:
    """Mark-to-market of the open lot, or ``None`` without a broker mark."""
    if view is None or mark is None:
        return None
    direction = 1.0 if view.side == _LONG else -1.0
    return (mark - view.entry) * view.qty * direction


def _aggregate_unrealized(rows: tuple[PositionRow, ...]) -> float | None:
    """Sum the open rows' marks; ``None`` when ANY open row lacks one (no partial)."""
    open_rows = tuple(row for row in rows if row.qty != 0.0)
    if not open_rows:
        return 0.0
    if any(row.unrealized is None for row in open_rows):
        return None
    return sum(cast("float", row.unrealized) for row in open_rows)


def render_pf(report: PfReport, fmt: str, style: Styler = PLAIN) -> str:
    """Deterministic text tables (default) or a JSON document for one pf read.

    *style* carries ANSI roles for the TEXT view only; JSON is parsed by machines,
    so it is never styled and *style* is ignored for it.
    """
    if fmt == "json":
        return _render_json(report)
    return _render_text(report, style)


#: The colour of each merged-position state — deliberately sparing: only a state
#: that needs acting on (a wedged order, a refusal) is coloured loudly, and an
#: ordinary open lot or a clean fill stays quiet.
_STATE_ROLES: Mapping[str, Role] = {
    "open": "strong",
    "closed": "dim",
    "filled": "good",
    "working": "warn",
    "pending": "warn",
    "unfilled": "warn",
    "unresolved": "bad",
    "rejected": "bad",
    "foreign": "warn",
}

#: The broker's owner marker: a foreign lot on a shared account is a caution, not
#: an error (usually it is another scope's or the human's).
_OWNER_ROLES: Mapping[str, Role] = {"ours": "good", "NOT-OURS": "warn"}


_ORDER_COLS = (
    Col("symbol", role="strong"),
    Col("side"),
    Col("order_id"),
    Col("filled", ">"),
    Col("status"),
    Col("ref"),
    Col("owner", roles=_OWNER_ROLES),
)
_SCOPE_COLS = (
    Col("scope"),
    Col("rev", ">"),
    Col("id"),
    Col("name"),
    Col("created"),
    Col("last_cycle"),
)
_STATS_COLS = (
    Col("scope"),
    Col("initial", ">"),
    Col("cash", ">"),
    Col("open_cost", ">"),
    Col("realized", ">", sign=True),
    Col("unreal", ">", sign=True),
    Col("total", ">", sign=True),
    Col("ret%", ">", sign=True),
    Col("comm", ">"),
    Col("lots", ">"),
    Col("trd", ">"),
    Col("win", ">"),
    Col("loss", ">"),
)
_POSITION_COLS = (
    Col("scope"),
    Col("symbol", role="strong"),
    Col("side"),
    Col("state", roles=_STATE_ROLES),
    Col("qty", ">"),
    Col("entry", ">"),
    Col("last", ">"),
    Col("upnl", ">", sign=True),
    Col("rpnl", ">", sign=True),
    Col("sl", ">"),
    Col("tp", ">"),
    Col("acct_id"),
    Col("acct_qty", ">"),
    Col("acct_avg", ">"),
    Col("mktvalue", ">"),
    Col("owner", roles=_OWNER_ROLES),
)


def _render_json(report: PfReport) -> str:
    """JSON at the edge — reuse the shared encoder for Timestamps/Enums.

    The per-store ``lots``/``orders``/``trades`` arrays stay RAW (a machine
    consumer wants the rows, not the merge), while ``stats`` and ``positions``
    carry the same derived numbers the text view renders.
    """
    doc = {
        "as_of": report.as_of,
        "broker": asdict(report.broker) if report.broker is not None else None,
        "stores": [_store_dict(store) for store in report.stores],
        "stats": [asdict(scope_stats(store, report.broker)) for store in report.stores],
        "positions": [
            asdict(row)
            for store in report.stores
            for row in position_rows(store, report.broker)
        ],
        "divergence": list(_divergence(report)),
    }
    return json.dumps(doc, default=_json_default, indent=2)


def _store_dict(store: StoreSide) -> dict[str, object]:
    """The store side as a plain mapping (audit rows + orders/trades flattened)."""
    return {
        "scope": store.scope,
        "db_path": store.db_path,
        "strategies": [asdict(row) for row in store.strategy_rows],
        "cash": store.cash,
        "initial_capital": store.initial_capital,
        "lots": [asdict(lot) for lot in store.lots],
        "orders": [_intent_dict(record) for record in store.orders],
        "trades": [asdict(trade) for trade in store.trades],
    }


def _intent_dict(record: IntentRecord) -> dict[str, object]:
    """A compact order view: identity, state, ref and the stuck counter."""
    return {
        "symbol": record.key.symbol,
        "action": record.key.action.value,
        "position_id": record.key.position_id,
        "state": record.state.value,
        "attempt": record.attempt,
        "order_ref": record.order_ref,
        "order_id": record.order_id,
        "tif": record.tif,
        "stuck_cycles": record.stuck_cycles,
        "decision_ts": record.decision_ts,
    }


def _render_text(report: PfReport, style: Styler = PLAIN) -> str:
    """Every side as one aggregated table (``src.bt.table``), no scalar run-on lines.

    All scopes fold into a SINGLE scopes / stats / positions table (each row
    carries its ``scope``), so a report over every scope stays the same height as
    one over a single config. ``positions`` is the ONE table the operator reads -
    each row is a symbol's lot, the newest order for it, the P&L its fills imply
    and, when the broker was read, the matching ACCOUNT lot (``acct_*``/``owner``)
    so the two books sit side by side without a second table. A broker lot no
    scope claims is appended as its own ``foreign`` row rather than hidden.

    *style* is applied sparingly and only through named roles (a title, a caution,
    a sign) - see ``src.shared.style``; the default writes no escape byte.
    """
    blocks: list[list[str]] = [[style.role(f"as_of: {report.as_of}", "note")]]
    if report.stores:
        blocks.append(_table("scopes", _SCOPE_COLS, _scope_rows(report.stores), style))
        blocks.append(
            _table(
                "stats",
                _STATS_COLS,
                _stats_rows(report.stores, report.broker),
                style,
            )
        )
    blocks.append(
        _table(
            "positions",
            _POSITION_COLS,
            _position_text_rows(report.stores, report.broker),
            style,
        )
    )
    if report.broker is not None:
        blocks.append(_broker_lines(report.broker, style))
    divergence = _divergence(report)
    if report.broker is None:
        blocks.append([style.role("divergence: - (broker not read)", "note")])
    else:
        blocks.append(
            [style.role(f"divergence: {item}", "warn") for item in divergence]
            or [style.role("divergence: none", "note")]
        )
    return "\n\n".join("\n".join(block) for block in blocks)


def _table(
    title: str,
    columns: tuple[Col, ...],
    rows: tuple[tuple[str, ...], ...],
    style: Styler,
) -> list[str]:
    """A titled table block, or ``"<title>: none"`` when there is nothing to show."""
    if not rows:
        return [style.role(f"{title}:", "title") + style.role(" none", "note")]
    return [
        style.role(f"{title}:", "title"),
        *render(Table(columns=columns, rows=rows), styler=style),
    ]


def _broker_lines(broker: BrokerSide, style: Styler = PLAIN) -> list[str]:
    """The broker block: one scalar line, then its working orders and warnings.

    The account's positions are NOT repeated here - they are folded into the
    ``positions`` table, where each store row carries its matching ``acct_*``
    cells and any lot no scope claims appears as its own ``foreign`` row.
    """
    scalars = [f"adapter={broker.adapter}", f"source={broker.source or '-'}"]
    if broker.account:
        scalars.append(f"account={broker.account}")
    if broker.net_liquidation is not None:
        scalars.append(f"net_liq={broker.net_liquidation:.2f}")
    if broker.cash is not None:
        scalars.append(f"cash={broker.cash:.2f}")
    lines = [style.role("broker:", "title") + "  " + "  ".join(scalars)]
    lines.extend(
        _table(
            "working orders",
            _ORDER_COLS,
            tuple(
                (
                    order.symbol,
                    order.side,
                    order.order_id,
                    f"{order.filled_qty:g}",
                    order.status,
                    order.order_ref,
                    "ours" if order in broker.ours_orders else "foreign",
                )
                for order in broker.working_orders
            ),
            style,
        )
    )
    lines.extend(style.role(f"warning: {w}", "warn") for w in broker.warnings)
    return lines


def _scope_rows(stores: tuple[StoreSide, ...]) -> tuple[tuple[str, ...], ...]:
    """One row per scope: identity and audit only (cash and P&L live in ``stats``).

    The counts that used to sit here (lots/orders/trades/sim) are gone: every one
    of them is now visible as rows in ``positions``, so repeating them as scalars
    was the redundancy this report was rebuilt to drop.
    """
    return tuple(_scope_row(store) for store in stores)


def _scope_row(store: StoreSide) -> tuple[str, ...]:
    """One scope's identity — the latest revision read once, then its cells."""
    latest = _latest(store)
    return (
        store.scope,
        str(len(store.strategy_rows)),
        latest.strategy_id[:12] if latest else "-",
        latest.name if latest else "-",
        _ts(latest.created_at if latest else None),
        _ts(latest.last_cycle_at if latest else None),
    )


def _stats_rows(
    stores: tuple[StoreSide, ...], broker: BrokerSide | None
) -> tuple[tuple[str, ...], ...]:
    """One P&L row per scope, derived from the store plus the broker's marks."""
    return tuple(_stats_row(scope_stats(store, broker)) for store in stores)


def _stats_row(stats: ScopeStats) -> tuple[str, ...]:
    """One scope's headline cells; a number we cannot honestly state renders ``-``."""
    return (
        stats.scope,
        f"{stats.initial_capital:.2f}",
        f"{stats.cash:.2f}",
        f"{stats.open_cost:.2f}",
        f"{stats.realized_pnl:.2f}",
        _money(stats.unrealized_pnl),
        _money(stats.total_pnl),
        "-" if stats.total_return is None else f"{stats.total_return * 100:.2f}",
        f"{stats.commission:.2f}",
        str(stats.open_lots),
        str(stats.trades),
        str(stats.wins),
        str(stats.losses),
    )


def _position_text_rows(
    stores: tuple[StoreSide, ...], broker: BrokerSide | None
) -> tuple[tuple[str, ...], ...]:
    """Every scope's merged position rows, each with its matching account cells.

    Store rows carry the ACCOUNT side of the same symbol (``acct_id``/``acct_qty``/
    ``acct_avg``/``mktvalue``/``owner``) so the two books read as one table. A
    broker lot no scope's book holds is appended as a ``foreign`` row (scope
    ``-``), so folding the tables never hides exposure another scope or a human
    opened on the shared account.
    """
    account: dict[str, list[BrokerLot]] = {}
    for lot in broker.positions if broker is not None else ():
        account.setdefault(lot.symbol, []).append(lot)
    owned_ids = {lot.id for store in stores for lot in _owned_lots(store)}
    rows: list[tuple[str, ...]] = []
    matched: set[str] = set()
    for store in stores:
        for row in position_rows(store, broker):
            lots = tuple(
                lot for lot in account.get(row.symbol, ()) if lot.id in owned_ids
            )
            matched.update(lot.id for lot in lots)
            rows.append(_position_cells(row, lots))
    for lots in account.values():
        for lot in lots:
            if lot.id not in matched:
                rows.append(_foreign_cells(lot))
    return tuple(rows)


def _position_cells(row: PositionRow, lots: tuple[BrokerLot, ...]) -> tuple[str, ...]:
    """A store row plus the ACCOUNT lot(s) our book pairs it with (``-`` if none)."""
    qty = sum(lot.qty for lot in lots)
    avg = sum(lot.qty * lot.avg_cost for lot in lots) / qty if qty else None
    return (
        row.scope,
        row.symbol,
        row.side,
        row.status,
        f"{row.qty:g}",
        _opt(row.entry),
        _opt(row.last),
        _money(row.unrealized),
        f"{row.realized:.2f}",
        _opt(row.stop_loss),
        _opt(row.take_profit),
        lots[0].id if lots else "-",
        f"{qty:g}" if lots else "-",
        _opt(avg),
        f"{sum(lot.market_value for lot in lots):.2f}" if lots else "-",
        "ours" if lots else "-",
    )


def _foreign_cells(lot: BrokerLot) -> tuple[str, ...]:
    """A broker lot no scope claims: shown whole, its account cells the only data."""
    return (
        "-",
        lot.symbol,
        lot.side,
        "foreign",
        f"{lot.qty:g}",
        _opt(lot.avg_cost),
        _opt(lot.last_price),
        "-",
        "-",
        "-",
        "-",
        lot.id,
        f"{lot.qty:g}",
        _opt(lot.avg_cost),
        f"{lot.market_value:.2f}",
        "ours" if lot.owned else "NOT-OURS",
    )


def _money(value: float | None) -> str:
    """A cash cell at two decimals, or ``-`` when the number is unknown."""
    return "-" if value is None else f"{value:.2f}"


def _latest(store: StoreSide) -> StrategyAudit | None:
    """The scope's newest audit revision, or ``None`` when it has never written."""
    return store.strategy_rows[-1] if store.strategy_rows else None


def _ts(value: pd.Timestamp | None) -> str:
    """A UTC timestamp at second precision (``-`` for an unwritten column)."""
    return "-" if value is None else value.strftime("%Y-%m-%d %H:%M:%S")


def _opt(value: float | None) -> str:
    """A price cell at the table's precision, or ``-`` when the lot has none."""
    return "-" if value is None else f"{value:.4f}"


def _divergence(report: PfReport) -> tuple[str, ...]:
    """Where the broker and our store disagree: lots both ways, and sim cash.

    The mismatches a human acts on: a broker lot we do not record, a store lot
    the broker does not show, and — for ``sim``, whose broker cash IS this
    scope's cash — a cash figure that moved on one side only. Ownership is the
    scope's open lots (the fill fold's ``position_id`` space).

    Cash is only compared for ``sim``: an IBKR account's cash is shared by every
    strategy on it (and the store's own cash is the authority), so a difference
    there says nothing about this scope. Without a broker read there is nothing
    to compare against and the list is empty.
    """
    broker = report.broker
    if broker is None or not report.stores:
        return ()
    broker_ids = {lot.id for lot in broker.positions}
    store_ids = {lot.id for store in report.stores for lot in _owned_lots(store)}
    ours_unmatched = tuple(
        f"broker lot {lot.id} {lot.symbol} not in our store"
        for lot in broker.positions
        if lot.id not in store_ids
    )
    # Keyed by id so a scope list that names the same lot twice reports it once;
    # the first sighting wins, which is the one carrying the symbol when any does.
    store_lots: dict[str, StoreLot] = {}
    for store in report.stores:
        for lot in _owned_lots(store):
            store_lots.setdefault(lot.id, lot)
    store_unmatched = tuple(
        f"store lot {lot.id} {lot.symbol or '-'} not at the broker"
        for lot in store_lots.values()
        if lot.id not in broker_ids
    )
    # A lot BOTH books hold but size differently: our fill fold says one number,
    # the (possibly hand-edited) broker book another. Keyed by id so it reports
    # once per lot; a tolerance keeps float noise out.
    broker_qty = {lot.id: lot.qty for lot in broker.positions}
    qty_mismatch = tuple(
        f"qty mismatch {lot.id} {lot.symbol or '-'}: ours={lot.qty} "
        f"account={broker_qty[lot.id]}"
        for lot in store_lots.values()
        if lot.id in broker_qty and abs(lot.qty - broker_qty[lot.id]) > _CASH_TOLERANCE
    )
    cash = _cash_divergence(broker, report.stores) if broker.adapter == "sim" else ()
    return ours_unmatched + qty_mismatch + store_unmatched + cash


#: Cash differences below this are rounding, not a divergence worth a line.
_CASH_TOLERANCE = 0.01


def _cash_divergence(
    broker: BrokerSide, stores: tuple[StoreSide, ...]
) -> tuple[str, ...]:
    """Cash the sim broker's book and our store disagree on, per scope."""
    if broker.cash is None:
        return ()
    return tuple(
        f"cash: broker {broker.cash:.2f} vs store {store.cash:.2f} for "
        f"{store.scope} (edited fixture, or a cycle that did not settle)"
        for store in stores
        if abs(broker.cash - store.cash) > _CASH_TOLERANCE
    )


__all__ = [
    "BrokerLot",
    "BrokerSide",
    "PfReport",
    "PositionRow",
    "ScopeStats",
    "StoreLot",
    "StoreSide",
    "position_rows",
    "scope_stats",
    "read_ibkr_broker",
    "read_sim_broker",
    "read_store",
    "render_pf",
]
