"""Side-by-side portfolio view: what the BROKER holds vs what OUR store records.

``ibkr live pf`` answers one operator question: do the broker's book and our
durable sqlite store agree? The two sides are read independently and rendered
together, so a divergence (a lot we own that the broker does not show, or a lot
the broker shows that our store does not claim) is visible rather than assumed.

The shapes here are DISPLAY-ONLY: :class:`BrokerSide` for ``ibkr`` carries the
account summary numbers (net liquidation, cash) purely for the reader — the store
side is the authority on per-scope cash, and the summary is the whole account's,
shared by every strategy. Building a report reads; it never writes a ledger row
or places an order. The only I/O is the async broker read at the edge
(``read_ibkr_broker``); everything else is pure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import cast

import pandas as pd

from src.bt.cmds._shared import _json_default
from src.bt.state import PortfolioState, Position
from src.bt.table import Col, Table, render
from src.data.ibkr.client import IbkrClient
from src.live.identity import ref_is_ours
from src.live.adapters.ibkr.mapping import IbkrPosition, parse_positions, parse_summary
from src.live.adapters.ibkr.orders import parse_working_order
from src.live.identity import IntentRecord, WorkingOrder
from src.live.ledger import ExecutionRecord, SqliteLedger, StrategyAudit
from src.live.portfolio_source import MockPortfolioSource
from src.live.result import Err
from src.live.types import FeedError, LiveConfig

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
    first); ``trades`` is every stored fill (oldest first). ``sim_open_ids`` is
    the sim lot ownership set — the sim path persists NO book rows, so for a sim
    scope this set (not ``lots``) is what proves ownership.
    """

    scope: str
    db_path: str
    strategy_rows: tuple[StrategyAudit, ...]
    cash: float
    initial_capital: float
    lots: tuple[StoreLot, ...]
    orders: tuple[IntentRecord, ...]
    trades: tuple[ExecutionRecord, ...]
    sim_open_ids: tuple[str, ...]


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
    ``initial_capital_of``, ``load_book``, ``load_open``, ``sim_open_ids``,
    ``strategies_of``); this never writes and never touches a model directly. The
    stored ``initial_capital`` wins for cash's seed; a scope that has never
    written falls back to *initial_capital* (``0.0`` without a config, so a
    scope read outside its own config invents no number).
    """
    book = ledger.load_book(scope)
    open_lots = tuple(
        StoreLot(
            id=str(row.conid),
            symbol=row.symbol,
            side=row.side,
            qty=row.qty,
            entry_price=row.entry_price,
            stop_loss=row.stop_loss,
            take_profit=row.take_profit,
            tag=row.tag,
            order_ref=row.order_ref,
        )
        for row in book.rows
        if row.is_open
    )
    return StoreSide(
        scope=scope,
        db_path=ledger.db_path,
        strategy_rows=ledger.strategies_of(scope),
        cash=ledger.cash_of(scope, initial_capital),
        initial_capital=ledger.initial_capital_of(scope) or initial_capital,
        lots=open_lots,
        orders=ledger.intents_of(scope),
        trades=ledger.executions_of(scope),
        sim_open_ids=tuple(sorted(ledger.sim_open_ids(scope))),
    )


async def read_sim_broker(cfg: LiveConfig, owned_ids: frozenset[str]) -> BrokerSide:
    """Read the sim/mock fixture as the broker side (no network).

    The fixture is the mock book, so a lot's ``id`` is its ``position_id`` and its
    side lives on ``Position.type`` (a fixture lot the strategy never opened is
    marked ``owned=False``, not ours). A fixture read failure is carried as a
    warning, never raised — the store side still renders.
    """
    source = MockPortfolioSource(cfg.portfolio_path)
    fetched = await source.fetch()
    if isinstance(fetched, Err):
        error = cast("FeedError", fetched.error)
        return BrokerSide(
            adapter="sim",
            source=cfg.portfolio_path,
            account="",
            net_liquidation=None,
            cash=None,
            positions=(),
            working_orders=(),
            ours_orders=(),
            warnings=(error.message,),
        )
    portfolio = fetched.value.portfolio
    lots = tuple(
        BrokerLot(
            symbol=pos.symbol,
            id=pos.position_id,
            qty=pos.qty,
            side=pos.type.value,
            avg_cost=pos.entry_price,
            last_price=pos.last_price,
            market_value=pos.qty * pos.last_price,
            owned=pos.position_id in owned_ids,
        )
        for pos in _sim_positions(portfolio)
    )
    return BrokerSide(
        adapter="sim",
        source=cfg.portfolio_path,
        account="",
        net_liquidation=None,
        cash=portfolio.cash,
        positions=lots,
        working_orders=(),
        ours_orders=(),
        warnings=(),
    )


def _sim_positions(portfolio: PortfolioState) -> tuple[Position, ...]:
    """Flatten a ``PortfolioState``'s per-symbol lot tuples in symbol order."""
    return tuple(
        pos for sym in sorted(portfolio.positions) for pos in portfolio.positions[sym]
    )


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
    """
    summary, summary_warnings = parse_summary(
        await client.portfolio_summary(account), account
    )
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
        warnings=summary_warnings + position_warnings,
    )


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


def render_pf(report: PfReport, fmt: str) -> str:
    """Deterministic text tables (default) or a JSON document for one pf read."""
    if fmt == "json":
        return _render_json(report)
    return _render_text(report)


_POSITION_COLS = (
    Col("symbol"),
    Col("id"),
    Col("side"),
    Col("qty", ">"),
    Col("avg", ">"),
    Col("last", ">"),
    Col("mktvalue", ">"),
    Col("owner"),
)
_ORDER_COLS = (
    Col("symbol"),
    Col("side"),
    Col("order_id"),
    Col("filled", ">"),
    Col("status"),
    Col("ref"),
    Col("owner"),
)
_STORE_COLS = (
    Col("scope"),
    Col("cash", ">"),
    Col("initial", ">"),
    Col("pos", ">"),
    Col("ord", ">"),
    Col("trd", ">"),
    Col("sim", ">"),
    Col("rev", ">"),
    Col("id"),
    Col("name"),
    Col("mode"),
    Col("created"),
    Col("last_cycle"),
)
_LOT_COLS = (
    Col("scope"),
    Col("symbol"),
    Col("id"),
    Col("side"),
    Col("qty", ">"),
    Col("entry", ">"),
    Col("sl", ">"),
    Col("tp", ">"),
    Col("tag"),
    Col("ref"),
)
_INTENT_COLS = (
    Col("scope"),
    Col("symbol"),
    Col("action"),
    Col("state"),
    Col("attempt", ">"),
    Col("order_id"),
    Col("ref"),
    Col("stuck", ">"),
)
_TRADE_COLS = (
    Col("scope"),
    Col("ts"),
    Col("symbol"),
    Col("conid"),
    Col("side"),
    Col("qty", ">"),
    Col("price", ">"),
    Col("comm", ">"),
    Col("cash", ">"),
)


def _render_json(report: PfReport) -> str:
    """JSON at the edge — reuse the shared encoder for Timestamps/Enums."""
    doc = {
        "as_of": report.as_of,
        "broker": asdict(report.broker) if report.broker is not None else None,
        "stores": [_store_dict(store) for store in report.stores],
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
        "sim_open_ids": list(store.sim_open_ids),
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


def _render_text(report: PfReport) -> str:
    """Every side as one aggregated table (``src.bt.table``), no scalar run-on lines.

    All scopes fold into a SINGLE stores table and SINGLE lots / orders / trades
    tables (each row carries its ``scope``), so a report over every scope stays
    the same height as one over a single config. The broker side, when read, is
    its own table block.
    """
    blocks: list[list[str]] = [[f"as_of: {report.as_of}"]]
    if report.stores:
        blocks.append(_store_block(report.stores))
    if report.broker is not None:
        blocks.append(_broker_lines(report.broker))
    blocks.append(_table("lots", _LOT_COLS, _lot_rows(report.stores)))
    blocks.append(_table("orders", _INTENT_COLS, _intent_rows(report.stores)))
    blocks.append(_table("trades", _TRADE_COLS, _trade_rows(report.stores)))
    divergence = _divergence(report)
    if report.broker is None:
        blocks.append(["divergence: - (broker not read)"])
    else:
        blocks.append(
            [f"divergence: {item}" for item in divergence] or ["divergence: none"]
        )
    return "\n\n".join("\n".join(block) for block in blocks)


def _table(
    title: str, columns: tuple[Col, ...], rows: tuple[tuple[str, ...], ...]
) -> list[str]:
    """A titled table block, or ``"<title>: none"`` when there is nothing to show."""
    if not rows:
        return [f"{title}: none"]
    return [f"{title}:", *render(Table(columns=columns, rows=rows))]


def _broker_lines(broker: BrokerSide) -> list[str]:
    """The broker block: one scalar line, then the positions and orders tables."""
    scalars = [f"adapter={broker.adapter}", f"source={broker.source or '-'}"]
    if broker.account:
        scalars.append(f"account={broker.account}")
    if broker.net_liquidation is not None:
        scalars.append(f"net_liq={broker.net_liquidation:.2f}")
    if broker.cash is not None:
        scalars.append(f"cash={broker.cash:.2f}")
    lines = ["broker: " + "  ".join(scalars)]
    lines.extend(
        _table(
            "positions",
            _POSITION_COLS,
            tuple(
                (
                    lot.symbol,
                    lot.id,
                    lot.side,
                    f"{lot.qty:g}",
                    f"{lot.avg_cost:.4f}",
                    f"{lot.last_price:.4f}",
                    f"{lot.market_value:.2f}",
                    "ours" if lot.owned else "NOT-OURS",
                )
                for lot in broker.positions
            ),
        )
    )
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
        )
    )
    lines.extend(f"warning: {w}" for w in broker.warnings)
    return lines


def _store_block(stores: tuple[StoreSide, ...]) -> list[str]:
    """One ``stores`` table over every scope, plus the single shared ``db`` line.

    The audit, cash and ownership scalars are columns, not a run-on ``key=value``
    header, so scopes line up and the report adds a row — never a paragraph — per
    scope. ``id`` is the config hash truncated for display; ``rev`` is the audit
    row count, so the truncated id still reads as the latest revision.
    """
    lines = ["stores:"]
    lines.extend(render(Table(columns=_STORE_COLS, rows=_store_rows(stores))))
    return lines


def _store_rows(stores: tuple[StoreSide, ...]) -> tuple[tuple[str, ...], ...]:
    """One row per scope: the latest audit revision's identity + the store's counts."""
    return tuple(_store_row(store) for store in stores)


def _store_row(store: StoreSide) -> tuple[str, ...]:
    """One scope's row — the latest revision read once, then its derived cells."""
    latest = _latest(store)
    return (
        store.scope,
        f"{store.cash:.2f}",
        f"{store.initial_capital:.2f}",
        str(len(store.lots)),
        str(len(store.orders)),
        str(len(store.trades)),
        str(len(store.sim_open_ids)),
        str(len(store.strategy_rows)),
        latest.strategy_id[:12] if latest else "-",
        latest.name if latest else "-",
        latest.mode if latest else "-",
        _ts(latest.created_at if latest else None),
        _ts(latest.last_cycle_at if latest else None),
    )


def _latest(store: StoreSide) -> StrategyAudit | None:
    """The scope's newest audit revision, or ``None`` when it has never written."""
    return store.strategy_rows[-1] if store.strategy_rows else None


def _ts(value: pd.Timestamp | None) -> str:
    """A UTC timestamp at second precision (``-`` for an unwritten column)."""
    return "-" if value is None else value.strftime("%Y-%m-%d %H:%M:%S")


def _lot_rows(stores: tuple[StoreSide, ...]) -> tuple[tuple[str, ...], ...]:
    """Every scope's open lots in one table, each row tagged with its scope."""
    return tuple(
        (
            store.scope,
            lot.symbol,
            lot.id,
            lot.side,
            f"{lot.qty:g}",
            f"{lot.entry_price:.4f}",
            _opt(lot.stop_loss),
            _opt(lot.take_profit),
            lot.tag or "-",
            lot.order_ref or "-",
        )
        for store in stores
        for lot in store.lots
    )


def _intent_rows(stores: tuple[StoreSide, ...]) -> tuple[tuple[str, ...], ...]:
    """Every scope's order intents in one table, newest first."""
    return tuple(
        (
            store.scope,
            record.key.symbol,
            record.key.action.value,
            record.state.value,
            str(record.attempt),
            record.order_id or "-",
            record.order_ref,
            str(record.stuck_cycles),
        )
        for store in stores
        for record in store.orders
    )


def _trade_rows(stores: tuple[StoreSide, ...]) -> tuple[tuple[str, ...], ...]:
    """Every scope's stored fills in one table, oldest first."""
    return tuple(
        (
            store.scope,
            _ts(trade.ts),
            trade.symbol or "-",
            str(trade.conid),
            trade.side,
            f"{trade.qty:g}",
            f"{trade.price:.4f}",
            f"{trade.commission:.2f}",
            f"{trade.cash_delta:.2f}",
        )
        for store in stores
        for trade in store.trades
    )


def _opt(value: float | None) -> str:
    """A price cell at the table's precision, or ``-`` when the lot has none."""
    return "-" if value is None else f"{value:.4f}"


def _divergence(report: PfReport) -> tuple[str, ...]:
    """Broker lots our store does not own, and store lots the broker does not show.

    The two mismatches a human acts on: a broker lot we do not record (a position
    our store does not claim) and a store lot the broker does not show (a book row
    with no matching broker position). Ownership is the UNION of the scope's book
    rows (conid space) and its sim lots (``position_id`` space) — the two adapters
    key their lots differently, and a sim scope has NO book rows at all.

    Compares the broker against the UNION of every scope in ``report.stores``:
    with no config the report covers them all, and a broker side is only ever
    built alongside those stores. Ownership is the union of each scope's book
    rows (conid space) and its sim lots (``position_id`` space) — the two
    adapters key their lots differently, and a sim scope has NO book rows at all.
    Without a broker read there is nothing to compare against and the list is
    empty.
    """
    broker = report.broker
    if broker is None or not report.stores:
        return ()
    broker_ids = {lot.id for lot in broker.positions}
    store_ids = {lot.id for store in report.stores for lot in store.lots} | {
        i for store in report.stores for i in store.sim_open_ids
    }
    ours_unmatched = tuple(
        f"broker lot {lot.id} {lot.symbol} not in our store"
        for lot in broker.positions
        if lot.id not in store_ids
    )
    store_unmatched = tuple(
        f"store lot {lot.id} {lot.symbol} not at the broker"
        for store in report.stores
        for lot in store.lots
        if lot.id not in broker_ids
    )
    return ours_unmatched + store_unmatched


__all__ = [
    "BrokerLot",
    "BrokerSide",
    "PfReport",
    "StoreLot",
    "StoreSide",
    "read_ibkr_broker",
    "read_sim_broker",
    "read_store",
    "render_pf",
]
