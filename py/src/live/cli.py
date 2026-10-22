"""Live CLI — ``ibkr live run <config.json>``: one batch reconcile cycle.

The caller (cron) drives cadence; this runs exactly ONE cycle. ``--dry-run``
reconciles and reports without placing anything. Everything the cycle reads
(config, adapter) is resolved here; the engine owns the pure core.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TextIO

import click
import pandas as pd

from src.config import read_live_config
from src.shared.style import COLOR, PLAIN, Styler
from src.data.ibkr.client import IbkrClient, IbkrError
from src.data.ibkr.gateway import IbkrGateway
from src.live.pipeline import PortfolioFetchError
from src.live.freshness import StaleDataError
from src.live.lease import CycleInProgressError
from src.live.ledger import LedgerReadError, SqliteLedger
from src.live.options import (
    LiveOptions,
    adapter_names,
    format_names,
    merge_live_options,
)
from src.live.adapters.sim.store import SimBookStore, resolve_sim_book_path
from src.live.pf import (
    BrokerSide,
    PfReport,
    StoreSide,
    read_ibkr_broker,
    read_sim_broker,
    read_store,
    render_pf,
)
from src.live.scope import (
    AdapterName,
    ScopeParts,
    config_name_of,
    config_hash as config_scope_hash,
    scope_of,
)

# The cycle orchestration lives in ``src.live.run_once`` (plan §3): ``src/bot``
# imports it directly, so this module is a CONSUMER, never the implementation.
# Importing the names here keeps ONE definition for ``live run``/``live pf``
# and their tests, rather than a second copy that would drift.
from src.live.run_once import (
    _read_json,
    load_strat_config,
    run_once,
)
from src.live.types import FeedError, LiveConfig

#: Adapters ``--adapter`` accepts, DERIVED from the ONE typing definition
#: (``scope.AdapterName``): a second literal list would let a name be accepted by
#: one layer and rejected by another.
_ADAPTERS = adapter_names()

#: Report formats ``--format`` accepts, derived the same way.
_FORMATS = format_names()

#: A sleep seam so a watch loop's cadence is injectable in tests (never slept
#: through the wall clock). ``time.sleep`` is the production default.
SleepFn = Callable[[float], None]


class SessionNotReady(RuntimeError):
    """The broker session could not be reached, or may not trade the account."""

    def __init__(self, error: FeedError) -> None:
        super().__init__(error.message)
        self.error = error


@click.group(name="live")
def live_group() -> None:
    """Live trading — one-shot reconcile cycles (cron it)."""


@click.command("run")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--dry-run",
    "dry_run",
    flag_value=True,
    default=None,
    help="Reconcile + report; place nothing. Unset: `[live] dry_run`, then false.",
)
@click.option(
    "--max-age",
    "-a",
    type=int,
    default=None,
    help="Staleness gate in days. Unset: `[live] max_age`, then 5.",
)
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(_FORMATS),
    default=None,
    help="Report format. Unset: `[live] format`, then `text`.",
)
@click.option(
    "--adapter",
    type=click.Choice(_ADAPTERS),
    default=None,
    help="Broker adapter to run through. Unset: `[live] adapter`, then `ibkr`.",
)
@click.option(
    "--allow-unsafe",
    is_flag=True,
    help=(
        "Exit 0 even when the cycle is unsafe (placement/resync error, a book "
        "divergence, or an unresolved/wedged/timed-out order). For callers that "
        "consume the report themselves; the default exits non-zero so cron can "
        "see it. DELIBERATELY flag-only: it is never a `[live]` setting."
    ),
)
@click.option(
    "--allow-oth",
    "allow_oth",
    flag_value=True,
    default=None,
    help=(
        "Allow placing BUY/SELL MKT orders outside NYSE trading hours "
        "(Mon-Fri 9:30-16:00 ET). Unset: `[live] allow_oth`, then refuse and warn."
    ),
)
def live_run(
    config_path: str,
    dry_run: bool | None,
    max_age: int | None,
    fmt: str | None,
    adapter: str | None,
    allow_unsafe: bool,
    allow_oth: bool | None,
) -> None:
    """Run ONE live cycle (cron-friendly). --dry-run reconciles without placing.

    Every knob except ``--allow-unsafe`` may also come from ``config.toml``'s
    ``[live]`` table; a flag always wins over the file.
    """
    try:
        # The merge is INSIDE the guard: a typo in ``[live] adapter`` is a config
        # error and must map to exit 1, not a traceback on a cron line.
        options = merge_live_options(
            adapter,
            allow_oth,
            read_live_config(),
            LiveOptions(),
            max_age=max_age,
            dry_run=dry_run,
            fmt=fmt,
        )
        code = run_once(
            config_path,
            options=options,
            allow_unsafe=allow_unsafe,
        )
    except (
        ValueError,
        StaleDataError,
        PortfolioFetchError,
        CycleInProgressError,
        LedgerReadError,
    ) as exc:
        # An unsized open (ValueError by default policy) and the typed cycle
        # failures are config/environment problems, not a traceback: they map to
        # the CLI's exit 1 rather than crashing cron.
        raise click.ClickException(str(exc)) from exc
    if code:
        raise click.exceptions.Exit(code)


live_group.add_command(live_run)


@click.command("pf")
@click.argument(
    "config_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(_FORMATS),
    default=None,
    help="Report format. Unset: `[live] format`, then `text`.",
)
@click.option(
    "--adapter",
    type=click.Choice(_ADAPTERS),
    default=None,
    help="Broker to read. Unset: `[live] adapter`. The read REUSES the running "
    "gateway session (`ibkr gw` owns login) — it never starts one.",
)
@click.option(
    "--watch",
    "watch_seconds",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help=(
        "Refresh the report every SECONDS until Ctrl-C (a polling view). "
        "Absent = one-shot, exactly as before."
    ),
)
def live_pf(
    config_path: str | None,
    fmt: str | None,
    adapter: str | None,
    watch_seconds: float | None,
) -> None:
    """Show scopes, their P&L and one merged positions table, plus the broker.

    With CONFIG_PATH the report covers that config's scope. WITHOUT it, EVERY
    scope the store knows about. Either way each scope renders its own lots,
    order intents and stored fills — MERGED into one ``positions`` row per
    symbol (lot, newest order and the P&L its fills imply), so the same order
    ref no longer appears in three tables; a ``stats`` table carries the
    headline P&L (realized / unrealized / total, cost basis, win-loss tally).

    ``--adapter`` and ``--format`` follow the SAME precedence ``live run`` uses:
    flag, then ``config.toml [live]``, then the code default. A broker read is
    not required for that: ``pf`` only ever READS an existing gateway session
    (``ibkr gw`` owns login and keepalive), so resolving ``ibkr`` here can never
    start one — it fails cleanly if no session is live.

    ``--watch SECONDS`` turns the one-shot report into a polling view: the store
    and broker are re-read and re-rendered every SECONDS until Ctrl-C (exit 0).
    The ledger is constructed ONCE for the whole session, so no tick runs DDL,
    takes a lease or re-gates. On a TTY the screen is a flicker-free alternate
    screen, restored on exit; off a TTY (a pipe or a cron log) each frame is
    appended after a separator, so ``>> log`` stays readable and NOT ONE escape
    byte is written. ``--watch`` with ``--format json`` is refused: a machine
    format has no use for an endless refresh.

    Nothing is placed and nothing is written: no lease is taken, no strategy row
    is ensured and no DDL runs (a read of an unwritten db is an empty store, not a
    creation). The exit code is 0 (report printed), 1 (a config/gateway/read
    failure) or 2 (usage) — there is no unsafe path here.
    """
    # Resolved BEFORE the ledger so a ``[live]`` typo is exit 1, not a traceback:
    # one merge feeds the report format AND the adapter the broker read uses.
    try:
        options = merge_live_options(
            adapter, None, read_live_config(), LiveOptions(), fmt=fmt
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    fmt = options.format
    adapter = options.adapter

    if watch_seconds is not None and fmt == "json":
        raise click.ClickException("--watch cannot be combined with --format json")
    ledger = SqliteLedger()
    if watch_seconds is not None:
        _watch_pf(ledger, config_path, adapter, watch_seconds)
        return
    try:
        report = (
            _all_scopes_report(ledger, adapter)
            if config_path is None
            else _config_report(ledger, config_path, adapter)
        )
    except (SessionNotReady, ValueError, LedgerReadError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(render_pf(report, fmt, _styler(fmt, sys.stdout.isatty())))


def _all_scopes_report(
    ledger: SqliteLedger,
    adapter: AdapterName | None,
) -> PfReport:
    """Every scope in the store, plus a broker read when an adapter is resolved."""
    scopes = ledger.scopes_of_store()
    stores = tuple(read_store(ledger, scope) for scope in scopes)
    broker = _pf_broker(ledger, adapter, None, None, stores)
    return PfReport(as_of=pd.Timestamp.now(tz="UTC"), stores=stores, broker=broker)


def _config_report(
    ledger: SqliteLedger,
    config_path: str,
    adapter: str | None,
) -> PfReport:
    """The config's scope, plus a broker read when one is resolved.

    ONE resolution feeds both halves: the scope minted from ``[live] adapter``
    and the broker actually read, so a report can never show a different backend
    than the one it addresses (``pf`` after ``run --adapter sim`` reads the sim
    scope AND the sim book).
    """
    cfg = load_strat_config(config_path)
    raw = _read_json(config_path)
    resolved = merge_live_options(
        adapter, None, read_live_config(), LiveOptions()
    ).adapter
    store = read_store(
        ledger, _config_scope(cfg, resolved), initial_capital=cfg.initial_capital
    )
    broker = _pf_broker(
        ledger,
        resolved,
        raw,
        cfg,
        (store,),
    )
    return PfReport(as_of=pd.Timestamp.now(tz="UTC"), stores=(store,), broker=broker)


def _config_scope(cfg: LiveConfig, resolved_adapter: AdapterName) -> str:
    """The scope a config addresses: ``<adapter>_<config_name>_<config_hash>``.

    The SAME mint ``live run`` uses, so ``pf``/``status`` read the scope a cycle
    writes. *resolved_adapter* is the caller's merge result, so the CLI
    ``--adapter`` flag selects the scope exactly as it does a run.
    """
    return scope_of(
        ScopeParts(resolved_adapter, config_name_of(cfg), config_scope_hash(cfg))
    )


def _pf_broker(
    ledger: SqliteLedger,
    adapter: AdapterName | None,
    raw: Mapping[str, object] | None,
    cfg: LiveConfig | None,
    stores: tuple[StoreSide, ...],
) -> BrokerSide | None:
    """Read the broker *adapter* names, or nothing when there is none.

    *adapter* is ALREADY resolved by the caller (flag > ``[live] adapter`` >
    default); this function never re-reads a flag or the file. ``None`` means
    store-only, which the watch path still uses for a config that names no
    broker. A ``sim`` read is the store's own account book
    (``read_sim_broker``) — no fixture, no gateway. An ``ibkr`` read REUSES the
    session ``ibkr gw`` owns: it resolves the account and reads the book, and
    never logs in.
    """
    resolved = adapter
    if resolved is None:
        return None
    if resolved == "sim":
        scope = stores[0].scope if stores else ""
        return read_sim_broker(
            SimBookStore(resolve_sim_book_path()),
            scope,
            ledger.owned_ids(scope),
        )
    owned = frozenset(lot.id for store in stores for lot in store.lots)
    return asyncio.run(
        _read_ibkr_pf(
            tuple(store.scope for store in stores),
            owned,
        )
    )


async def _open_ibkr_session() -> tuple[IbkrGateway, str]:
    """Attach to the session ``ibkr gw`` owns and resolve its account.

    Returns the OPEN gateway handle and the resolved account; the caller owns
    ``aclose``. This NEVER logs in and never starts a gateway: a down or logged-out
    session is a typed ``SessionNotReady`` (exit 1), never a traceback.
    """
    gateway = IbkrGateway(IbkrClient())
    try:
        account = await gateway.client.resolve_account()
        return gateway, account
    except IbkrError as exc:
        await gateway.aclose()
        raise SessionNotReady(FeedError(kind=exc.kind, message=str(exc))) from exc
    except BaseException:
        await gateway.aclose()
        raise


async def _read_ibkr_pf(
    scopes: tuple[str, ...],
    owned: frozenset[str],
) -> BrokerSide:
    """Resolve the account on the EXISTING session, then read the IBKR book once.

    The client is ALWAYS closed in the ``finally``; nothing is placed and no lease
    is taken. A gateway that is down or logged out surfaces as ``SessionNotReady``
    (exit 1), never a traceback — ``pf`` reports, it does not start a session.
    """
    gateway, account = await _open_ibkr_session()
    try:
        return await read_ibkr_broker(gateway.client, account, scopes, owned)
    except IbkrError as exc:
        raise SessionNotReady(FeedError(kind=exc.kind, message=str(exc))) from exc
    finally:
        await gateway.aclose()


#: The alternate-screen + cursor escapes a TTY refresh uses (never a pipe's).
_ALT_SCREEN_ON = "\x1b[?1049h"
_ALT_SCREEN_OFF = "\x1b[?1049l"
_CURSOR_HIDE = "\x1b[?25l"
_CURSOR_SHOW = "\x1b[?25h"
_CLEAR_HOME = "\x1b[H\x1b[2J"
#: The rule a NON-TTY watch appends between frames, so a log stays readable.
_TICK_SEPARATOR = "-" * 72


def _styler(fmt: str, tty: bool) -> Styler:
    """ANSI styling only for a human's terminal AND the human text format.

    Both halves are required: a pipe or cron log must receive text with no escape
    byte in it (a pager or ``grep`` would otherwise see codes), and ``--format
    json`` is parsed by a machine that a colour code would only corrupt.
    """
    return COLOR if fmt == "text" and tty else PLAIN


def _watch_pf(
    ledger: SqliteLedger,
    config_path: str | None,
    adapter: AdapterName,
    interval: float,
) -> None:
    """Re-read the store + broker and re-render every *interval* until Ctrl-C.

    The one-shot plumbing runs ONCE — the ledger is reused (so no tick can DDL),
    and an ``ibkr`` gateway handle is taken before the loop and only its book is
    re-read per tick. That handle REUSES the session ``ibkr gw`` already owns; it
    never logs in. A per-tick read failure is rendered as a typed error line and
    the loop keeps going. Teardown restores the screen on every exit path.
    """
    tty = sys.stdout.isatty()
    session = _open_watch_session(ledger, config_path, adapter, _styler("text", tty))
    try:
        watch_pf_loop(
            session.frame,
            interval,
            tty=tty,
            sleeper=time.sleep,
            out=sys.stdout,
        )
    finally:
        session.close()


@dataclass(frozen=True)
class _WatchSession:
    """A watch's per-tick frame plus its teardown (closes any open gateway)."""

    frame: Callable[[], str]
    close: Callable[[], None]


def _open_watch_session(
    ledger: SqliteLedger,
    config_path: str | None,
    adapter: AdapterName,
    style: Styler = PLAIN,
) -> _WatchSession:
    """Resolve the watch's frame + teardown ONCE, before the loop starts.

    An ``ibkr`` read (with or without a config) holds ONE gateway on ONE
    persistent event loop — only its book is re-read per tick, so the connection
    pool is never rebound to a fresh loop. That gateway is a REUSED session
    (``ibkr gw`` owns login), never one this process starts. A ``sim`` read
    needs no session at all (the store is re-read per tick). No lease is taken
    and no DDL runs: the frame calls the same read helpers the one-shot path
    uses.

    *adapter* is ALREADY resolved by the caller, so the watch and the one-shot
    report always address the same backend and mint the same scope.
    """
    if config_path is None:
        if adapter == "ibkr":
            return _ibkr_watch_session(
                ledger,
                ledger.scopes_of_store,
                0.0,
            )
        return _WatchSession(
            _store_frame(ledger, None, adapter, adapter, style),
            _noop,
        )
    cfg = load_strat_config(config_path)
    if adapter != "ibkr":
        return _WatchSession(_store_frame(ledger, cfg, adapter, adapter, style), _noop)
    return _ibkr_watch_session(
        ledger,
        lambda: (_config_scope(cfg, adapter),),
        cfg.initial_capital,
        style,
    )


def _ibkr_watch_session(
    ledger: SqliteLedger,
    scopes_fn: Callable[[], tuple[str, ...]],
    initial_capital: float,
    style: Styler = PLAIN,
) -> _WatchSession:
    """Hold ONE ibkr session open on a persistent loop, then read only the book.

    The loop is held for the whole watch: an ``httpx.AsyncClient`` binds to the
    loop that first runs it, so a fresh ``asyncio.run`` per tick would rebind the
    pool. ``close`` releases the client and the loop on every exit path.
    *scopes_fn* is re-evaluated per tick, so a scope written mid-watch joins the
    read — the same coverage the one-shot path gives at its own instant.
    """
    loop = asyncio.new_event_loop()
    try:
        gateway, account = loop.run_until_complete(_open_ibkr_session())
    except BaseException:
        loop.close()
        raise

    def frame() -> str:
        scopes = scopes_fn()
        stores = tuple(
            read_store(ledger, scope, initial_capital=initial_capital)
            for scope in scopes
        )
        owned = frozenset(lot.id for store in stores for lot in store.lots)
        broker = loop.run_until_complete(
            read_ibkr_broker(gateway.client, account, scopes, owned)
        )
        return render_pf(
            PfReport(as_of=pd.Timestamp.now(tz="UTC"), stores=stores, broker=broker),
            "text",
            style,
        )

    def close() -> None:
        loop.run_until_complete(gateway.aclose())
        loop.close()

    return _WatchSession(frame, close)


def _store_frame(
    ledger: SqliteLedger,
    cfg: LiveConfig | None,
    adapter: str | None,
    resolved_adapter: AdapterName,
    style: Styler = PLAIN,
) -> Callable[[], str]:
    """A store (+ sim broker) frame builder for the scopes a watch covers.

    *adapter* is the broker to read per tick; *resolved_adapter* is the scope-
    selecting adapter (the config's own when *cfg* is None and no flag was
    given), so the two never disagree.
    """

    def frame() -> str:
        stores, broker = _store_and_broker(ledger, cfg, adapter, resolved_adapter)
        return render_pf(
            PfReport(as_of=pd.Timestamp.now(tz="UTC"), stores=stores, broker=broker),
            "text",
            style,
        )

    return frame


def _store_and_broker(
    ledger: SqliteLedger,
    cfg: LiveConfig | None,
    adapter: str | None,
    resolved_adapter: AdapterName,
) -> tuple[tuple[StoreSide, ...], BrokerSide | None]:
    """Re-read the store side and (for ``sim``) the store's account book, per tick.

    Only the store is re-read here — a non-sim broker would need a live session,
    which the watch holds open itself. Mirrors ``_all_scopes_report``/
    ``_config_report`` for the store half; a sim read is the same store read
    (``read_sim_broker``), so a watch never touches a fixture.
    """
    if cfg is None:
        stores = tuple(read_store(ledger, scope) for scope in ledger.scopes_of_store())
    else:
        scope = _config_scope(cfg, resolved_adapter)
        stores = (read_store(ledger, scope, initial_capital=cfg.initial_capital),)
    if adapter != "sim":
        return stores, None
    scope = stores[0].scope if stores else ""
    return stores, read_sim_broker(
        SimBookStore(resolve_sim_book_path()), scope, ledger.owned_ids(scope)
    )


def watch_pf_loop(
    frame: Callable[[], str],
    interval: float,
    *,
    tty: bool,
    out: TextIO,
    sleeper: SleepFn = time.sleep,
) -> None:
    """Paint *frame* every *interval* seconds until Ctrl-C, then exit 0.

    On a TTY the frame is drawn on the alternate screen with the cursor hidden,
    so a refresh never flickers into scrollback; OFF a TTY each frame is appended
    after a separator rule and NOT ONE escape byte is written. A failing frame is
    rendered as an ``error:`` line — a watch outlives a transient read failure.
    The screen and cursor are restored in ``finally``, so Ctrl-C (and any exit
    path) leaves the terminal as it was found. *sleeper* is the cadence seam.
    """
    first = True
    try:
        if tty:
            out.write(_ALT_SCREEN_ON + _CURSOR_HIDE)
            out.flush()
        while True:
            _paint(_safe_frame(frame), tty=tty, first=first, out=out)
            first = False
            sleeper(interval)
    except KeyboardInterrupt:
        return
    finally:
        if tty:
            out.write(_CURSOR_SHOW + _ALT_SCREEN_OFF)
        out.flush()


def _paint(text: str, *, tty: bool, first: bool, out: TextIO) -> None:
    """Draw one frame: a cleared alternate screen, or an appended rule + frame."""
    if tty:
        out.write(_CLEAR_HOME + text + "\n")
    else:
        prefix = "" if first else _TICK_SEPARATOR + "\n"
        out.write(prefix + text + "\n")
    out.flush()


def _safe_frame(frame: Callable[[], str]) -> str:
    """One tick's frame; a per-tick read failure is a typed error line, not a stop."""
    try:
        return frame()
    except Exception as exc:
        return f"error: {exc}"


def _noop() -> None:
    """A teardown that owns nothing (a store-only or sim watch has no session)."""
    return None


live_group.add_command(live_pf)
