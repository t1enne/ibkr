"""Live CLI — ``ibkr live run <config.json>``: one batch reconcile cycle.

The caller (cron) drives cadence; this runs exactly ONE cycle. ``--dry-run``
reconciles and reports without placing anything. Everything the cycle reads
(config, adapter) is resolved here; the engine owns the pure core.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, TextIO, cast

import click
import pandas as pd
import peewee

from src.bt import load_strategy
from src.bt.cmds._shared import _json_default
from src.shared.style import COLOR, PLAIN, Styler
from src.bt.table import Col, Table, render as render_table
from src.bt.state import ActionType, PortfolioState
from src.bt.types import StrategyConfig
from src.config import live_adapter
from src.data.ibkr.client import IbkrClient, IbkrError
from src.data.ibkr.gateway import IbkrGateway
from src.live.adapter import LiveAdapter, resolve_adapter, resolve_adapter_name
from src.live.pure import OrderResult
from src.live.engine import (
    CycleReport,
    PortfolioFetchError,
    StaleDataError,
    run_cycle,
    unsafe_outcomes,
)
from src.live.lease import CycleInProgressError
from src.live.identity import OPEN_STATES, IntentKey, IntentState
from src.live.ledger import LedgerReadError, SqliteLedger, config_hash
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
from src.live.divergence import Divergence
from src.live.types import (
    FeedError,
    LiveConfig,
    cost_provenance,
)

#: Sizing modes the shared ``SizingParams`` layer accepts.
SizeMode = Literal["equity", "cash", "fixed"]
_SIZE_MODES = frozenset({"equity", "cash", "fixed"})
#: The keys ``StrategyConfig`` itself defines — everything else is a live-only key.
_STRATEGY_FIELDS = frozenset(f.name for f in fields(StrategyConfig))
#: Adapters ``--adapter`` accepts. Phase 2 ships ``ibkr`` read-only.
_ADAPTERS = ("sim", "ibkr")

#: A sleep seam so a watch loop's cadence is injectable in tests (never slept
#: through the wall clock). ``time.sleep`` is the production default.
SleepFn = Callable[[float], None]

#: The exit code for an UNSAFE cycle (see ``CycleReport.is_unsafe``). Distinct
#: from click's ``1`` (ClickException — config/stale-data/gateway failures) and
#: ``2`` (UsageError), so cron can tell "the broker may be holding something we
#: cannot see" apart from "the run could not start".
_UNSAFE_EXIT_CODE = 3


def _stderr_log(message: str) -> None:
    """Route the IBKR edge's diagnostics to stderr, never the report's stdout.

    A scaled cohort and a cash-refused open are operator notices: they belong in
    the cron log beside the run, not interleaved with the report the next
    consumer parses off stdout.
    """
    click.echo(message, err=True)


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
@click.option("--dry-run", is_flag=True, help="Reconcile + report; place nothing.")
@click.option("--max-age", "-a", type=int, default=5, show_default=True)
@click.option(
    "--format", "-F", "fmt", type=click.Choice(["text", "json"]), default="text"
)
@click.option(
    "--adapter",
    type=click.Choice(_ADAPTERS),
    default=None,
    help="Broker adapter to run through. Unset, the config decides (its `adapter` "
    "key, then the back-compat `broker` key), defaulting to `ibkr`.",
)
@click.option(
    "--allow-unsafe",
    is_flag=True,
    help=(
        "Exit 0 even when the cycle is unsafe (placement/resync error, a book "
        "divergence, or an unresolved/wedged/timed-out order). For callers that "
        "consume the report themselves; the default exits non-zero so cron can "
        "see it."
    ),
)
def live_run(
    config_path: str,
    dry_run: bool,
    max_age: int,
    fmt: str,
    adapter: str | None,
    allow_unsafe: bool,
) -> None:
    """Run ONE live cycle (cron-friendly). --dry-run reconciles without placing."""
    cfg = load_live_config(config_path)
    raw = _read_json(config_path)
    try:
        resolved = resolve_adapter_name(adapter, raw, cfg.strategy_params)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    # Audit key vs ownership key (plan §1.4): the config hash identifies the
    # revision that wrote the scope; the scope keys the book, the cOID prefix and
    # the lease. The scope is minted ONCE here from the ORIGINAL raw config, never
    # the strategy-only temp projection (a random path must not change identity).
    strategy_id = config_hash(raw)
    scope = scope_of(ScopeParts(resolved, config_name_of(cfg), config_scope_hash(cfg)))
    strategy = _strategy_config(config_path, raw)
    ledger = SqliteLedger()
    if not dry_run:  # a dry run writes nothing (not even the strategy row)
        ledger.ensure_strategy(strategy_id, scope, strategy.name)
        ledger.ensure_cash(scope, cfg.initial_capital)
    gateway: IbkrGateway | None = None
    client: IbkrClient | None = None
    if resolved == "ibkr":
        gateway = IbkrGateway(IbkrClient())
        client = gateway.client
    # The adapter is built from the resolved name only: no branch on the
    # backend's internals survives here, so a third adapter needs no edit.
    backend = resolve_adapter(
        cfg, resolved, scope, ledger, dry_run, _stderr_log, client=client
    )
    # Plan §7.3: label which source produced this run's costs. A resolved IBKR run
    # books the broker's exact per-execution commission but still SIZES on the sim
    # model — the report states both, so the mix is never ambiguous.
    cost = cost_provenance(resolved)
    try:
        # The live file is a strategy config PLUS live-only keys; the screen
        # bridge (``load_strategy``) is strict and rejects those extras, so it
        # gets a strategy-only projection written to a temp dir and cleaned up
        # on exit.
        with tempfile.TemporaryDirectory(prefix="ibkr-live-") as tmp:
            normalized_path = _write_strategy_config(strategy, tmp)
            report = replace(
                asyncio.run(
                    _run_cycle(
                        cfg,
                        adapter=backend,
                        ledger=ledger,
                        strategy_id=strategy_id,
                        scope=scope,
                        config_path=normalized_path,
                        max_age_days=max_age,
                        dry_run=dry_run,
                        gateway=gateway,
                    )
                ),
                cost=cost,
            )
    except (
        StaleDataError,
        PortfolioFetchError,
        CycleInProgressError,
        LedgerReadError,
        ValueError,
    ) as exc:
        # ValueError: an unsized open raises by default policy — a traceback is
        # not a CLI contract.
        raise click.ClickException(str(exc)) from exc
    click.echo(render_report(report, fmt, scope))
    _housekeeping(ledger, dry_run)
    # The report already prints placement_error/resync_error/the unsafe outcomes,
    # so the exit code is the machine-readable signal — never a duplicated message.
    # stdout stays the parseable contract (a JSON document for ``--format json``);
    # the exit code is out-of-band. Checked AFTER the housekeeping so a prune
    # failure cannot change the verdict.
    if report.is_unsafe():
        if not allow_unsafe:
            raise click.exceptions.Exit(_UNSAFE_EXIT_CODE)
        # D2: ``--allow-unsafe`` forces exit 0, so the ONLY trace of an unsafe
        # cycle is this note. It must name what it suppressed and be derived from
        # the report (never a fixed string), or one cron line silently neutralises
        # every placement/resync/wedge/timeout/divergence alert.
        _stderr_log(
            f"--allow-unsafe: exit 0 forced for an UNSAFE cycle; suppressed "
            f"outcomes: {', '.join(unsafe_outcomes(report))}"
        )


async def _run_cycle(
    cfg: LiveConfig,
    *,
    adapter: LiveAdapter,
    ledger: SqliteLedger,
    strategy_id: str,
    scope: str,
    config_path: str,
    max_age_days: int,
    dry_run: bool,
    gateway: IbkrGateway | None,
) -> CycleReport:
    """Run the cycle, closing the gateway client when it is done.

    Reachability, login and session freshness are ``ibkr gw``'s job: this only
    owns the client's lifetime (the portfolio source shares it, so it outlives
    the read and is released here rather than leaked).
    """
    try:
        return await run_cycle(
            cfg,
            adapter,
            ledger=ledger,
            strategy_id=strategy_id,
            scope=scope,
            config_path=config_path,
            max_age_days=max_age_days,
            dry_run=dry_run,
        )
    finally:
        # The client is shared with the portfolio source and outlives the read;
        # close it when the cycle is done rather than leaking the pool.
        if gateway is not None:
            await gateway.aclose()


def resolve_broker(
    raw: Mapping[str, object], params: Mapping[str, object]
) -> tuple[str, bool]:
    """The ONE read of the config's ``broker`` key: ``(name, was_named_explicitly)``.

    ``StrategyConfig.broker``, ``LiveConfig.broker`` and ``live pf``'s probe
    previously resolved the key in three places and disagreed when it lived in
    ``strategy_params``. They now all go through this: flat/top-level wins, then
    ``strategy_params``, then ``config.toml``'s ``[live] adapter``. The
    explicit-named flag comes from the same scan, so it can never contradict the
    resolved name.

    ``live run`` no longer consults this for adapter SELECTION (its ``--adapter``
    always carries a value, ``ibkr`` by default); it is the description of the
    config, used by ``live pf`` and by the sim broker's own construction.
    """
    for source in (raw, params):
        if "broker" in source:
            value = source["broker"]
            if value not in _ADAPTERS:
                raise ValueError(
                    f"broker must be one of {sorted(_ADAPTERS)}, got {value!r}"
                )
            return cast("str", value), True
    return live_adapter(), False


live_group.add_command(live_run)


@click.command("abandon")
@click.option("--scope", required=True, help="Ownership scope of the wedged key.")
@click.option("--symbol", required=True, help="Symbol of the wedged key.")
@click.option("--action", type=click.Choice(["long", "short", "close"]), required=True)
@click.option("--position-id", default=None, help="Target lot for a close key.")
@click.option("--yes", "confirmed", is_flag=True, help="Acknowledge the warning.")
def live_abandon(
    scope: str, symbol: str, action: str, position_id: str | None, confirmed: bool
) -> None:
    """Clear ONE wedged OPEN intent key so the next cycle may re-mint it.

    WARNING (irreversible): this clears only OUR durable record. It does NOT
    cancel anything at the broker. If an order for the key is still live
    broker-side, the next cycle will place a DUPLICATE. Verify broker-side FIRST
    (the order is cancelled/expired), then pass --yes.
    """
    if not confirmed:
        raise click.UsageError(
            "abandon clears our durable record only and does NOT cancel the broker "
            "order; if it is still live the next cycle duplicates it. Verify "
            "broker-side, then re-run with --yes."
        )
    ledger = SqliteLedger()
    key = IntentKey(
        scope=scope,
        symbol=symbol,
        action=ActionType(action),
        position_id=position_id,
    )
    record = ledger.load(key)
    if record is None:
        raise click.ClickException(f"no intent record for {scope}/{symbol}/{action}")
    if record.state not in OPEN_STATES:
        click.echo(f"key {scope}/{symbol}/{action} is already {record.state.value}")
        return
    ledger.close(key, IntentState.UNFILLED, record.order_id, pd.Timestamp.now(tz="UTC"))
    click.echo(
        f"abandoned {scope}/{symbol}/{action}: {record.state.value} -> unfilled "
        f"(order_id={record.order_id or 'unknown'}) — next cycle may re-mint"
    )


live_group.add_command(live_abandon)


@click.command("pf")
@click.argument(
    "config_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--format", "-F", "fmt", type=click.Choice(["text", "json"]), default="text"
)
@click.option(
    "--adapter",
    type=click.Choice(_ADAPTERS),
    default=None,
    help="Broker to read; a filter over which broker side is included. "
    "Absent = store-only (no broker read).",
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
    fmt: str,
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

    ``--adapter`` is a FILTER, not a requirement: given, it selects the broker
    to read (``ibkr`` account-wide, or the ``sim`` book straight from the store)
    and the report includes the broker block + divergence; absent, the report is
    store-only and no broker is touched.

    ``--watch SECONDS`` turns the one-shot report into a polling view: the store
    and broker are re-read and re-rendered every SECONDS until Ctrl-C (exit 0).
    The gateway is authenticated and the ledger constructed ONCE for the whole
    session, so no tick runs DDL, takes a lease or re-gates. On a TTY the screen
    is a flicker-free alternate screen, restored on exit; off a TTY (a pipe or a
    cron log) each frame is appended after a separator, so ``>> log`` stays
    readable and NOT ONE escape byte is written. ``--watch`` with ``--format
    json`` is refused: a machine format has no use for an endless refresh.

    Nothing is placed and nothing is written: no lease is taken, no strategy row
    is ensured and no DDL runs (a read of an unwritten db is an empty store, not a
    creation). The exit code is 0 (report printed), 1 (a config/gateway/read
    failure) or 2 (usage) — there is no unsafe path here.
    """
    if watch_seconds is not None and fmt == "json":
        raise click.ClickException("--watch cannot be combined with --format json")
    ledger = SqliteLedger()
    if watch_seconds is not None:
        _watch_pf(ledger, config_path, adapter, watch_seconds)
        return
    try:
        report = (
            _all_scopes_report(
                ledger,
                adapter,
            )
            if config_path is None
            else _config_report(
                ledger,
                config_path,
                adapter,
            )
        )
    except (SessionNotReady, ValueError, LedgerReadError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(render_pf(report, fmt, _styler(fmt, sys.stdout.isatty())))


def _all_scopes_report(
    ledger: SqliteLedger,
    adapter: str | None,
) -> PfReport:
    """Every scope in the store, plus a broker read when ``--adapter`` names one."""
    scopes = ledger.scopes_of_store()
    stores = tuple(read_store(ledger, scope) for scope in scopes)
    broker = _pf_broker(ledger, adapter, None, None, stores)
    return PfReport(as_of=pd.Timestamp.now(tz="UTC"), stores=stores, broker=broker)


def _config_report(
    ledger: SqliteLedger,
    config_path: str,
    adapter: str | None,
) -> PfReport:
    """The config's scope, plus a broker read when one is resolved."""
    cfg = load_live_config(config_path)
    raw = _read_json(config_path)
    store = read_store(ledger, _config_scope(cfg), initial_capital=cfg.initial_capital)
    broker = _pf_broker(
        ledger,
        adapter,
        raw,
        cfg,
        (store,),
    )
    return PfReport(as_of=pd.Timestamp.now(tz="UTC"), stores=(store,), broker=broker)


def _config_scope(cfg: LiveConfig) -> str:
    """The scope a config addresses: ``<adapter>_<config_name>_<config_hash>``.

    The SAME mint ``live run`` uses, so ``pf``/``status`` read the scope a cycle
    writes. A config's own ``adapter`` key decides the segment (``sim`` when it
    never named one), matching ``resolve_adapter_name``'s precedence minus the
    CLI flag — a diagnostic read is not a run.
    """
    return scope_of(
        ScopeParts(cfg.adapter, config_name_of(cfg), config_scope_hash(cfg))
    )


def _pf_broker(
    ledger: SqliteLedger,
    adapter: str | None,
    raw: Mapping[str, object] | None,
    cfg: LiveConfig | None,
    stores: tuple[StoreSide, ...],
) -> BrokerSide | None:
    """Resolve the broker to read (if any) and read it.

    ``--adapter`` wins; otherwise a config's EXPLICITLY-named ``broker`` key; a
    config that never named one reads NO broker (store-only), so ``live pf
    <strategy.json>`` works with no configuration beyond the strategy. A ``sim``
    read is the store's own account book (``read_sim_broker``) — no fixture, no
    gateway.
    """
    resolved = adapter
    if resolved is None and raw is not None and cfg is not None:
        name, named = resolve_broker(raw, cfg.strategy_params)
        resolved = name if named else None
    if resolved is None:
        return None
    if resolved == "sim":
        return read_sim_broker(ledger, stores[0].scope if stores else "")
    owned = frozenset(lot.id for store in stores for lot in store.lots)
    return asyncio.run(
        _read_ibkr_pf(
            tuple(store.scope for store in stores),
            owned,
        )
    )


async def _open_ibkr_session() -> tuple[IbkrGateway, str]:
    """Open ONE authenticated session and resolve its account.

    Returns the OPEN gateway and the resolved account; the caller owns
    ``aclose``. A failure is a typed ``SessionNotReady`` (exit 1), never a
    traceback.
    """
    gateway = IbkrGateway(IbkrClient())
    try:
        account = await gateway.client.resolve_account()
        return gateway, account
    except BaseException:
        await gateway.aclose()
        raise


async def _read_ibkr_pf(
    scopes: tuple[str, ...],
    owned: frozenset[str],
) -> BrokerSide:
    """Open a session, resolve the account, then read the IBKR book once.

    The client is ALWAYS closed in the ``finally``; nothing is placed and no lease
    is taken.
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
    adapter: str | None,
    interval: float,
) -> None:
    """Re-read the store + broker and re-render every *interval* until Ctrl-C.

    The one-shot plumbing runs ONCE — the ledger is reused (so no tick can DDL),
    and an ``ibkr`` session is authenticated before the loop and only its book is
    re-read per tick. A per-tick read failure is rendered as a typed error line
    and the loop keeps going. Teardown restores the screen on every exit path.
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
    adapter: str | None,
    style: Styler = PLAIN,
) -> _WatchSession:
    """Resolve the watch's frame + teardown ONCE, before the loop starts.

    An ``ibkr`` read (with or without a config) opens ONE gateway on ONE
    persistent event loop — the client is authenticated here and only its book is
    re-read per tick, so the connection pool is never rebound to a fresh loop. A
    store-only or ``sim`` read needs no session at all (the store is re-read per
    tick). No lease is taken and no DDL runs: the frame calls the same read
    helpers the one-shot path uses.
    """
    if config_path is None:
        if adapter == "ibkr":
            return _ibkr_watch_session(
                ledger,
                ledger.scopes_of_store,
                0.0,
            )
        return _WatchSession(_store_frame(ledger, None, adapter, style), _noop)
    cfg = load_live_config(config_path)
    raw = _read_json(config_path)
    name = adapter
    if name is None:
        resolved, named = resolve_broker(raw, cfg.strategy_params)
        name = resolved if named else None
    if name != "ibkr":
        return _WatchSession(
            _store_frame(ledger, cfg if name is not None else None, name, style),
            _noop,
        )
    return _ibkr_watch_session(
        ledger,
        lambda: (_config_scope(cfg),),
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
    style: Styler = PLAIN,
) -> Callable[[], str]:
    """A store (+ sim broker) frame builder for the scopes a watch covers."""

    def frame() -> str:
        stores, broker = _store_and_broker(ledger, cfg, adapter)
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
        scope = _config_scope(cfg)
        stores = (read_store(ledger, scope, initial_capital=cfg.initial_capital),)
    if adapter != "sim":
        return stores, None
    return stores, read_sim_broker(ledger, stores[0].scope if stores else "")


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


def _write_strategy_config(strategy: StrategyConfig, tmp: str) -> str:
    """Dump a strategy-only projection of *strategy* into *tmp*; return its path.

    ``load_strategy`` validates through ``StrategyConfig(**data)`` and rejects
    unknown top-level keys, so the screen bridge must see ONLY the keys
    ``StrategyConfig`` defines. Live-only keys (sizing) never reach the temp
    file.
    """
    path = Path(tmp) / "strategy.json"
    path.write_text(json.dumps(asdict(strategy), default=_json_default))
    return str(path)


def _housekeeping(ledger: SqliteLedger, dry_run: bool) -> None:
    """Prune aged-out closed rows; never fail the cycle over cleanup.

    A dry run writes nothing, so it skips the prune (the caller already skipped
    ``ensure_strategy``).
    """
    if dry_run:
        return
    try:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
        ledger.prune_closed(cast("pd.Timestamp", cutoff))
    except peewee.OperationalError as exc:  # housekeeping is non-fatal
        # peewee wraps sqlite3 errors (a lock, a busy DB) as peewee.OperationalError
        # — NOT a sqlite3.Error — so this is the type a prune failure actually
        # raises. The ledger's own reads catch the same type. LedgerReadError is
        # not caught: prune_closed takes the WRITE path and never raises it.
        _stderr_log(f"housekeeping: prune skipped ({exc})")


def load_live_config(path: str) -> LiveConfig:
    """Parse + validate the live JSON -> ``LiveConfig`` (strategy + sizing).

    Strategy fields are validated through ``StrategyConfig`` (via
    ``load_strategy`` for a pure file); the live-only sizing keys are read from
    the raw dict (and a key neither knows, such as a stale ``mode``, is ignored).
    Sizing accepts a nested ``"sizing"`` object OR flat keys (top-level, then
    ``strategy_params``, where real ``strats/*.json`` keep sizing); flat wins when
    both are given.
    """
    raw = _read_json(path)
    strategy = _strategy_config(path, raw)
    params = strategy.strategy_params
    size_mode, size, alloc = _sizing(raw, params)
    raw_broker, _ = resolve_broker(raw, params)
    commission = _float_or((raw, params), "commission", strategy.commission)
    spread_bps = _float_or((raw, params), "spread_bps", strategy.spread_bps)
    slippage_bps = _float_or((raw, params), "slippage_bps", strategy.slippage_bps)
    per_share = _float_or_none(
        (raw, params), "commission_per_share", strategy.commission_per_share
    )
    commission_min = _float_or((raw, params), "commission_min", strategy.commission_min)
    commission_max_pct = _float_or_none(
        (raw, params), "commission_max_pct", strategy.commission_max_pct
    )
    return LiveConfig(
        strategy_type=strategy.strategy_type,
        symbols=tuple(strategy.symbols),
        initial_capital=strategy.initial_capital,
        strategy_params=params,
        bars=tuple(strategy.bars),
        warmup=strategy.warmup,
        commission=commission,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        commission_per_share=per_share,
        commission_min=commission_min,
        commission_max_pct=commission_max_pct,
        size_mode=size_mode,
        size=size,
        max_symbol_allocation=alloc,
        adapter=cast("AdapterName", _resolve_adapter_key(raw, params)),
        broker=cast("Literal['sim', 'ibkr']", raw_broker),
        config_name=strategy.name,
    )


def _resolve_adapter_key(
    raw: Mapping[str, object], params: Mapping[str, object]
) -> str:
    """The config's own adapter, by the shared precedence (CLI flag excluded).

    Delegates to :func:`resolve_adapter_name` with no flag, so a config's
    ``adapter`` key, then the back-compat ``broker`` key, then ``ibkr`` are read
    in ONE place — the CLI flag only overrides it at the call site.
    """
    return resolve_adapter_name(None, raw, params)


def render_report(report: CycleReport, fmt: str, scope: str = "") -> str:
    """Deterministic text table (default) or JSON document for one cycle.

    *scope* is the resolved ownership key, printed in the header so the operator
    sees WHICH book this cycle touched (``<adapter>_<name>_<hash>``) without
    re-deriving it from the config.
    """
    if fmt == "json":
        return _render_json(report, scope)
    return _render_text(report, scope)


def _render_json(report: CycleReport, scope: str = "") -> str:
    """JSON at the edge — reuse the shared encoder for Timestamps/Enums."""
    doc = {
        "as_of": report.as_of,
        "scope": scope,
        "costs": asdict(report.cost),
        "signals": [asdict(s) for s in report.signals],
        "intents": [asdict(i) for i in report.intents],
        "results": [_result_dict(r) for r in report.results],
        "placement_error": (
            asdict(report.placement_error) if report.placement_error else None
        ),
        "resync_error": (asdict(report.resync_error) if report.resync_error else None),
        "portfolio_before": _portfolio_dict(report.portfolio_before),
    }
    return json.dumps(doc, default=_json_default, indent=2)


_SIGNAL_COLS = (
    Col("symbol"),
    Col("action"),
    Col("score", ">"),
    Col("price", ">"),
)
_INTENT_COLS = (
    Col("symbol"),
    Col("action"),
    Col("qty", ">"),
    Col("ref", ">"),
    Col("reason"),
)
_RESULT_COLS = (
    Col("symbol"),
    Col("action"),
    Col("outcome"),
    Col("kind"),
    Col("qty", ">"),
    Col("filled", ">"),
    Col("message"),
)


def _divergence_lines(divergences: tuple[Divergence, ...]) -> list[str]:
    """One line per book disagreement, naming the side that is short.

    The operator's next step is a manual reconciliation, so the line states both
    quantities: "ours" is the fill fold (engine-owned), "account" is what the
    broker reported or the operator edited.
    """
    return [
        f"divergence: {d.symbol} {d.position_id or '-'} {d.kind} "
        f"ours={d.ours_qty:g} account={d.account_qty:g}"
        for d in divergences
    ]


def _render_text(report: CycleReport, scope: str = "") -> str:
    """The cycle as block-aligned tables (``src.bt.table``), not run-on rows.

    Scalars stay lines; every list — signals, intents, order results — is one
    table with a title, so a wide order's message column reads instead of
    fighting a ``key=value`` header. The last column of the results table is the
    message, which is allowed to grow; the rest stay tight.
    """
    blocks: list[list[str]] = [[f"scope: {scope}", f"as_of: {report.as_of}"]]
    if report.divergences:
        blocks.append(_divergence_lines(report.divergences))
    blocks.append(
        [f"costs: bookkeeping={report.cost.bookkeeping} sizing={report.cost.sizing}"]
    )
    blocks.append(
        _title_table(
            "signals",
            _SIGNAL_COLS,
            tuple(
                (sig.symbol, str(sig.action), f"{sig.score:.4f}", f"{sig.price:.4f}")
                for sig in report.signals
            ),
        )
    )
    blocks.append(
        _title_table(
            "intents",
            _INTENT_COLS,
            tuple(
                (
                    intent.symbol,
                    intent.action.value,
                    f"{intent.qty:g}",
                    f"{intent.ref_price:.4f}",
                    intent.reason,
                )
                for intent in report.intents
            ),
        )
    )
    blocks.append(
        _title_table(
            "orders",
            _RESULT_COLS,
            tuple(
                (
                    result.intent.symbol,
                    result.intent.action.value,
                    result.outcome.value,
                    result.error_kind or "-",
                    f"{result.intent.qty:g}",
                    f"{result.filled_qty:g}" if result.filled_qty is not None else "-",
                    f"{result.message}{_partial_note(result)}",
                )
                for result in report.results
            ),
        )
    )
    if report.resync_error is not None:
        error = report.resync_error
        blocks.append([f"resync_error: {error.kind}: {error.message}"])
    if report.placement_error is not None:
        error = report.placement_error
        blocks.append([f"placement_error: {error.kind}: {error.message}"])
    blocks.append([f"cash: {report.portfolio_before.cash:.2f}"])
    blocks.append(
        [
            f"summary: {len(report.signals)} signals, "
            f"{len(report.intents)} intents, {len(report.results)} orders"
        ]
    )
    return "\n\n".join("\n".join(block) for block in blocks)


def _title_table(
    title: str, columns: tuple[Col, ...], rows: tuple[tuple[str, ...], ...]
) -> list[str]:
    """A titled table block, or ``"<title>: none"`` when there is nothing to show."""
    if not rows:
        return [f"{title}: none"]
    return [f"{title}:", *render_table(Table(columns=columns, rows=rows))]


def _result_dict(result: OrderResult) -> dict[str, object]:
    """Compact order-result view (the full fill is redundant at the edge)."""
    return {
        "symbol": result.intent.symbol,
        "action": result.intent.action.value,
        "ok": result.ok,
        "qty": result.intent.qty,
        "message": result.message,
        "position_id": result.position_id,
        "outcome": result.outcome.value,
        "kind": result.error_kind,
        "filled": result.filled_qty,
        "shortfall": _shortfall(result),
    }


def _shortfall(result: OrderResult) -> float | None:
    """Unfilled shares against the ticket's ask, or ``None`` when unknown."""
    if result.filled_qty is None:
        return None
    return max(0.0, result.intent.qty - result.filled_qty)


def _partial_note(result: OrderResult) -> str:
    """The filled/short annotation for a GENUINE partial, else an empty string.

    A partial entry is not chased (the posture diff compares sides, never sizes),
    so this shortfall is the only trace that the live position came in under what
    the sizer asked for. Reported rather than acted on: under-filling errs toward
    LESS exposure than intended, and silently is the thing to avoid.

    Only a fill STRICTLY between nothing and the ask is a partial: a zero-fill
    refusal or timeout carries ``filled_qty=0.0`` and is not "partial", so
    annotating it would leave the label meaning nothing.
    """
    filled = result.filled_qty
    if filled is None or not 0.0 < filled < result.intent.qty:
        return ""
    return f" partial={filled:g}/{result.intent.qty:g} short={result.intent.qty - filled:g}"


def _portfolio_dict(portfolio: PortfolioState) -> dict[str, object]:
    """Cash + per-symbol lot counts — never the trades/equity_curve wholesale."""
    return {
        "cash": portfolio.cash,
        "positions": {sym: len(lots) for sym, lots in portfolio.positions.items()},
    }


def _read_json(path: str) -> dict[str, object]:
    """Read a JSON object; a parse/shape failure becomes ``ValueError``."""
    try:
        raw = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config must be a JSON object")
    return raw


def _strategy_config(path: str, raw: Mapping[str, object]) -> StrategyConfig:
    """Validate the strategy fields, tolerating live-only top-level keys.

    ``load_strategy`` is ``StrategyConfig(**json)`` and rejects unknown keys, but
    a live config legitimately carries extra live-only keys. A pure file still
    goes through ``load_strategy`` (the canonical loader); otherwise we build the
    same dataclass from the recognised subset.
    """
    if set(raw) <= _STRATEGY_FIELDS:
        return load_strategy(path)
    subset = {k: raw[k] for k in raw if k in _STRATEGY_FIELDS}
    # dataclass kwargs are field-typed; entries are the recognised StrategyConfig fields.
    return StrategyConfig(**cast("dict[str, Any]", subset))


def _sizing(
    raw: Mapping[str, object], params: Mapping[str, object]
) -> tuple[SizeMode, float, float]:
    """Resolve `size_mode`/`size`/`max_symbol_allocation` (flat wins over nested)."""
    nested = raw.get("sizing")
    sources: list[Mapping[str, object]] = [raw]
    if isinstance(nested, Mapping):
        sources.append(cast("Mapping[str, object]", nested))
    sources.append(params)
    mode = _pick(sources, ("size_mode", "sizing_mode"), "equity")
    if mode not in _SIZE_MODES:
        raise ValueError(
            f"size_mode must be one of {sorted(_SIZE_MODES)}, got {mode!r}"
        )
    size = _float_or(sources, "size", 0.0)
    alloc = _float_or(sources, "max_symbol_allocation", 1.0)
    return cast("SizeMode", mode), size, alloc


def _pick(
    sources: Sequence[Mapping[str, object]],
    keys: tuple[str, ...],
    default: object,
) -> object:
    """First present key across *sources* in order, else *default*."""
    for source in sources:
        for key in keys:
            if key in source:
                return source[key]
    return default


def _float_or(
    sources: Sequence[Mapping[str, object]], key: str, default: float
) -> float:
    value = _pick(sources, (key,), default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number, got {value!r}")
    return float(value)


def _float_or_none(
    sources: Sequence[Mapping[str, object]], key: str, default: float | None
) -> float | None:
    """Like ``_float_or`` but tolerates a null/absent value (optional knob)."""
    value = _pick(sources, (key,), default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number or null, got {value!r}")
    return float(value)
