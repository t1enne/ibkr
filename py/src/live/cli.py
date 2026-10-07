"""Live CLI — ``ibkr live run <config.json>``: one batch reconcile cycle.

The caller (cron) drives cadence; this runs exactly ONE cycle. ``--dry-run``
reconciles and reports without placing anything. Everything the cycle reads
(config, mock book) is resolved here; the engine owns the pure core.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Literal, cast

import click
import pandas as pd
import peewee

from src.bt import load_strategy
from src.bt.cmds._shared import _json_default
from src.bt.table import Col, Table, render as render_table
from src.bt.state import ActionType, PortfolioState
from src.bt.state.factories import create_initial_portfolio
from src.bt.types import StrategyConfig
from src.data.ibkr.client import IbkrClient, IbkrError
from src.data.ibkr.gateway import IbkrGateway
from src.live.adapters.ibkr.authz import authorize
from src.live.adapters.ibkr.broker import IbkrBroker
from src.live.adapters.ibkr.portfolio_source import IbkrPortfolioSource
from src.live.broker import LiveBroker, OrderResult, SimulatedBroker
from src.live.engine import (
    CycleReport,
    PortfolioFetchError,
    StaleDataError,
    run_cycle,
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
from src.live.portfolio_source import MockPortfolioSource, PortfolioSource
from src.live.result import Err
from src.live.types import (
    CostProvenance,
    FeedError,
    LiveConfig,
    cost_provenance,
    exec_params_of,
)

#: Sizing modes the shared ``SizingParams`` layer accepts.
SizeMode = Literal["equity", "cash", "fixed"]
_SIZE_MODES = frozenset({"equity", "cash", "fixed"})
#: The keys ``StrategyConfig`` itself defines — everything else is a live-only key.
_STRATEGY_FIELDS = frozenset(f.name for f in fields(StrategyConfig))
#: Adapters ``--adapter`` accepts. Phase 2 ships ``ibkr`` read-only.
_ADAPTERS = ("sim", "ibkr")

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


class GatewayNotReady(RuntimeError):
    """The broker gateway is not ready; the cycle never ran."""

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
    default="ibkr",
    help="Broker adapter; defaults to the config's `broker` key.",
)
@click.option(
    "--allow-live",
    is_flag=True,
    help="Required to read a LIVE account (mode: live).",
)
@click.option(
    "--no-gateway",
    is_flag=True,
    help="Skip the gateway readiness check (trust an externally kept-alive gateway).",
)
@click.option(
    "--allow-unsafe",
    is_flag=True,
    help=(
        "Exit 0 even when the cycle is unsafe (placement/resync error, or an "
        "unresolved/wedged/timed-out order). For callers that consume the "
        "report themselves; the default exits non-zero so cron can see it."
    ),
)
def live_run(
    config_path: str,
    dry_run: bool,
    max_age: int,
    fmt: str,
    adapter: str | None,
    allow_live: bool,
    no_gateway: bool,
    allow_unsafe: bool,
) -> None:
    """Run ONE live cycle (cron-friendly). --dry-run reconciles without placing."""
    cfg = load_live_config(config_path)
    raw = _read_json(config_path)
    resolved = resolve_adapter(adapter, raw, cfg)
    # Scope key derives from the ORIGINAL raw config, never the temp projection
    # (the strategy-only file lives at a random path and must not change scope).
    strategy_id = config_hash(raw)
    strategy = _strategy_config(config_path, raw)
    scope = cfg.scope or strategy.name
    ledger = SqliteLedger()
    if not dry_run:  # a dry run writes nothing (not even the strategy row)
        ledger.ensure_strategy(strategy_id, scope, strategy.name, cfg.mode)
        ledger.ensure_cash(scope, cfg.initial_capital)
    gateway: IbkrGateway | None = None
    broker: LiveBroker
    if resolved == "ibkr":
        gateway = IbkrGateway(IbkrClient())
        source: PortfolioSource = IbkrPortfolioSource(
            gateway.client,
            scope=scope,
            ledger=ledger,
            initial_capital=cfg.initial_capital,
            dry_run=dry_run,
        )
        # The real routing edge. ``dry_run`` is passed through as defence in
        # depth: even if the guard below were skipped, this broker places nothing.
        broker = IbkrBroker(
            gateway.client,
            scope=scope,
            intents=ledger,
            params=exec_params_of(cfg),
            exposure=ledger,
            dry_run=dry_run,
            log=_stderr_log,
        )
    else:
        if not cfg.portfolio_path:
            raise click.UsageError(
                "config requires portfolio_path (mock portfolio fixture)"
            )
        source = MockPortfolioSource(cfg.portfolio_path)
        broker = SimulatedBroker(
            create_initial_portfolio(cfg.initial_capital, pd.Timestamp.now()),
            exec_params_of(cfg),
            _stderr_log,
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
            report = asyncio.run(
                _run_cycle(
                    cfg,
                    source=source,
                    broker=broker,
                    ledger=ledger,
                    strategy_id=strategy_id,
                    scope=scope,
                    config_path=normalized_path,
                    max_age_days=max_age,
                    dry_run=dry_run,
                    gateway=gateway,
                    allow_live=allow_live,
                    no_gateway=no_gateway,
                    cost=cost,
                )
            )
    except (
        StaleDataError,
        PortfolioFetchError,
        GatewayNotReady,
        CycleInProgressError,
        LedgerReadError,
        ValueError,
    ) as exc:
        # ValueError: an unsized open raises by default policy — a traceback is
        # not a CLI contract.
        raise click.ClickException(str(exc)) from exc
    click.echo(render_report(report, fmt))
    _housekeeping(ledger, dry_run)
    # The report already prints placement_error/resync_error/the unsafe outcomes,
    # so the exit code is the machine-readable signal — never a duplicated message.
    # stdout stays the parseable contract (a JSON document for ``--format json``);
    # the exit code is out-of-band. Checked AFTER the housekeeping so a prune
    # failure cannot change the verdict.
    if not allow_unsafe and report.is_unsafe():
        raise click.exceptions.Exit(_UNSAFE_EXIT_CODE)


async def _run_cycle(
    cfg: LiveConfig,
    *,
    source: PortfolioSource,
    broker: LiveBroker,
    ledger: SqliteLedger,
    strategy_id: str,
    scope: str,
    config_path: str,
    max_age_days: int,
    dry_run: bool,
    gateway: IbkrGateway | None,
    allow_live: bool,
    no_gateway: bool,
    cost: CostProvenance,
) -> CycleReport:
    """Gate the cycle, then run it: resolve account + authz before any read.

    The gateway adapter is what this adds over the sim path, and it runs BEFORE
    ``run_cycle``: a cycle that cannot reach an authenticated broker session, or
    that is not permitted to trade the account it found, must fail without ever
    touching the screen or the book. ``no_gateway`` (plan §7.5) trusts an
    externally kept-alive gateway and skips the readiness probe — loudly, never
    silently.
    """
    if gateway is not None:
        try:
            account = await gateway.client.resolve_account()
        except IbkrError as exc:
            raise GatewayNotReady(FeedError(kind="auth", message=str(exc))) from exc
        decision = authorize(
            mode=cfg.mode,
            account=account,
            allow_live=allow_live,
            dry_run=dry_run,
        )
        if isinstance(decision, Err):
            raise GatewayNotReady(cast("FeedError", decision.error))
        if no_gateway:
            _stderr_log(
                "gateway readiness check skipped (--no-gateway; "
                "trusting an externally kept-alive gateway)"
            )
        else:
            ready = await gateway.ensure_ready()
            if isinstance(ready, Err):
                raise GatewayNotReady(cast("FeedError", ready.error))
    try:
        return await run_cycle(
            cfg,
            source=source,
            broker=broker,
            ledger=ledger,
            strategy_id=strategy_id,
            scope=scope,
            config_path=config_path,
            max_age_days=max_age_days,
            dry_run=dry_run,
            cost=cost,
        )
    finally:
        # The client is shared with the portfolio source and outlives the read;
        # close it when the cycle is done rather than leaking the pool.
        if gateway is not None:
            await gateway.aclose()


def resolve_adapter(
    cli_adapter: str | None,
    raw: Mapping[str, object],
    cfg: LiveConfig,
) -> str:
    """Which adapter this run uses (plan §7.7).

    The ``--adapter`` flag wins; otherwise the config's ``broker`` key (the phase
    1.5 ``StrategyConfig.broker`` field — its first real consumer). ``mode: live``
    must NAME its adapter in one of those two places: falling through to the
    ``sim`` default would let a live run quietly never touch the broker, so it is
    a hard error instead.
    """
    if cli_adapter in _ADAPTERS:
        return cli_adapter
    _, named = resolve_broker(raw, cfg.strategy_params)
    if named:
        return cfg.broker
    if cfg.mode == "live":
        raise click.UsageError(
            "mode 'live' must name its adapter: pass --adapter or set "
            "'broker' in the config (no default is assumed for a live run)"
        )
    return cfg.broker


def resolve_broker(
    raw: Mapping[str, object], params: Mapping[str, object]
) -> tuple[str, bool]:
    """The ONE read of the config's ``broker`` key: ``(name, was_named_explicitly)``.

    ``StrategyConfig.broker`` (the phase-1.5 field), ``LiveConfig.broker`` and
    ``resolve_adapter``'s "was it named?" probe previously resolved the key in
    three places and disagreed when it lived in ``strategy_params``. They now all
    go through this: flat/top-level wins, then ``strategy_params``, then the
    ``sim`` default. The explicit-named flag comes from the same scan, so it can
    never contradict the resolved name.
    """
    for source in (raw, params):
        if "broker" in source:
            value = source["broker"]
            if value not in _ADAPTERS:
                raise ValueError(
                    f"broker must be one of {sorted(_ADAPTERS)}, got {value!r}"
                )
            return cast("str", value), True
    return "sim", False


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
    "--allow-live",
    is_flag=True,
    help="Required to read a LIVE account (implies mode: live without a config).",
)
@click.option(
    "--no-gateway",
    is_flag=True,
    help="Skip the gateway readiness check (trust an externally kept-alive gateway).",
)
def live_pf(
    config_path: str | None,
    fmt: str,
    adapter: str | None,
    allow_live: bool,
    no_gateway: bool,
) -> None:
    """Show scopes and their orders/trades, plus the broker side when read.

    With CONFIG_PATH the report covers that config's scope. WITHOUT it, EVERY
    scope the store knows about. Either way each scope renders its own lots,
    order intents and stored fills (all read from OUR durable store).

    ``--adapter`` is a FILTER, not a requirement: given, it selects the broker
    to read (``ibkr`` account-wide, or a config's ``sim`` fixture) and the report
    includes the broker block + divergence; absent, the report is store-only and
    no broker is touched — so a bare strategy config reads without a
    ``portfolio_path``. Without a config, a live account needs ``--allow-live``.

    Nothing is placed and nothing is written: no lease is taken, no strategy row
    is ensured and no DDL runs (a read of an unwritten db is an empty store, not a
    creation). The exit code is 0 (report printed), 1 (a config/gateway/read
    failure) or 2 (usage) — there is no unsafe path here.
    """
    ledger = SqliteLedger()
    try:
        report = (
            _all_scopes_report(ledger, adapter, allow_live, no_gateway)
            if config_path is None
            else _config_report(ledger, config_path, adapter, allow_live, no_gateway)
        )
    except (GatewayNotReady, ValueError, LedgerReadError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(render_pf(report, fmt))


def _all_scopes_report(
    ledger: SqliteLedger,
    adapter: str | None,
    allow_live: bool,
    no_gateway: bool,
) -> PfReport:
    """Every scope in the store, plus a broker read when ``--adapter`` names one."""
    scopes = ledger.scopes_of_store()
    stores = tuple(read_store(ledger, scope) for scope in scopes)
    broker = _pf_broker(adapter, None, None, stores, allow_live, no_gateway)
    return PfReport(as_of=pd.Timestamp.now(tz="UTC"), stores=stores, broker=broker)


def _config_report(
    ledger: SqliteLedger,
    config_path: str,
    adapter: str | None,
    allow_live: bool,
    no_gateway: bool,
) -> PfReport:
    """The config's scope, plus a broker read when one is resolved."""
    cfg = load_live_config(config_path)
    raw = _read_json(config_path)
    store = read_store(ledger, cfg.scope, initial_capital=cfg.initial_capital)
    broker = _pf_broker(adapter, raw, cfg, (store,), allow_live, no_gateway)
    return PfReport(as_of=pd.Timestamp.now(tz="UTC"), stores=(store,), broker=broker)


def _pf_broker(
    adapter: str | None,
    raw: Mapping[str, object] | None,
    cfg: LiveConfig | None,
    stores: tuple[StoreSide, ...],
    allow_live: bool,
    no_gateway: bool,
) -> BrokerSide | None:
    """Resolve the broker to read (if any) and read it.

    ``--adapter`` wins; otherwise a config's EXPLICITLY-named ``broker`` key; a
    config that never named one reads NO broker (store-only), so ``live pf
    <strategy.json>`` works without a ``portfolio_path``. A ``sim`` read without
    a ``portfolio_path`` degrades to a warning, never a usage error. Without a
    config the mode is ``live`` when ``--allow-live`` and ``paper`` otherwise.
    """
    resolved = adapter
    if resolved is None and raw is not None and cfg is not None:
        name, named = resolve_broker(raw, cfg.strategy_params)
        resolved = name if named else None
    if resolved is None:
        return None
    if resolved == "sim":
        if cfg is None or not cfg.portfolio_path:
            return _skipped_broker(
                "sim", "no portfolio_path in config (broker read skipped)"
            )
        owned = frozenset(
            i
            for store in stores
            for i in (*(lot.id for lot in store.lots), *store.sim_open_ids)
        )
        return asyncio.run(read_sim_broker(cfg, owned))
    mode: Literal["paper", "live"] = (
        cfg.mode if cfg is not None else ("live" if allow_live else "paper")
    )
    owned = frozenset(lot.id for store in stores for lot in store.lots)
    return asyncio.run(
        _read_ibkr_pf(
            mode,
            tuple(store.scope for store in stores),
            owned,
            allow_live,
            no_gateway,
        )
    )


def _skipped_broker(adapter: str, reason: str) -> BrokerSide:
    """A broker block that carries only a warning — no positions, no orders."""
    return BrokerSide(
        adapter=adapter,
        source="",
        account="",
        net_liquidation=None,
        cash=None,
        positions=(),
        working_orders=(),
        ours_orders=(),
        warnings=(reason,),
    )


async def _read_ibkr_pf(
    mode: Literal["paper", "live"],
    scopes: tuple[str, ...],
    owned: frozenset[str],
    allow_live: bool,
    no_gateway: bool,
) -> BrokerSide:
    """Gate then read the IBKR book: resolve account + authz before any read.

    Mirrors ``_run_cycle``'s gate order exactly — reach an authenticated session,
    prove *mode* may read the account, then probe readiness — so a pf read is
    refused under the same rules as a cycle. A read failure is a typed
    ``GatewayNotReady`` (exit 1), never a traceback. The client is ALWAYS closed
    in the ``finally``; nothing is placed and no lease is taken.
    """
    gateway = IbkrGateway(IbkrClient())
    try:
        try:
            account = await gateway.client.resolve_account()
        except IbkrError as exc:
            raise GatewayNotReady(FeedError(kind="auth", message=str(exc))) from exc
        decision = authorize(
            mode=mode, account=account, allow_live=allow_live, dry_run=True
        )
        if isinstance(decision, Err):
            raise GatewayNotReady(cast("FeedError", decision.error))
        if no_gateway:
            _stderr_log(
                "gateway readiness check skipped (--no-gateway; "
                "trusting an externally kept-alive gateway)"
            )
        else:
            ready = await gateway.ensure_ready()
            if isinstance(ready, Err):
                raise GatewayNotReady(cast("FeedError", ready.error))
        try:
            return await read_ibkr_broker(gateway.client, account, scopes, owned)
        except IbkrError as exc:
            raise GatewayNotReady(FeedError(kind=exc.kind, message=str(exc))) from exc
    finally:
        await gateway.aclose()


live_group.add_command(live_pf)


def _write_strategy_config(strategy: StrategyConfig, tmp: str) -> str:
    """Dump a strategy-only projection of *strategy* into *tmp*; return its path.

    ``load_strategy`` validates through ``StrategyConfig(**data)`` and rejects
    unknown top-level keys, so the screen bridge must see ONLY the keys
    ``StrategyConfig`` defines. Live-only keys (``portfolio_path``, ``mode``,
    sizing) never reach the temp file.
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
    except peewee.OperationalError:  # housekeeping is non-fatal
        # peewee wraps sqlite3 errors (a lock, a busy DB) as peewee.OperationalError
        # — NOT a sqlite3.Error — so this is the type a prune failure actually
        # raises. The ledger's own reads catch the same type. LedgerReadError is
        # not caught: prune_closed takes the WRITE path and never raises it.
        pass


def load_live_config(path: str) -> LiveConfig:
    """Parse + validate the live JSON -> ``LiveConfig`` (strategy + sizing).

    Strategy fields are validated through ``StrategyConfig`` (via
    ``load_strategy`` for a pure file); live-only keys — ``portfolio_path``,
    ``mode``, sizing — are read from the raw dict. Sizing accepts a nested
    ``"sizing"`` object OR flat keys (top-level, then ``strategy_params``, where
    real ``strats/*.json`` keep sizing); flat wins when both are given.
    """
    raw = _read_json(path)
    strategy = _strategy_config(path, raw)
    params = strategy.strategy_params
    size_mode, size, alloc = _sizing(raw, params)
    raw_mode = _pick((raw, params), ("mode",), "paper")
    if raw_mode not in ("paper", "live"):
        raise ValueError(f"mode must be 'paper' or 'live', got {raw_mode!r}")
    raw_broker, _ = resolve_broker(raw, params)
    portfolio_path = _pick((raw, params), ("portfolio_path",), "")
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
        portfolio_path=portfolio_path if isinstance(portfolio_path, str) else "",
        mode=cast("Literal['paper', 'live']", raw_mode),
        broker=cast("Literal['sim', 'ibkr']", raw_broker),
        scope=_scope(raw, params, strategy),
    )


def _scope(
    raw: Mapping[str, object],
    params: Mapping[str, object],
    strategy: StrategyConfig,
) -> str:
    """The ownership scope: the config's ``scope`` key, else the strategy name.

    A stable strategy identity that survives a config edit (plan §4) — the
    per-scope book key and the cOID attribution prefix on a shared account.
    """
    value = _pick((raw, params), ("scope",), None)
    return (
        value if isinstance(value, str) and value else (strategy.scope or strategy.name)
    )


def render_report(report: CycleReport, fmt: str) -> str:
    """Deterministic text table (default) or JSON document for one cycle."""
    if fmt == "json":
        return _render_json(report)
    return _render_text(report)


def _render_json(report: CycleReport) -> str:
    """JSON at the edge — reuse the shared encoder for Timestamps/Enums."""
    doc = {
        "as_of": report.as_of,
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


def _render_text(report: CycleReport) -> str:
    """The cycle as block-aligned tables (``src.bt.table``), not run-on rows.

    Scalars stay lines; every list — signals, intents, order results — is one
    table with a title, so a wide order's message column reads instead of
    fighting a ``key=value`` header. The last column of the results table is the
    message, which is allowed to grow; the rest stay tight.
    """
    blocks: list[list[str]] = [[f"as_of: {report.as_of}"]]
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
