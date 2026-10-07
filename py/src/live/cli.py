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
    default=None,
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
def live_run(
    config_path: str,
    dry_run: bool,
    max_age: int,
    fmt: str,
    adapter: str | None,
    allow_live: bool,
    no_gateway: bool,
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


def _render_text(report: CycleReport) -> str:
    lines = [f"as_of: {report.as_of}"]
    lines.append(
        f"costs: bookkeeping={report.cost.bookkeeping} sizing={report.cost.sizing}"
    )
    for sig in report.signals:
        lines.append(
            f"signal {sig.symbol} {sig.action} "
            f"score={sig.score:.4f} price={sig.price:.4f}"
        )
    for intent in report.intents:
        lines.append(
            f"intent {intent.symbol} {intent.action.value} "
            f"qty={intent.qty:g} @ {intent.ref_price:.4f} ({intent.reason})"
        )
    for result in report.results:
        status = result.outcome.value
        kind = f" (kind={result.error_kind})" if result.error_kind else ""
        lines.append(
            f"order {result.intent.symbol} {result.intent.action.value} "
            f"{status}{kind} {result.message}"
        )
    if report.resync_error is not None:
        error = report.resync_error
        lines.append(f"resync_error: {error.kind}: {error.message}")
    if report.placement_error is not None:
        error = report.placement_error
        lines.append(f"placement_error: {error.kind}: {error.message}")
    lines.append(f"cash: {report.portfolio_before.cash:.2f}")
    lines.append(
        f"summary: {len(report.signals)} signals, "
        f"{len(report.intents)} intents, {len(report.results)} orders"
    )
    return "\n".join(lines)


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
    }


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
