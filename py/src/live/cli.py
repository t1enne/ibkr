"""Live CLI — ``ibkr live run <config.json>``: one batch reconcile cycle.

The caller (cron) drives cadence; this runs exactly ONE cycle. ``--dry-run``
reconciles and reports without placing anything. Everything the cycle reads
(config, mock book) is resolved here; the engine owns the pure core.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Literal, cast

import click
import pandas as pd

from src.bt import load_strategy
from src.bt.cmds._shared import _json_default
from src.bt.state import PortfolioState
from src.bt.state.factories import create_execution_params, create_initial_portfolio
from src.bt.types import StrategyConfig
from src.live.broker import OrderResult, SimulatedBroker
from src.live.engine import (
    CycleReport,
    PortfolioFetchError,
    StaleDataError,
    run_cycle,
)
from src.live.ledger import SqliteLedger, config_hash
from src.live.portfolio_source import MockPortfolioSource
from src.live.types import LiveConfig

#: Sizing modes the shared ``SizingParams`` layer accepts.
SizeMode = Literal["equity", "cash", "fixed"]
_SIZE_MODES = frozenset({"equity", "cash", "fixed"})
#: The keys ``StrategyConfig`` itself defines — everything else is a live-only key.
_STRATEGY_FIELDS = frozenset(f.name for f in fields(StrategyConfig))


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
def live_run(config_path: str, dry_run: bool, max_age: int, fmt: str) -> None:
    """Run ONE live cycle (cron-friendly). --dry-run reconciles without placing."""
    cfg = load_live_config(config_path)
    raw = _read_json(config_path)
    strategy_id = config_hash(raw)
    strategy = _strategy_config(config_path, raw)
    ledger = SqliteLedger()
    ledger.ensure_strategy(strategy_id, strategy.name, cfg.mode)
    if not cfg.portfolio_path:
        raise click.UsageError(
            "config requires portfolio_path (mock portfolio fixture)"
        )
    source = MockPortfolioSource(cfg.portfolio_path)
    broker = SimulatedBroker(
        create_initial_portfolio(cfg.initial_capital, pd.Timestamp.now()),
        create_execution_params(),
        click.echo,
    )
    try:
        report = asyncio.run(
            run_cycle(
                cfg,
                source=source,
                broker=broker,
                ledger=ledger,
                strategy_id=strategy_id,
                config_path=config_path,
                max_age_days=max_age,
                dry_run=dry_run,
            )
        )
    except (StaleDataError, PortfolioFetchError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(render_report(report, fmt))
    _housekeeping(ledger)


live_group.add_command(live_run)


def _housekeeping(ledger: SqliteLedger) -> None:
    """Prune aged-out closed rows; never fail the cycle over cleanup."""
    try:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
        ledger.prune_closed(cast("pd.Timestamp", cutoff))
    except Exception:  # housekeeping is non-fatal
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
    portfolio_path = _pick((raw, params), ("portfolio_path",), "")
    return LiveConfig(
        strategy_type=strategy.strategy_type,
        symbols=tuple(strategy.symbols),
        initial_capital=strategy.initial_capital,
        strategy_params=params,
        bars=tuple(strategy.bars),
        warmup=strategy.warmup,
        size_mode=size_mode,
        size=size,
        max_symbol_allocation=alloc,
        portfolio_path=portfolio_path if isinstance(portfolio_path, str) else "",
        mode=cast("Literal['paper', 'live']", raw_mode),
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
        "signals": [asdict(s) for s in report.signals],
        "intents": [asdict(i) for i in report.intents],
        "results": [_result_dict(r) for r in report.results],
        "portfolio_before": _portfolio_dict(report.portfolio_before),
    }
    return json.dumps(doc, default=_json_default, indent=2)


def _render_text(report: CycleReport) -> str:
    lines = [f"as_of: {report.as_of}"]
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
        status = "ok" if result.ok else "rejected"
        lines.append(
            f"order {result.intent.symbol} {result.intent.action.value} "
            f"{status} {result.message}"
        )
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
