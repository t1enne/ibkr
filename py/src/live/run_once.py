"""One live cycle behind a callable — the ``live run`` body (plan §3).

``live_run`` (``src.live.cli``) delegates here and ``src.bot`` imports this
module directly, so both get the SAME cycle and every monkeypatch seam lands on
ONE implementation. Config loading, ledger/adapter construction, the exit-code
mapping and report rendering live here; ``pipeline.run_cycle`` owns the pure
core.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Literal, cast, get_args

import click
from src.bt import load_strategy
from src.bt.cmds._shared import _json_default
from src.bt.state import PortfolioState
from src.bt.table import Col, Table, render as render_table
from src.bt.types import StrategyConfig
from src.config import live_adapter
from src.data.ibkr.client import IbkrClient
from src.data.ibkr.gateway import IbkrGateway
from src.live.adapter import LiveAdapter, resolve_adapter, resolve_adapter_name
from src.live.divergence import Divergence
from src.live.console import console_log
from src.live.ledger import SqliteLedger, config_hash
from src.live.pipeline import run_cycle
from src.live.pure import OrderResult
from src.live.report import CycleReport, unsafe_outcomes
from src.live.scope import (
    AdapterName,
    ScopeParts,
    config_name_of,
    config_hash as config_scope_hash,
    scope_of,
)
from src.live.types import CostProvenance, LiveConfig, cost_provenance
from src.shared.runtime_log import info as _log_info

#: Sizing modes the shared ``SizingParams`` layer accepts.
SizeMode = Literal["equity", "cash", "fixed"]
_SIZE_MODES = frozenset({"equity", "cash", "fixed"})
#: The keys ``StrategyConfig`` itself defines — everything else is a live-only key.
_STRATEGY_FIELDS = frozenset(f.name for f in fields(StrategyConfig))
#: The adapters a ``broker`` key may name, derived from the ONE typing literal.
_ADAPTERS = get_args(AdapterName)


_UNSAFE_EXIT_CODE = 3


_stderr_log = console_log


def run_once(
    config_path: str,
    *,
    dry_run: bool = False,
    max_age_days: int,
    fmt: str = "text",
    adapter: str | None = None,
    allow_unsafe: bool = False,
    allow_oth: bool = False,
) -> int:
    """Run ONE live cycle for *config_path*; return its machine exit code.

    ``0`` for a clean cycle, ``_UNSAFE_EXIT_CODE`` for a cycle that may hold
    something the operator cannot see. Transient failures propagate as the typed
    exceptions ``live_run`` translates — the caller retries (bot §5) or fails a
    cron line. ``allow_unsafe`` forces ``0`` for an unsafe cycle.
    ``allow_oth`` when true, the broker may auto-confirm orders placed outside
    trading hours. Default (false) rejects them.
    """
    cfg = load_live_config(config_path)
    raw = _read_json(config_path)
    resolved = resolve_adapter_name(adapter, raw, cfg.strategy_params)
    # Audit key vs ownership key (plan §1.4): the config hash identifies the
    # revision that wrote the scope; the scope keys the book, the cOID prefix and
    # the lease. The scope is minted ONCE here from the ORIGINAL raw config, never
    # the strategy-only temp projection (a random path must not change identity).
    strategy_id = config_hash(raw)
    scope = scope_of(ScopeParts(resolved, config_name_of(cfg), config_scope_hash(cfg)))
    strategy = _strategy_config(config_path, raw)
    ledger = SqliteLedger()
    if not dry_run:  # a dry run writes nothing (not even the strategy row)
        ledger.ensure_strategy(strategy_id, scope, strategy.name, cfg.initial_capital)
    backend, gateway = _build_backend(
        cfg, resolved, scope, ledger, dry_run, allow_oth=allow_oth
    )
    # Plan §7.3: label which source produced this run's costs. A resolved IBKR run
    # books the broker's exact per-execution commission but still SIZES on the sim
    # model — the report states both, so the mix is never ambiguous.
    cost = cost_provenance(resolved)
    report = _execute(
        cfg,
        backend=backend,
        ledger=ledger,
        strategy=strategy,
        strategy_id=strategy_id,
        scope=scope,
        max_age_days=max_age_days,
        dry_run=dry_run,
        gateway=gateway,
        cost=cost,
    )
    click.echo(render_report(report, fmt, scope))
    _log_info(f"cycle exit {_exit_code(report, allow_unsafe)}")
    return _exit_code(report, allow_unsafe)


def _exit_code(report: CycleReport, allow_unsafe: bool) -> int:
    """The machine-readable exit code for one cycle (stdout is the contract).

    The report already prints the error/unsafe outcomes, so the code is
    out-of-band and never duplicates a message. ``--allow-unsafe`` forces 0 and
    logs what it suppressed, derived from the report (never a fixed string).
    """
    if not report.is_unsafe():
        return 0
    if not allow_unsafe:
        return _UNSAFE_EXIT_CODE
    _stderr_log(f"allow-unsafe suppressed: {', '.join(unsafe_outcomes(report))}")
    return 0


def _build_backend(
    cfg: LiveConfig,
    resolved: AdapterName,
    scope: str,
    ledger: SqliteLedger,
    dry_run: bool,
    allow_oth: bool = False,
) -> tuple[LiveAdapter, IbkrGateway | None]:
    """The adapter plus the IBKR gateway whose client it shares (or ``None``).

    Built from the resolved NAME only — no branch on a backend's internals — so
    a third adapter needs no edit here. The gateway is returned so the cycle can
    close it; a non-IBKR backend gets ``None``.
    """
    client: IbkrClient | None = None
    gateway: IbkrGateway | None = None
    if resolved == "ibkr":
        gateway = IbkrGateway(IbkrClient())
        client = gateway.client
    adapter = resolve_adapter(
        cfg,
        resolved,
        scope,
        ledger,
        dry_run,
        _stderr_log,
        client=client,
        allow_oth=allow_oth,
    )
    return adapter, gateway


def _execute(
    cfg: LiveConfig,
    *,
    backend: LiveAdapter,
    ledger: SqliteLedger,
    strategy: StrategyConfig,
    strategy_id: str,
    scope: str,
    max_age_days: int,
    dry_run: bool,
    gateway: IbkrGateway | None,
    cost: CostProvenance,
) -> CycleReport:
    """Run one cycle over a strategy-only projection of the config, with costs.

    The live file carries live-only keys that ``load_strategy`` rejects, so it
    gets a strategy-only projection written to a temp dir and cleaned up on exit.
    """
    with tempfile.TemporaryDirectory(prefix="ibkr-live-") as tmp:
        normalized_path = _write_strategy_config(strategy, tmp)
        return replace(
            asyncio.run(
                _run_cycle(
                    cfg,
                    adapter=backend,
                    ledger=ledger,
                    strategy_id=strategy_id,
                    scope=scope,
                    config_path=normalized_path,
                    max_age_days=max_age_days,
                    dry_run=dry_run,
                    gateway=gateway,
                )
            ),
            cost=cost,
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

    Precedence: flat/top-level, then ``strategy_params``, then ``config.toml``'s
    ``[live] adapter``. The explicit-named flag comes from the same scan, so it
    can never contradict the resolved name. ``live run`` does NOT consult this
    for adapter SELECTION (its ``--adapter`` always carries a value); it is the
    config description used by ``live pf`` and the sim broker.
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


def load_live_config(path: str) -> LiveConfig:
    """Parse + validate the live JSON -> ``LiveConfig`` (strategy + sizing).

    Strategy fields go through ``StrategyConfig``; the live-only sizing keys are
    read from the raw dict (a key neither knows is ignored). Sizing accepts a
    nested ``"sizing"`` object OR flat keys (top-level, then
    ``strategy_params``); flat wins when both are given.
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
        adapter=resolve_adapter_name(None, raw, params),
        broker=cast("Literal['sim', 'ibkr']", raw_broker),
        config_name=strategy.name,
    )


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
        # A book disagreement is an exit-3 reason, so a machine consumer must be
        # able to read WHY the cycle was unsafe from the document alone: the
        # exit code is out-of-band and cannot name the lot.
        "divergences": [asdict(d) for d in report.divergences],
        # Advisory read findings flow through too (never an exit-3 reason).
        "warnings": list(report.warnings),
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
    if report.warnings:
        blocks.append([f"warning: {w}" for w in report.warnings])
    if report.divergences:
        blocks.append(_divergence_lines(report.divergences))
    blocks.append(
        [f"costs: bookkeeping={report.cost.bookkeeping} sizing={report.cost.sizing}"]
    )
    blocks.append(_signal_block(report))
    blocks.append(_intent_block(report))
    blocks.append(_order_block(report))
    for error, label in (
        (report.resync_error, "resync_error"),
        (report.placement_error, "placement_error"),
    ):
        if error is not None:
            blocks.append([f"{label}: {error.kind}: {error.message}"])
    blocks.append([f"cash: {report.portfolio_before.cash:.2f}"])
    blocks.append(
        [
            f"summary: {len(report.signals)} signals, "
            f"{len(report.intents)} intents, {len(report.results)} orders"
        ]
    )
    return "\n\n".join("\n".join(block) for block in blocks)


def _signal_block(report: CycleReport) -> list[str]:
    """One row per emitted signal."""
    return _title_table(
        "signals",
        _SIGNAL_COLS,
        tuple(
            (sig.symbol, str(sig.action), f"{sig.score:.4f}", f"{sig.price:.4f}")
            for sig in report.signals
        ),
    )


def _intent_block(report: CycleReport) -> list[str]:
    """One row per order intent."""
    return _title_table(
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


def _order_block(report: CycleReport) -> list[str]:
    """One row per order result, with the partial/fill note in the message."""
    return _title_table(
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
    """The filled/short annotation for a GENUINE partial, else ``""``.

    A partial is never chased, so this shortfall is the only trace that the live
    position came in under the ask; under-filling errs toward LESS exposure, so
    it is reported, not acted on. Only a fill STRICTLY between 0 and the ask is
    a partial — a zero-fill refusal or timeout carries ``filled_qty=0.0``.
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


__all__ = [
    "load_live_config",
    "render_report",
    "resolve_broker",
    "run_once",
]
