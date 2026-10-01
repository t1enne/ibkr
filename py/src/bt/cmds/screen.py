"""`bt screen` command — run a real strategy through the engine and surface
its current-bar (manual-trade) intent, including explicit close signals.

A screen is a backtest whose *intent* is surfaced, not whose fills matter: the
strategy's own ``on_candle`` runs over the configured feed, the engine's signal
observer captures every fresh emission before ``_finalize`` discards it, and the
driver projects per-symbol posture into a ranked table (``text``) or a stable
machine-readable intent document (``json``) for live-trading consumption.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import click

from src.bt.cmds._shared import _json_default
from src.bt.table import render_from_dicts

if TYPE_CHECKING:
    from src.bt.engine.candle_store import CandleStore

#: Column order for the printed table (common metrics appended after the core).
#: ``date`` shows the bar the live posture came from (the signal's generation
#: date), so a stale/retained setup is instantly visible as an old date next
#: to current price context. Blank when a symbol has no posture (flat).
TABLE_COLS = ["symbol", "action", "score", "signals", "date"]


@click.command(name="screen")
@click.argument("strategy_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--warmup",
    "-w",
    type=int,
    default=None,
    show_default=False,
    help="Trailing history to load in days (default: driver's WARMUP_DAYS). "
    "A screen only needs enough bars to warm the strategy + display indicators, "
    "never the config's multi-year backtest span.",
)
@click.option(
    "--max-age",
    "-a",
    type=int,
    default=5,
    show_default=True,
    help="Only report postures whose setting bar is within N days of the "
    "symbol's own latest bar. 0 = no limit.",
)
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format: text (ranked table) or json (machine-readable intent "
    "for a live trading system; actionable long/short/close signals only).",
)
def screen(strategy_file: str, warmup: int | None, max_age: int, fmt: str) -> None:
    """Run a strategy and surface its current-bar intent (opens AND closes).

    STRATEGY_FILE: the same JSON strategy config a ``bt run`` consumes. The
    strategy module runs over its configured feed; each symbol's latest emitted
    intent is projected as an action + score (1.0 = fresh on the newest bar,
    0.8 = an older/retained setup). Actions are ``long``/``short`` (open or
    reorient), ``close`` (explicit exit) and ``flat`` (no signal — text only).
    Ranked by score desc.

    Output is an intent rank only — screens never trade, so a high score means
    "the condition fired", not "expected profit" (pre-cost by design).

    ``-F json`` emits only actionable (``long``/``short``/``close``) rows with
    the setting signal's executable fields (price, qty, stop_loss, take_profit,
    position_id, tag) so a live layer can act without replaying the engine.
    """
    from src.bt.screen.run_strategy import (
        COMMON_COLS,
        common_metrics,
        render_screen_json,
        run_screen_from_strategy,
    )

    run = run_screen_from_strategy(
        strategy_file,
        max_age_days=max_age or None,
    )

    if fmt == "json":
        payload = render_screen_json(run, strategy=strategy_file)
        click.echo(json.dumps(payload, indent=2, default=_json_default))
        return

    rows, state = run.rows, run.state
    table_rows: list[dict[str, str]] = []
    for r in rows:
        frame = _symbol_frame(state.candles, r.symbol)
        feats = common_metrics(frame) if frame is not None else {}
        # Posture-stale on purpose: ``date`` shows the LAST posture-setting
        # bar (where the signal actually fired), not newest tape, so retained
        # reason strings can't be read as fresh signals.
        shown = str(r.sig_ts) if r.sig_ts is not None else str(r.ts)
        table_rows.append(
            {
                "symbol": r.symbol,
                "action": r.action,
                "score": f"{r.score:.3f}",
                "signals": ", ".join(r.signals),
                "date": shown,
                **{k: _fmt(feats.get(k)) for k in COMMON_COLS},
            }
        )

    if not table_rows:
        click.echo("No recent signals.")
        return

    for line in render_from_dicts(
        TABLE_COLS + list(COMMON_COLS), table_rows, align="<"
    ):
        click.echo(line)


def _symbol_frame(candles: "CandleStore", symbol: str):
    """:return: the symbol's base-interval frame from the store, or ``None``."""
    base_iv = next((iv for (_, iv) in candles.keys()), "1d")
    try:
        return candles.get((symbol, base_iv))
    except KeyError:
        return None


def _fmt(v: float | None) -> str:
    """Format a metric float; None/NaN renders as empty."""
    if v is None or _isna(v):
        return ""
    return f"{v:.2f}"


def _isna(f: float) -> bool:
    import pandas as pd

    return bool(pd.isna(f))


def register(group: click.Group) -> None:
    """Register this command onto the bt group."""
    group.add_command(screen)
