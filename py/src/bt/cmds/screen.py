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
    from src.bt.screen.run_strategy import ScreenRow

#: Column order for the printed table: the executable fields of the posture-
#: setting signal, so the row is an order ticket, not a metric sheet. ``date``
#: is the bar the signal fired on (a stale/retained setup shows an old date).
#: ``price`` is the signal-time price; ``qty`` absolute shares (0 = engine-
#: sized); ``sl``/``tp`` blank when the strategy set none.
TABLE_COLS = [
    "symbol",
    "action",
    "score",
    "price",
    "qty",
    "sl",
    "tp",
    "date",
    "signals",
]

#: Column order for the ``--trades`` table (the run's executed fills): entry and
#: exit stamps make an unclosed lot visible (blank exit fields), the engine's
#: final-bar flatten included.
TRADE_COLS = [
    "symbol",
    "position",
    "qty",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "pnl",
    "close_reason",
]


@click.command(name="screen")
@click.argument("strategy_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--warmup",
    "-w",
    type=int,
    default=None,
    show_default=False,
    help="Trailing history to load in days (default: the config's warmup). "
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
@click.option(
    "--trades",
    "trades",
    is_flag=True,
    default=False,
    help="Include the run's executed trades (open -> close); default off.",
)
def screen(
    strategy_file: str,
    warmup: int | None,
    max_age: int,
    fmt: str,
    trades: bool,
) -> None:
    """Run a strategy and surface its current-bar intent (opens AND closes).

    STRATEGY_FILE: the same JSON strategy config a ``bt run`` consumes. The
    strategy module runs over its configured feed; each symbol's latest emitted
    intent is projected as an action + score (1.0 = fresh on the newest bar,
    0.8 = an older/retained setup). Actions are ``long``/``short`` (open or
    reorient), ``close`` (explicit exit) and ``flat`` (no signal — text only).
    Ranked by score desc.

    Each row prints the setting signal's executable fields — signal-time price,
    qty, stop-loss, take-profit, and the bar it fired on — so the table is a
    ticket, not a metric sheet. Output is an intent rank only: screens never
    trade, so a high score means "the condition fired", not "expected profit"
    (pre-cost by design).

    ``-F json`` emits only actionable (``long``/``short``/``close``) rows with
    the same executable fields plus ``position_id``/``tag``, so a live layer
    can act without replaying the engine.

    ``--trades`` (default off) additionally surfaces the run's executed trades:
    a second text table after the intent table, or a ``trades`` key in the json
    payload. The screen runs the real engine, so these are its fills — the
    final-bar ``_finalize`` flatten included, not strategy intent.
    """
    from src.bt.output import trade_json
    from src.bt.screen.run_strategy import (
        render_screen_json,
        run_screen_from_strategy,
        trade_table_row,
    )

    run = run_screen_from_strategy(
        strategy_file,
        max_age_days=max_age or None,
        include_trades=trades,
        warmup_days=warmup,
    )

    if fmt == "json":
        payload: dict[str, object] = dict(
            render_screen_json(run, strategy=strategy_file)
        )
        if trades:
            payload["trades"] = [trade_json(t) for t in run.trades]
        click.echo(json.dumps(payload, indent=2, default=_json_default))
        return

    table_rows = [_ticket_row(r) for r in run.rows]
    if not table_rows:
        click.echo("No recent signals.")
    else:
        for line in render_from_dicts(TABLE_COLS, table_rows, align="<"):
            click.echo(line)
    if trades:
        _echo_trades([trade_table_row(t) for t in run.trades])


def _echo_trades(rows: list[dict[str, str]]) -> None:
    """Print the executed-trade table under the intent table (nothing if empty)."""
    if not rows:
        return
    click.echo()
    for line in render_from_dicts(TRADE_COLS, rows, align="<"):
        click.echo(line)


def _ticket_row(r: "ScreenRow") -> dict[str, str]:
    """One screen row -> the signal's executable fields (blank on non-signal)."""
    # Posture-stale on purpose: ``date`` shows the LAST posture-setting bar
    # (where the signal actually fired), not newest tape, so retained reason
    # strings can't be read as fresh signals.
    return {
        "symbol": r.symbol,
        "action": r.action,
        "score": f"{r.score:.3f}",
        "price": _num(r.price),
        "qty": _num(r.qty),
        "sl": _num(r.stop_loss),
        "tp": _num(r.take_profit),
        "date": str(r.sig_ts) if r.sig_ts is not None else str(r.ts),
        "signals": ", ".join(r.signals),
    }


def _num(v: float | None) -> str:
    """Price/level float -> fixed decimals; None renders blank (no sl/tp)."""
    return "" if v is None else f"{v:.2f}"


def register(group: click.Group) -> None:
    """Register this command onto the bt group."""
    group.add_command(screen)
