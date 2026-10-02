"""`bt sweep` command."""

from __future__ import annotations

import json

import click

from src.bt.cmds._shared import parse_param_grid


@click.command(name="sweep")
@click.argument("strategy_file", type=click.Path(exists=True))
@click.argument("param_grid", type=str)
@click.option(
    "--sort-by",
    default="annual_return",
    help="PortfolioResult metric to rank combos by (e.g. sharpe_ratio).",
)
@click.option(
    "--limit",
    "top_n",
    type=int,
    default=None,
    help="Show only the top N combos (default: all).",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Parallelize combo backtests across N worker processes (default 1).",
)
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Report format.",
)
def sweep(
    strategy_file: str,
    param_grid: str,
    sort_by: str,
    top_n: int | None,
    workers: int,
    fmt: str,
):
    """Sweep a strategy's params over a grid and rank backtest results.

    PARAM_GRID: a partial config JSON that deep-merges into the strategy
    config. Top-level keys override StrategyConfig fields; a nested
    "strategy_params" object merges into the strategy's own params. Any value
    that is a list is swept (cartesian product over all sweepable leaves);
    scalars override once.

    Example:
      ibkr bt sweep mystrat.json '{"strategy_params":{"position_size":[0.8,0.95], "drift_tolerance":[0.01,0.05]}}'
    """
    from src.bt import load_strategy
    from src.bt.sweep import (
        render_sweep_report,
        run_sweep,
        sweep_report_to_json,
    )

    try:
        grid: dict = parse_param_grid(param_grid)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    cfg = load_strategy(strategy_file)

    # Stream results to stdout as each combo finishes, then re-render the
    # ranked table once at the end.
    def _stream_result(idx: int, total: int, overrides: dict, pf) -> None:
        params_desc = " ".join(f"{k}={v}" for k, v in overrides.items())
        click.echo(
            f"[{idx + 1}/{total}] {params_desc}  "
            f"ann={pf.annual_return:.2%}  sharpe={pf.sharpe_ratio:.2f}  "
            f"maxdd={pf.max_drawdown:.2%}  trades={len(pf.trades)}"
        )

    results = run_sweep(
        cfg,
        grid,
        sort_metric=sort_by,
        on_result=None if fmt == "json" else _stream_result,
        workers=workers,
    )

    if fmt == "json":
        click.echo(json.dumps(sweep_report_to_json(results), indent=2))
    else:
        click.echo()
        click.echo(render_sweep_report(results, sort_metric=sort_by, limit=top_n))


def register(group: click.Group) -> None:
    """Register this command onto the bt group."""
    group.add_command(sweep)
