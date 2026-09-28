"""`bt optimize` command."""

from __future__ import annotations

import json

import click

from src.bt.cmds._shared import cli_ts, parse_param_grid


@click.command(name="optimize")
@click.argument("strategy_file", type=click.Path(exists=True))
@click.argument("param_grid", type=str)
@click.option(
    "--is-end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Anchor IS/OOS split date (single-split mode).",
)
@click.option(
    "--folds",
    type=int,
    default=None,
    help="Walk-forward fold count (overrides --is-end).",
)
@click.option(
    "--min-is-years",
    type=float,
    default=5.0,
    help="Min IS window length for walk-forward.",
)
@click.option(
    "--sort-by",
    default="sharpe_ratio",
    help="PortfolioResult metric maximized on each fold's IS window.",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Parallelize folds across N worker processes (default 1).",
)
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Report format.",
)
def optimize(
    strategy_file: str,
    param_grid: str,
    is_end,
    folds,
    min_is_years,
    sort_by: str,
    workers: int,
    fmt: str,
):
    """Walk-forward optimize: tune params per fold's IS, validate on its OOS.

    Per fold: sweep PARAM_GRID on the in-sample window, pick the best combo
    by --sort-by (default sharpe_ratio), lock it, and run the out-of-sample
    window with those params. OOS metrics were never optimized against.

    Honest about overfitting: tuning per fold curve-fits the IS window; the
    OOS result prices that cost. If mean OOS Sharpe holds up across folds,
    the edge is likely real. If IS is strong but OOS collapses, the grid is
    fitting noise.

    PARAM_GRID: same JSON shape as `bt sweep`
    (list-valued leaves are swept; scalars override once).

    Example:
      uv run ibkr bt optimize strat.json '{"strategy_params":{"ma_slow":[50,100,200]}}' --folds 4

    Requires exactly one of --is-end or --folds.
    """
    from src.bt import load_strategy
    from src.bt.optimize import (
        optimize_report_to_json,
        render_optimize_report,
        run_optimize,
    )
    from src.bt.split import anchor_split, walk_forward_folds

    cfg = load_strategy(strategy_file)

    def _stream_fold(fold, best_params: dict, is_metrics: dict, oos) -> None:
        if fmt == "json":
            return  # keep stdout a single parseable JSON document
        params_desc = " ".join(f"{k}={v}" for k, v in best_params.items())
        click.echo(
            f"[fold {fold.index + 1}]  "
            f"IS {fold.is_start.date()}→{fold.is_end.date()} | "
            f"OOS {fold.oos_start.date()}→{fold.oos_end.date()}\n"
            f"  params: {params_desc or '(none)'}\n"
            f"  IS sharpe={is_metrics['sharpe_ratio']:.2f}  "
            f"OOS sharpe={oos.sharpe_ratio:.2f}  "
            f"OOS ann={oos.annual_return:.2%}  "
            f"OOS maxdd={oos.max_drawdown:.2%}"
        )

    try:
        grid: dict = parse_param_grid(param_grid)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    try:
        if folds is not None:
            folds_list = walk_forward_folds(cfg, folds, min_is_years=min_is_years)
        elif is_end is not None:
            folds_list = anchor_split(cfg, cli_ts(is_end))
        else:
            raise click.UsageError(
                "Provide one of --is-end (single split) or --folds (walk-forward)."
            )

        results, agg = run_optimize(
            cfg,
            folds_list,
            grid,
            sort_metric=sort_by,
            on_result=_stream_fold,
            workers=workers,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    if fmt == "json":
        click.echo(json.dumps(optimize_report_to_json(results, agg), indent=2))
    else:
        click.echo(render_optimize_report(results, agg))


def register(group: click.Group) -> None:
    """Register this command onto the bt group."""
    group.add_command(optimize)
