"""`bt split` command."""

from __future__ import annotations

import json

import click

from src.bt.cmds._shared import cli_ts


@click.command(name="split")
@click.argument("strategy_file", type=click.Path(exists=True))
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
def split(
    strategy_file: str,
    is_end,
    folds,
    min_is_years,
    workers: int,
    fmt: str,
):
    """Walk-forward / single-split in-sample vs out-of-sample validation.

    Evaluates the strategy's FIXED params across IS/OOS windows. Reports
    out-of-sample Sharpe and IS to OOS degradation. Does NOT re-tune params.
    """
    from src.bt import load_strategy
    from src.bt.split import (
        anchor_split,
        render_split_report,
        run_split,
        split_report_to_dict,
        walk_forward_folds,
    )

    cfg = load_strategy(strategy_file)

    def _stream_fold(fold, is_result, oos_result) -> None:
        click.echo(
            f"[fold {fold.index + 1}]  "
            f"IS {fold.is_start.date()}→{fold.is_end.date()} | "
            f"OOS {fold.oos_start.date()}→{fold.oos_end.date()}\n"
            f"  IS sharpe={is_result.sharpe_ratio:.2f}  ann={is_result.annual_return:.2%}\n"
            f"  OOS sharpe={oos_result.sharpe_ratio:.2f}  ann={oos_result.annual_return:.2%}  "
            f"maxdd={oos_result.max_drawdown:.2%}"
        )

    try:
        if folds is not None:
            folds_list = walk_forward_folds(
                cfg,
                folds,
                min_is_years=min_is_years,
            )
        elif is_end is not None:
            folds_list = anchor_split(cfg, cli_ts(is_end))
        else:
            raise click.UsageError(
                "Provide one of --is-end (single split) or --folds (walk-forward)."
            )

        report = run_split(cfg, folds_list, on_result=_stream_fold, workers=workers)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    if fmt == "json":
        click.echo(json.dumps(split_report_to_dict(report), indent=2))
    else:
        click.echo(render_split_report(report))


def register(group: click.Group) -> None:
    """Register this command onto the bt group."""
    group.add_command(split)
