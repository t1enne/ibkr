"""`bt run` command."""

from __future__ import annotations

import json

import click

from src.bt.cmds._shared import _json_default


@click.command(name="run")
@click.argument("strategy_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(["jsonl", "json", "text", "plot"]),
    default="text",
    help="Output format: jsonl (equity curve lines), json (full result), "
    "text (summary), or plot (chart payload inline for dashboards)",
)
@click.option(
    "-o",
    "--output",
    "output",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Write output to FILE instead of stdout (any format).",
)
@click.option(
    "--plot",
    "plot",
    is_flag=True,
    default=False,
    help="Run the backtest then open the Streamlit dashboard over the result.",
)
def run(strategy_file: str, fmt: str, output: str | None, plot: bool):
    """Run a backtest from a strategy JSON config file.

    STRATEGY_FILE: JSON config with symbols, dates, strategy params.

    --plot runs the backtest then launches the Streamlit dashboard over the
    result (overrides --format/--output).

    Output:
      text  — human-readable summary table.
      json  — full structured result as one JSON document (metrics, trades,
              equity curve, benchmarks).
      jsonl — equity curve points as JSON lines, then a final metrics+trades
              record.
      plot  — one JSON document of metrics, per-symbol candle bars, and
              trades, sized for candlestick dashboards.
    """
    from src.bt import load_strategy, run_backtest_results
    from src.bt import get_backtest_results_analysis
    from src.bt.dashboard import launch_dashboard
    from src.bt.output import (
        render_result_json,
        render_result_jsonl,
        render_plot_json,
    )

    config = load_strategy(strategy_file)
    results = run_backtest_results(config)
    if plot:
        launch_dashboard(render_plot_json(results))
        return
    if fmt == "plot":
        payload = render_plot_json(results)
    elif fmt == "json":
        payload = render_result_json(results)
    elif fmt == "jsonl":
        lines = render_result_jsonl(results)
        text = "\n".join(json.dumps(line, default=_json_default) for line in lines)
        _emit(text + "\n", output)
        return
    else:
        click.echo(
            get_backtest_results_analysis(
                results.pf, benchmark_curves=results.benchmark_curves
            )
        )
        return
    _emit(json.dumps(payload, indent=2, default=_json_default), output)


def _emit(text: str, output: str | None) -> None:
    """Write text to an output file if given, else stdout."""
    if output:
        with open(output, "w") as fh:
            fh.write(text)
    else:
        click.echo(text)


def register(group: click.Group) -> None:
    """Register this command onto the bt group."""
    group.add_command(run)
