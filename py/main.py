"""IBKR Trading Library — agent-friendly CLI.

Usage:
    ibkr bt run strategy.json
    ibkr data query AAPL --from 2026-01-01
    ibkr data dl AAPL MSFT --from 2026-01-01

Pipe composition:
    ibkr data query AAPL | ibkr bt run strategy.json
"""

import click

from src.bt.cli import bt_group
from src.data.cli import data_group
from src.gw.cli import gw_group
from src.live.cli import live_group
from src.research.cli import research_cmd


@click.group()
def main():
    """IBKR — composable CLI for market data and backtesting."""


main.add_command(data_group)
main.add_command(bt_group)
main.add_command(research_cmd)
main.add_command(live_group)
main.add_command(gw_group)


if __name__ == "__main__":
    main()
