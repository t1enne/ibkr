"""Data CLI group — wires the `data dl/query/preview` command modules."""

from __future__ import annotations

import click

from src.data.dl import register as register_dl
from src.data.preview import register as register_preview
from src.data.query import register as register_query
from src.data.fundamentals.dl import register as register_fundamentals_dl


@click.group(name="data")
def data_group():
    """Market data download and query."""


@click.group(name="fundamentals")
def fundamentals_group():
    """SEC EDGAR fundamentals (sparse fiscal rows in the local DB)."""


register_query(data_group)
register_dl(data_group)
register_preview(data_group)
register_fundamentals_dl(fundamentals_group)
data_group.add_command(fundamentals_group)
