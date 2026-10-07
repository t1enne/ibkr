"""Data CLI group — wires the `data dl/query/preview` command modules."""

from __future__ import annotations

import click

from src.data.dl import register as register_dl
from src.data.preview import register as register_preview
from src.data.query import register as register_query


@click.group(name="data")
def data_group():
    """Market data + SEC fundamentals download and query."""


register_query(data_group)
register_dl(data_group)
register_preview(data_group)
