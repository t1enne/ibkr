"""Tests for src.bt.table — critical formatters."""

from src.bt.table import Col, Table, render, render_from_dicts


def test_render_single_column():
    t = Table(columns=(Col("Name", "<"),), rows=(("Alice",), ("Bob",), ("Charlie",)))
    assert render(t) == ["Name   ", "-------", "Alice  ", "Bob    ", "Charlie"]


def test_render_from_dicts():
    headers = ["Name", "Score"]
    rows = [{"Name": "Alice", "Score": "95"}, {"Name": "Bob", "Score": "87"}]
    result = render_from_dicts(headers, rows)
    assert result[0].startswith("Name")
    assert len(result) == 4  # header + sep + 2 rows


def test_render_multiline_cell_expands_row_height():
    """A cell with embedded newlines grows its row; each line is padded."""
    t = Table(
        columns=(Col("params", "<"), Col("Sharpe", ">")),
        rows=(("a=1\nb=2", "1.00"),),
    )
    lines = render(t)
    # header + sep + 2 physical lines for the one logical row
    assert len(lines) == 4
    assert lines[0].startswith("params")
    assert lines[2].startswith("a=1")
    assert lines[3].startswith("b=2")
    assert lines[2].endswith("  1.00")  # metric right-aligned on first line only
    assert lines[3].endswith("      ")  # blank continuation is filled
