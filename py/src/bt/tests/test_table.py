"""Tests for src.bt.table — critical formatters."""

from src.shared.style import COLOR, strip_ansi
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


def test_render_styles_after_padding_so_widths_cannot_drift():
    """Regression: an ANSI code has no width, so it must land AFTER the padding.

    The coloured render is byte-identical once the codes are stripped, and every
    line keeps the plain render's length — the property a styled table needs.
    """
    table = Table(
        columns=(Col("sym", role="strong"), Col("pnl", ">", sign=True)),
        rows=(("AAPL", "98.00"), ("MSFT", "-12.00"), ("SHOP", "-")),
    )
    plain = render(table)
    styled = render(table, styler=COLOR)

    assert "\x1b[" not in "".join(plain)
    assert [strip_ansi(line) for line in styled] == plain
    # Visible width is unchanged by styling — the raw bytes are longer, the
    # terminal's columns are not, which is the whole reason padding comes first.
    assert [len(strip_ansi(line)) for line in styled] == [len(line) for line in plain]
    assert "\x1b[32m" in styled[2] and "\x1b[31m" in styled[3]


def test_render_maps_a_columns_role_from_the_cell_text():
    """A value-dependent column styles each cell by what that cell says."""
    table = Table(
        columns=(Col("state", roles={"open": "strong", "rejected": "bad"}),),
        rows=(("open",), ("rejected",), ("closed",)),
    )
    styled = render(table, styler=COLOR)
    assert styled[2].startswith("\x1b[1mopen")
    assert styled[3].startswith("\x1b[31mrejected")
    assert "\x1b[" not in styled[4]  # unmapped text is left alone
    assert styled[4].strip() == "closed"
