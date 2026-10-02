"""Minimal declarative table formatting. No dependencies. Pure functions."""

from dataclasses import dataclass, field
from typing import Sequence

type Align = str  # "<" | ">" | "^"


@dataclass(frozen=True)
class Col:
    """Single column definition. Label doubles as header text."""

    label: str
    align: Align = "<"
    fmt: str = ""


@dataclass(frozen=True)
class Table:
    """Declarative table: columns + rows of strings, render later."""

    columns: tuple[Col, ...]
    rows: tuple[tuple[str, ...], ...] = field(default_factory=tuple)


def _pad(value: str, width: int, align: Align) -> str:
    fill = width - len(value)
    if align == ">":
        return " " * fill + value
    if align == "^":
        left = fill // 2
        return " " * left + value + " " * (fill - left)
    return value + " " * fill  # "<"


def _lines(cell: str) -> list[str]:
    """Physical lines of a cell — embedded newlines expand the row height."""
    return cell.split("\n")


def render(table: Table, *, sep: str = "  ") -> list[str]:
    """Render a Table into lines. Returns list of str (no trailing newlines).

    Column widths auto-expand to fit the widest physical line (label or cell),
    and a cell containing newlines grows its row to the tallest cell — each
    physical line is padded independently, so short cells are blank-filled.
    """
    cols = table.columns
    widths = [len(c.label) for c in cols]

    for row in table.rows:
        for i, cell in enumerate(row):
            for line in _lines(cell):
                widths[i] = max(widths[i], len(line))

    lines: list[str] = []

    # Header
    header = sep.join(_pad(c.label, widths[i], c.align) for i, c in enumerate(cols))
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows — one physical line per cell line, row height = tallest cell.
    for row in table.rows:
        cell_lines = [_lines(cell) for cell in row]
        height = max((len(cl) for cl in cell_lines), default=1)
        for r in range(height):
            lines.append(
                sep.join(
                    _pad(
                        cell_lines[i][r] if r < len(cell_lines[i]) else "",
                        widths[i],
                        cols[i].align,
                    )
                    for i in range(len(cols))
                )
            )

    return lines


def render_from_dicts(
    headers: Sequence[str],
    rows: Sequence[dict[str, str]],
    *,
    align: Align = "<",
    sep: str = "  ",
) -> list[str]:
    """Convenience: build and render a Table from list-of-dicts.

    All columns share the same alignment. For mixed alignment, use Table + Col directly.
    """
    if not rows:
        return []

    cols = tuple(Col(label=h, align=align) for h in headers)
    str_rows = tuple(tuple(str(row.get(h, "")) for h in headers) for row in rows)
    return render(Table(columns=cols, rows=str_rows), sep=sep)
