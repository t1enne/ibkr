"""Tests for src.shared.style — the role palette and its width-neutral contract."""

from __future__ import annotations

import pytest

from src.shared.style import COLOR, PLAIN, Styler, strip_ansi


def test_plain_is_a_no_op_so_a_pipe_can_never_see_a_code() -> None:
    """Regression: the default styler must not emit ONE escape byte."""
    for text in ("scope", "", "  ", "-1.00"):
        assert PLAIN.role(text, "bad") == text
        assert PLAIN.signed(text) == text


def test_color_wraps_a_role_and_resets_it() -> None:
    """Every styled span is closed, so a code cannot bleed into the next cell."""
    styled = COLOR.role("AAPL", "strong")
    assert styled == "\x1b[1mAAPL\x1b[0m"
    assert strip_ansi(styled) == "AAPL"


def test_an_empty_cell_is_left_empty() -> None:
    """A blank cell gets no codes: an empty span would add width for nothing."""
    assert COLOR.role("   ", "good") == "   "


@pytest.mark.parametrize(
    ("text", "codes"),
    [("98.00", "\x1b[32m"), ("-5.00", "\x1b[31m")],
)
def test_signed_colours_a_number_by_its_own_sign(text: str, codes: str) -> None:
    """A positive figure is good and a negative one is bad."""
    styled = COLOR.signed(text)
    assert styled == f"{codes}{text}\x1b[0m"
    assert strip_ansi(styled) == text  # the colour adds no visible character


def test_a_genuine_zero_is_coloured_neither_way() -> None:
    """Regression: 0.00 is not a win — only an unparsable cell means "unknown"."""
    assert COLOR.signed("0.00") == "0.00"


def test_signed_dims_a_placeholder_instead_of_calling_it_zero() -> None:
    """Regression: an unknown figure (``-``) is dim, never read as a result."""
    assert COLOR.signed("-") == "\x1b[2m-\x1b[0m"


def test_styler_is_frozen_and_defaults_to_off() -> None:
    """Styling is opt-in: a caller that never decided gets no codes."""
    assert Styler().enabled is False
    assert (PLAIN.enabled, COLOR.enabled) == (False, True)
