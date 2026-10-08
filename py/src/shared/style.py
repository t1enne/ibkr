"""Terminal styling: ANSI roles, a width-safe styler, and the PLAIN/COLOR pair.

Pure and side-effect free — nothing here reads ``sys.stdout``, a tty or an env
var. The CALLER decides whether styling is on (a human's terminal gets
``COLOR``, a pipe or a cron log gets ``PLAIN``) and passes the styler down into
rendering. So a report's bytes stay a function of its arguments: redirecting
output cannot make escape codes appear.

A styler is applied AFTER a table cell has been padded (``src.bt.table``), never
before: escape codes have no width, so padding an already-styled string is what
breaks a column's alignment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal, Mapping

#: What a piece of text IS, not what colour it should be. A caller names a role
#: so the palette can move without touching every call site.
type Role = Literal[
    "title", "header", "strong", "dim", "note", "good", "bad", "warn", "accent"
]

_RESET: Final[str] = "\x1b[0m"
_BOLD: Final[str] = "\x1b[1m"
_DIM: Final[str] = "\x1b[2m"
_ITALIC: Final[str] = "\x1b[3m"
_RED: Final[str] = "\x1b[31m"
_GREEN: Final[str] = "\x1b[32m"
_YELLOW: Final[str] = "\x1b[33m"
_CYAN: Final[str] = "\x1b[36m"

#: The codes each role opens with (weight first, then the colour).
_CODES: Final[Mapping[Role, str]] = {
    "title": f"{_BOLD}{_CYAN}",
    "header": _BOLD,
    "strong": _BOLD,
    "dim": _DIM,
    "note": f"{_ITALIC}{_DIM}",
    "good": _GREEN,
    "bad": _RED,
    "warn": _YELLOW,
    "accent": _CYAN,
}

_ANSI_RE: Final[re.Pattern[str]] = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """*text* without any SGR sequence — what the reader actually sees."""
    return _ANSI_RE.sub("", text)


@dataclass(frozen=True)
class Styler:
    """Applies a role's codes to text, or hands it back untouched.

    ``enabled=False`` is the safe default and makes every method a no-op, so a
    caller that never decided (a test, a machine format, a log) cannot leak an
    escape byte by accident.
    """

    enabled: bool = False

    def role(self, text: str, role: Role) -> str:
        """Wrap *text* in *role*'s codes (a no-op while styling is off)."""
        return self._wrap(text, _CODES[role])

    def signed(self, text: str) -> str:
        """Colour a formatted NUMBER cell by its sign; placeholders stay dim.

        The cell was already formatted by the report, so parsing it back holds no
        surprises: a value that does not parse (``-``) is "unknown", never zero,
        and a genuine zero is left uncoloured rather than called a win.
        """
        if not self.enabled:
            return text
        value = _as_float(text)
        if value is None:
            return self._wrap(text, _DIM)
        if value > 0:
            return self._wrap(text, _GREEN)
        if value < 0:
            return self._wrap(text, _RED)
        return text

    def _wrap(self, text: str, codes: str) -> str:
        """*text* inside *codes*; an empty cell stays empty (no stray reset)."""
        if not self.enabled or not text.strip():
            return text
        return f"{codes}{text}{_RESET}"


def _as_float(text: str) -> float | None:
    """The number *text* holds, or ``None`` when it holds no number."""
    try:
        return float(text.strip())
    except ValueError:
        return None


#: Styling OFF — a pipe, a cron log, a machine format.
PLAIN: Final[Styler] = Styler(enabled=False)
#: Styling ON — a human's terminal.
COLOR: Final[Styler] = Styler(enabled=True)

__all__ = ["COLOR", "PLAIN", "Role", "Styler", "strip_ansi"]
