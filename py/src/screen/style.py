"""Terminal styling utilities for screen output.

Provides ANSI escape codes (``Style``) and a set of static formatting
helpers (``Fmt``) for colored, aligned, percentage-aware terminal
rendering of financial screen results.

Usage:

    from src.screen.style import Style, Fmt

    # ANSI colors
    print(f"{Style.GREEN}green text{Style.RESET}")

    # Formatting helpers
    print(Fmt.score(0.85))
    print(Fmt.signal("long"))
    print(Fmt.price(198.50, "long"))
    print(Fmt.float_val(2.3, "rvol"))
    print(Fmt.float_val(5.0, "distance_52w"))  # → "+5.00%"

All helpers are ``@staticmethod`` s on the ``Fmt`` class, so they can
be imported individually or used as ``Fmt.score(...)``.
"""

from __future__ import annotations

from typing import Final


# ── ANSI color codes ───────────────────────────────────────────


class Style:
    """ANSI escape codes for terminal coloring.

    Usage::

        print(f"{Style.GREEN}green{Style.RESET}")
        print(f"{Style.BOLD}{Style.BRIGHT_WHITE}header{Style.RESET}")

    All members are plain strings — they can be concatenated, added to
    f-strings, or used directly in ``print()`` / ``write()`` calls.
    """

    RESET: str = "\033[0m"
    BOLD: str = "\033[1m"
    DIM: str = "\033[2m"

    # Foreground colors
    GREEN: str = "\033[32m"
    RED: str = "\033[31m"
    YELLOW: str = "\033[33m"
    CYAN: str = "\033[36m"
    WHITE: str = "\033[37m"
    BRIGHT_WHITE: str = "\033[97m"

    # Background colors
    BG_GREEN: str = "\033[42m"
    BG_RED: str = "\033[41m"
    BG_YELLOW: str = "\033[43m"
    BG_RESET: str = "\033[49m"


# ── Percentage-key configuration ───────────────────────────────

# Metadata keys whose float values are **already stored as percentages**
# (e.g. ``distance_52w=5.0`` means 5 %).  These keys get a ``%`` suffix
# and sign-coloring when rendered.
PCT_DISPLAY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "distance_52w",
        "pct_off_high",
        "max_drawdown",
        "return_pct",
        "above_50dma",
        "above_200dma",
        "confidence",
        "risk_score",
        "growth_score",
        "inflation_score",
        "breadth_score",
        "liquidity_score",
        "confirmation",
        "avg_momentum",
    }
)

# Column widths for the standard screen table (in characters).
_RANK_W: int = 5
_SYM_W: int = 8
_SIG_W: int = 8
_SCORE_W: int = 8
_PRICE_W: int = 10
_META_W: int = 14

# Score thresholds for color-coding
_SCORE_HIGH: float = 0.5
_SCORE_MID: float = 0.3


# ── Helpers ────────────────────────────────────────────────────


def _strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences from *s* (used for width calculation)."""
    import re

    return re.sub(r"\033\[[0-9;]*m", "", s)


def _color_for_sign(val: float) -> str:
    """Return ``Style.GREEN`` if *val* > 0, ``Style.RED`` if < 0, else ``""``."""
    if val > 0:
        return Style.GREEN
    if val < 0:
        return Style.RED
    return ""


# ── Public formatting API ──────────────────────────────────────


class Fmt:
    """Static formatting helpers for colored terminal output.

    Every method returns a string with embedded ANSI escape codes.
    Call ``repr()`` on the result to inspect the raw codes.
    """

    # ── Colored values ─────────────────────────────────────────

    @staticmethod
    def score(score: float) -> str:
        """Format a screen score with green / yellow / red coloring.

        * ≥ 0.50 → green
        * ≥ 0.30 → yellow
        * < 0.30 → red

        Padded to ``_SCORE_W`` characters.
        """
        if score >= _SCORE_HIGH:
            color = Style.GREEN
        elif score >= _SCORE_MID:
            color = Style.YELLOW
        else:
            color = Style.RED
        return f"{color}{score:<{_SCORE_W}.2f}{Style.RESET}"

    @staticmethod
    def price(price: float, signal: str) -> str:
        """Format a dollar price.

        * ``"long"`` signal → green
        * otherwise → white

        Padded to ``_PRICE_W`` characters (including the ``$`` prefix).
        """
        color = Style.GREEN if signal == "long" else Style.WHITE
        return f"{color}${price:<{_PRICE_W - 1}.2f}{Style.RESET}"

    @staticmethod
    def signal(signal: str) -> str:
        """Format a signal string as a colored badge.

        * ``"long"`` → green ``LONG``
        * ``"short"`` → red ``SHORT``
        * ``"neutral"`` → dim ``neutral``

        Padded to ``_SIG_W`` characters.
        """
        if signal == "long":
            return f"{Style.GREEN}LONG{' ' * (_SIG_W - 4)}{Style.RESET}"
        if signal == "short":
            return f"{Style.RED}SHORT{' ' * (_SIG_W - 5)}{Style.RESET}"
        return f"{Style.DIM}neutral{' ' * (_SIG_W - 7)}{Style.RESET}"

    # ── Generic float ──────────────────────────────────────────

    @staticmethod
    def float_val(val: float, key: str) -> str:
        """Format a single float value with ANSI coloring and alignment.

        Behaviour depends on *key*:

        * **Percentage keys** (see ``PCT_DISPLAY_KEYS``) — already stored as
          percentages; rendered as ``+5.00%`` or ``-12.00%`` with sign-color.
        * **General floats** — positive → green, negative → red, zero →
          uncolored.  Adaptive precision:

          * ``abs(val) >= 1000`` → 1 decimal
          * ``abs(val) >= 1`` → 3 decimals
          * otherwise → 4 decimals

        Always padded to ``_META_W`` (14) characters.
        """
        if key in PCT_DISPLAY_KEYS:
            return Fmt._pct(val)

        color = _color_for_sign(val)

        if abs(val) >= 1000:
            raw = f"{val:<{_META_W}.1f}"
        elif abs(val) >= 1:
            raw = f"{val:<{_META_W}.3f}"
        else:
            raw = f"{val:<{_META_W}.4f}"

        return f"{color}{raw}{Style.RESET}" if color else raw

    @staticmethod
    def _pct(val: float) -> str:
        """Format a value that is already a percentage (e.g. 5.0 → ``+5.00%``)."""
        color = Style.GREEN if val >= 0 else Style.RED
        raw = f"{val:<+8.2f}%"
        return f"{color}{raw:<{_META_W}}{Style.RESET}"

    # ── Cell wrappers (symbol, rank, plain value) ──────────────

    @staticmethod
    def rank(idx: int) -> str:
        """Dimmed rank number padded to ``_RANK_W`` chars."""
        return f"{Style.DIM}{idx:<{_RANK_W}}{Style.RESET}"

    @staticmethod
    def symbol(sym: str) -> str:
        """Bright-white symbol padded to ``_SYM_W`` chars."""
        return f"{Style.BRIGHT_WHITE}{sym:<{_SYM_W}}{Style.RESET}"

    @staticmethod
    def plain_val(val: object) -> str:
        """White string value padded to ``_META_W`` chars (for non-float metadata)."""
        return f"{Style.WHITE}{str(val):<{_META_W}}{Style.RESET}"

    # ── Structural helpers ─────────────────────────────────────

    @staticmethod
    def header_label(label: str) -> str:
        """Bold column header."""
        return f"{Style.BOLD}{label}{Style.RESET}"

    @staticmethod
    def dim(s: str) -> str:
        """Dimmed text (separators, footers)."""
        return f"{Style.DIM}{s}{Style.RESET}"

    @staticmethod
    def title(s: str) -> str:
        """Bold bright-white title text."""
        return f"{Style.BOLD}{Style.BRIGHT_WHITE}{s}{Style.RESET}"

    @staticmethod
    def warning(s: str) -> str:
        """Yellow warning text."""
        return f"{Style.YELLOW}{s}{Style.RESET}"
