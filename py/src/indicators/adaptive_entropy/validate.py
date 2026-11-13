"""Input validation shared by the online and batch adaptive-entropy paths.

The model carries its adaptive EMA and Wilder ATR as *running scalars*
(``online.py``) or a single forward pass (``pure.py``), so one non-finite bar
poisons it for the rest of the run: the ATR becomes ``nan``, every band
comparison is then ``False``, and ``trend`` silently freezes at its last value.
That failure is invisible in a report — a regime gate that never refuses a trade
looks exactly like a gate that always allows one. Rejecting junk input loudly is
the cheaper failure, so both paths validate before touching model state, which
also keeps them in step: a bar the batch path refuses is a bar ``observe``
refuses.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_WHY = (
    "the model keeps its adaptive EMA and ATR as running scalars, so a single "
    "non-finite bar freezes `trend` at its last value for the rest of the run; "
    "drop the bar or forward-fill it before feeding the model"
)


class NonFiniteBarError(ValueError):
    """A feed contained a non-finite close/high/low."""


def _describe(name: str, value: object) -> str:
    return f"{name}={value!r}"


def require_finite_bar(close: float, high: float, low: float, *, bar: int) -> None:
    """Raise :class:`NonFiniteBarError` unless all three prices are finite.

    ``bar`` is the caller's positional bar index, so the message points at the
    offending row rather than at the whole feed.
    """
    bad = [
        _describe(name, value)
        for name, value in (("close", close), ("high", high), ("low", low))
        if not np.isfinite(value)
    ]
    if bad:
        raise NonFiniteBarError(
            f"adaptive entropy: non-finite input at bar {bar}: {', '.join(bad)}; {_WHY}"
        )


def require_finite_series(
    close: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
) -> None:
    """Raise :class:`NonFiniteBarError` on the first non-finite series value.

    Reports every offending series with its first bad index and value, so a
    union-calendar feed (blank rows where a symbol did not trade) is obvious.
    """
    bad: list[str] = []
    for name, values in (("close", close), ("high", high), ("low", low)):
        idx = np.flatnonzero(~np.isfinite(values))
        if idx.size:
            first = int(idx[0])
            bad.append(
                f"{_describe(name, float(values[first]))} at index {first} "
                f"({idx.size} non-finite)"
            )
    if bad:
        raise NonFiniteBarError(
            f"adaptive entropy: non-finite input: {'; '.join(bad)}; {_WHY}"
        )
