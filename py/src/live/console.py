"""Structured styled log for live cycles — one line per event, click-styled.

Replaces raw ``click.echo(message, err=True)`` with a :class:`LiveConsole`
that formats each event type with an icon + colour, so the operator sees a
compact running log instead of unstructured text.
"""

from __future__ import annotations

import datetime

import click


def _ts() -> str:
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


class LiveConsole:
    """One-shot cycle console — call its methods as events happen.

    Every method writes to stderr (the cycle's report goes to stdout), styled
    with ``click.style``. The console is stateless: each event is printed
    immediately, never buffered.
    """

    __slots__ = ()

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _ts_prefix() -> str:
        return click.style(_ts(), dim=True)

    @staticmethod
    def _sym(symbol: str, action: str, *rest: str) -> str:
        """``SYMBOL ACTION [rest...]`` — the common event stub."""
        parts = [f"{click.style(symbol, bold=True)} {action}"]
        parts.extend(rest)
        return " ".join(parts)

    # -- placement events -------------------------------------------------------

    def placed(
        self,
        symbol: str,
        action: str,
        qty: float,
        price: float,
        order_id: str = "",
        ref: str = "",
    ) -> None:
        """Order filled/placed successfully."""
        rest = f"@{price:.4f}" if price else ""
        ident = f"order_id={order_id}" if order_id else ""
        click.echo(
            self._ts_prefix()
            + click.style(" ✓", fg="green")
            + " "
            + self._sym(symbol, action, f"{qty:g}", rest, ident),
            err=True,
        )

    def submitted(self, symbol: str, action: str, qty: float, order_ref: str) -> None:
        """Order submitted, waiting for fill."""
        click.echo(
            self._ts_prefix()
            + click.style(" ▶", fg="cyan")
            + " "
            + self._sym(symbol, action, f"{qty:g}", f"cOID={order_ref}"),
            err=True,
        )

    def adopted(self, symbol: str, action: str, order_id: str, ref: str) -> None:
        """Working order adopted instead of re-submitted."""
        click.echo(
            self._ts_prefix()
            + click.style(" ↩", fg="blue")
            + " "
            + self._sym(symbol, action, "adopted", f"order_id={order_id}"),
            err=True,
        )

    def confirming(
        self, symbol: str, action: str, reply_id: str, message: str = ""
    ) -> None:
        """Auto-confirming a reply (STP challenge, OTH prompt)."""
        msg = f"({message})" if message else ""
        click.echo(
            self._ts_prefix()
            + click.style(" ↻", fg="cyan")
            + " "
            + self._sym(symbol, action, f"confirm {reply_id}", msg),
            err=True,
        )

    def rounded(self, symbol: str, qty: float, whole: int) -> None:
        """Fractional qty rounded to whole shares."""
        click.echo(
            self._ts_prefix()
            + click.style(" ~", fg="white", dim=True)
            + " "
            + self._sym(symbol, "rounded", f"{qty:g}→{whole}"),
            err=True,
        )

    # -- failures / rejections --------------------------------------------------

    def rejected(self, symbol: str, action: str, reason: str) -> None:
        """Order rejected (broker refused, cash insufficient)."""
        click.echo(
            self._ts_prefix()
            + click.style(" ✗", fg="red")
            + " "
            + self._sym(symbol, action, click.style(reason, fg="red")),
            err=True,
        )

    def unresolved(self, symbol: str, action: str, detail: str) -> None:
        """Order state unknown after submit — may or may not have landed."""
        click.echo(
            self._ts_prefix()
            + click.style(" ?", fg="yellow", bold=True)
            + " "
            + self._sym(symbol, action, click.style(detail, fg="yellow")),
            err=True,
        )

    def unprotected(self, symbol: str, detail: str) -> None:
        """Lot is naked — no protective bracket resting."""
        click.echo(
            self._ts_prefix()
            + click.style(" ⚠", fg="red", bold=True)
            + " "
            + self._sym(symbol, "unprotected", click.style(detail, fg="red")),
            err=True,
        )

    def close_refused(self, symbol: str, reason: str) -> None:
        """Close refused (bracket could not be cancelled)."""
        click.echo(
            self._ts_prefix()
            + click.style(" ✗", fg="red")
            + " "
            + self._sym(symbol, "close refused", click.style(reason, fg="red")),
            err=True,
        )

    # -- cohort / structural ----------------------------------------------------

    def cohort_scaled(
        self, scale: float, requested: float, budget: float, members: str
    ) -> None:
        """Open cohort was cash-scaled."""
        click.echo(
            self._ts_prefix()
            + click.style(" §", fg="yellow")
            + f" cohort scaled x{scale:.4f}: {members} reduced "
            + click.style(f"${requested:.2f}→${budget:.2f}", fg="yellow"),
            err=True,
        )

    def dropped(self, symbol: str, reason: str) -> None:
        """Intent dropped (structural: unfunded, close-failed)."""
        click.echo(
            self._ts_prefix()
            + click.style(" ⊘", fg="white", dim=True)
            + " "
            + self._sym(symbol, "dropped", click.style(reason, dim=True)),
            err=True,
        )

    # -- protection -------------------------------------------------------------

    def protect_resting(self, symbol: str, legs: str) -> None:
        """Bracket already resting — re-reported, never re-sent."""
        click.echo(
            self._ts_prefix()
            + click.style(" ✓", fg="green")
            + " "
            + self._sym(symbol, "protect", f"bracket resting: {legs}"),
            err=True,
        )

    def protect_armed(self, symbol: str, details: str) -> None:
        """Protective bracket placed."""
        click.echo(
            self._ts_prefix()
            + click.style(" ✓", fg="green")
            + " "
            + self._sym(symbol, "protect", details),
            err=True,
        )

    def cancel_bracket(self, symbol: str, count: int) -> None:
        """Bracket legs cancelled."""
        click.echo(
            self._ts_prefix()
            + click.style(" ✕", fg="magenta")
            + " "
            + self._sym(symbol, f"cancel {count} leg(s)"),
            err=True,
        )

    def cancel_failed(self, symbol: str, detail: str) -> None:
        """Bracket cancel refusal — close blocked."""
        click.echo(
            self._ts_prefix()
            + click.style(" ✗", fg="red")
            + " "
            + self._sym(symbol, "cancel failed", click.style(detail, fg="red")),
            err=True,
        )

    # -- raw fallback -----------------------------------------------------------

    def raw(self, message: str) -> None:
        """Unstructured fallback for rare or transient messages."""
        click.echo(click.style(f" {_ts()} {message}", dim=True), err=True)


#: A ``LiveConsole`` as a plain Callable — the shape the broker's ``log`` seam
#: accepts (``Callable[[str], None]``). Each message is re-parsed to call the
#: right method; the ``raw`` path is the fallback when parsing fails.
#:
#: For now, the adapter still calls ``log(message)`` directly. The console
#: routes known patterns to the structured method; everything else hits ``raw``.
def console_log(message: str) -> None:
    """Parsed log callable — routes one string to the right styled method.

    This is the function assigned to the broker's ``log`` seam. It pattern-matches
    the message strings the broker produces and calls the appropriate structured
    method.
    """
    cns = _CONSOLE
    # Recurring broker log patterns (src/live/adapters/ibkr/broker.py)
    m = message
    if m.startswith("rounded "):
        # rounded {symbol} qty {qty} -> {whole} (whole shares)
        parts = m.split()
        if len(parts) >= 5:
            cns.rounded(
                parts[1], float(parts[3]), int(parts[5]) if len(parts) > 5 else 0
            )
        else:
            cns.raw(m)
    elif m.startswith("adopting already-working order "):
        # adopting already-working order {id} for {ref}
        cns.raw(m)
    elif m.startswith("cohort scaled "):
        # cohort scaled x{scale}: requested {r} > budget {b}; reduced {members}
        cns.cohort_scaled(0, 0, 0, m)  # simplified; raw for now
    elif m.startswith("order failed "):
        # order failed {symbol}: {kind}: {message}
        rest = m[len("order failed ") :]
        parts = rest.split(":", 2)
        sym = parts[0].strip()
        kind = parts[1].strip() if len(parts) > 1 else ""
        detail = parts[2].strip() if len(parts) > 2 else ""
        if kind == "rejected":
            cns.rejected(sym, "?", detail)
        elif kind == "unresolved":
            cns.unresolved(sym, "?", detail)
        else:
            cns.raw(m)
    elif m.startswith("UNPROTECTED "):
        parts = m[len("UNPROTECTED ") :].split(":", 1)
        sym = parts[0].strip()
        rest = parts[1].strip() if len(parts) > 1 else ""
        cns.unprotected(sym, rest)
    elif m.startswith("close ") and "refused" in m:
        parts = m.split("refused:", 1)
        sym = parts[0].replace("close ", "").strip()
        rest = parts[1].strip() if len(parts) > 1 else ""
        cns.close_refused(sym, rest)
    elif m.startswith("dropped open "):
        parts = m[len("dropped open ") :].split(":", 1)
        sym = parts[0].strip()
        rest = parts[1].strip() if len(parts) > 1 else ""
        cns.dropped(sym, rest)
    elif m.startswith("dropped open "):
        parts = m[len("dropped open ") :].split(":", 1)
        sym = parts[0].strip()
        rest = parts[1].strip() if len(parts) > 1 else ""
        cns.dropped(sym, rest)
    elif m.startswith("submitted "):
        # submitted {action} {symbol} qty={qty} cOID={ref}
        # Parsing: "submitted long TRI qty=81.7627 cOID=xxx order_id=yyy"
        parts = m.split()
        if len(parts) >= 4:
            action = parts[1]
            sym = parts[2]
            qty_str = parts[3].replace("qty=", "")
            try:
                qty = float(qty_str)
            except ValueError:
                qty = 0.0
            ref = parts[4].replace("cOID=", "") if len(parts) > 4 else ""
            cns.submitted(sym, action, qty, ref)
        else:
            cns.raw(m)
    elif m.startswith("confirming "):
        # confirming {ref} reply {reply_id}: {message}
        parts = m.split()
        if len(parts) >= 4:
            cns.confirming(
                "",
                "",
                parts[3] if len(parts) > 3 else "",
                parts[4] if len(parts) > 4 else "",
            )
        else:
            cns.raw(m)
    elif "already working as " in m:
        ref_end = m.find(" already working as ")
        ref = m[:ref_end] if ref_end > 0 else ""
        order_id_start = m.find("order_id=")
        cns.adopted(
            "?",
            "?",
            m[order_id_start:].replace("order_id=", "") if order_id_start >= 0 else "",
            ref,
        )
    elif m.startswith("protect ") and "already resting" in m:
        sym = m.split()[1] if len(m.split()) > 1 else "?"
        cns.protect_resting(sym, m)
    elif m.startswith("protect ") and "bracket resting" in m:
        # protect {symbol} bracket resting: ...
        sym = m.split()[1] if len(m.split()) > 1 else "?"
        cns.protect_armed(sym, m)
    elif m.startswith("cancel ") and "confirmed absent" in m:
        # cancel {symbol} {n} leg(s) confirmed absent from the working read
        parts = m.split()
        sym = parts[1] if len(parts) > 1 else "?"
        try:
            count = int(parts[2]) if len(parts) > 2 else 0
        except ValueError:
            count = 0
        cns.cancel_bracket(sym, count)
    elif m.startswith("cancel ") and "cannot prove they are gone" in m:
        parts = m.split()
        sym = parts[1] if len(parts) > 1 else "?"
        cns.cancel_failed(sym, m)
    elif m.startswith("synthetic close of naked "):
        parts = m[len("synthetic close of naked ") :].split(" failed: ", 1)
        sym = parts[0].strip() if parts else "?"
        cns.unprotected(
            sym, parts[1].strip() if len(parts) > 1 else "synthetic close failed"
        )
    else:
        cns.raw(m)


_CONSOLE: LiveConsole = LiveConsole()


def get_console() -> LiveConsole:
    """The shared console instance — every component uses the same one."""
    return _CONSOLE
