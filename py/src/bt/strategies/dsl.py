"""Pine-flavoured declarative strategy framework.

A decorated strategy is a tiny pure function of a :class:`StrategyContext` --
it describes *what* to do per candle, not *how* the engine delivers data. The
framework owns the plumbing a hand-rolled strategy would re-invent everywhere:

* candle iteration, cursor advancement, per-symbol signal bucketing,
* OHLCV + indicator prefetch into a cursor-safe ``ctx.ta`` (``TaContext``),
* cross-call state via ``ctx.shared`` (when declared ``stateful=True``),
* signal construction from ``ctx.long/close/...``.

A decorated strategy still exposes ``on_candle(state, candle, params)`` +
``STRATEGY_TYPE`` + a no-op ``reset_global`` shim, so it plugs straight into
auto-discovery, ``bt split`` and ``bt sweep`` -- no engine fork.

The DSL is the **only** authoring surface. ``@strategy`` is the universal entry
point; a strategy that needs arbitrary per-candle behavior beyond the ``ctx``
shortcuts reads ``BacktestState`` / ``state.candles`` directly through
``ctx.state`` rather than dropping into a second, parallel authoring style.

Shape::

    STRATEGY_TYPE = "ema_cross"

    @strategy(bars="1d")
    def on_candle(ctx: StrategyContext):
        fast = ctx.ta.ema("AAPL", 9)
        slow = ctx.ta.ema("AAPL", 21)
        if ctx.cross_over(fast, slow):
            ctx.long("AAPL", size=0.1, sl=0.04, tp=0.08)
        elif ctx.cross_under(fast, slow):
            ctx.close("AAPL")

The decorator is a pure adapter: it wraps the plain decision function in an
``on_candle(state, candle, params)`` that builds a ``StrategyContext``,
collects the signals the user's ``ctx.long/close`` calls emit, and returns
them.
"""

from __future__ import annotations

import sys
from types import FunctionType, MappingProxyType
from typing import Any, Callable, Literal, TYPE_CHECKING, cast

from src.bt.state import ActionType, TradeSignal, BacktestState, Candle, Position
from src.bt.strategies.fundamentals_context import Fundamentals
from src.bt.strategies.series import SeriesView
from src.bt.strategies.ta_context import OhlcvView, TaContext
from src.bt.strategies.utils import sl_tp_from_pct

if TYPE_CHECKING:
    pass


class WarmupTradeError(RuntimeError):
    """Raised when a strategy tries to open a position during the warmup span.

    Warmup bars are walked so strategy state (``ctx.shared`` accumulators,
    ``ctx.ta`` series) is warm when trading begins — no fill can occur on them,
    so emitting an entry there is a programming error, not a dropped signal.
    """


class StrategyContext:
    """Per-candle decision surface handed to a decorated strategy.

    Every data access is cursor-safe: ``ctx.ohlcv`` and ``ctx.ta`` read through
    the engine's ``TaContext``, which shares the ``CandleStore`` cursor and can
    never expose a future bar. ``ctx.state`` is the raw
    ``BacktestState`` for power needs (portfolio / position lookup) the DSL
    shortcuts don't cover -- the DSL allows it, it just makes the common path
    safe by construction.

    Sizing: ``size`` is a 0..1 fraction of *initial* capital converted to an
    absolute share count (``size * initial_capital / close``) — a fixed-percent
    order, not scaled by available cash. When ``size`` is omitted, the signal is
    emitted with ``qty=0`` and the engine's shared sizing layer computes the
    share count from ``SizingParams`` (equity/cash base, size). Risk-targeted
    sizing (``risk_pct`` + stop/ATR) lives in the strategy via
    :func:`src.bt.size.pure.risk_sized_qty` and is expressed as a back-solved
    ``size``. ``sl``/``tp`` are fractional
    percentages (e.g. ``0.04`` = 4%) converted to absolute per-trade stop/target
    prices.
    """

    __slots__ = (
        "_state",
        "_candle",
        "_ta",
        "_fundamentals",
        "_params",
        "_symbols",
        "_interval",
        "_signals",
        "_shared",
        "_readonly",
        "_phase",
    )

    def __init__(
        self,
        state: BacktestState,
        candle: Candle,
        params: Any,
        ta: TaContext,
        symbols: tuple[str, ...],
        interval: str,
        fundamentals: Fundamentals | None = None,
    ) -> None:
        self._state = state
        self._candle = candle
        self._ta = ta
        self._fundamentals = fundamentals
        self._params = params
        self._symbols = symbols
        self._interval = interval
        self._signals: list[TradeSignal] = []
        self._shared: dict | None = None
        # Post-run plot contexts are read-only (set by ``for_plot``); mutating
        # methods assert against this so a ``plot`` can't emit a phantom signal.
        self._readonly: bool = False
        # Engine phase for this bar: "warmup" (before ``trading_start``) or
        # "trade". Signal emission is a hard error during warmup.
        self._phase: str = "trade"

    @classmethod
    def for_plot(
        cls,
        results: Any,  # BacktestResults — Any avoids a state-layer import cycle
        *,
        symbol: str,
        interval: str,
        params: Any,
    ) -> "StrategyContext":
        """Read-only context for a post-run ``plot(ctx, params)`` call.

        The cursor sits at the end of the feed, so ``ctx.ta`` series are the full
        history. ``ctx.shared`` is bound unconditionally to the run's strategy
        state (a stateful strategy's pivot cache) and exposed write-protected.
        """
        data = results.data
        ta = getattr(data, "ta", None)
        if not isinstance(ta, TaContext):
            raise RuntimeError(
                "plot() requires a prefetched TaContext; run through "
                "`src.bt.engine.backtest.run` (it builds `ta` from data) so "
                "results.data.ta is set."
            )
        if not data.is_exhausted:
            raise RuntimeError(
                "plot() requires a terminal cursor: `results.data` was captured "
                "mid-backtest, so ctx.ta series would be silently truncated and "
                "the chart plausible-but-wrong."
            )
        state = results.final_state
        df = data[(symbol, interval)]
        tail = df.iloc[-1]
        candle = Candle(
            timestamp=df.index[-1],
            symbol=symbol,
            open=float(tail["open"]),
            high=float(tail["high"]),
            low=float(tail["low"]),
            close=float(tail["close"]),
            volume=float(tail["volume"]),
            interval=interval,
        )
        ctx = cls(
            state=state,
            candle=candle,
            params=params,
            ta=ta,
            symbols=symbols_from(state),
            interval=interval,
            fundamentals=None,
        )
        ctx.shared = data.strategy_state or {}
        ctx._readonly = True
        ctx._phase = "trade"
        return ctx

    @property
    def shared(self) -> dict:
        """Framework-owned cross-cancel storage (only when the strategy is
        declared ``@strategy(stateful=True)``). Persists across candles within
        a run and is cleared by ``reset_global()`` between split/sweep windows.

        Raises when the strategy wasn't declared stateful -- calling this from a
        stateless strategy is a footgun, so fail loudly rather than silently
        sharing nothing.

        On a read-only plot context the returned mapping is a shallow
        ``MappingProxyType`` guard (nested dicts stay mutable) -- a guard against
        accidental writes from ``plot``, not a sandbox.
        """
        if self._shared is None:
            raise RuntimeError(
                "ctx.shared requires the strategy to be declared "
                "`@strategy(stateful=True)` (cross-call state is the DSL's "
                "GLOBAL replacement)."
            )
        if self._readonly:
            # Shallow guard (nested dicts stay mutable) -- cast so the writable
            # call sites keep a `dict` type without a separate accessor.
            return cast("dict", MappingProxyType(self._shared))
        return self._shared

    @shared.setter
    def shared(self, holder: dict) -> None:
        """Bind the framework-owned state holder (the adapter wires this for
        stateful strategies). Mirrors the ``shared`` getter so the adapter
        doesn't poke ``self._shared`` directly.
        """
        self._shared = holder

    # -- public read-only accessors ------------------------------------------

    @property
    def state(self) -> "BacktestState":
        return self._state

    @property
    def phase(self) -> Literal["warmup", "trade"]:
        """Engine phase for the current bar: ``"warmup"`` or ``"trade"``.

        Bars strictly before ``trading_start`` are warmup bars: the engine
        walks them with the strategy invoked (so ``ctx.shared`` accumulators
        and ``ctx.ta`` series fill) but the strategy must NOT trade. A
        strategy that is not ready yet returns early on warmup bars::

            if ctx.phase == "warmup":
                return   # accumulate only; begin trading at trading_start

        Calling ``ctx.long``/``ctx.short`` during warmup raises: emitting a
        trade signal for a bar where no fill can happen is a programming error,
        and silently discarding it would leave a stateful strategy's own
        bookkeeping diverged from the engine's book.
        """
        return cast("Literal['warmup', 'trade']", self._phase)

    @property
    def candle(self) -> "Candle":
        return self._candle

    @property
    def timestamp(self):
        return self._candle.timestamp

    @property
    def params(self) -> Any:
        return self._params

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def interval(self) -> str:
        return self._interval

    @property
    def ta(self) -> TaContext:
        return self._ta

    @property
    def fundamentals(self) -> Fundamentals:
        """Cursor-safe SEC fundamentals accessor (``ctx.fundamentals``).

        Raises when the run didn't load a fundamentals store, rather than
        returning an empty one: a strategy that reads fundamentals deserves to
        know the data channel is absent (empty series would look like "this
        company has no filings" and silently disable its filter).
        """
        if self._fundamentals is None:
            raise RuntimeError(
                "ctx.fundamentals requires a fundamentals store; run through "
                "`src.bt.engine.backtest.run` (it loads one per config symbol "
                "set) so `state.candles.fundamentals` is set."
            )
        return self._fundamentals

    # -- data access -----------------------------------------------------------

    def ohlcv(self, sym: str, interval: str | None = None) -> OhlcvView:
        return self._ta.ohlcv(sym, interval or self._interval)

    def price(self, sym: str) -> float:
        """Current close for ``sym`` (cursor-safe O(1))."""
        return self._ta.close(sym, self._interval)[-1]

    def position(self, sym: str) -> "Position | None":
        """In-position? Returns the newest open :class:`Position` for ``sym``.

        For multi-lot bookkeeping use :meth:`quantity`, :meth:`position_ids`
        and :meth:`avg_entry` — this convenience accessor returns the most
        recently opened lot (``tuple[-1]``) and ``None`` when ``sym`` is flat.
        """
        tup = self._state.portfolio.positions.get(sym)
        if not tup:
            return None
        return tup[-1]

    # -- multi-position aggregated reads --------------------------------------

    def quantity(self, sym: str) -> float:
        """Net signed size for ``sym`` across all lots.

        Long lots count positive, short lots negative. One unambiguous answer
        to "how big"; a strategy that wants netting semantics computes this and
        opens the delta itself via ``long`` + ``partial_close``.
        """
        from src.bt.portfolio.pure import net_quantity

        return net_quantity(self._state.portfolio, sym)

    def position_ids(self, sym: str) -> tuple[str, ...]:
        """Ordered handles of the active lots for ``sym``.

        The list-of-trades surface backtesting.py ships as ``self.trades`` and
        Pine tracks per entry-name — the DSL previously hid this. Pass an id
        to :meth:`partial_close` as ``lot=...`` to target that lot.
        """
        return tuple(
            p.position_id for p in self._state.portfolio.positions.get(sym, ())
        )

    def avg_entry(self, sym: str) -> float | None:
        """Quantity-weighted average entry price across ``sym``'s lots."""
        from src.bt.portfolio.pure import avg_entry

        return avg_entry(self._state.portfolio, sym)

    def current_equity(self) -> float:
        """Live mark-to-market equity: cash + unrealized across all open lots.

        Flat book -> equals cash (grows/shrinks with realized PnL). Used as the
        base for ``size_mode="equity"`` sizing so a ``size`` is a live fraction
        of the current book rather than a fixed amount of seed capital.
        """
        portfolio = self._state.portfolio
        positions_value = sum(
            pos.qty * pos.last_price
            for lots in portfolio.positions.values()
            for pos in lots
        )
        return float(portfolio.cash + positions_value)

    # -- lot-targeted mutation -------------------------------------------------

    def partial_close(
        self,
        sym: str,
        qty: float,
        lot: str = "",
        tag: str = "",
        reason: str = "partial close",
    ) -> None:
        """Release a fraction of a specific lot in ``sym``.

        ``qty`` is a fraction ``(0, 1]`` of the target lot's current quantity
        to shed (``0.25`` = release a quarter of the lot's shares). ``lot``
        targets by ``position_id``; ``tag`` targets by the ``ctx.long(...,
        tag=...)`` label; for neither, the newest lot is used. Fills as a
        ``rebalance`` reduce, realizing PnL on the released shares and keeping
        the surviving shares' cost basis (see ``_rebalance_position``). A no-op
        when the lot is flat or ``qty`` <= 0.
        """
        lots = self._state.portfolio.positions.get(sym, ())
        from src.bt.portfolio.pure import resolve_lot

        assert not self._readonly, "plot() must be read-only"
        target = resolve_lot(lots, lot=lot, tag=tag)
        if target is None or qty <= 0:
            return
        release = round(target.qty * qty, 4)
        if release <= 0:
            return
        price = self.price(sym)
        self._signals.append(
            TradeSignal(
                action=ActionType.rebalance,
                symbol=sym,
                timestamp=self._candle.timestamp,
                price=price,
                qty=-release,
                reason=reason,
                position_id=target.position_id,
            )
        )

    # -- signal emission ---------------------------------------------------------

    def long(
        self,
        sym: str,
        size: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        tag: str = "",
        reason: Any = "long",
        size_mode: Literal["capital", "equity"] = "capital",
    ) -> None:
        """Open a long position in ``sym`` ``size`` fraction of capital.

        Always opens a **fresh lot** — this is Pine ``entry`` semantics, never
        a netting adjust. ``size`` is a 0..1 fraction of a capital base
        converted to an absolute share count (``size * base / price``) before
        emission — a fixed-size order, not scaled by available cash.

        ``size_mode='capital'`` (default) sizes off *initial* capital (legacy;
        fixed dollar amount). ``size_mode='equity'`` sizes off *current* MTM
        equity so a ``size`` is a live fraction that grows/shrinks with PnL and
        keeps capital utilization high — the share count is recomputed from the
        live book at emit time. When ``size`` is omitted the engine's shared
        sizing layer sizes the position (signal emitted with ``qty=0``),
        regardless of ``size_mode``.
        ``sl``/``tp`` are fractional percentages converted to absolute levels.
        ``tag`` is an optional strategy-facing lot label (Pine entry-name
        analogue, e.g. ``"spy-r1"``) stored on the :class:`Position` so
        ``partial_close(..., tag=...)`` is readable lot targeting instead of
        raw ``position_id`` strings.
        """
        assert not self._readonly, "plot() must be read-only"
        self._emit(ActionType.long, sym, size, sl, tp, reason, tag, size_mode)

    def short(
        self,
        sym: str,
        size: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        tag: str = "",
        reason: Any = "short",
        size_mode: Literal["capital", "equity"] = "capital",
    ) -> None:
        """Open a short position in ``sym`` ``size`` fraction of capital.

        ``size`` is a 0..1 fraction of a capital base converted to an absolute
        share count (``size * base / price``) before emission — a fixed-size
        order, not scaled by available cash.

        ``size_mode='capital'`` (default) sizes off *initial* capital (legacy;
        fixed dollar amount). ``size_mode='equity'`` sizes off *current* MTM
        equity so ``size`` is a live fraction that grows/shrinks with PnL and
        keeps capital utilization high — the share count is recomputed from the
        live book at emit time. When ``size`` is omitted the engine's shared
        sizing layer sizes the position (signal emitted with ``qty=0``),
        regardless of ``size_mode``. ``sl``/``tp`` are fractional percentages
        converted to absolute levels.
        ``tag`` is an optional strategy-facing lot label (Pine entry-name
        analogue) stored on the :class:`Position` for readable ``partial_close``
        lot targeting.
        """
        assert not self._readonly, "plot() must be read-only"
        self._emit(ActionType.short, sym, size, sl, tp, reason, tag, size_mode)

    def close(
        self, sym: str, reason: Any = "close", guard_price: float | None = None
    ) -> None:
        """Close **every** open lot in ``sym`` (invoke-all).

        Emits one position-targeted ``close`` signal per open lot — matching
        the design's "``close`` = invoke-all, ``long`` = always-new" rule with no
        ``exclusive_orders`` ambiguity. A no-op when ``sym`` is flat.

        ``guard_price``, when given, models an intra-bar stop trigger: each
        close signal carries it plus the position side so the engine fills at
        the adverse worse-of ``(guard, next_open)`` (see ``execute_signal``)
        instead of naively at next open.

        The multiple emits share a symbol bucket; each carries its own
        ``position_id`` and fills at next bar's open, so the engine's per-symbol
        drain closes each lot independently (no dangling ``close`` with
        ``position_id=None``).
        """
        lots = self._state.portfolio.positions.get(sym, ())
        assert not self._readonly, "plot() must be read-only"
        if not lots:
            return
        price = self.price(sym)
        for pos in lots:
            self._signals.append(
                TradeSignal(
                    action=ActionType.close,
                    symbol=sym,
                    timestamp=self._candle.timestamp,
                    price=price,
                    reason=reason,
                    position_id=pos.position_id,
                    fill_guard_price=guard_price,
                    fill_guard_is_long=(
                        None if guard_price is None else pos.type == ActionType.long
                    ),
                )
            )

    def set_stops(
        self,
        sym: str,
        sl: float | None = None,
        tp: float | None = None,
        reason: Any = "stops",
        tag: str = "",
    ) -> None:
        """Arm/adjust SL/TP levels for EVERY open lot in ``sym`` (invoke-all).

        Emits one ``stop_update`` ``TradeSignal`` per open lot, each targeting
        its own ``position_id``; the portfolio layer ratchets the levels
        (tighten-only, never widen). ``sl``/``tp`` here are ABSOLUTE PRICES —
        unlike ``ctx.long(sl=)``, whose ``sl`` is a fraction of the entry price.
        ``None`` for a leg leaves it unchanged. A no-op when ``sym`` is flat.

        Timing: the level takes effect from the NEXT bar. The engine drains the
        update out of the fill path and applies it AFTER the current bar's own
        risk check, so a level computed on bar t can never stop out the position
        on bar t itself (no intra-bar look-ahead); Stage 8 of bar t+1 then fires
        it intrabar at the trigger (gap-adjusted).
        """
        lots = self._state.portfolio.positions.get(sym, ())
        assert not self._readonly, "plot() must be read-only"
        if not lots:
            return
        price = self.price(sym)
        for pos in lots:
            self._signals.append(
                TradeSignal(
                    action=ActionType.stop_update,
                    symbol=sym,
                    timestamp=self._candle.timestamp,
                    price=price,
                    qty=0.0,
                    stop_loss=sl,
                    take_profit=tp,
                    reason=reason,
                    fill_at_next_open=False,
                    position_side=pos.type,
                    position_id=pos.position_id,
                    tag=tag,
                )
            )

    def _emit(
        self,
        action: ActionType,
        sym: str,
        size: float | None,
        sl: float | None,
        tp: float | None,
        reason: Any,
        tag: str = "",
        size_mode: Literal["capital", "equity"] = "capital",
    ) -> None:
        self._assert_tradable(action, sym)
        price = self.price(sym)
        is_long = action == ActionType.long
        sl_price, tp_price = sl_tp_from_pct(
            price, sl or 0.0, tp or 0.0, is_long=is_long
        )
        # Explicit ``size`` -> 0..1 fraction of a capital base -> absolute share
        # count. ``size_mode`` picks the base: ``"capital"`` = initial capital
        # (legacy fixed-dollar behavior); ``"equity"`` = live MTM equity, so a
        # ``size`` tracks the growing/shrinking book (compounds with PnL, keeps
        # utilization high). Omitted ``size`` -> qty=0, sized by the engine's
        # shared sizing layer regardless of ``size_mode``.
        base = (
            self._state.portfolio.initial_capital
            if size_mode == "capital"
            else self.current_equity()
        )
        qty = 0.0 if size is None else round(size * base / price, 4)
        self._signals.append(
            TradeSignal(
                action=action,
                symbol=sym,
                timestamp=self._candle.timestamp,
                price=price,
                qty=qty,
                reason=reason,
                stop_loss=sl_price,
                take_profit=tp_price,
                tag=tag,
            )
        )

    def _assert_tradable(self, action: ActionType, sym: str) -> None:
        """Reject a position-opening emission on a warmup bar.

        Warmup bars exist so accumulators fill without trading; a ``long``/
        ``short`` emitted there could never fill (the engine skips execution
        during warmup), and discarding it would leave the strategy's own
        ``ctx.shared`` bookkeeping permanently out of sync with the engine's
        book. Fail loudly instead: a strategy that is not ready must return
        early on ``ctx.phase == "warmup"``.
        """
        if self._phase != "warmup":
            return
        raise WarmupTradeError(
            f"{action.value} signal for {sym!r} emitted at {self._candle.timestamp} "
            "during the warmup window (before trading_start). Warmup bars fill "
            "strategy state only — no fill can happen there. Guard the decision "
            "with `if ctx.phase == 'warmup': return` (or move the entry behind "
            "the strategy's own bar-count readiness gate)."
        )

    # -- Pine built-ins (pure; operate on cursor-truncated views / floats) ------

    def nz(self, v: float, fallback: float = 0.0) -> float:
        return v if v == v else fallback

    def cross_over(self, a, b) -> bool:
        return _cross(a, b, over=True)

    def cross_under(self, a, b) -> bool:
        return _cross(a, b, over=False)

    def change(self, series: SeriesView, bars: int = 1) -> float:
        return series.change(bars)

    def barssince(self, pred: Callable[[int], bool], max_bars: int = 500) -> float:
        """Pine ``barssince`` — bars ago the ``pred(offset)`` was last True.

        ``pred(i)`` tests the bar ``i`` bars in the past (``i=0`` = current
        candle). Returns NaN when ``pred`` is never True within ``max_bars``.
        """
        for i in range(max_bars):
            if pred(i):
                return float(i)
        return float("nan")


# ---------------------------------------------------------------------------
# decorator + wrapper
# ---------------------------------------------------------------------------


class _StrategyAdapter:
    """Callable engine hook produced by ``@strategy``.

    Wraps a user ``def on_candle(ctx)`` so it exposes the engine contract
    ``on_candle(state, candle, params) -> list[TradeSignal]`` while building a
    cursor-safe :class:`StrategyContext` for the decision body. Created per
    decoration; holds the cross-call state holder (when stateful).
    """

    __slots__ = (
        "ctx_fn",
        "params_cls",
        "interval",
        "stateful",
        "__name__",
    )

    def __init__(
        self,
        fn: FunctionType,
        bars: str,
        stateful: bool,
    ) -> None:
        self.ctx_fn = fn
        self.params_cls = None
        self.interval = bars
        self.stateful = stateful
        self.__name__ = "on_candle"

    def reset(self) -> None:
        """Wired as the module's ``reset_global`` for split/sweep back-compat.

        With per-run state holders (minted fresh by the engine for every run
        window), there is no module-level dict to clear — a new window simply
        gets a new holder. Kept as a no-op so old ``bt split``/``bt sweep``
        callers that invoke ``reset_global()`` between windows keep working
        without racing on shared state.
        """
        return None

    def __call__(
        self,
        state: "BacktestState",
        candle: "Candle",
        params: Any,
    ) -> list[TradeSignal]:
        ta = getattr(state.candles, "ta", None)
        if not isinstance(ta, TaContext):
            raise RuntimeError(
                "DSL strategy requires a prefetched TaContext; run through "
                "`src.bt.engine.backtest.run` (it builds `ta` from data) so "
                f"state.candles.ta is set for module {self.ctx_fn.__module__}."
            )
        # Fundamentals are optional (a strategy may never read them) and arrive
        # on the store as a loose `Any` — the DSL narrows it here. A non-Fundamentals
        # value means someone attached the wrong object, so fail instead of
        # silently serving an empty store.
        raw_fundamentals = getattr(state.candles, "fundamentals", None)
        fundamentals: Fundamentals | None = None
        if raw_fundamentals is not None:
            if not isinstance(raw_fundamentals, Fundamentals):
                raise RuntimeError(
                    "state.candles.fundamentals is not a Fundamentals store "
                    f"(got {type(raw_fundamentals).__name__})."
                )
            fundamentals = raw_fundamentals
        holder = None
        if self.stateful:
            holder = getattr(state.candles, "strategy_state", None)
            if not isinstance(holder, dict):
                raise RuntimeError(
                    "Stateful DSL strategy requires a per-run state holder; run "
                    "through `src.bt.engine.backtest.run` (it mints a fresh "
                    "holder per window) so `state.candles.strategy_state` is a "
                    f"dict for module {self.ctx_fn.__module__}."
                )
        # The engine only fires on_candle on base-interval candles (HTF-only
        # candles skip signal generation), so ``candle.interval`` is the true
        # signal interval at every call -- i.e. ``config.bars[0]`` regardless of
        # the (per-strategy) decorator ``bars``. Derive the context interval from
        # the candle so a config using a non-``bars`` base bar (e.g. "1h") stays
        # consistent with the TaContext base interval. ``self.interval`` is kept
        # for introspection/back-compat only.
        ctx = StrategyContext(
            state=state,
            candle=candle,
            params=params,
            ta=ta,
            symbols=symbols_from(state),
            interval=candle.interval or self.interval,
            fundamentals=fundamentals,
        )
        if holder is not None:
            ctx.shared = holder
        # Mirror the engine's phase so a strategy can tell a state-warming bar
        # from a tradable one (``ctx.phase == "warmup"``).
        ctx._phase = getattr(state.candles, "phase", "trade")
        self.ctx_fn(ctx)
        return ctx._signals


def strategy(bars: str = "1d", stateful: bool = False):
    """Decorate ``def on_candle(ctx: StrategyContext)`` into an engine hook.

    Returns an adapter (callable ``on_candle(state, candle, params) ->
    list[TradeSignal]``) wrapping the plain decision function with a
    ``StrategyContext``.

    Cross-candle state (``stateful=True``) is **per-run**: the engine mints a
    fresh holder for every ``run``/window and the adapter reads it from
    ``state.candles.strategy_state``. Nothing is shared at module scope, so a
    stateless OR stateful DSL strategy is thread-safe across concurrent
    ``run_split``/``run_sweep``/``run_optimize`` workers. ``reset_global`` is
    kept as a no-op shim for split/sweep back-compat.

    Args:
        bars: signal interval served by ``ctx`` (matches the config base bar).
        stateful: when True, persist cross-call state in ``ctx.shared`` (a
            per-run dict, fresh for every run window).

    Invariant: a module-level ``plot`` (if present) is NOT part of the engine
    contract. It is never called from ``on_candle`` and is never invoked by the
    engine; ``output.render_plot_json`` calls it post-run, once per symbol.
    """

    def decorate(fn: FunctionType):
        adapter = _StrategyAdapter(fn, bars, stateful)
        module = sys.modules.get(fn.__module__)
        if module is not None:
            adapter.params_cls = getattr(module, "Params", None)
            strategy_type = getattr(module, "STRATEGY_TYPE", None)
            if strategy_type is None:
                strategy_type = getattr(fn, "STRATEGY_TYPE", None)
            if strategy_type is not None:
                setattr(module, "STRATEGY_TYPE", strategy_type)
            # reset_global is a no-op back-compat shim. State is per-run (minted
            # fresh by the engine per window), so cross-window bleed is already
            # impossible without any module-level clear.
            setattr(module, "reset_global", adapter.reset)
        return adapter

    return decorate


def symbols_from(state: "BacktestState") -> tuple[str, ...]:
    """Symbols present in the candle store's accumulator (deduped, ordered).

    Dedups by symbol across every interval key (base + any HTF), preserving the
    store's deterministic insertion order. O(S) via a set, not an O(S²) list scan.
    By the time ``on_candle`` fires, the store is populated for all configured
    symbols, so this is authoritative at call sites.
    """
    seen: set[str] = set()
    result: list[str] = []
    for k in state.candles.keys():
        sym = k[0]
        if sym not in seen:
            seen.add(sym)
            result.append(sym)
    return tuple(result)


# ---------------------------------------------------------------------------
# cross helpers
# ---------------------------------------------------------------------------


def _read_position(v) -> float:
    return v[-1] if isinstance(v, SeriesView) else float(v)


def _read_previous(v) -> float:
    return v[-2] if isinstance(v, SeriesView) else float(v)


def _cross(a, b, over: bool) -> bool:
    """True when ``a`` crossed ``b`` on the current bar in the given direction.

    SeriesViews read the current + previous cursor-truncated values (O(1), no
    lookahead); raw floats compare against themselves (a degenerate, usually
    false, single-value cross).
    """
    a_cur, b_cur = _read_position(a), _read_position(b)
    a_prev, b_prev = _read_previous(a), _read_previous(b)
    if over:
        return a_prev <= b_prev and a_cur > b_cur
    return a_prev >= b_prev and a_cur < b_cur


__all__ = [
    "strategy",
    "StrategyContext",
    "WarmupTradeError",
    "SeriesView",
    "OhlcvView",
    "TaContext",
    "Fundamentals",
    "symbols_from",
]
