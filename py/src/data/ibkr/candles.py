import asyncio
from datetime import datetime, timedelta, date, timezone
from typing import Callable, Optional, cast

from peewee import fn
from src.data.types import CandleSchema, db
from ib_rest_api_client.api.trading_market_data import get_iserver_marketdata_history
from ib_rest_api_client.models import IserverHistoryBidAskResponse
from ib_rest_api_client.models import GetIserverMarketdataHistoryDirection

from .client import default_client
from .shared import get_contract_info
from .rate_limiter import with_retry
from src.data.types import CandleDict
from src.data.xcal import is_non_trading_day
from src.shared.runtime_log import (
    error as _log_error,
    info as _log_info,
    warn as _log_warn,
)

# Maximum number of candles per API request
MAX_CANDLES_PER_REQUEST = 1000

# Maximum period (in days) for a single IBKR API request.
# The API rejects periods longer than 365 days. For bar sizes shorter
# than 1d (e.g. 1h), the effective limit may be lower, but 365d is the
# documented maximum for daily bars and serves as an upper bound.
MAX_PERIOD_DAYS = 365

# Pause (seconds) between chunked history requests in _fetch_candles_iterative.
# IBKR caps each response at ~2000 bars, so a multi-year hourly backfill issues
# many back-to-back calls that can trip the gateway's server-side rate limiter
# (HTTP 503). This delay spreads the requests to stay under that threshold.
CHUNK_DELAY_S = 1.0

# Upper bound (seconds) for the adaptive inter-chunk delay. Doubled on repeated
# HTTP 503 (rate-limit) responses so a sustained burst backs off up to this cap
# before giving up, then decays back toward CHUNK_DELAY_S once requests succeed.
CHUNK_DELAY_MAX_S = 60.0


def _chunked_insert(insert_data: list[CandleDict], batch_size: int = 500) -> None:
    """Insert candles in batches to avoid SQLite's variable limit."""
    for i in range(0, len(insert_data), batch_size):
        batch = insert_data[i : i + batch_size]
        with db.atomic():
            CandleSchema.insert_many(batch).on_conflict_ignore().execute()


def date_to_timestamp(d: date) -> int:
    return int(datetime.combine(d, datetime.min.time()).timestamp() * 1000)


def timestamp_to_datetime(ts: int) -> datetime:
    """Convert ms epoch timestamp (UTC) to naive UTC datetime."""
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).replace(tzinfo=None)


def _is_all_non_trading(gap_start: datetime, gap_end: datetime) -> bool:
    """Check if every date in [gap_start, gap_end] is a non-trading day.

    The boundaries are partial by construction, so neither end is judged as a
    whole day:

    - **Start day** is skipped always. A gap begins at the last stored bar —
      mid-session — so the remainder of that day was never expected to exist.
      Deliberately date-based: the universe spans markets with different session
      hours, so no single close time would be right.
    - **Any day from today onward** is skipped. Such a day may still be trading,
      or may have closed hours ago without the vendor settling it — neither is
      missing history.

    Examples (today = Mon, last bar Fri 15:30, gap runs to Tue):
      → Sun 10:23:  Sat+Sun are non-trading, nothing else in range → True
      → Mon 23:59:  Mon is today, unsettled → True
      → Tue 09:00:  Tue is not today, unsettled → True
      → Thu 09:00:  Wed is past and a trading day → False
    """
    # A day is owed bars only once it is safely behind us. Any day from today
    # onward may still be trading or may have closed hours ago without the
    # vendor settling it, so the whole unsettled tail owes nothing.
    settled_before = _today_utc()

    d = gap_start.date()
    end = gap_end.date()
    while d <= end:
        dt = datetime.combine(d, datetime.min.time())
        # The gap starts mid-session at the last stored bar, so the rest of
        # that day is not owed.
        if d == gap_start.date():
            d += timedelta(days=1)
            continue
        # Skip any day that has not settled yet.
        if d >= settled_before:
            d += timedelta(days=1)
            continue
        if not is_non_trading_day(dt):
            return False
        d += timedelta(days=1)
    return True


def _today_utc() -> date:
    """UTC today — the clock domain of every stored timestamp.

    ``timestamp_to_datetime`` yields naive UTC, so the forward edge must be
    judged in UTC too; a local date runs a day ahead east of Greenwich. A seam,
    so a test can pin the forward-edge clock.
    """
    return datetime.now(timezone.utc).date()


def calculate_gaps(
    requested_from: datetime,
    requested_to: datetime,
    oldest_existing: Optional[int],
    newest_existing: Optional[int],
) -> list[tuple[datetime, datetime]]:
    """Calculate gaps between requested date range and existing data.

    Pure function that returns a list of (start, end) tuples representing
    missing date ranges that need to be fetched.

    Detects TWO kinds of gaps:
    1. Backward gap: before the oldest existing candle
    2. Forward gap: after the newest existing candle

    Gaps that consist entirely of non-trading days (weekends + holidays)
    are filtered out.

    Internal gaps (missing days BETWEEN existing candles) are detected
    separately by find_internal_gaps() — that function operates at day
    granularity, while this function uses exact timestamp precision.

    Args:
        requested_from: Start of the requested datetime range
        requested_to: End of the requested datetime range
        oldest_existing: Oldest timestamp in DB (ms), or None if empty
        newest_existing: Newest timestamp in DB (ms), or None if empty

    Returns:
        List of (start_datetime, end_datetime) tuples for gaps to fetch,
        sorted chronologically.
    """
    gaps: list[tuple[datetime, datetime]] = []

    if oldest_existing is None and newest_existing is None:
        return [(requested_from, requested_to)]

    if oldest_existing is None:
        oldest_existing = newest_existing
    if newest_existing is None:
        newest_existing = oldest_existing

    existing_from_ts = cast(int, oldest_existing)
    existing_to_ts = cast(int, newest_existing)

    existing_from = timestamp_to_datetime(existing_from_ts)
    existing_to = timestamp_to_datetime(existing_to_ts)

    # 1. Backward gap: strictly before the oldest existing candle
    if requested_from < existing_from:
        if not _is_all_non_trading(requested_from, existing_from):
            gaps.append((requested_from, existing_from))

    # 2. Forward gap: strictly after the newest existing candle
    if existing_to < requested_to:
        if not _is_all_non_trading(existing_to, requested_to):
            gaps.append((existing_to, requested_to))

    return gaps


def find_internal_gaps(
    ticker: str,
    requested_from: datetime,
    requested_to: datetime,
    max_gap_hours: float = 48.0,
) -> list[tuple[datetime, datetime]]:
    """Find missing data INTERNAL to the existing range by scanning
    consecutive candle timestamps.

    Unlike calculate_gaps() which only looks at the edges (oldest/newest),
    this function detects a gap between two existing candles where the
    time difference exceeds max_gap_hours — indicating a full trading
    day (or more) of missing data.

    This is needed because IBKR's API may fail to return certain days
    during a fetch, or a previous sync may have completed but skipped
    a trading day (e.g. due to API limits or transient errors that were
    swallowed).

    Args:
        ticker: Symbol to check for internal gaps
        requested_from: Lower bound for gap detection
        requested_to: Upper bound for gap detection
        max_gap_hours: Max allowed hours between consecutive candles.
                        Default 48h (covers weekend Fri 21:00 → Mon 15:30).

    Returns:
        List of (gap_start, gap_end) tuples for internal gaps found.
    """
    from peewee import SQL

    # Get all distinct trading days within the requested range
    days: list[str] = [
        r.date
        for r in CandleSchema.select(
            SQL("DATE(datetime(timestamp/1000, 'unixepoch'))").alias("date")
        )
        .where(
            CandleSchema.ticker == ticker,
            CandleSchema.timestamp >= int(requested_from.timestamp() * 1000),
            CandleSchema.timestamp <= int(requested_to.timestamp() * 1000),
        )
        .distinct()
        .order_by(SQL("date"))
    ]

    if not days:
        return []

    gaps: list[tuple[datetime, datetime]] = []
    for i in range(len(days) - 1):
        curr = datetime.strptime(days[i], "%Y-%m-%d")
        nxt = datetime.strptime(days[i + 1], "%Y-%m-%d")
        gap_days = (nxt - curr).days

        if gap_days == 1:
            # Adjacent trading days — no gap
            continue

        # Scan each missing day to find the first and last non-holiday/non-weekend days
        missing_start: datetime | None = None
        missing_end: datetime | None = None

        for offset in range(1, gap_days):
            missing_day = curr + timedelta(days=offset)
            if is_non_trading_day(missing_day):
                # If we were tracking a gap range, close it before the holiday
                if missing_start is not None:
                    if missing_end is not None:
                        gaps.append((missing_start, missing_end))
                    missing_start = None
                    missing_end = None
                continue

            # This is a real missing trading day
            if missing_start is None:
                missing_start = missing_day
            missing_end = missing_day + timedelta(days=1)  # end is exclusive

        # Close any remaining gap range
        if missing_start is not None and missing_end is not None:
            gaps.append((missing_start, missing_end))

    return gaps


def _merge_and_sort_gaps(
    gaps: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Merge overlapping/adjacent gaps and sort chronologically.

    Args:
        gaps: List of (start, end) gap tuples (unsorted, possibly overlapping)

    Returns:
        Merged, sorted list of non-overlapping gaps
    """
    if not gaps:
        return []

    # Sort by start time
    sorted_gaps = sorted(gaps, key=lambda g: g[0])
    merged: list[tuple[datetime, datetime]] = [sorted_gaps[0]]

    for start, end in sorted_gaps[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping or adjacent — merge
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def get_existing_range_sync(ticker: str) -> tuple[Optional[int], Optional[int]]:
    """Returns (oldest_timestamp, newest_timestamp) for ticker, or (None, None) if empty.

    Synchronous — use get_existing_range() from async contexts to avoid blocking the event loop.
    """
    result = (
        CandleSchema.select(
            fn.MIN(CandleSchema.timestamp).alias("oldest"),
            fn.MAX(CandleSchema.timestamp).alias("newest"),
        )
        .where(CandleSchema.ticker == ticker)
        .first()
    )

    if result is None or result.oldest is None:
        return None, None
    return result.oldest, result.newest


async def get_existing_range(ticker: str) -> tuple[Optional[int], Optional[int]]:
    """Async wrapper around get_existing_range_sync to avoid blocking the event loop."""
    return await asyncio.to_thread(get_existing_range_sync, ticker)


async def _fetch_candles_iterative(
    conid: int,
    ticker: str,
    bar: str,
    from_datetime: datetime,
    to_datetime: datetime,
    on_chunk: Optional[Callable[[list[CandleDict]], None]] = None,
) -> list[CandleDict]:
    """Iteratively fetch candles from IBKR API for a given datetime range.

    Walks BACKWARD from ``to_datetime`` down toward ``from_datetime`` in chunks
    capped at MAX_PERIOD_DAYS, using ``direction=-1`` (backward) anchored at the
    current walk cursor.

    Empirically, IBKR's history endpoint serves backward requests reliably
    (``direction=-1``), while the forward variant (``direction=1``) returns HTTP
    500 ``Chart data unavailable`` for all requests on common gateways. Anchoring
    ``startTime`` at the newest boundary and walking backward also keeps the
    cursor advancing monotonically toward the past, so multi-year backfills stay
    within the per-request ``period`` cap without drifting stale cursors.

    Each API response is a ``chunk`` of candles. When ``on_chunk`` is supplied it
    is called with every chunk as soon as it arrives — letting the caller write
    candles to the DB immediately instead of buffering the whole backfill in
    memory. If ``on_chunk`` is None (e.g. for pure tests of the fetch loop), the
    chunks are accumulated in memory and returned. The return value is the list
    of fetched candles in chronological order in both cases.
    """
    first_chunk = True
    current_to = to_datetime
    accumulated_data: list[CandleDict] = []
    # Adaptive pacing. A multi-year backfill issues many sequential history calls
    # (IBKR caps each response at 2000 bars and/or 365d), and a burst of
    # back-to-back requests trips the gateway's server-side rate limiter (HTTP
    # 503). We keep a short base delay between chunks and throttle it up on a 503
    # so a sustained burst backs off instead of aborting the whole symbol.
    delay = CHUNK_DELAY_S

    while current_to > from_datetime:
        if not first_chunk:
            await asyncio.sleep(delay)
        first_chunk = False

        delta = current_to - from_datetime
        days_to_fetch = max(1, delta.days + (1 if delta.seconds > 0 else 0))

        if delta.days > MAX_PERIOD_DAYS:
            _log_info(f"range {from_datetime}→{current_to}>{MAX_PERIOD_DAYS}d chunks")
        days_to_fetch = min(days_to_fetch, MAX_PERIOD_DAYS)

        resp = await get_iserver_marketdata_history.asyncio_detailed(
            client=default_client().rest,
            conid=conid,
            bar=bar,
            period=f"{days_to_fetch}d",
            start_time=current_to.strftime("%Y%m%d-%H:%M:%S"),
            direction=GetIserverMarketdataHistoryDirection.VALUE_NEGATIVE_1,
        )
        r = resp.parsed

        if resp.status_code == 503:
            # Gateway rate-limit throttle: grow the inter-chunk delay and retry
            # the same window after backing off, rather than aborting the whole
            # multi-year backfill over a transient burst.
            delay = min(delay * 2.0, CHUNK_DELAY_MAX_S)
            _log_warn(f"{ticker} 503 throttled to {delay}s, retry window {current_to}")
            await asyncio.sleep(delay)
            continue

        if (
            resp.status_code != 200
            or not isinstance(r, IserverHistoryBidAskResponse)
            or not r.data
        ):
            _log_error(f"{ticker} bad response status={resp.status_code}")
            raise Exception(f"Unexpected response: status={resp.status_code}")

        # Success — gently decay the delay back toward the base.
        if delay > CHUNK_DELAY_S:
            delay = max(CHUNK_DELAY_S, delay / 2.0)

        sorted_data = sorted(r.data, key=lambda x: x.t)
        if not sorted_data or not sorted_data[0].t:
            _log_info(f"{ticker}: no more data")
            break

        oldest_ts = cast(int, sorted_data[0].t)

        new_candles = cast(
            list[CandleDict],
            [
                {
                    "conid": conid,
                    "ticker": ticker,
                    "timestamp": item.t,
                    "open": item.o,
                    "high": item.h,
                    "low": item.l,
                    "close": item.c,
                    "volume": item.v,
                }
                for item in sorted_data
            ],
        )
        # Write the chunk as soon as it arrives when a sink is provided;
        # otherwise buffer it (backward-compatible path for tests).
        if on_chunk is not None:
            on_chunk(new_candles)
        else:
            accumulated_data = new_candles + accumulated_data

        oldest_datetime = timestamp_to_datetime(oldest_ts)
        _log_info(f"{ticker} got {len(sorted_data)} candles, oldest {oldest_datetime}")

        # We've reached or passed the from_datetime boundary
        if oldest_datetime <= from_datetime:
            break
        # Safety: if oldest_datetime hasn't advanced, avoid infinite loop
        if oldest_datetime >= current_to:
            break
        current_to = oldest_datetime

    return accumulated_data


@with_retry(max_retries=3, base_delay_ms=1000, max_delay_ms=10000)
async def candles(
    conid: int,
    from_datetime: datetime,
    to_datetime: Optional[datetime] = None,
    bar: str = "1h",
    force_fetch: bool = False,
    ticker: Optional[str] = None,
    on_chunk: Optional[Callable[[list[CandleDict]], None]] = None,
) -> list[tuple[datetime, datetime]]:
    """Fetch candles from IBKR API with rate limiting and retry support.

    Args:
        conid: Contract ID
        from_datetime: Start datetime for fetching
        to_datetime: End datetime for fetching (defaults to now)
        bar: Bar size (e.g., "1h", "1d")
        force_fetch: If True, ignore existing data and fetch all (default: False)
        ticker: Optional ticker symbol. If not provided, will fetch from API.
        on_chunk: Called with each fetched chunk as it arrives, so a caller can
            write candles without buffering a multi-year backfill. ``None``
            writes each chunk to the DB (the default, standalone behaviour).

    Returns:
        The merged gap ranges still missing AFTER the fetch — empty means the
        symbol holds the full requested window. One slice is the contract the
        operator report and a bot's health both read.
    """
    if to_datetime is None:
        to_datetime = datetime.now()

    if ticker is None:
        symbol_info = await get_contract_info(conid)
        ticker = symbol_info.ticker

    oldest_existing, newest_existing = await get_existing_range(ticker)

    gaps: list[tuple[datetime, datetime]]
    if force_fetch:
        gaps = [(from_datetime, to_datetime)]
    else:
        # Edge gaps: before oldest / after newest
        gaps = calculate_gaps(
            from_datetime, to_datetime, oldest_existing, newest_existing
        )
        # Internal gaps: missing days BETWEEN existing candles
        internal_gaps = await asyncio.to_thread(
            find_internal_gaps, ticker, from_datetime, to_datetime
        )
        gaps.extend(internal_gaps)
        # Deduplicate and sort
        gaps = _merge_and_sort_gaps(gaps)

    if not gaps:
        return []

    _log_info(f"{ticker} {len(gaps)} gap(s) {gaps[0][0]}→{gaps[-1][1]}")

    def _write_chunk(chunk: list[CandleDict]) -> None:
        # Write each chunk as it arrives so a multi-year backfill never buffers
        # the whole range in memory.
        try:
            _chunked_insert(chunk, batch_size=500)
        except Exception:
            _log_error(f"bulk insert failed for {ticker}")
            raise

    sink = on_chunk if on_chunk is not None else _write_chunk
    for gap_start, gap_end in gaps:
        await _fetch_candles_iterative(
            conid, ticker, bar, gap_start, gap_end, on_chunk=sink
        )

    return await remaining_gaps(ticker, from_datetime, to_datetime)


async def remaining_gaps(
    ticker: str,
    from_datetime: datetime,
    to_datetime: datetime,
) -> list[tuple[datetime, datetime]]:
    """Gaps still missing for *ticker* over the window, after a fetch.

    Re-reads the stored edges and re-scans internals, so it reports what a
    later reader would actually find rather than what a fetch claimed to write.
    """
    oldest, newest = await get_existing_range(ticker)
    edge = calculate_gaps(from_datetime, to_datetime, oldest, newest)
    internal = await asyncio.to_thread(
        find_internal_gaps, ticker, from_datetime, to_datetime
    )
    return _merge_and_sort_gaps(edge + internal)


def gap_range(
    gaps: tuple[tuple[datetime, datetime], ...] | list[tuple[datetime, datetime]],
) -> str:
    """Short range string for a gap list, e.g. "2026-10-13→2026-10-15"."""
    if not gaps:
        return "-"
    start = gaps[0][0].strftime("%Y-%m-%d")
    end = gaps[-1][1].strftime("%Y-%m-%d")
    return f"{start}→{end}"


async def candles_batch(
    conids: list[int],
    from_datetime: datetime,
    to_datetime: Optional[datetime] = None,
    max_concurrent: int = 1,
    bar: str = "1h",
    on_chunk: Optional[Callable[[list[CandleDict]], None]] = None,
    on_item: Optional[
        Callable[[int, Optional[list[tuple[datetime, datetime]]]], None]
    ] = None,
) -> dict[int, list[tuple[datetime, datetime]] | None]:
    """Fetch candles for multiple conids with rate limiting and batching.

    Args:
        conids: List of conids to fetch candles for
        from_datetime: Start datetime for fetching
        to_datetime: End datetime for fetching (defaults to now)
        max_concurrent: Maximum concurrent requests
        bar: Bar size (e.g., "1h", "1d")
        on_chunk: Sink for fetched chunks (default: write straight to the DB)
        on_item: Called once per conid with its gaps still missing, or ``None``
            when the fetch raised — the only per-symbol signal a progress bar
            and a report need.

    Returns:
        ``{conid: gaps still missing}`` for every conid; a ``None`` value means
        that symbol's fetch raised. ``candles()`` already computed those gaps, so
        they are passed through rather than re-scanned here.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_limit(
        c: int,
    ) -> tuple[int, list[tuple[datetime, datetime]] | None]:
        gaps: list[tuple[datetime, datetime]] | None = None
        try:
            async with semaphore:
                gaps = await candles(
                    c, from_datetime, to_datetime, bar, on_chunk=on_chunk
                )
        except Exception as e:
            _log_error(f"failed to fetch conid {c}: {e}")
            gaps = None
        if on_item is not None:
            on_item(c, gaps)
        return c, gaps

    results = await asyncio.gather(*[fetch_with_limit(c) for c in conids])

    return dict(results)
