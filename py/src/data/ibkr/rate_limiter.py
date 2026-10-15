import asyncio
from functools import wraps
from typing import Callable, TypeVar, Awaitable, ParamSpec
from dataclasses import dataclass

from src.shared.runtime_log import retry as _log_retry
from src.shared.runtime_log import success as _log_success

P = ParamSpec("P")
T = TypeVar("T")

# Monotonic clock — overridable in tests via patch()
_monotonic_sleep = asyncio.sleep


def _set_monotonic_sleep(sleep_fn: Callable[[float], Awaitable[None]]) -> None:
    """Override the sleep implementation (for tests using a mock clock)."""
    global _monotonic_sleep  # noqa: PLW0603
    _monotonic_sleep = sleep_fn


@dataclass(frozen=True)
class RateLimitConfig:
    max_concurrent: int = 2
    min_delay_ms: int = 200
    max_retries: int = 3
    base_delay_ms: int = 500
    max_delay_ms: int = 10000


def with_retry(
    max_retries: int | None = None,
    base_delay_ms: int | None = None,
    max_delay_ms: int | None = None,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
):
    config = RateLimitConfig(
        max_retries=max_retries or 3,
        base_delay_ms=base_delay_ms or 500,
        max_delay_ms=max_delay_ms or 10000,
    )

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: BaseException | None = None
            name = func.__name__  # ty: ignore[unresolved-attribute]
            for attempt in range(config.max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        _log_success(name, f"on attempt {attempt + 1}")
                    return result
                except retryable_exceptions as e:
                    last_exception = e
                    _log_retry(
                        name,
                        attempt + 1,
                        config.max_retries + 1,
                        str(e).split(":")[0].strip(),
                    )
                    if attempt < config.max_retries:
                        delay = (
                            min(
                                config.base_delay_ms * (2**attempt),
                                config.max_delay_ms,
                            )
                            / 1000.0
                        )
                        await _monotonic_sleep(delay)
                    else:
                        raise last_exception  # type: ignore[possibly-undefined]
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator
