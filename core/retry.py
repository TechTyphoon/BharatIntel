"""
BharatIntel — Retry Decorator with Exponential Backoff

Usage:
    @with_retry(max_attempts=3, backoff_base=2.0, exceptions=(httpx.TimeoutException,))
    async def fetch_feed(url: str) -> str:
        ...
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Sequence, Type

from core.logger import get_logger

log = get_logger("retry")


def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    exceptions: Sequence[Type[BaseException]] = (Exception,),
) -> Callable:
    """
    Decorator for async functions. Retries on specified exceptions
    with exponential backoff: backoff_base^attempt seconds.

    After exhausting attempts, re-raises the last exception.
    """
    caught = tuple(exceptions)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except caught as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        log.error(
                            "retry_exhausted",
                            function=fn.__qualname__,
                            attempt=attempt,
                            error=str(exc),
                        )
                        raise
                    delay = backoff_base ** attempt
                    log.warning(
                        "retry_attempt",
                        function=fn.__qualname__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay_seconds=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]  # unreachable but satisfies type checker

        return wrapper

    return decorator
