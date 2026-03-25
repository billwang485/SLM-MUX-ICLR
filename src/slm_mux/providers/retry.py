"""Shared retry logic with exponential backoff and jitter."""

import functools
import logging
import random
import time
from typing import Any, Callable, Optional, Sequence, Type

logger = logging.getLogger(__name__)

# Sentinel used to detect HTTP status codes on exceptions.
_SERVER_ERROR_RANGE = range(500, 600)
_RATE_LIMIT_STATUS = 429


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Best-effort extraction of an HTTP status code from an exception."""
    # OpenAI / httpx style
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    if isinstance(status, int):
        return status
    # Fallback: scan string representation
    text = str(exc).lower()
    for code in (429, 500, 502, 503, 504):
        if f" {code}" in text or f"status {code}" in text or f"http {code}" in text:
            return code
    return None


def retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: Sequence[Type[Exception]] = (Exception,),
    on_retry: Optional[Callable[..., Any]] = None,
) -> Callable:
    """Decorator that retries a function with exponential backoff and jitter.

    Parameters
    ----------
    max_retries:
        Maximum number of retry attempts for non-transient errors.
        Transient errors (HTTP 429, 5xx) are retried indefinitely.
    base_delay:
        Initial delay in seconds before the first retry.
    max_delay:
        Cap on the computed backoff delay (before status-specific overrides).
    retryable_exceptions:
        Tuple of exception types that should trigger a retry.
    on_retry:
        Optional callback ``(attempt, exception, delay) -> None`` invoked
        before each sleep.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except tuple(retryable_exceptions) as exc:
                    status = _extract_status_code(exc)
                    is_server_error = status is not None and status in _SERVER_ERROR_RANGE
                    is_rate_limit = status == _RATE_LIMIT_STATUS
                    is_transient = is_server_error or is_rate_limit

                    # For non-transient errors, respect max_retries
                    if not is_transient and attempt >= max_retries:
                        logger.error(
                            "retry_with_backoff: %s failed after %d attempts. "
                            "Last error: %s",
                            func.__qualname__,
                            attempt,
                            exc,
                        )
                        raise

                    # Compute delay
                    exp_backoff = min(
                        max_delay,
                        base_delay * (2 ** (attempt - 1)),
                    )
                    jitter = random.random()

                    if is_server_error:
                        # Server errors: wait longer to let backend recover
                        delay = max(60.0, exp_backoff) + jitter
                    elif is_rate_limit:
                        # Rate-limit: shorter sleep, will retry indefinitely
                        delay = max(base_delay, exp_backoff * 0.5) + jitter
                    else:
                        delay = exp_backoff + jitter

                    logger.warning(
                        "retry_with_backoff: %s attempt %d failed (status=%s, "
                        "transient=%s). Retrying in %.1fs. Error: %s",
                        func.__qualname__,
                        attempt,
                        status,
                        is_transient,
                        delay,
                        exc,
                    )

                    if on_retry is not None:
                        try:
                            on_retry(attempt, exc, delay)
                        except Exception:
                            pass

                    time.sleep(delay)

        return wrapper

    return decorator
