"""Thread-safe rate limiting utilities.

All limiters can be used as context managers::

    qps = QPSLimiter(10)
    with qps:
        make_request()

    rpm = RPMLimiter(600)
    with rpm:
        make_request()

    conc = ConcurrencyLimiter(8)
    with conc:
        make_request()
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class QPSLimiter:
    """Thread-safe queries-per-second limiter.

    Enforces a minimum interval of ``1 / max_qps`` seconds between
    successive ``acquire()`` calls (process-wide).
    """

    def __init__(self, max_qps: float) -> None:
        if max_qps <= 0:
            raise ValueError(f"max_qps must be positive, got {max_qps}")
        self._min_interval = 1.0 / max_qps
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def acquire(self) -> None:
        """Block until enough time has passed since the last request."""
        with self._lock:
            now = time.monotonic()
            wait = (self._last_ts + self._min_interval) - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last_ts = now

    def __enter__(self) -> "QPSLimiter":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


class RPMLimiter:
    """Thread-safe requests-per-minute limiter using a sliding window.

    Tracks a 60-second window and sleeps when the budget is exhausted.
    """

    _WINDOW_SECONDS = 60.0

    def __init__(self, max_rpm: float) -> None:
        if max_rpm <= 0:
            raise ValueError(f"max_rpm must be positive, got {max_rpm}")
        self._max_rpm = max_rpm
        self._lock = threading.Lock()
        self._window_start = 0.0
        self._request_count = 0.0

    def acquire(self) -> None:
        """Block until a slot is available within the current 60s window."""
        with self._lock:
            now = time.monotonic()

            # Reset window if uninitialized or expired
            if self._window_start <= 0.0 or (now - self._window_start) >= self._WINDOW_SECONDS:
                self._window_start = now
                self._request_count = 0.0

            # If budget exhausted, sleep until the window resets
            if self._request_count >= self._max_rpm:
                sleep_for = (self._window_start + self._WINDOW_SECONDS) - now
                if sleep_for > 0:
                    logger.debug(
                        "RPMLimiter: budget exhausted (%d/%d), sleeping %.1fs",
                        int(self._request_count),
                        int(self._max_rpm),
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                # Start a new window
                now = time.monotonic()
                self._window_start = now
                self._request_count = 0.0

            self._request_count += 1.0

    def __enter__(self) -> "RPMLimiter":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


class ConcurrencyLimiter:
    """Semaphore-based concurrency limiter.

    Caps the number of concurrent operations to ``max_concurrent``.
    """

    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent <= 0:
            raise ValueError(f"max_concurrent must be positive, got {max_concurrent}")
        self._max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self._inflight = 0

    @property
    def inflight(self) -> int:
        """Current number of in-flight operations (informational)."""
        with self._lock:
            return self._inflight

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire a slot.  Returns True on success, False on timeout."""
        acquired = self._semaphore.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                self._inflight += 1
        return acquired

    def release(self) -> None:
        """Release a previously acquired slot."""
        with self._lock:
            self._inflight -= 1
        self._semaphore.release()

    def __enter__(self) -> "ConcurrencyLimiter":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
