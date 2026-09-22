"""Circuit breaker for external API clients."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(self, failures_before_open: int = 5, reset_seconds: int = 60) -> None:
        self.failures_before_open = failures_before_open
        self.reset_seconds = reset_seconds
        self.failures = 0
        self.opened_at: float | None = None

    def call(self, func, *args, **kwargs):
        if self.opened_at is not None:
            if time.monotonic() - self.opened_at < self.reset_seconds:
                raise RuntimeError("Circuit breaker is open; backing off")
            logger.info("Circuit breaker half-open; retrying")
            self.opened_at = None
            self.failures = 0
        try:
            result = func(*args, **kwargs)
        except Exception:
            self.failures += 1
            logger.warning("Circuit breaker failure %d", self.failures)
            if self.failures >= self.failures_before_open:
                self.opened_at = time.monotonic()
                logger.warning("Circuit breaker opened; pausing for %ds", self.reset_seconds)
            raise
        self.failures = 0
        return result
