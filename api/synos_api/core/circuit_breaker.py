import time
import logging
from enum import Enum
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreakerOpenException(Exception):
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = State.CLOSED
        self.failures = 0
        self.last_failure_time = 0

    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == State.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = State.HALF_OPEN
                logger.info("CircuitBreaker: Entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenException("Circuit is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == State.HALF_OPEN:
                self.reset()
            return result
        except Exception as e:
            self._handle_failure()
            raise e

    def _handle_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = State.OPEN
            logger.warning("CircuitBreaker: Circuit OPENED due to failures")

    def reset(self):
        self.state = State.CLOSED
        self.failures = 0
        logger.info("CircuitBreaker: Circuit CLOSED (Recovered)")

# Global instance for scanner
scanner_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
