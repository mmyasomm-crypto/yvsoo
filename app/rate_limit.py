"""
Simple in-memory sliding-window rate limiter, keyed per user id.

Good enough for a single-process v1 deployment. If the API is ever
scaled to multiple processes/pods, swap this for a Redis-backed
implementation (same interface) so limits are shared across instances.
"""
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, int]:
        now = time.time()
        with self._lock:
            q = self._hits[key]
            cutoff = now - self.window_seconds
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                retry_after = int(self.window_seconds - (now - q[0])) + 1
                return False, retry_after
            q.append(now)
            return True, 0
