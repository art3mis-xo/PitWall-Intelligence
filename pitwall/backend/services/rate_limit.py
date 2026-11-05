# pitwall/backend/services/rate_limit.py
"""In-memory limits on endpoints that call Groq, so a public deployment can't drain the quota.

Two limits, both set by environment variables:
  GROQ_LIMIT_PER_IP    requests per visitor IP per window   (default 10)
  GROQ_LIMIT_WINDOW_S  window length in seconds              (default 600)
  GROQ_DAILY_CAP       requests per UTC day, all visitors    (default 300)

State lives in this process, which is enough for a single server instance.
A request counts once even if the agent makes several Groq calls for it.
"""
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import Request


class GroqLimitExceeded(Exception):
    pass


def client_ip(request: Request) -> str:
    # Behind a hosting proxy (e.g. Render) the socket peer is the proxy. The proxy appends
    # the real client address as the LAST X-Forwarded-For entry; earlier entries are
    # client-supplied and can be forged. The daily cap still applies either way.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


class GroqRateLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self._hits: dict[str, deque] = defaultdict(deque)
        self._day = None
        self._day_count = 0

    def check(self, ip: str) -> None:
        """Record one request from `ip`, or raise GroqLimitExceeded without recording it."""
        per_ip = int(os.getenv("GROQ_LIMIT_PER_IP", "10"))
        window = float(os.getenv("GROQ_LIMIT_WINDOW_S", "600"))
        daily_cap = int(os.getenv("GROQ_DAILY_CAP", "300"))
        now = time.monotonic()
        today = datetime.now(timezone.utc).date()

        with self._lock:
            if today != self._day:
                self._day, self._day_count = today, 0
            hits = self._hits[ip]
            while hits and now - hits[0] > window:
                hits.popleft()

            if self._day_count >= daily_cap:
                raise GroqLimitExceeded("The AI assistant has reached its daily limit. Please try again tomorrow.")
            if len(hits) >= per_ip:
                raise GroqLimitExceeded("Too many AI requests. Please wait a few minutes and try again.")

            hits.append(now)
            self._day_count += 1


limiter = GroqRateLimiter()
