"""
FastAPI middleware components for rate limiting, request tracking, and timing.

Middleware is applied in main.py and runs on every request before reaching route handlers.
"""

import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utils.errors import RateLimitExceededError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple token-bucket rate limiter per session_id.

    Tracks request counts per session in a sliding window.
    Returns 429 when limit is exceeded.
    """

    def __init__(self, app, max_requests: int = 30, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only rate-limit chat endpoints
        if not request.url.path.startswith("/api/v1/chat"):
            return await call_next(request)

        # Extract session_id: try JSON body first, then query params, then client IP
        session_id = request.headers.get("X-Request-ID", "") or \
                     f"{request.client.host}:{request.url.path}" if request.client else "unknown"

        # Clean old entries
        now = time.time()
        cutoff = now - self.window_seconds
        # Also clean up all buckets periodically to prevent memory leak
        self._buckets = {
            k: [t for t in v if t > cutoff]
            for k, v in self._buckets.items() if v
        }

        # Check limit
        bucket = self._buckets.get(session_id, [])
        if len(bucket) >= self.max_requests:
            raise RateLimitExceededError(
                detail=f"请求太频繁（{self.max_requests}次/{self.window_seconds}秒），请稍后再试"
            )

        # Record request
        if session_id not in self._buckets:
            self._buckets[session_id] = []
        self._buckets[session_id].append(now)
        return await call_next(request)


class TimingMiddleware(BaseHTTPMiddleware):
    """Adds X-Response-Time header to all responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time"] = f"{elapsed_ms:.2f}ms"
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Ensures every request/response has an X-Request-ID header.
    Generates a UUID if not provided by the client.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request and response with method, path, status, and timing.

    Output format: [ACCESS] POST /api/v1/chat → 200 (1234ms)
    Errors:        [ACCESS] POST /api/v1/chat → 500 (567ms) ERROR: ...
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        method = request.method
        path = request.url.path

        # Skip health check noise
        skip_paths = ("/api/v1/health", "/api/v1/health/ready")
        should_log = path not in skip_paths

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if should_log:
                status = response.status_code
                if status >= 400:
                    logger.warning(
                        f"[ACCESS] {method} {path} → {status} ({elapsed_ms:.0f}ms)"
                    )
                else:
                    logger.info(
                        f"[ACCESS] {method} {path} → {status} ({elapsed_ms:.0f}ms)"
                    )
            return response

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"[ACCESS] {method} {path} → ERROR ({elapsed_ms:.0f}ms): {str(e)}"
            )
            raise
