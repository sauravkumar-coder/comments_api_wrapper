"""
Security middleware — API key authentication, rate limiting, and input
sanitisation.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Callable, Dict, Tuple

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import get_settings

logger = logging.getLogger("rm_quality_api.security")


# ═══════════════════════════════════════════════════════════════════════════════
#  API Key Authentication
# ═══════════════════════════════════════════════════════════════════════════════

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str | None:
    """
    FastAPI dependency that validates the ``X-API-Key`` header.

    If ``RM_API_KEY`` is empty (default), authentication is disabled and all
    requests are allowed through.
    """
    settings = get_settings()

    # If no key is configured, auth is disabled
    if not settings.api_key:
        return None

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key


# ═══════════════════════════════════════════════════════════════════════════════
#  Rate Limiting (Token-Bucket per client IP)
# ═══════════════════════════════════════════════════════════════════════════════

class _TokenBucket:
    """Simple per-IP token bucket."""

    def __init__(self, max_tokens: int, refill_seconds: int) -> None:
        self.max_tokens = max_tokens
        self.refill_seconds = refill_seconds
        self._buckets: Dict[str, Tuple[float, float]] = {}  # ip → (tokens, last_refill)

    def allow(self, client_ip: str) -> bool:
        now = time.monotonic()
        tokens, last_refill = self._buckets.get(
            client_ip, (float(self.max_tokens), now)
        )

        # Refill tokens based on elapsed time
        elapsed = now - last_refill
        tokens = min(
            self.max_tokens,
            tokens + (elapsed / self.refill_seconds) * self.max_tokens,
        )

        if tokens >= 1.0:
            self._buckets[client_ip] = (tokens - 1.0, now)
            return True
        else:
            self._buckets[client_ip] = (tokens, now)
            return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces per-IP rate limiting using a token bucket.

    Exempt paths: ``/health``, ``/version``, ``/docs``, ``/openapi.json``.
    """

    _EXEMPT_PATHS = {"/health", "/version", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        settings = get_settings()
        self._bucket = _TokenBucket(
            max_tokens=settings.rate_limit_requests,
            refill_seconds=settings.rate_limit_window_seconds,
        )

    async def dispatch(self, request: Request, call_next: Callable):  # type: ignore[no-untyped-def]
        if request.url.path in self._EXEMPT_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        if not self._bucket.allow(client_ip):
            logger.warning("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please retry later."},
            )

        return await call_next(request)


# ═══════════════════════════════════════════════════════════════════════════════
#  Input Sanitisation
# ═══════════════════════════════════════════════════════════════════════════════

def sanitise_remark(text: str) -> str:
    """
    Basic input sanitisation — strip null bytes and control characters
    that could cause downstream issues.
    """
    # Remove null bytes
    text = text.replace("\x00", "")
    # Limit length (defence-in-depth — Pydantic also validates)
    return text[:10_000]
