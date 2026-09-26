"""Resilient async HTTP provider base with rate limiting, retries, circuit breaker, and cache.

Each academic API adapter inherits from ResilientProvider and gets its own:
  - rate limiter (aiolimiter, per-provider limits)
  - circuit breaker (aiobreaker, per-provider)
  - concurrency semaphore (per-provider)
  - persistent cache (diskcache, per-provider)
  - retry with exponential backoff + jitter (tenacity)
  - structured logging via httpx event hooks

The layering (outer to inner):
  cache -> circuit breaker -> retry -> semaphore -> rate limiter -> httpx
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import aiobreaker
import diskcache
import httpx
from aiolimiter import AsyncLimiter
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from research_explorer.logging_setup import get_logger
from research_explorer.redaction import redact_secrets

log = get_logger("providers")


class RetryableHTTPError(Exception):
    """Raised on 429 / 5xx / network errors — worth retrying."""

    def __init__(self, status: int, url: str, retry_after: str | None = None):
        self.status = status
        self.url = redact_secrets(url)
        self.retry_after = retry_after
        super().__init__(f"{status} {self.url}")


class TransientProviderError(Exception):
    """Typed transient failure: circuit open or retries exhausted.

    Callers that must distinguish temporary failure from definitive absence
    (e.g. the research kernel frontier) depend on this being raised rather than
    silently degraded to ``None``/``[]``. It never contains raw credentials.
    """

    def __init__(
        self,
        provider: str,
        reason: str,
        url: str | None = None,
        status: int | None = None,
    ):
        self.provider = provider
        self.reason = reason
        self.url = redact_secrets(url) if url else None
        self.status = status
        detail = self.url or provider
        super().__init__(f"transient provider failure ({reason}) for {detail}")


RETRYABLE = (
    RetryableHTTPError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.ProtocolError,
)


def _raise_for_retryable(response: httpx.Response) -> None:
    """Raise RetryableHTTPError on 429/5xx.

    Ordinary 4xx responses are returned to the caller so ``get_json`` can map
    them to definitive absence without ever counting toward the circuit
    breaker (see STAB-2).
    """
    if response.status_code == 429 or 500 <= response.status_code < 600:
        raise RetryableHTTPError(
            response.status_code,
            str(response.url),
            response.headers.get("retry-after"),
        )


def _wait_retry_after(retry_state) -> float:
    """Honor Retry-After header if present, else exponential backoff with jitter."""
    exc = retry_state.outcome.exception()
    if isinstance(exc, RetryableHTTPError) and exc.retry_after:
        try:
            return max(1.0, min(float(exc.retry_after), 300.0))
        except (TypeError, ValueError):
            pass
    return wait_random_exponential(multiplier=1, max=60)(retry_state)


class ResilientProvider:
    """Base class for resilient async HTTP providers.

    Subclasses implement search/get_paper/get_references/get_citations
    by calling self.get_json(path, **params) and parsing the response.
    """

    supports_fulltext: bool = False

    async def search(self, query: str, limit: int = 10) -> list:
        raise NotImplementedError

    async def get_paper(self, paper_id: str, id_type: str = "auto") -> Any | None:
        raise NotImplementedError

    async def get_references(self, paper_id: str, limit: int = 50) -> list:
        raise NotImplementedError

    async def get_citations(self, paper_id: str, limit: int = 50) -> list:
        raise NotImplementedError

    async def get_abstract(self, paper_id: str) -> str | None:
        raise NotImplementedError

    async def get_fulltext_and_refs(
        self, paper_id: str, max_chars: int = 16000, ref_limit: int = 100
    ) -> tuple[str, list[str]] | None:
        """Return (truncated full text, raw bibliography entries) or None if unsupported."""
        return None

    def __init__(
        self,
        name: str,
        base_url: str,
        *,
        rate: int = 1,
        period: int = 1,
        concurrency: int = 5,
        timeout: float = 30.0,
        connect: float = 5.0,
        cache_ttl: int = 86400,
        cache_dir: str = "data/.cache",
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        strict_transient: bool = False,
    ):
        self.name = name
        self.base_url = base_url
        self.cache_ttl = cache_ttl
        self.strict_transient = strict_transient

        self.limiter = AsyncLimiter(rate, period)
        self.sem = asyncio.Semaphore(concurrency)
        self.breaker = aiobreaker.CircuitBreaker(
            fail_max=5,
            timeout_duration=timedelta(seconds=60),
            exclude=[httpx.HTTPStatusError],
        )

        cache_path = f"{cache_dir}/{name}"
        self.cache = diskcache.Cache(cache_path, size_limit=2**30)  # 1 GB

        headers = {
            "User-Agent": "research-explorer/0.1 (mailto:research@example.com)",
            "Accept": "application/json",
        }
        if api_key:
            headers["x-api-key"] = api_key
        if extra_headers:
            headers.update(extra_headers)

        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=connect, pool=10.0),
            limits=httpx.Limits(
                max_keepalive_connections=concurrency,
                max_connections=concurrency,
            ),
            http2=True,
            headers=headers,
            event_hooks={
                "request": [self._log_request],
                "response": [self._log_response],
            },
        )

    async def _log_request(self, request: httpx.Request) -> None:
        request.extensions["t0"] = time.perf_counter()
        log.debug(
            "http.request",
            provider=self.name,
            method=request.method,
            url=redact_secrets(str(request.url)),
        )

    async def _log_response(self, response: httpx.Response) -> None:
        t0 = response.request.extensions.get("t0")
        elapsed = round((time.perf_counter() - t0) * 1000, 2) if t0 else None
        log.debug(
            "http.response",
            provider=self.name,
            status=response.status_code,
            elapsed_ms=elapsed,
            rl_remaining=response.headers.get("x-ratelimit-remaining"),
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=_wait_retry_after,
        retry=retry_if_exception_type(RETRYABLE),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    async def _attempt_get(self, path: str, **params: Any) -> httpx.Response:
        async with self.sem, self.limiter:
            resp = await self.client.get(path, params=params)
            _raise_for_retryable(resp)
            return resp

    @contextlib.asynccontextmanager
    async def strict_outcomes(self) -> AsyncIterator[None]:
        """Temporarily raise TransientProviderError instead of degrading to None."""
        previous = self.strict_transient
        self.strict_transient = True
        try:
            yield
        finally:
            self.strict_transient = previous

    async def get_json(self, path: str, **params: Any) -> dict | list | None:
        """GET path with caching, circuit breaker, retry, rate limit, and concurrency control.

        Returns cached response on hit. On a definitive non-200 response (an
        ordinary 4xx) returns None. On a transient failure — circuit open,
        retry-exhausted 429/5xx, timeout, or network error — returns None when
        ``strict_transient`` is False (backwards-compatible graceful
        degradation) and raises :class:`TransientProviderError` when it is True.
        """
        key = self._cache_key(path, params)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        try:
            resp = await self.breaker.call(self._attempt_get, path, **params)
        except aiobreaker.CircuitBreakerError as exc:
            if self.strict_transient:
                raise TransientProviderError(self.name, "circuit_open", path) from exc
            log.warning("circuit_open", provider=self.name, path=redact_secrets(path))
            return None
        except RETRYABLE as exc:
            if self.strict_transient:
                url = getattr(exc, "url", None) or path
                raise TransientProviderError(
                    self.name,
                    "retry_exhausted",
                    url,
                    status=getattr(exc, "status", None),
                ) from exc
            log.warning(
                "request_failed", provider=self.name, path=redact_secrets(path)
            )
            return None

        if resp.status_code == 200:
            data = resp.json()
            self.cache.set(key, data, expire=self.cache_ttl)
            return data
        return None

    def _cache_key(self, path: str, params: dict) -> str:
        norm = json.dumps(
            {"p": path, "q": sorted(params.items())},
            sort_keys=True,
        )
        return hashlib.sha256(norm.encode()).hexdigest()

    async def aclose(self) -> None:
        """Close the HTTP client and cache."""
        await self.client.aclose()
        self.cache.close()
