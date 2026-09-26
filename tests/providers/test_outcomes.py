"""STAB-2: explicit provider outcome semantics."""

from __future__ import annotations

import tempfile
from datetime import datetime
from types import SimpleNamespace

import aiobreaker
import httpx
import pytest

from research_explorer.providers.base import (
    ResilientProvider,
    RetryableHTTPError,
    TransientProviderError,
    _raise_for_retryable,
)

_SENTINEL = "SENTINEL-SECRET-123"


class _ProbeProvider(ResilientProvider):
    """Provider whose transport always fails transiently (no network)."""

    def __init__(self, cache_dir: str, *, strict: bool = False) -> None:
        super().__init__(
            name="probe",
            base_url="https://api.example.test/v1",
            cache_dir=cache_dir,
            strict_transient=strict,
        )

    async def _attempt_get(self, path: str, **params: object) -> httpx.Response:
        raise RetryableHTTPError(
            429, f"https://api.example.test{path}?api_key={_SENTINEL}"
        )


@pytest.fixture
def cache_dir() -> str:
    return tempfile.mkdtemp(prefix="probe-cache-")


def _request(url: str = "https://api.example.test/v1") -> httpx.Request:
    return httpx.Request("GET", url)


def test_ordinary_4xx_is_not_retryable() -> None:
    # Must not raise: 4xx is definitive absence, not a transient failure.
    _raise_for_retryable(httpx.Response(404, request=_request()))


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_transient_statuses_are_retryable(status: int) -> None:
    with pytest.raises(RetryableHTTPError):
        _raise_for_retryable(httpx.Response(status, request=_request()))


def test_breaker_excludes_4xx_but_counts_transient() -> None:
    provider = ResilientProvider(
        "probe", "https://api.example.test", cache_dir=tempfile.mkdtemp()
    )
    excluded = provider.breaker.excluded_exceptions
    assert httpx.HTTPStatusError in excluded
    assert RetryableHTTPError not in excluded


async def test_lenient_provider_degrades_to_none(cache_dir: str) -> None:
    provider = _ProbeProvider(cache_dir, strict=False)
    try:
        assert await provider.get_json("/paper/x") is None
    finally:
        await provider.aclose()


async def test_strict_provider_raises_typed_transient(cache_dir: str) -> None:
    provider = _ProbeProvider(cache_dir, strict=True)
    try:
        with pytest.raises(TransientProviderError) as exc_info:
            await provider.get_json("/paper/x")
        assert exc_info.value.reason == "retry_exhausted"
        assert exc_info.value.status == 429
        assert _SENTINEL not in str(exc_info.value)
    finally:
        await provider.aclose()


async def test_strict_provider_raises_on_open_circuit(cache_dir: str) -> None:
    provider = _ProbeProvider(cache_dir, strict=True)

    class _OpenBreaker:
        async def call(self, fn, *args, **kwargs):
            raise aiobreaker.CircuitBreakerError("open", datetime.now())

    provider.breaker = _OpenBreaker()  # type: ignore[assignment]
    try:
        with pytest.raises(TransientProviderError) as exc_info:
            await provider.get_json("/paper/x")
        assert exc_info.value.reason == "circuit_open"
    finally:
        await provider.aclose()


async def test_strict_scope_toggles_and_restores(cache_dir: str) -> None:
    provider = _ProbeProvider(cache_dir, strict=False)
    try:
        assert provider.strict_transient is False
        async with provider.strict_outcomes():
            assert provider.strict_transient is True
        assert provider.strict_transient is False
    finally:
        await provider.aclose()


def test_retryable_error_redacts_secret() -> None:
    exc = RetryableHTTPError(429, f"https://x.test?api_key={_SENTINEL}")
    assert _SENTINEL not in str(exc)
    assert _SENTINEL not in (exc.url or "")


def test_transient_error_has_expected_attributes() -> None:
    exc = TransientProviderError("openalex", "circuit_open", "https://x.test")
    assert SimpleNamespace(reason=exc.reason, provider=exc.provider) == SimpleNamespace(
        reason="circuit_open", provider="openalex"
    )


class _NotFoundProvider(ResilientProvider):
    """Provider whose transport returns an ordinary 404 (definitive absence)."""

    def __init__(self, cache_dir: str, *, strict: bool = False) -> None:
        super().__init__(
            name="probe",
            base_url="https://api.example.test/v1",
            cache_dir=cache_dir,
            strict_transient=strict,
        )

    async def _attempt_get(self, path: str, **params: object) -> httpx.Response:
        return httpx.Response(
            404, request=httpx.Request("GET", f"https://api.example.test/v1{path}")
        )


@pytest.mark.parametrize("strict", [False, True])
async def test_ordinary_404_is_definitive_absence_not_transient(
    cache_dir: str, strict: bool
) -> None:
    provider = _NotFoundProvider(cache_dir, strict=strict)
    try:
        assert await provider.get_json("/paper/missing") is None
    finally:
        await provider.aclose()
