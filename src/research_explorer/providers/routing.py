"""Seed identifier detection and provider routing (STAB-1).

A seed identifier is classified as a DOI, an arXiv id, or an opaque native id.
DOI seeds route to an enabled DOI-capable provider (OpenAlex, then Semantic
Scholar, unless configuration defines an explicit order). arXiv seeds route to
the arXiv provider when enabled. A clear configuration error is raised when no
capable provider is enabled — the caller must never silently fall back to an
unrelated provider or construct an unused one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from research_explorer.providers.base import ResilientProvider
from research_explorer.resolution.resolver import (
    is_valid_arxiv,
    is_valid_doi,
    normalize_arxiv,
    normalize_doi,
)


class SeedKind(str, Enum):
    DOI = "doi"
    ARXIV = "arxiv"
    NATIVE = "native"


class SeedRoutingError(ValueError):
    """Raised when no enabled provider can serve the seed identifier."""


@dataclass(frozen=True)
class SeedRef:
    kind: SeedKind
    value: str
    raw: str


DOI_CAPABLE = ("openalex", "semantic_scholar")
ARXIV_CAPABLE = ("arxiv",)


def detect_seed(seed: str) -> SeedRef:
    """Classify a raw seed identifier and return its normalized value."""
    raw = seed.strip()
    doi = normalize_doi(raw)
    if is_valid_doi(doi):
        return SeedRef(SeedKind.DOI, doi, raw)
    arxiv = normalize_arxiv(raw)
    if is_valid_arxiv(arxiv):
        return SeedRef(SeedKind.ARXIV, arxiv, raw)
    return SeedRef(SeedKind.NATIVE, raw, raw)


def _first_enabled(
    order: tuple[str, ...], providers: dict[str, ResilientProvider]
) -> ResilientProvider | None:
    for name in order:
        provider = providers.get(name)
        if provider is not None:
            return provider
    return None


def route_seed_provider(
    seed: str,
    providers: dict[str, ResilientProvider],
    explicit_order: list[str] | None = None,
) -> tuple[ResilientProvider, SeedRef]:
    """Select the provider for a seed, or raise for a clear config error."""
    ref = detect_seed(seed)

    if ref.kind is SeedKind.DOI:
        order = tuple(explicit_order) if explicit_order else DOI_CAPABLE
        provider = _first_enabled(order, providers)
        if provider is None:
            raise SeedRoutingError(
                f"No DOI-capable provider enabled for seed {seed!r}. "
                f"Enable one of {list(order)} in providers.active."
            )
        return provider, ref

    if ref.kind is SeedKind.ARXIV:
        # An arXiv identifier is only served by the arXiv provider; the DOI
        # routing order must never redirect it to a non-arXiv provider.
        provider = _first_enabled(ARXIV_CAPABLE, providers)
        if provider is None:
            raise SeedRoutingError(
                f"No arXiv-capable provider enabled for seed {seed!r}. "
                f"Enable one of {list(ARXIV_CAPABLE)} in providers.active."
            )
        return provider, ref

    order = tuple(explicit_order) if explicit_order else tuple(providers)
    provider = _first_enabled(order, providers)
    if provider is None:
        raise SeedRoutingError(
            f"No provider enabled for native seed {seed!r}. "
            "Enable at least one provider in providers.active."
        )
    return provider, ref


def provider_by_name(
    providers: dict[str, ResilientProvider],
    preferred: str,
    fallback: str | None = None,
) -> ResilientProvider:
    """Return an enabled provider by name without raising a bare ``KeyError``.

    Falls back to ``fallback`` when the preferred provider is disabled, then to
    any enabled provider. Raises a clear configuration error when none is
    enabled, so callers never index a missing mapping key.
    """
    provider = providers.get(preferred)
    if provider is not None:
        return provider
    if fallback is not None and fallback != preferred:
        provider = providers.get(fallback)
        if provider is not None:
            return provider
    if not providers:
        raise SeedRoutingError("No providers enabled for evidence acquisition.")
    return next(iter(providers.values()))
