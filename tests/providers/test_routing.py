"""STAB-1: seed identifier detection and provider routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_explorer.providers.routing import (
    SeedKind,
    SeedRoutingError,
    detect_seed,
    route_seed_provider,
)


def _providers(*names: str) -> dict:
    return {name: SimpleNamespace(name=name) for name in names}


def test_detect_doi_with_url_prefix() -> None:
    ref = detect_seed("https://doi.org/10.1038/NRN3241")
    assert ref.kind is SeedKind.DOI
    assert ref.value == "10.1038/nrn3241"


def test_detect_arxiv_new_style_with_version() -> None:
    ref = detect_seed("arXiv:2301.00001v2")
    assert ref.kind is SeedKind.ARXIV
    assert ref.value == "2301.00001"


def test_detect_native_id() -> None:
    ref = detect_seed("W1973275441")
    assert ref.kind is SeedKind.NATIVE
    assert ref.value == "W1973275441"


def test_doi_prefers_openalex() -> None:
    providers = _providers("arxiv", "openalex", "semantic_scholar")
    provider, ref = route_seed_provider("10.1038/nrn3241", providers)
    assert provider.name == "openalex"
    assert ref.kind is SeedKind.DOI


def test_doi_falls_back_to_semantic_scholar() -> None:
    providers = _providers("arxiv", "semantic_scholar")
    provider, _ = route_seed_provider("10.1038/nrn3241", providers)
    assert provider.name == "semantic_scholar"


def test_doi_explicit_order_wins() -> None:
    providers = _providers("openalex", "semantic_scholar")
    provider, _ = route_seed_provider(
        "10.1038/nrn3241", providers, explicit_order=["semantic_scholar", "openalex"]
    )
    assert provider.name == "semantic_scholar"


def test_doi_without_capable_provider_raises() -> None:
    providers = _providers("arxiv", "pubmed")
    with pytest.raises(SeedRoutingError):
        route_seed_provider("10.1038/nrn3241", providers)


def test_arxiv_routes_to_arxiv() -> None:
    providers = _providers("openalex", "arxiv")
    provider, ref = route_seed_provider("2301.00001", providers)
    assert provider.name == "arxiv"
    assert ref.kind is SeedKind.ARXIV


def test_arxiv_without_arxiv_provider_raises() -> None:
    providers = _providers("openalex", "semantic_scholar")
    with pytest.raises(SeedRoutingError):
        route_seed_provider("2301.00001", providers)


def test_native_seed_uses_first_enabled() -> None:
    providers = _providers("pubmed", "arxiv")
    provider, ref = route_seed_provider("22595786", providers)
    assert ref.kind is SeedKind.NATIVE
    assert provider.name == "pubmed"


def test_no_providers_raises() -> None:
    with pytest.raises(SeedRoutingError):
        route_seed_provider("10.1038/nrn3241", {})
