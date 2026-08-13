"""Provider factory — builds provider instances from config."""

from __future__ import annotations

from research_explorer.config import Config, get_api_key
from research_explorer.providers.arxiv import ArxivProvider
from research_explorer.providers.base import ResilientProvider
from research_explorer.providers.openalex import OpenAlexProvider
from research_explorer.providers.pubmed import PubMedProvider
from research_explorer.providers.semantic_scholar import SemanticScholarProvider


def build_provider(name: str, config: Config) -> ResilientProvider:
    """Build a provider instance by name, configured from the Config."""
    entry = config.providers.entries.get(name)
    api_key_env = entry.api_key_env if entry else None
    api_key = get_api_key(api_key_env) if api_key_env else None
    cache_dir = config.storage.cache_dir
    rate = entry.rate if entry else 1
    period = entry.period if entry else 1
    concurrency = entry.concurrency if entry else 5
    timeout = entry.timeout if entry else 30.0

    if name == "semantic_scholar":
        return SemanticScholarProvider(
            api_key=api_key, cache_dir=cache_dir, rate=rate, period=period,
            concurrency=concurrency, timeout=timeout,
        )
    if name == "openalex":
        return OpenAlexProvider(
            api_key=api_key, cache_dir=cache_dir, rate=rate, period=period,
            concurrency=concurrency, timeout=timeout,
        )
    if name == "pubmed":
        return PubMedProvider(
            api_key=api_key, cache_dir=cache_dir, rate=rate, period=period,
            concurrency=concurrency, timeout=timeout,
        )
    if name == "arxiv":
        return ArxivProvider(
            cache_dir=cache_dir, rate=rate, period=period,
            concurrency=concurrency, timeout=timeout,
        )
    raise ValueError(f"Unknown provider: {name}")


def build_all_providers(config: Config) -> dict[str, ResilientProvider]:
    """Build all active providers from config."""
    return {name: build_provider(name, config) for name in config.providers.active}
