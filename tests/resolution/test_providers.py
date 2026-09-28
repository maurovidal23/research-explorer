from pathlib import Path
from typing import cast

from research_explorer.config import Config
from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.graph.store import GraphStore
from research_explorer.providers.arxiv import ArxivProvider
from research_explorer.providers.base import ResilientProvider
from research_explorer.references.builder import build_reference_builder
from research_explorer.resolution.providers import (
    ArxivVerification,
    build_verification_providers,
)


class FakeArxivProvider:
    name = "arxiv"

    def __init__(self) -> None:
        self.paper: Paper | None = None
        self.results: list[PaperSummary] = []
        self.paper_calls: list[tuple[str, str]] = []
        self.search_calls: list[tuple[str, int]] = []

    async def get_paper(self, paper_id: str, id_type: str = "auto") -> Paper | None:
        self.paper_calls.append((paper_id, id_type))
        return self.paper

    async def search(self, query: str, limit: int = 10) -> list[PaperSummary]:
        self.search_calls.append((query, limit))
        return self.results


async def test_arxiv_verification_resolves_native_identifier() -> None:
    provider = FakeArxivProvider()
    provider.paper = Paper(
        id="2106.09685",
        arxiv_id="2106.09685",
        title="LoRA",
        provider="arxiv",
    )
    adapter = ArxivVerification(cast(ArxivProvider, provider))

    result = await adapter.lookup_arxiv("2106.09685")

    assert result is not None
    assert result.arxiv_id == "2106.09685"
    assert provider.paper_calls == [("2106.09685", "arxiv")]


async def test_arxiv_verification_filters_doi_search_results() -> None:
    provider = FakeArxivProvider()
    provider.results = [
        PaperSummary(
            id="1",
            doi="10.1/other",
            title="Other",
            provider="arxiv",
        ),
        PaperSummary(
            id="2",
            doi="https://doi.org/10.1/target",
            title="Target",
            provider="arxiv",
        ),
    ]
    adapter = ArxivVerification(cast(ArxivProvider, provider), title_limit=3)

    result = await adapter.lookup_doi("10.1/TARGET")

    assert result is not None and result.id == "2"
    assert provider.search_calls == [("doi:10.1/TARGET", 3)]


def test_build_verification_providers_supports_arxiv() -> None:
    provider = FakeArxivProvider()
    adapters = build_verification_providers(
        ["arxiv"],
        {"arxiv": cast(ResilientProvider, provider)},
    )

    assert [adapter.name for adapter in adapters] == ["arxiv"]


def test_reference_builder_uses_arxiv_when_it_is_the_only_provider(
    tmp_path: Path,
) -> None:
    config = Config()
    config.providers.active = ["arxiv"]
    store = GraphStore(str(tmp_path / "graph.db"))
    provider = FakeArxivProvider()

    builder = build_reference_builder(
        config,
        {"arxiv": cast(ResilientProvider, provider)},
        store,
        cast(object, object()),
    )

    assert builder is not None
    assert [adapter.name for adapter in builder.resolver.providers] == ["arxiv"]
    store.close()
