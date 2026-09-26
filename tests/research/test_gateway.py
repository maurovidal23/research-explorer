"""Provider outcome classification at the kernel evidence gateway (STAB-2)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator

import pytest

from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.graph.store import GraphStore
from research_explorer.providers.base import TransientProviderError
from research_explorer.research.loop import GraphEvidenceGateway
from research_explorer.research.models import ProviderOutcome


class FakeProvider:
    name = "openalex"
    supports_fulltext = False

    def __init__(self, *, result: Paper | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.strict = False
        self.native_calls: list[str] = []

    @contextlib.asynccontextmanager
    async def strict_outcomes(self) -> AsyncIterator[None]:
        previous = self.strict
        self.strict = True
        try:
            yield
        finally:
            self.strict = previous

    async def get_paper(self, native_id: str) -> Paper | None:
        self.native_calls.append(native_id)
        if self.error is not None:
            raise self.error
        return self.result

    async def aclose(self) -> None:
        return None


class FulltextProvider(FakeProvider):
    name = "arxiv"
    supports_fulltext = True

    async def get_fulltext_and_refs(
        self, native_id: str, max_chars: int, ref_limit: int
    ) -> tuple[str, list[str]]:
        return "full paper content", ["Reference with arXiv:2301.00002"]


class FakeReferenceMapper:
    async def map_references(
        self, paper: Paper, question: str, limit: int
    ) -> list[PaperSummary]:
        assert paper.fulltext == "full paper content"
        assert paper.ref_entries == ["Reference with arXiv:2301.00002"]
        assert question == "research question"
        return [
            PaperSummary(
                id="2301.00002", title="Mapped reference", provider="arxiv"
            )
        ]


class FailingReferenceMapper:
    async def map_references(
        self, paper: Paper, question: str, limit: int
    ) -> list[PaperSummary]:
        raise ValueError("truncated JSON")


@pytest.fixture
def graph(tmp_path) -> Iterator[GraphStore]:
    store = GraphStore(tmp_path / "graph.db")
    yield store
    store.close()


def _gateway(graph: GraphStore, provider: FakeProvider) -> GraphEvidenceGateway:
    return GraphEvidenceGateway(graph, {"openalex": provider}, "openalex")


def _paper() -> Paper:
    return Paper(
        id="W1",
        provider="openalex",
        title="Seed paper",
        abstract="an abstract",
        references=[PaperSummary(id="W2", title="Reference", provider="openalex")],
        citations=[PaperSummary(id="W3", title="Citation", provider="openalex")],
    )


async def test_success_caches_paper_and_builds_candidates(graph: GraphStore) -> None:
    provider = FakeProvider(result=_paper())
    acquisition = await _gateway(graph, provider).acquire("openalex:W1", turn=1)
    assert acquisition.outcome is ProviderOutcome.SUCCESS
    assert acquisition.evidence is not None
    assert acquisition.evidence.paper_id == "openalex:W1"
    assert acquisition.evidence.content_hash
    assert provider.native_calls == ["W1"]
    assert graph.get_paper_summary("openalex:W1") is not None
    candidates = {c.paper_id: c for c in acquisition.candidates}
    assert candidates["openalex:W2"].mode == "ref"
    assert candidates["openalex:W3"].mode == "cites"


async def test_definitive_absence(graph: GraphStore) -> None:
    provider = FakeProvider(result=None)
    acquisition = await _gateway(graph, provider).acquire("openalex:W9", turn=1)
    assert acquisition.outcome is ProviderOutcome.ABSENT


async def test_typed_transient_failure_is_classified(graph: GraphStore) -> None:
    provider = FakeProvider(
        error=TransientProviderError("openalex", "retry_exhausted", status=503)
    )
    acquisition = await _gateway(graph, provider).acquire("openalex:W1", turn=1)
    assert acquisition.outcome is ProviderOutcome.TRANSIENT
    assert acquisition.provider == "openalex"


async def test_unexpected_error_is_treated_as_transient(graph: GraphStore) -> None:
    provider = FakeProvider(error=RuntimeError("socket reset"))
    acquisition = await _gateway(graph, provider).acquire("openalex:W1", turn=1)
    assert acquisition.outcome is ProviderOutcome.TRANSIENT


async def test_paper_exists_reads_graph_store(graph: GraphStore) -> None:
    gateway = _gateway(graph, FakeProvider())
    assert gateway.paper_exists("openalex:missing") is False
    graph.cache_paper(_paper())
    assert gateway.paper_exists("openalex:W1") is True


async def test_fulltext_bibliography_builds_arxiv_candidates(
    graph: GraphStore,
) -> None:
    paper = Paper(id="2301.00001", provider="arxiv", title="Seed")
    provider = FulltextProvider(result=paper)
    gateway = GraphEvidenceGateway(
        graph,
        {"arxiv": provider},
        "arxiv",
        reference_mapper=FakeReferenceMapper(),
        question="research question",
    )
    acquisition = await gateway.acquire("arxiv:2301.00001", turn=1)
    assert acquisition.outcome is ProviderOutcome.SUCCESS
    assert acquisition.content == "full paper content"
    assert [candidate.paper_id for candidate in acquisition.candidates] == [
        "arxiv:2301.00002"
    ]


async def test_explicit_arxiv_id_survives_mapping_failure(
    graph: GraphStore,
) -> None:
    paper = Paper(id="2301.00001", provider="arxiv", title="Seed")
    provider = FulltextProvider(result=paper)
    gateway = GraphEvidenceGateway(
        graph,
        {"arxiv": provider},
        "arxiv",
        reference_mapper=FailingReferenceMapper(),
    )
    acquisition = await gateway.acquire("arxiv:2301.00001", turn=1)
    assert acquisition.outcome is ProviderOutcome.SUCCESS
    assert [candidate.paper_id for candidate in acquisition.candidates] == [
        "arxiv:2301.00002"
    ]
