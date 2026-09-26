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
