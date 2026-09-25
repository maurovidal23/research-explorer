"""Tests for canonical neighbor expansion (traversal), using mocked sources
and a real IdentityResolver backed by mocked verification providers.

Covers: both directions, per-direction limits, OpenAlex -> Semantic Scholar
fallback, cross-provider dedup, rejected/unverifiable candidate pollution
prevention, title-only resolution, arXiv id resolution, and replay payloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_explorer.config import ResolutionConfig
from research_explorer.graph.models import PaperSummary
from research_explorer.graph.store import GraphStore
from research_explorer.replay.trace import RunTracer, RunTraceStore
from research_explorer.resolution.resolver import IdentityResolver
from research_explorer.resolution.traversal import NeighborExpander

SEED_NID = "openalex:Wseed"
SEED_DOI = "10.1234/seed"


class MockSource:
    """Mock OpenAlex/S2 provider exposing the expander's raw fetch surface."""

    def __init__(
        self,
        name: str,
        *,
        citations: list[PaperSummary] | None = None,
        references: list[PaperSummary] | None = None,
        batch: list[PaperSummary] | None = None,
    ) -> None:
        self.name = name
        self.citations = list(citations or [])
        self.references = list(references or [])
        self.batch = list(batch or [])
        self.cit_calls: list[str] = []
        self.ref_calls: list[str] = []
        self.paper_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    async def get_citations(self, key: str, limit: int = 50) -> list[PaperSummary]:
        self.cit_calls.append(key)
        return list(self.citations[:limit])

    async def get_references(self, key: str, limit: int = 50) -> list[PaperSummary]:
        self.ref_calls.append(key)
        return list(self.references[:limit])

    async def get_paper(self, key: str, id_type: str = "auto") -> None:
        self.paper_calls.append(key)
        return None

    async def get_works_batch(
        self, openalex_ids: list[str], limit: int = 50
    ) -> list[PaperSummary]:
        self.batch_calls.append(list(openalex_ids))
        return list(self.batch[:limit])


class MockResolverProvider:
    """Mock VerificationProvider for the deterministic IdentityResolver."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.doi_results: dict[str, PaperSummary | None] = {}
        self.arxiv_results: dict[str, PaperSummary | None] = {}
        self.title_results: dict[str, list[PaperSummary]] = {}
        self.doi_calls: list[str] = []
        self.arxiv_calls: list[str] = []
        self.title_calls: list[str] = []

    async def lookup_doi(self, doi: str) -> PaperSummary | None:
        self.doi_calls.append(doi)
        return self.doi_results.get(doi)

    async def lookup_arxiv(self, arxiv_id: str) -> PaperSummary | None:
        self.arxiv_calls.append(arxiv_id)
        return self.arxiv_results.get(arxiv_id)

    async def search_title(self, title: str) -> list[PaperSummary]:
        self.title_calls.append(title)
        return list(self.title_results.get(title, []))


def summary(**overrides: object) -> PaperSummary:
    values: dict[str, object] = {
        "id": "W1",
        "title": "Some work",
        "year": 2020,
        "authors": ["A. Author"],
        "provider": "openalex",
    }
    values.update(overrides)
    return PaperSummary(**values)


def openalex_summary(nid: str, *, doi: str = "", **overrides: object) -> PaperSummary:
    values: dict[str, object] = {
        "id": nid.removeprefix("openalex:"),
        "provider": "openalex",
        "title": f"Work {nid}",
        "year": 2020,
        "authors": [f"Author {nid}"],
    }
    if doi:
        values["doi"] = doi
    values.update(overrides)
    return PaperSummary(**values)


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    s = GraphStore(str(tmp_path / "graph.db"))
    seed = PaperSummary(id="Wseed", provider="openalex", doi=SEED_DOI,
                        title="Seed paper", year=2020, authors=["S. Seed"])
    s.cache_summary_for_nid(SEED_NID, seed)
    yield s
    s.close()


def config(**overrides: object) -> ResolutionConfig:
    values: dict[str, object] = {
        "enabled": True,
        "primary": "openalex",
        "fallbacks": ["semantic_scholar"],
        "incoming_enabled": True,
        "outgoing_enabled": True,
        "incoming_limit": 50,
        "outgoing_limit": 50,
        "title_search_limit": 5,
        "min_title_similarity": 0.82,
        "min_author_overlap": 0.34,
        "min_confidence": 0.75,
        "year_tolerance": 2,
    }
    values.update(overrides)
    return ResolutionConfig(**values)


def build(
    store: GraphStore,
    sources: dict[str, MockSource],
    resolvers: list[MockResolverProvider],
    cfg: ResolutionConfig,
) -> NeighborExpander:
    resolver = IdentityResolver(providers=resolvers, store=store)
    return NeighborExpander(providers=sources, resolver=resolver, store=store, config=cfg)


async def test_expand_both_directions_records_canonical_edges(
    store: GraphStore,
) -> None:
    oa = MockSource(
        name="openalex",
        citations=[openalex_summary("Wc1", doi="10.1234/c1")],
        references=[
            openalex_summary("Wr1", doi="10.1234/r1"),
            openalex_summary("Wr2", doi="10.1234/r2"),
        ],
    )
    s2 = MockSource(name="semantic_scholar")
    vp = MockResolverProvider("openalex")
    vp.doi_results = {
        "10.1234/c1": openalex_summary("Wc1", doi="10.1234/c1"),
        "10.1234/r1": openalex_summary("Wr1", doi="10.1234/r1"),
        "10.1234/r2": openalex_summary("Wr2", doi="10.1234/r2"),
    }
    expander = build(store, {"openalex": oa, "semantic_scholar": s2}, [vp], config())

    result = await expander.expand(SEED_NID)

    assert result.incoming.direction == "incoming"
    assert result.incoming.provider == "openalex"
    assert result.incoming.node_ids == ["openalex:Wc1"]
    assert result.outgoing.provider == "openalex"
    assert result.outgoing.node_ids == ["openalex:Wr1", "openalex:Wr2"]
    assert oa.cit_calls == ["Wseed"]
    assert oa.ref_calls == ["Wseed"]
    assert s2.cit_calls == []

    incoming = store.get_edge_provenance(dst=SEED_NID)
    assert [r["src"] for r in incoming] == ["openalex:Wc1"]
    assert all(r["direction"] == "cited_by" for r in incoming)
    assert all(r["provider"] == "openalex" for r in incoming)

    outgoing = store.get_edge_provenance(src=SEED_NID)
    dsts = [r["dst"] for r in outgoing]
    assert dsts == ["openalex:Wr1", "openalex:Wr2"]
    assert all(r["direction"] == "references" for r in outgoing)
    assert store.get_references(SEED_NID) == ["openalex:Wr1", "openalex:Wr2"]


async def test_fallback_to_semantic_scholar_when_primary_empty(store: GraphStore) -> None:
    oa = MockSource(name="openalex", citations=[])
    s2 = MockSource(
        name="semantic_scholar",
        citations=[summary(id="Sc1", provider="semantic_scholar", doi="10.1234/c1")],
    )
    vp_s2 = MockResolverProvider("semantic_scholar")
    vp_s2.doi_results["10.1234/c1"] = summary(
        id="Sc1", provider="semantic_scholar", doi="10.1234/c1"
    )
    expander = build(
        store, {"openalex": oa, "semantic_scholar": s2},
        [MockResolverProvider("openalex"), vp_s2], config(),
    )

    result = await expander.expand(SEED_NID)

    assert result.incoming.provider == "semantic_scholar"
    assert result.incoming.fallback_used is True
    assert result.incoming.node_ids == ["s2:Sc1"]
    assert oa.cit_calls == ["Wseed"]
    assert s2.cit_calls == [f"DOI:{SEED_DOI}"]
    incoming = store.get_edge_provenance(dst=SEED_NID)
    assert [(r["src"], r["provider"]) for r in incoming] == [("s2:Sc1", "semantic_scholar")]


async def test_outgoing_disabled_skips_fetch(store: GraphStore) -> None:
    oa = MockSource(name="openalex", references=[openalex_summary("Wr1")])
    expander = build(
        store, {"openalex": oa}, [MockResolverProvider("openalex")],
        config(outgoing_enabled=False),
    )

    result = await expander.expand(SEED_NID)

    assert result.outgoing.node_ids == []
    assert result.outgoing.provider is None
    assert oa.ref_calls == []
    assert store.get_references(SEED_NID) == []


async def test_cross_provider_duplicate_candidates_dedup(store: GraphStore) -> None:
    same_work_a = openalex_summary("Wa", doi="10.1234/same", title="The same work")
    same_work_b = summary(
        id="Sb", provider="semantic_scholar", doi="10.1234/same", title="The same work"
    )
    oa = MockSource(name="openalex", references=[same_work_a, same_work_b])
    vp = MockResolverProvider("openalex")
    vp.doi_results["10.1234/same"] = openalex_summary(
        "Wa", doi="10.1234/same", title="The same work"
    )
    expander = build(store, {"openalex": oa}, [vp], config())

    result = await expander.expand(SEED_NID)

    assert result.outgoing.discovered == 1
    assert result.outgoing.node_ids == ["openalex:Wa"]
    assert store.get_references(SEED_NID) == ["openalex:Wa"]
    assert len(store.get_edge_provenance(src=SEED_NID)) == 1


async def test_rejected_candidates_never_pollute_graph(store: GraphStore) -> None:
    oa = MockSource(
        name="openalex",
        references=[
            openalex_summary("Wok", doi="10.1234/ok"),
            openalex_summary("Wbad", doi="10.1234/bad"),
        ],
    )
    vp = MockResolverProvider("openalex")
    vp.doi_results["10.1234/ok"] = openalex_summary("Wok", doi="10.1234/ok")
    vp.doi_results["10.1234/bad"] = openalex_summary(
        "Wbad", doi="10.1234/bad", title="Completely unrelated"
    )
    expander = build(store, {"openalex": oa}, [vp], config())

    result = await expander.expand(SEED_NID)

    assert result.outgoing.node_ids == ["openalex:Wok"]
    assert result.outgoing.rejected == 1
    assert store.get_references(SEED_NID) == ["openalex:Wok"]
    assert store.get_paper_summary("openalex:Wbad") is None
    assert store.get_canonical_id("doi:10.1234/bad") is None
    assert len(store.get_edge_provenance(src=SEED_NID)) == 1


async def test_unhydrated_stub_skipped_without_calling_resolver(store: GraphStore) -> None:
    stub = openalex_summary("Wghost", title="")
    oa = MockSource(name="openalex", references=[stub], batch=[])
    vp = MockResolverProvider("openalex")
    expander = build(store, {"openalex": oa}, [vp], config())

    result = await expander.expand(SEED_NID)

    assert result.outgoing.discovered == 1
    assert result.outgoing.rejected == 1
    assert result.outgoing.node_ids == []
    assert vp.doi_calls == []
    assert vp.title_calls == []
    assert store.get_references(SEED_NID) == []


async def test_title_only_candidate_resolves_via_title_search(store: GraphStore) -> None:
    title_only = summary(
        id="Wx", provider="openalex", doi=None,
        title="Graph Attention Networks", year=2018, authors=["P. Velickovic"],
    )
    oa = MockSource(name="openalex", references=[title_only])
    vp = MockResolverProvider("openalex")
    vp.title_results["Graph Attention Networks"] = [title_only]
    expander = build(store, {"openalex": oa}, [vp], config())

    result = await expander.expand(SEED_NID)

    assert result.outgoing.node_ids == ["openalex:Wx"]
    assert vp.title_calls == ["Graph Attention Networks"]
    assert store.get_paper_summary("openalex:Wx") is not None


async def test_arxiv_candidate_resolves_to_versionless_canonical(store: GraphStore) -> None:
    arxiv = summary(
        id="1905.07697", provider="arxiv", doi=None, arxiv_id="1905.07697",
        title="Graph Attention Networks", year=2018, authors=["P. Velickovic"],
    )
    oa = MockSource(name="openalex", references=[arxiv])
    vp_oa = MockResolverProvider("openalex")
    vp_s2 = MockResolverProvider("semantic_scholar")
    vp_s2.arxiv_results["1905.07697"] = arxiv
    expander = build(store, {"openalex": oa}, [vp_oa, vp_s2], config())

    result = await expander.expand(SEED_NID)

    assert result.outgoing.node_ids == ["arxiv:1905.07697"]
    assert vp_oa.arxiv_calls == ["1905.07697"]
    assert vp_s2.arxiv_calls == ["1905.07697"]
    assert store.get_references(SEED_NID) == ["arxiv:1905.07697"]


async def test_expand_respects_per_direction_limits(store: GraphStore) -> None:
    oa = MockSource(
        name="openalex",
        citations=[
            openalex_summary("Wc1", doi="10.1234/c1"),
            openalex_summary("Wc2", doi="10.1234/c2"),
        ],
        references=[
            openalex_summary("Wr1", doi="10.1234/r1"),
            openalex_summary("Wr2", doi="10.1234/r2"),
            openalex_summary("Wr3", doi="10.1234/r3"),
        ],
    )
    vp = MockResolverProvider("openalex")
    for wid, doi in [
        ("Wc1", "10.1234/c1"), ("Wc2", "10.1234/c2"),
        ("Wr1", "10.1234/r1"), ("Wr2", "10.1234/r2"), ("Wr3", "10.1234/r3"),
    ]:
        vp.doi_results[doi] = openalex_summary(wid, doi=doi)
    expander = build(store, {"openalex": oa}, [vp], config(incoming_limit=1, outgoing_limit=2))

    result = await expander.expand(SEED_NID)

    assert result.incoming.node_ids == ["openalex:Wc1"]
    assert result.outgoing.node_ids == ["openalex:Wr1", "openalex:Wr2"]
    assert oa.cit_calls == ["Wseed"]
    assert len(store.get_edge_provenance(dst=SEED_NID)) == 1
    assert len(store.get_edge_provenance(src=SEED_NID)) == 2


async def test_replay_payloads_for_expansion_and_fallback(store: GraphStore, tmp_path: Path) -> None:
    oa = MockSource(name="openalex", citations=[])
    s2 = MockSource(
        name="semantic_scholar",
        citations=[
            summary(id="Sd1", provider="semantic_scholar", doi="10.1234/d1"),
            summary(id="Sd2", provider="semantic_scholar", doi="10.1234/d2"),
        ],
    )
    vp_s2 = MockResolverProvider("semantic_scholar")
    vp_s2.doi_results["10.1234/d1"] = summary(
        id="Sd1", provider="semantic_scholar", doi="10.1234/d1"
    )
    vp_s2.doi_results["10.1234/d2"] = summary(
        id="Sd2", provider="semantic_scholar", doi="10.1234/d2",
        title="Completely unrelated",
    )
    expander = build(
        store, {"openalex": oa, "semantic_scholar": s2},
        [MockResolverProvider("openalex"), vp_s2], config(),
    )

    trace = RunTraceStore(str(tmp_path / "replay.db"))
    run_id = trace.create_run(SEED_NID, "graph attention")
    tracer = RunTracer(trace, run_id)
    try:
        result = await expander.expand(SEED_NID, tracer=tracer)
        events = trace.list_events(run_id)
    finally:
        trace.close()

    types = [e["type"] for e in events]
    assert types == [
        "provider_fallback",
        "resolution_started",
        "resolution_resolved",
        "resolution_started",
        "resolution_rejected",
        "neighbors_expanded",
        "provider_fallback",
        "neighbors_expanded",
    ]

    fallback = events[0]["payload"]
    assert fallback["node_id"] == SEED_NID
    assert fallback["direction"] == "incoming"
    assert fallback["from_provider"] == "openalex"
    assert fallback["to_provider"] == "semantic_scholar"
    assert fallback["reason"] == "empty"

    started = [e for e in events if e["type"] == "resolution_started"]
    assert len(started) == 2
    assert {e["payload"]["direction"] for e in started} == {"incoming"}
    assert {e["payload"]["provider"] for e in started} == {"semantic_scholar"}

    resolved = [e for e in events if e["type"] == "resolution_resolved"]
    assert resolved[0]["payload"]["canonical_id"] == "s2:Sd1"
    assert resolved[0]["payload"]["provider"] == "semantic_scholar"
    assert resolved[0]["payload"]["confidence"] == pytest.approx(1.0)
    assert "method" in resolved[0]["payload"]

    rejected = [e for e in events if e["type"] == "resolution_rejected"]
    assert rejected[0]["payload"]["title"] == "Some work"
    assert rejected[0]["payload"]["reason"] == "title_mismatch"

    incoming_expanded = [
        e for e in events
        if e["type"] == "neighbors_expanded" and e["payload"]["direction"] == "incoming"
    ]
    assert len(incoming_expanded) == 1
    expanded = incoming_expanded[0]["payload"]
    assert expanded["node_id"] == SEED_NID
    assert expanded["provider"] == "semantic_scholar"
    assert expanded["discovered"] == 2
    assert expanded["resolved"] == 1
    assert expanded["rejected"] == 1
    assert expanded["fallback_used"] is True

    assert result.incoming.node_ids == ["s2:Sd1"]


async def test_replay_payload_has_outgoing_resolution_started(store: GraphStore, tmp_path: Path) -> None:
    oa = MockSource(name="openalex", references=[openalex_summary("Wr1", doi="10.1234/r1")])
    vp = MockResolverProvider("openalex")
    vp.doi_results["10.1234/r1"] = openalex_summary("Wr1", doi="10.1234/r1")
    expander = build(store, {"openalex": oa}, [vp], config())

    trace = RunTraceStore(str(tmp_path / "replay.db"))
    run_id = trace.create_run(SEED_NID, "q")
    tracer = RunTracer(trace, run_id)
    try:
        await expander.expand(SEED_NID, tracer=tracer)
        events = trace.list_events(run_id)
    finally:
        trace.close()

    outgoing_resolved = [
        e for e in events
        if e["type"] == "resolution_resolved" and e["payload"]["direction"] == "outgoing"
    ]
    assert len(outgoing_resolved) == 1
    payload = outgoing_resolved[0]["payload"]
    assert payload["node_id"] == SEED_NID
    assert payload["canonical_id"] == "openalex:Wr1"
    assert payload["method"] == "doi_lookup"
    assert payload["provider"] == "openalex"

    expanded = [e for e in events if e["type"] == "neighbors_expanded"]
    outgoing = [e for e in expanded if e["payload"]["direction"] == "outgoing"]
    assert len(outgoing) == 1
    assert outgoing[0]["payload"]["provider"] == "openalex"
    assert outgoing[0]["payload"]["discovered"] == 1
    assert outgoing[0]["payload"]["resolved"] == 1
    assert outgoing[0]["payload"]["rejected"] == 0
