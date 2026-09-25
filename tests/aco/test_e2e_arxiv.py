"""Combined end-to-end test: arXiv seed resolution -> cross-provider expansion
-> deterministic ACO selection, plus fallback/duplicate/budget scenarios.

The seed starts as the *versioned* arXiv id ``1905.07697v2``; resolution and
provider parsing must collapse it to the versionless canonical ``arxiv:1905.07697``
before expansion, and expansion must push both directions (references via the
OpenAlex primary, cited-by via the Semantic Scholar fallback) through the
deterministic resolver into the shared frontier, which then drives ACO selection.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from research_explorer.aco.colony import Colony
from research_explorer.aco.frontier import SharedFrontier
from research_explorer.aco.transition import select_candidate
from research_explorer.agents.explorer import ExplorerAgent
from research_explorer.agents.state import AgentState
from research_explorer.config import Config, ResolutionConfig
from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.graph.store import GraphStore
from research_explorer.replay.trace import RunTracer, RunTraceStore
from research_explorer.resolution.resolver import IdentityResolver, normalize_arxiv
from research_explorer.resolution.traversal import NeighborExpander

SEED = "arxiv:1905.07697v2"
SEED_VERSIONLESS = "arxiv:1905.07697"


def summary(**overrides: Any) -> PaperSummary:
    values: dict[str, Any] = {
        "id": "W1",
        "title": "Some work",
        "year": 2020,
        "authors": ["A. Author"],
        "provider": "openalex",
    }
    values.update(overrides)
    return PaperSummary(**values)


class MockSource:
    """Mock OpenAlex/S2 expansion source exposing the raw fetch surface."""

    def __init__(
        self,
        name: str,
        *,
        citations: list[PaperSummary] | None = None,
        references: list[PaperSummary] | None = None,
        batch: list[PaperSummary] | None = None,
        fail_citations: bool = False,
        fail_references: bool = False,
        paper_id: str | None = None,
    ) -> None:
        self.name = name
        self.citations = list(citations or [])
        self.references = list(references or [])
        self.batch = list(batch or [])
        self.cit_calls: list[str] = []
        self.ref_calls: list[str] = []
        self.batch_calls: list[list[str]] = []
        self.fail_citations = fail_citations
        self.fail_references = fail_references
        self.paper_id = paper_id

    async def get_citations(self, key: str, limit: int = 50) -> list[PaperSummary]:
        self.cit_calls.append(key)
        if self.fail_citations:
            raise RuntimeError("transport failure")
        return list(self.citations[:limit])

    async def get_references(self, key: str, limit: int = 50) -> list[PaperSummary]:
        self.ref_calls.append(key)
        if self.fail_references:
            raise RuntimeError("transport failure")
        return list(self.references[:limit])

    async def get_paper(self, key: str, id_type: str = "auto") -> Paper | None:
        if self.paper_id is None:
            return None
        return Paper(id=self.paper_id, title="", provider="openalex", doi=None)

    async def get_works_batch(self, ids: list[str], limit: int = 50) -> list[PaperSummary]:
        self.batch_calls.append(list(ids))
        return list(self.batch[:limit])


class MockVP:
    """Mock resolver VerificationProvider (deterministic IdentityResolver)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.doi: dict[str, PaperSummary | None] = {}
        self.arxiv: dict[str, PaperSummary | None] = {}
        self.title: dict[str, list[PaperSummary]] = {}
        self.doi_calls: list[str] = []
        self.arxiv_calls: list[str] = []
        self.title_calls: list[str] = []

    async def lookup_doi(self, doi: str) -> PaperSummary | None:
        self.doi_calls.append(doi)
        return self.doi.get(doi)

    async def lookup_arxiv(self, arxiv_id: str) -> PaperSummary | None:
        self.arxiv_calls.append(arxiv_id)
        return self.arxiv.get(arxiv_id)

    async def search_title(self, title: str) -> list[PaperSummary]:
        self.title_calls.append(title)
        return list(self.title.get(title, []))


def resolution_config() -> ResolutionConfig:
    return ResolutionConfig(
        enabled=True,
        primary="openalex",
        fallbacks=["semantic_scholar"],
        incoming_enabled=True,
        outgoing_enabled=True,
        incoming_limit=50,
        outgoing_limit=50,
        title_search_limit=5,
    )


def seed_graph(store: GraphStore) -> None:
    """Cache the arXiv seed's metadata exactly as a fetched/parsed paper would."""
    store.cache_summary_for_nid(
        SEED_VERSIONLESS,
        PaperSummary(
            id="1905.07697",
            title="Graph Attention Networks",
            year=2018,
            authors=["P. Velickovic"],
            arxiv_id="1905.07697",
            doi="10.48550/arxiv.1905.07697",
            provider="arxiv",
        ),
    )


async def test_arxiv_1905_07697v2_end_to_end(tmp_path: Path) -> None:
    """Assert the full version-to-canonical, expand, fallback, select chain."""
    store = GraphStore(str(tmp_path / "g.db"))
    seed_graph(store)

    oa = MockSource(
        name="openalex",
        references=[summary(id="Wref1", doi="10.1234/ref1"),
                    summary(id="Wref2", doi="10.1234/ref2")],
        paper_id="W90507697",
    )
    s2 = MockSource(
        name="semantic_scholar",
        citations=[summary(id="S0", provider="semantic_scholar", doi="10.1234/cit1")],
    )
    vp_oa = MockVP("openalex")
    vp_s2 = MockVP("semantic_scholar")
    vp_oa.doi["10.1234/ref1"] = summary(id="Wref1", doi="10.1234/ref1")
    vp_oa.doi["10.1234/ref2"] = summary(id="Wref2", doi="10.1234/ref2")
    vp_s2.doi["10.1234/cit1"] = summary(id="S0", provider="semantic_scholar", doi="10.1234/cit1")

    expander = NeighborExpander(
        providers={"openalex": oa, "semantic_scholar": s2},
        resolver=IdentityResolver(providers=[vp_oa, vp_s2], store=store),
        store=store,
        config=resolution_config(),
    )

    trace = RunTraceStore(str(tmp_path / "r.db"))
    run_id = trace.create_run(SEED, "graph attention")
    tracer = RunTracer(trace, run_id)

    result = await expander.expand(SEED_VERSIONLESS, tracer=tracer)

    # Outgoing references via OpenAlex primary; incoming cited-by via S2 fallback.
    assert result.outgoing.node_ids == ["openalex:Wref1", "openalex:Wref2"]
    assert result.outgoing.provider == "openalex"
    assert result.outgoing.fallback_used is False
    assert result.incoming.node_ids == ["s2:S0"]
    assert result.incoming.provider == "semantic_scholar"
    assert result.incoming.fallback_used is True

    # Directed edges recorded with provenance, in the correct orientation.
    outgoing = store.get_edge_provenance(src=SEED_VERSIONLESS)
    assert [r["dst"] for r in outgoing] == ["openalex:Wref1", "openalex:Wref2"]
    assert all(r["direction"] == "references" and r["provider"] == "openalex" for r in outgoing)
    incoming = store.get_edge_provenance(dst=SEED_VERSIONLESS)
    assert [r["src"] for r in incoming] == ["s2:S0"]
    assert all(r["direction"] == "cited_by" and r["provider"] == "semantic_scholar" for r in incoming)

    # Replay stream contains the full ordered resolution narrative.
    types = [e["type"] for e in trace.list_events(run_id)]
    assert "provider_fallback" in types
    assert "resolution_started" in types
    assert "resolution_resolved" in types
    assert "neighbors_expanded" in types
    assert "resolution_rejected" not in types

    # Feed the expanded candidates into deterministic ACO selection.
    frontier = SharedFrontier()
    frontier.add(result.outgoing.node_ids, source=SEED_VERSIONLESS, mode="ref")
    frontier.add(result.incoming.node_ids, source=SEED_VERSIONLESS, mode="cites")
    agent = _make_agent(frontier, store)
    selection = await agent._select_candidate()
    assert selection is not None
    assert set(selection.probabilities) == {
        "openalex:Wref1", "openalex:Wref2", "s2:S0",
    }
    assert sum(selection.probabilities.values()) == pytest.approx(1.0)
    trace.close()
    store.close()


async def test_fallback_on_unusable_provider_result(tmp_path: Path) -> None:
    """A *raising* (unusable) primary must fall through to the fallback cleanly."""
    store = GraphStore(str(tmp_path / "g.db"))
    seed_graph(store)

    oa = MockSource(name="openalex", citations=[], fail_citations=True)
    s2 = MockSource(
        name="semantic_scholar",
        citations=[summary(id="S1", provider="semantic_scholar", doi="10.1234/c1")],
    )
    vp_s2 = MockVP("semantic_scholar")
    vp_s2.doi["10.1234/c1"] = summary(id="S1", provider="semantic_scholar", doi="10.1234/c1")

    expander = NeighborExpander(
        providers={"openalex": oa, "semantic_scholar": s2},
        resolver=IdentityResolver(providers=[MockVP("openalex"), vp_s2], store=store),
        store=store,
        config=resolution_config(),
    )

    trace = RunTraceStore(str(tmp_path / "r.db"))
    run_id = trace.create_run(SEED_VERSIONLESS, "q")
    tracer = RunTracer(trace, run_id)
    result = await expander.expand(SEED_VERSIONLESS, tracer=tracer)
    events = trace.list_events(run_id)

    assert result.incoming.node_ids == ["s2:S1"]
    assert result.incoming.fallback_used is True
    assert result.incoming.provider == "semantic_scholar"
    fb = [e for e in events if e["type"] == "provider_fallback" and e["payload"]["direction"] == "incoming"]
    assert fb and fb[0]["payload"]["reason"] == "unavailable"
    trace.close()
    store.close()


async def test_malformed_and_duplicate_identifiers_handled(tmp_path: Path) -> None:
    """Bogus DOI/arXiv ids are unverifiable and fall to title; duplicates collapse."""
    store = GraphStore(str(tmp_path / "g.db"))
    seed_graph(store)

    bad_doi = summary(id="Wbad", doi="10.999/bogus!!", title="Nope")
    dup = [
        summary(id="Wd1", doi="10.1234/same", title="Same work"),
        summary(id="Wd2", doi="10.1234/same", title="Same work"),
    ]
    oa = MockSource(name="openalex", references=[bad_doi, *dup], paper_id="W90507697")
    vp = MockVP("openalex")
    vp.doi["10.1234/same"] = summary(id="Wd1", doi="10.1234/same", title="Same work")

    expander = NeighborExpander(
        providers={"openalex": oa},
        resolver=IdentityResolver(providers=[vp], store=store),
        store=store,
        config=resolution_config(),
    )

    result = await expander.expand(SEED_VERSIONLESS)

    # Duplicate candidates collapse; malformed id is rejected (never pollutes).
    assert result.outgoing.node_ids == ["openalex:Wd1"]
    assert result.outgoing.discovered == 2
    assert result.outgoing.rejected == 1
    assert store.get_paper_summary("openalex:Wbad") is None
    assert store.get_references(SEED_VERSIONLESS) == ["openalex:Wd1"]
    store.close()


async def test_versioned_arxiv_normalizes_to_versionless() -> None:
    assert normalize_arxiv("arXiv:1905.07697v2") == "1905.07697"
    assert normalize_arxiv("1905.07697v2") == "1905.07697"
    assert normalize_arxiv("https://arxiv.org/abs/1905.07697v2") == "1905.07697"


# ---- Colonie determinism -----------------------------------------------------


def _minimal_colony(graph: GraphStore, seed: int) -> Colony:
    colony = Colony.__new__(Colony)
    colony.graph = graph
    colony.cfg = Config()
    colony.colony_seed = seed
    colony.shared_visited = set()
    colony.shared_frontier = SharedFrontier()
    colony.expander = None
    return colony


def test_colony_seed_rng_is_reproducible() -> None:
    c1 = _minimal_colony(GraphStore.__new__(GraphStore), 7)
    c2 = _minimal_colony(GraphStore.__new__(GraphStore), 7)
    r1 = c1._agent_rng(0)
    r2 = c2._agent_rng(0)
    assert r1.random() == r2.random()

    c3 = _minimal_colony(GraphStore.__new__(GraphStore), 8)
    assert c3._agent_rng(0).random() != r1.random()


# ---- helper ------------------------------------------------------------------


class _DummyProvider:
    name = "openalex"
    supports_fulltext = False

    async def get_paper(self, native: str) -> Paper | None:
        return None

    async def get_fulltext_and_refs(self, native: str, max_chars: int = 0, ref_limit: int = 0):
        return None


def _make_agent(
    frontier: SharedFrontier,
    graph: GraphStore,
    caste: str = "mixto",
    seed: int = 42,
) -> ExplorerAgent:
    agent = ExplorerAgent.__new__(ExplorerAgent)
    agent.state = AgentState(id="agent-0", pos=SEED_VERSIONLESS, caste=caste, budget=5)
    agent.cfg = Config()
    agent.seed_embedding = None
    agent.graph = graph
    agent._shared_visited = set()
    agent._frontier = frontier
    agent.expander = None
    agent.tracer = None
    agent.rng = random.Random(seed)
    agent.provider = _DummyProvider()
    agent.providers = {"openalex": agent.provider}
    agent.embedding = None
    agent._eta_cache = {}
    agent._llm_priority = {}
    return agent


def test_deterministic_selection_uses_seeded_rng() -> None:
    """Same frontier + seed -> same chosen candidate across re-runs."""
    from research_explorer.aco.transition import Candidate

    cands = [Candidate(pid=f"p{i}", src="s", mode="ref") for i in range(5)]
    r1 = select_candidate(cands, 1.0, 2.0, 0.0, random.Random(123))
    r2 = select_candidate(cands, 1.0, 2.0, 0.0, random.Random(123))
    assert r1.chosen == r2.chosen


async def test_metadata_only_transit_feeds_deterministic_selection(tmp_path: Path) -> None:
    """Full-text-unavailable metadata transit expands refs AND cites, which then
    become selectable candidates via deterministic ACO selection."""
    from types import SimpleNamespace as Ns

    store = GraphStore(str(tmp_path / "g.db"))
    seed_graph(store)
    meta = "openalex:Wmeta"
    store.cache_summary(PaperSummary(id="Wmeta", title="Meta only", provider="openalex", doi="10.1/meta"))

    class Expander:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def expand(self, node_id, paper=None, extracted=None, tracer=None):
            self.calls.append(node_id)
            return Ns(
                node_id=node_id,
                incoming=Ns(direction="incoming", provider="openalex",
                            fallback_used=False, discovered=1, rejected=0,
                            node_ids=["openalex:Wcit1"]),
                outgoing=Ns(direction="outgoing", provider="openalex",
                            fallback_used=False, discovered=2, rejected=0,
                            node_ids=["openalex:Wref1", "openalex:Wref2"]),
            )

    expander = Expander()
    frontier = SharedFrontier()
    frontier.add([meta], source=SEED_VERSIONLESS, mode="ref")
    agent = _make_agent(frontier, store)
    agent.expander = expander
    agent.state.mark_discovered(SEED_VERSIONLESS)

    await agent._metadata_transit(meta, SEED_VERSIONLESS, "ref")

    # Neighbors added in both directions under the correct mode/source.
    assert frontier.sources["openalex:Wcit1"] == (meta, "cites")
    assert frontier.sources["openalex:Wref1"] == (meta, "ref")
    assert frontier.sources["openalex:Wref2"] == (meta, "ref")

    # The expanded candidates are now selectable through deterministic ACO.
    selection = await agent._select_candidate()
    assert selection is not None
    assert set(selection.probabilities) == {"openalex:Wcit1", "openalex:Wref1", "openalex:Wref2"}
    assert sum(selection.probabilities.values()) == pytest.approx(1.0)
    store.close()
