"""Seed-discovery telemetry and empty-frontier classification.

Regression coverage for TUI-REL-4/TUI-REL-5: the tracer is attached before seed
discovery, every discovery pass emits structured started/completed/failed
semantics, and an empty initial frontier maps to exactly one primary reason.
"""

from __future__ import annotations

from types import SimpleNamespace

from research_explorer.aco.colony import Colony
from research_explorer.config import Config
from research_explorer.events.models import (
    REASON_NO_NEIGHBORS_DISCOVERED,
    REASON_NO_TRAVERSABLE_IDENTIFIERS,
    REASON_REFERENCE_EXTRACTION_FAILED,
    REASON_REFERENCE_MAPPING_INCOMPLETE,
    REASON_SEED_DISCOVERY_FAILED,
)
from research_explorer.events.projection import RunProjection
from research_explorer.graph.embeddings import EmbeddingService
from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.graph.store import GraphStore
from research_explorer.replay.trace import RunTracer, RunTraceStore
from research_explorer.tui import text as render

SEED_NID = "arxiv:1905.07697"


async def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [[0.5, 0.5, 0.5] for _ in texts]


class _SeedProvider:
    name = "arxiv"
    supports_fulltext = True

    def __init__(
        self,
        paper: Paper | None,
        ref_entries: list | None = None,
        raise_on_get: bool = False,
    ) -> None:
        self.paper = paper
        self.ref_entries = list(ref_entries or [])
        self.raise_on_get = raise_on_get

    async def get_paper(self, native: str, id_type: str = "auto") -> Paper | None:
        if self.raise_on_get:
            raise RuntimeError("transport exploded api_key=sk-sentinel-777")
        return self.paper

    async def get_fulltext_and_refs(self, native: str, max_chars: int = 0, ref_limit: int = 0):
        return ("full text body", list(self.ref_entries))

    async def aclose(self) -> None:
        return None


class _FakeLLM:
    def __init__(self, response: str = '{"narrative": "n", "references": []}') -> None:
        self.response = response

    async def chat(self, messages, **kwargs) -> str:
        return self.response

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await _fake_embed(texts)


def _seed_paper() -> Paper:
    return Paper(
        id="1905.07697",
        title="Graph Attention Networks",
        provider="arxiv",
        abstract="An abstract",
    )


def _colony(tmp_path, provider: _SeedProvider, llm: _FakeLLM) -> tuple[Colony, GraphStore]:
    graph = GraphStore(str(tmp_path / "g.db"))
    seed = graph.get_paper(SEED_NID)
    if seed is None:
        graph.cache_paper(_seed_paper())
    cfg = Config()
    cfg.aco.colony_size = 3
    embedding = EmbeddingService(graph, embedder=_fake_embed)
    colony = Colony(
        graph=graph,
        llm=llm,  # type: ignore[arg-type]
        embedding=embedding,
        provider=provider,  # type: ignore[arg-type]
        providers={"arxiv": provider},  # type: ignore[dict-item]
        config=cfg,
    )
    return colony, graph


async def test_no_neighbors_discovered(tmp_path) -> None:
    colony, graph = _colony(tmp_path, _SeedProvider(None), _FakeLLM())
    await colony.initialize(SEED_NID, "q")
    assert colony.init_reason == REASON_NO_NEIGHBORS_DISCOVERED
    assert len(colony.agents) == 3
    graph.close()


async def test_no_traversable_identifiers(tmp_path) -> None:
    paper = _seed_paper()
    paper.references = [PaperSummary(id="untitled", title="No ids", provider="unknown")]
    colony, graph = _colony(tmp_path, _SeedProvider(paper), _FakeLLM())
    await colony.initialize(SEED_NID, "q")
    assert colony.init_reason == REASON_NO_TRAVERSABLE_IDENTIFIERS
    graph.close()


async def test_reference_extraction_failed(tmp_path) -> None:
    paper = _seed_paper()
    provider = _SeedProvider(paper, ref_entries=[{"title": "Ref without ids"}])
    colony, graph = _colony(tmp_path, provider, _FakeLLM())
    await colony.initialize(SEED_NID, "q")
    assert colony.init_reason == REASON_REFERENCE_EXTRACTION_FAILED
    graph.close()


async def test_seed_discovery_failed_is_contained(tmp_path) -> None:
    store = RunTraceStore(str(tmp_path / "r.db"))
    run_id = store.create_run(SEED_NID, "q")
    tracer = RunTracer(store, run_id)
    colony, graph = _colony(tmp_path, _SeedProvider(None, raise_on_get=True), _FakeLLM())
    await colony.initialize(SEED_NID, "q", tracer=tracer)
    assert colony.init_reason == REASON_SEED_DISCOVERY_FAILED
    assert colony.init_failures
    assert "sk-sentinel-777" not in " ".join(colony.init_failures)
    events = store.list_events(run_id)
    failed = [e for e in events if e["type"] == "neighbor_discovery_failed"]
    assert len(failed) == 3
    assert all(e["payload"]["paper_id"] == SEED_NID for e in failed)
    store.close()
    graph.close()


async def _events_for(tmp_path, provider, llm):
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = RunTraceStore(str(tmp_path / "r.db"))
    run_id = store.create_run(SEED_NID, "q")
    tracer = RunTracer(store, run_id)
    colony, graph = _colony(tmp_path, provider, llm)
    await colony.initialize(SEED_NID, "q", tracer=tracer)
    events = store.list_events(run_id)
    store.close()
    graph.close()
    return colony, events


async def test_each_reason_has_expected_telemetry(tmp_path) -> None:
    # no_neighbors_discovered: a definitive-absence seed yields empty outcomes.
    colony, events = await _events_for(
        tmp_path / "a", _SeedProvider(None), _FakeLLM()
    )
    assert colony.init_reason == REASON_NO_NEIGHBORS_DISCOVERED
    completed = [e for e in events if e["type"] == "neighbor_discovery_completed"]
    assert len(completed) == 3
    assert all(e["payload"]["refs"] == 0 and e["payload"]["traversable"] == 0 for e in completed)
    assert all(e["payload"]["oleada"] == 0 for e in completed)

    # no_traversable_identifiers: neighbors exist but carry no usable id.
    paper = _seed_paper()
    paper.references = [PaperSummary(id="untitled", title="No ids", provider="unknown")]
    colony, events = await _events_for(
        tmp_path / "b", _SeedProvider(paper), _FakeLLM()
    )
    assert colony.init_reason == REASON_NO_TRAVERSABLE_IDENTIFIERS
    completed = [e for e in events if e["type"] == "neighbor_discovery_completed"]
    assert all(e["payload"]["refs"] == 1 and e["payload"]["traversable"] == 0 for e in completed)

    # reference_extraction_failed: a bibliography exists but extraction is empty.
    paper = _seed_paper()
    provider = _SeedProvider(paper, ref_entries=[{"title": "Ref without ids"}])
    colony, events = await _events_for(tmp_path / "c", provider, _FakeLLM())
    assert colony.init_reason == REASON_REFERENCE_EXTRACTION_FAILED
    completed = [e for e in events if e["type"] == "neighbor_discovery_completed"]
    assert all(e["payload"]["refs"] == 0 for e in completed)

    # seed_discovery_failed: the failing agent emits started + failed.
    colony, events = await _events_for(
        tmp_path / "d", _SeedProvider(None, raise_on_get=True), _FakeLLM()
    )
    assert colony.init_reason == REASON_SEED_DISCOVERY_FAILED
    started = [e for e in events if e["type"] == "neighbor_discovery_started"]
    failed = [e for e in events if e["type"] == "neighbor_discovery_failed"]
    assert len(started) == 3 and len(failed) == 3
    assert all(e["payload"]["paper_id"] == SEED_NID for e in failed)


async def test_seed_discovery_telemetry_is_structured(tmp_path) -> None:
    paper = _seed_paper()
    provider = _SeedProvider(paper, ref_entries=[{"title": "Ref without ids"}])
    store = RunTraceStore(str(tmp_path / "r.db"))
    run_id = store.create_run(SEED_NID, "q")
    tracer = RunTracer(store, run_id)
    colony, graph = _colony(tmp_path, provider, _FakeLLM())
    await colony.initialize(SEED_NID, "q", tracer=tracer)

    events = store.list_events(run_id)
    started = [e for e in events if e["type"] == "neighbor_discovery_started"]
    completed = [e for e in events if e["type"] == "neighbor_discovery_completed"]
    assert len(started) == 3 and len(completed) == 3
    for event in started + completed:
        payload = event["payload"]
        assert payload["paper_id"] == SEED_NID
        assert payload["agent_id"]
        assert payload["turn"] == 0
    assert all(e["payload"]["traversable"] == 0 for e in completed)

    projection = RunProjection.from_events(events)
    discoveries = [n for n in projection.state.timeline if n.kind == "discovery"]
    assert len(discoveries) == 3
    assert all(n.wave == 0 and n.turn == 0 for n in discoveries)
    store.close()
    graph.close()


def test_classification_prefers_failure_only_without_frontier(tmp_path) -> None:
    """A concurrent initialization failure is not masked, but a live frontier wins."""
    from research_explorer.agents.explorer import DiscoveryOutcome

    colony, graph = _colony(tmp_path, _SeedProvider(None), _FakeLLM())
    colony._classify_seed_discovery(
        [RuntimeError("boom"), DiscoveryOutcome(traversable=2)]
    )
    assert colony.init_reason == ""
    colony._classify_seed_discovery([RuntimeError("boom"), DiscoveryOutcome()])
    assert colony.init_reason == REASON_SEED_DISCOVERY_FAILED
    graph.close()


def test_classification_reports_incomplete_seed_mapping(tmp_path) -> None:
    """A pending/partial seed mapping is its own terminal reason, not 'no neighbors'."""
    from research_explorer.agents.explorer import DiscoveryOutcome

    colony, graph = _colony(tmp_path, _SeedProvider(None), _FakeLLM())
    colony._classify_seed_discovery([DiscoveryOutcome(mapping_incomplete=True)])
    assert colony.init_reason == REASON_REFERENCE_MAPPING_INCOMPLETE
    graph.close()


async def test_seed_mapping_incomplete_is_reported(tmp_path) -> None:
    """A partial seed mapping must survive into the colony terminal reason."""
    from research_explorer.references.models import ReferenceAccounting

    class _PartialBuilder:
        async def build(self, paper, tracer=None) -> ReferenceAccounting:
            return ReferenceAccounting(
                source_id=SEED_NID,
                job_id="j",
                status="partial",
                observed=1,
                processed=1,
                mapped=1,
                provisional=1,
                resolved_summaries=[
                    PaperSummary(id="Wkept", title="Kept", provider="openalex")
                ],
            )

    class _EmptyExpander:
        async def expand(self, node_id, paper=None, extracted=None, tracer=None):
            empty = SimpleNamespace(
                node_ids=[], discovered=0, rejected=0, provider=None,
                fallback_used=False,
            )
            return SimpleNamespace(
                node_id=node_id,
                incoming=SimpleNamespace(**vars(empty), direction="incoming"),
                outgoing=SimpleNamespace(**vars(empty), direction="outgoing"),
            )

    paper = _seed_paper()
    provider = _SeedProvider(paper, ref_entries=["Author. A reference. 2020."])
    colony, graph = _colony(tmp_path, provider, _FakeLLM())
    colony.reference_builder = _PartialBuilder()
    colony.expander = _EmptyExpander()

    await colony.initialize(SEED_NID, "q")

    assert colony.init_reason == REASON_REFERENCE_MAPPING_INCOMPLETE
    graph.close()


async def test_frozen_empty_frontier_regression(tmp_path) -> None:
    """Frozen reproduction of run 7a67d30b7f39.

    Three agents initialize from an arXiv seed, no traversable neighbors
    survive, the backend emits a completed degraded outcome, the TUI renders the
    reason, and a no-winner report retains the run evidence.
    """
    from research_explorer.orchestrator.report import build_report

    paper = _seed_paper()
    provider = _SeedProvider(paper, ref_entries=[{"title": "Ref without ids"}])
    store = RunTraceStore(str(tmp_path / "r.db"))
    run_id = store.create_run(SEED_NID, "graph attention")
    tracer = RunTracer(store, run_id)
    tracer.emit(
        "orchestrator_start",
        run_id=run_id,
        seed=SEED_NID,
        query="graph attention",
        colony_size=3,
        K=2,
        k_per_turn=2,
    )
    colony, graph = _colony(tmp_path, provider, _FakeLLM())
    await colony.initialize(SEED_NID, "graph attention", tracer=tracer)
    assert colony.init_reason == REASON_REFERENCE_EXTRACTION_FAILED

    tracer.emit(
        "warning",
        classification="warning",
        outcome="degraded",
        reason_code=colony.init_reason,
        reason=colony.init_reason_text,
    )
    tracer.emit(
        "no_winner",
        run_id=run_id,
        status="completed",
        outcome="degraded",
        reason_code=colony.init_reason,
        reason=colony.init_reason_text,
        elapsed=23.0,
        total_fetches=2,
        total_waves=1,
    )

    from research_explorer.events.models import RunEvent

    persisted = [
        RunEvent(seq=e["seq"], type=e["type"], payload=e["payload"], ts=e["ts"])
        for e in store.list_events(run_id)
    ]
    projection = RunProjection.from_events(persisted)
    state = projection.state
    assert state.status == "completed"
    assert state.outcome == "degraded"
    assert state.run_id == run_id
    assert state.winner_agent == ""
    assert len(state.warnings) == 1
    assert state.reason_code == REASON_REFERENCE_EXTRACTION_FAILED
    assert "reference extraction" in state.terminal_reason
    timeline = render.render_timeline(state, state.selected_entry_id)
    metadata = render.render_metadata(state)
    assert "reference extraction" in timeline
    assert "reference extraction" in metadata
    assert "degraded" in render.render_header(state)

    report = build_report(
        config=Config(),
        colony=colony,
        scheduler=SimpleNamespace(oleada_count=1, total_fetches=2, history=[]),
        convergence=SimpleNamespace(state=SimpleNamespace(quality_history=[])),
        graph=graph,
        seed_paper_id=SEED_NID,
        seed_query="graph attention",
        elapsed=23.0,
        outcome="degraded",
        terminal_reason=colony.init_reason_text,
    )
    assert "No winning narrative was produced" in report
    assert "**Outcome:** degraded" in report
    assert "reference extraction" in report
    assert "Graph Attention Networks" in report
    store.close()
    graph.close()
