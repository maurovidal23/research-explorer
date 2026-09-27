"""Seed-discovery telemetry and empty-frontier classification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_explorer.aco.colony import Colony
from research_explorer.aco.diagnostics import SeedDiscovery, classify_empty_frontier
from research_explorer.aco.frontier import SharedFrontier
from research_explorer.agents.explorer import ExplorerAgent
from research_explorer.events.models import (
    REASON_NO_NEIGHBORS_DISCOVERED,
    REASON_NO_TRAVERSABLE_IDENTIFIERS,
    REASON_REFERENCE_EXTRACTION_FAILED,
    REASON_SEED_DISCOVERY_FAILED,
)


class _RecordingTracer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, type: str, **payload) -> int:
        self.events.append((type, payload))
        return len(self.events)


def _record(**kwargs) -> SeedDiscovery:
    base = {"agent_id": "a0"}
    base.update(kwargs)
    return SeedDiscovery(**base)


def test_classify_empty_frontier_reason_codes() -> None:
    assert classify_empty_frontier([]) == REASON_SEED_DISCOVERY_FAILED
    assert (
        classify_empty_frontier([_record(failed=True), _record(failed=True)])
        == REASON_SEED_DISCOVERY_FAILED
    )
    assert (
        classify_empty_frontier([_record(refs=0, cits=0)])
        == REASON_NO_NEIGHBORS_DISCOVERED
    )
    assert (
        classify_empty_frontier([_record(refs=0, cits=0, fulltext_seed=True)])
        == REASON_REFERENCE_EXTRACTION_FAILED
    )
    assert (
        classify_empty_frontier([_record(refs=3, cits=1, traversable=0)])
        == REASON_NO_TRAVERSABLE_IDENTIFIERS
    )
    assert classify_empty_frontier([_record(refs=3, cits=1, traversable=2)]) is None


def _fake_colony(supports_fulltext: bool) -> Colony:
    colony = Colony.__new__(Colony)
    provider = SimpleNamespace(name="arxiv", supports_fulltext=supports_fulltext)
    colony.graph = SimpleNamespace(get_paper=lambda pid: None)
    colony.llm = SimpleNamespace()
    colony.embedding = SimpleNamespace()
    colony.provider = provider
    colony.providers = {"arxiv": provider}
    colony.cfg = SimpleNamespace(
        aco=SimpleNamespace(colony_size=2, max_concurrent=1, k_per_turn=1),
        budget=SimpleNamespace(max_fetches=10),
    )
    colony.colony_seed = None
    colony.agents = []
    colony.seed_id = ""
    colony.seed_query = ""
    colony.seed_embedding = None
    colony.shared_visited = set()
    colony.shared_frontier = SharedFrontier()
    colony.expander = None
    colony._seed_discovery_records = []
    return colony


async def test_colony_attaches_tracer_before_seed_discovery(monkeypatch) -> None:
    colony = _fake_colony(supports_fulltext=False)
    seen: list[object] = []

    async def fake_discover(self, paper_id, paper=None, extracted=None) -> None:
        seen.append(self.tracer)
        self.state.set_local_neighbors(paper_id, ["arxiv:2401.00002"], [])

    monkeypatch.setattr(ExplorerAgent, "_discover_neighbors", fake_discover)
    tracer = _RecordingTracer()

    await colony.initialize("arxiv:2401.00001", "question", tracer=tracer)

    assert all(item is tracer for item in seen)
    starts = [e for e in tracer.events if e[0] == "neighbor_discovery_started"]
    completes = [e for e in tracer.events if e[0] == "neighbor_discovery_completed"]
    assert len(starts) == 2
    assert len(completes) == 2
    for _type, payload in completes:
        assert payload["oleada"] == 0
        assert payload["turn"] == 0
        assert payload["seed"] == "arxiv:2401.00001"
        assert payload["paper_id"] == "arxiv:2401.00001"
        assert payload["traversable"] == 1
    assert colony.empty_frontier_reason() is None


async def test_seed_discovery_failure_is_contained_and_redacted(monkeypatch) -> None:
    colony = _fake_colony(supports_fulltext=False)

    async def boom(self, paper_id, paper=None, extracted=None) -> None:
        raise RuntimeError("fetch failed api_key=sk-sentinel-1234")

    monkeypatch.setattr(ExplorerAgent, "_discover_neighbors", boom)
    tracer = _RecordingTracer()

    await colony.initialize("arxiv:2401.00001", "question", tracer=tracer)

    failed = [e for e in tracer.events if e[0] == "neighbor_discovery_failed"]
    assert len(failed) == 2
    assert "sk-sentinel-1234" not in str(failed)
    assert colony.empty_frontier_reason() == REASON_SEED_DISCOVERY_FAILED


async def test_empty_frontier_reason_reflects_non_traversable_neighbors(monkeypatch) -> None:
    colony = _fake_colony(supports_fulltext=True)

    async def fake_discover(self, paper_id, paper=None, extracted=None) -> None:
        self.state.set_local_neighbors(paper_id, ["unknown:Some Title"], [])

    monkeypatch.setattr(ExplorerAgent, "_discover_neighbors", fake_discover)
    tracer = _RecordingTracer()

    await colony.initialize("arxiv:2401.00001", "question", tracer=tracer)

    assert colony.empty_frontier_reason() == REASON_NO_TRAVERSABLE_IDENTIFIERS


@pytest.mark.parametrize(
    ("fulltext", "expected"),
    [
        (False, REASON_NO_NEIGHBORS_DISCOVERED),
        (True, REASON_REFERENCE_EXTRACTION_FAILED),
        (True, REASON_NO_TRAVERSABLE_IDENTIFIERS),
    ],
)
async def test_empty_frontier_telemetry_is_structured(
    monkeypatch, fulltext: bool, expected: str
) -> None:
    colony = _fake_colony(supports_fulltext=fulltext)

    async def fake_discover(self, paper_id, paper=None, extracted=None) -> None:
        if expected == REASON_NO_TRAVERSABLE_IDENTIFIERS:
            self.state.set_local_neighbors(paper_id, ["unknown:Title"], [])
        else:
            self.state.set_local_neighbors(paper_id, [], [])

    monkeypatch.setattr(ExplorerAgent, "_discover_neighbors", fake_discover)
    tracer = _RecordingTracer()

    await colony.initialize("arxiv:2401.00001", "question", tracer=tracer)

    assert colony.empty_frontier_reason() == expected
    completes = [e for e in tracer.events if e[0] == "neighbor_discovery_completed"]
    assert len(completes) == 2
    for _type, payload in completes:
        assert set(payload) >= {"agent_id", "paper_id", "oleada", "turn", "refs", "cits", "traversable"}
        assert payload["oleada"] == 0
        assert payload["turn"] == 0
    assert not any(e[0] == "llm_operation_completed" for e in tracer.events)


def test_classifier_is_deterministic() -> None:
    assert classify_empty_frontier([_record()]) == REASON_NO_NEIGHBORS_DISCOVERED
