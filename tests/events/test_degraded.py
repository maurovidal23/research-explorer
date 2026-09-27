"""Degraded (no-winner) run projection and replay compatibility."""

from __future__ import annotations

import time

import pytest

from research_explorer.config import Config
from research_explorer.events.models import (
    GENERIC_NO_WINNER_REASON,
    REASON_LABELS,
    REASON_NO_NEIGHBORS_DISCOVERED,
    REASON_NO_TRAVERSABLE_IDENTIFIERS,
    REASON_REFERENCE_EXTRACTION_FAILED,
    REASON_SEED_DISCOVERY_FAILED,
    RunEvent,
)
from research_explorer.events.projection import RunProjection
from research_explorer.events.sink import CallbackSink
from research_explorer.graph.models import Paper
from research_explorer.orchestrator.runner import Orchestrator
from research_explorer.providers.routing import SeedKind, SeedRef
from research_explorer.replay.trace import RunTracer, RunTraceStore
from research_explorer.tui import text as render

VOLATILE = {"events", "started_at"}


def _stable_state(state) -> dict:
    data = state.model_dump(mode="json")
    for key in VOLATILE:
        data.pop(key, None)
    return data


def _started() -> RunEvent:
    return RunEvent(
        seq=1,
        type="orchestrator_start",
        payload={"run_id": "degraded-1", "seed": "arxiv:2401.00001", "colony_size": 3, "K": 2},
    )


def _enriched_no_winner(seq: int = 4) -> RunEvent:
    return RunEvent(
        seq=seq,
        type="no_winner",
        payload={
            "run_id": "degraded-1",
            "status": "completed",
            "outcome": "degraded",
            "reason_code": REASON_NO_TRAVERSABLE_IDENTIFIERS,
            "reason": "neighbors were discovered but none exposed a traversable identifier",
            "elapsed": 23.0,
            "total_fetches": 3,
            "waves": 1,
        },
    )


def test_enriched_no_winner_is_completed_degraded_with_one_warning() -> None:
    projection = RunProjection.from_events(
        [
            _started(),
            RunEvent(
                seq=2,
                type="warning",
                payload={
                    "reason": "neighbors were discovered but none exposed a traversable identifier",
                    "reason_code": REASON_NO_TRAVERSABLE_IDENTIFIERS,
                    "classification": "degraded",
                    "phase": "seed_discovery",
                },
            ),
            _enriched_no_winner(),
        ]
    )
    state = projection.state
    assert state.status == "completed"
    assert state.outcome == "degraded"
    assert state.winner_agent == ""
    assert state.terminal_reason_code == REASON_NO_TRAVERSABLE_IDENTIFIERS
    assert "traversable" in state.terminal_reason
    assert len(state.warnings) == 1
    assert not state.failures

    timeline = render.render_timeline(state, state.selected_entry_id)
    assert "traversable" in timeline
    metadata = render.render_metadata(state)
    assert "outcome: degraded" in metadata
    assert "terminal_reason_code: no_traversable_identifiers" in metadata
    assert "traversable" in render.render_footer(state)


def test_legacy_unenriched_no_winner_gets_one_generic_warning() -> None:
    projection = RunProjection.from_events(
        [
            _started(),
            RunEvent(seq=2, type="no_winner", payload={"run_id": "degraded-1", "elapsed": 23.0}),
        ]
    )
    state = projection.state
    assert state.status == "completed"
    assert state.outcome == "degraded"
    assert state.terminal_reason == GENERIC_NO_WINNER_REASON
    assert state.warnings.count(GENERIC_NO_WINNER_REASON) == 1
    assert len(state.warnings) == 1


def test_legacy_unenriched_no_winner_is_idempotent_on_redelivery() -> None:
    event = RunEvent(seq=2, type="no_winner", payload={"run_id": "degraded-1"})
    projection = RunProjection.from_events([_started(), event, event])
    assert projection.state.warnings == [GENERIC_NO_WINNER_REASON]


def test_degraded_live_and_replay_projections_are_equivalent(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("arxiv:2401.00001", "q")
    live_events: list[RunEvent] = []
    tracer = RunTracer(store, run_id, sink=CallbackSink(live_events.append))

    tracer.emit(
        "orchestrator_start",
        run_id=run_id,
        seed="arxiv:2401.00001",
        query="q",
        colony_size=3,
        K=2,
    )
    for agent_id in ("a0", "a1", "a2"):
        tracer.emit(
            "neighbor_discovery_started",
            agent_id=agent_id,
            paper_id="arxiv:2401.00001",
            oleada=0,
            turn=0,
        )
        tracer.emit(
            "neighbor_discovery_completed",
            agent_id=agent_id,
            paper_id="arxiv:2401.00001",
            oleada=0,
            turn=0,
            refs=2,
            cits=0,
            traversable=0,
        )
    reason = "neighbors were discovered but none exposed a traversable identifier"
    tracer.emit("warning", reason=reason, reason_code=REASON_NO_TRAVERSABLE_IDENTIFIERS, classification="degraded")
    tracer.emit(
        "no_winner",
        run_id=run_id,
        status="completed",
        outcome="degraded",
        reason_code=REASON_NO_TRAVERSABLE_IDENTIFIERS,
        reason=reason,
        elapsed=23.0,
        total_fetches=3,
        waves=1,
    )

    live = RunProjection.from_events(live_events)
    persisted = [
        RunEvent(seq=e["seq"], type=e["type"], payload=e["payload"], ts=e["ts"])
        for e in store.list_events(run_id)
    ]
    replayed = RunProjection.from_events(persisted)

    assert _stable_state(live.state) == _stable_state(replayed.state)
    assert replayed.state.status == "completed"
    assert replayed.state.outcome == "degraded"
    assert replayed.state.winner_agent == ""
    assert len(replayed.state.warnings) == 1
    store.close()


# ---- Orchestrator completion telemetry (TUI-REL-5, required test 5) -----------


class _FakeGraph:
    def __init__(self) -> None:
        self.cached: list[Paper] = []

    def cache_paper(self, paper: Paper) -> None:
        self.cached.append(paper)

    def get_paper(self, paper_id: str) -> None:
        return None


class _FakeProvider:
    name = "arxiv"
    supports_fulltext = False

    def __init__(self, paper: Paper) -> None:
        self._paper = paper

    async def get_paper(self, key: str) -> Paper:
        return self._paper


class _FakeColony:
    def __init__(self, reason_code: str) -> None:
        self.agents: list = []
        self._reason = reason_code
        self.best_agent = None
        self.best_quality = 0.0
        self.best_narrative = ""
        self.best_snapshot_agent = ""
        self.best_snapshot_oleada = 0
        self.seed_id = ""

    async def initialize(self, seed_id: str, seed_query: str, tracer=None) -> None:
        self.seed_id = seed_id
        self.tracer_seen = tracer

    def empty_frontier_reason(self) -> str:
        return self._reason

    def active_candidates(self) -> list:
        return []

    def pheromone_concentration(self) -> float:
        return 0.0


class _FakeConvergence:
    def should_stop(self, *args) -> bool:
        return True


class _FakeScheduler:
    total_fetches = 3
    oleada_count = 1


async def _run_degraded(reason_code: str, tmp_path) -> list[RunEvent]:
    store = RunTraceStore(tmp_path / "trace.db")
    run_id = store.create_run("arxiv:2401.12345", "q")
    live: list[RunEvent] = []
    tracer = RunTracer(store, run_id, sink=CallbackSink(live.append))

    orch = Orchestrator.__new__(Orchestrator)
    orch.cfg = Config()
    orch.graph = _FakeGraph()
    orch.colony = _FakeColony(reason_code)
    orch.scheduler = _FakeScheduler()
    orch.convergence = _FakeConvergence()
    orch.trace = store
    orch.tracer = tracer
    orch._elapsed = 0.0
    orch._terminal_reason_code = ""
    orch._run_outcome = ""
    paper = Paper(id="2401.12345", title="Seed", provider="arxiv")
    seed_ref = SeedRef(SeedKind.ARXIV, "arxiv:2401.12345", "2401.12345", "2401.12345")

    await orch._run_impl(
        "arxiv:2401.12345",
        "q",
        time.monotonic(),
        run_id,
        tracer,
        _FakeProvider(paper),
        seed_ref,
    )
    assert orch._run_outcome == "degraded"
    assert orch._terminal_reason_code == reason_code
    # The durable run finishes completed even when degraded.
    assert store.get_run(run_id)["status"] == "completed"
    store.close()
    return live


@pytest.mark.parametrize(
    "reason_code",
    [
        REASON_NO_NEIGHBORS_DISCOVERED,
        REASON_NO_TRAVERSABLE_IDENTIFIERS,
        REASON_REFERENCE_EXTRACTION_FAILED,
        REASON_SEED_DISCOVERY_FAILED,
    ],
)
async def test_orchestrator_emits_enriched_no_winner_per_reason(
    reason_code: str, tmp_path
) -> None:
    live = await _run_degraded(reason_code, tmp_path)

    warnings = [e for e in live if e.type == "warning"]
    no_winners = [e for e in live if e.type == "no_winner"]
    assert len(warnings) == 1
    assert len(no_winners) == 1

    warning = warnings[0]
    assert warning.payload["reason_code"] == reason_code
    assert warning.payload["reason"] == REASON_LABELS[reason_code]
    assert warning.payload["classification"] == "degraded"
    assert warning.payload["phase"] == "seed_discovery"

    payload = no_winners[0].payload
    assert payload["run_id"]
    assert payload["status"] == "completed"
    assert payload["outcome"] == "degraded"
    assert payload["reason_code"] == reason_code
    assert payload["reason"] == REASON_LABELS[reason_code]
    assert payload["elapsed"] >= 0.0
    assert payload["total_fetches"] == 3
    assert payload["waves"] == 1

    state = RunProjection.from_events(live).state
    assert state.status == "completed"
    assert state.outcome == "degraded"
    assert state.terminal_reason_code == reason_code
    assert len(state.warnings) == 1
    assert not state.failures


def _bare_orchestrator(store: RunTraceStore, run_id: str) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch.run_id = run_id
    orch.trace = store
    orch.tracer = None
    return orch


def test_mark_cancelled_does_not_relabel_completed_run(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "trace.db")
    run_id = store.create_run("arxiv:2401.00001", "q")
    store.finish_run(run_id, "completed", best_quality=1.0)

    _bare_orchestrator(store, run_id).mark_cancelled()

    assert store.get_run(run_id)["status"] == "completed"
    store.close()


def test_mark_cancelled_does_not_relabel_failed_run(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "trace.db")
    run_id = store.create_run("arxiv:2401.00001", "q")
    store.finish_run(run_id, "failed")

    _bare_orchestrator(store, run_id).mark_cancelled()

    assert store.get_run(run_id)["status"] == "failed"
    store.close()


def test_mark_cancelled_persists_cancelled_for_running_run(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "trace.db")
    run_id = store.create_run("arxiv:2401.00001", "q")

    _bare_orchestrator(store, run_id).mark_cancelled()

    assert store.get_run(run_id)["status"] == "cancelled"
    store.close()


async def test_durable_completion_precedes_terminal_event_publication(tmp_path) -> None:
    """The trace store must be terminal before any terminal event is published."""
    store = RunTraceStore(tmp_path / "trace.db")
    run_id = store.create_run("arxiv:2401.12345", "q")
    observed: list[str] = []

    class _StatusSink:
        def publish(self, event: RunEvent) -> None:
            if event.type in ("no_winner", "orchestrator_complete"):
                run = store.get_run(run_id)
                observed.append(run["status"] if run else "missing")

    tracer = RunTracer(store, run_id, sink=_StatusSink())
    orch = Orchestrator.__new__(Orchestrator)
    orch.cfg = Config()
    orch.graph = _FakeGraph()
    orch.colony = _FakeColony(REASON_NO_TRAVERSABLE_IDENTIFIERS)
    orch.scheduler = _FakeScheduler()
    orch.convergence = _FakeConvergence()
    orch.trace = store
    orch.tracer = tracer
    orch._elapsed = 0.0
    orch._terminal_reason_code = ""
    orch._run_outcome = ""
    paper = Paper(id="2401.12345", title="Seed", provider="arxiv")
    seed_ref = SeedRef(SeedKind.ARXIV, "arxiv:2401.12345", "2401.12345", "2401.12345")

    await orch._run_impl(
        "arxiv:2401.12345",
        "q",
        time.monotonic(),
        run_id,
        tracer,
        _FakeProvider(paper),
        seed_ref,
    )

    assert observed == ["completed"]
    store.close()
