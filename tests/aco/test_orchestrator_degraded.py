"""Orchestrator-level degraded completion telemetry (TUI-REL-4/5/6).

The colony tests assert per-agent *initialization* telemetry; these drive the
real ``Orchestrator._run_impl`` so the *completion* contract — a preceding
warning followed by an enriched ``no_winner``, a run persisted as ``completed``,
and a replay projection that renders the terminal reason — is executable for
every empty-frontier classification. A regression in ``orchestrator/runner.py``
(or in the report's outcome/terminal-reason wiring) fails these tests even if
the projection fixtures stay green.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from research_explorer.config import Config
from research_explorer.events.models import (
    OUTCOME_DEGRADED,
    REASON_NO_NEIGHBORS_DISCOVERED,
    REASON_NO_TRAVERSABLE_IDENTIFIERS,
    REASON_NO_WINNER,
    REASON_REFERENCE_EXTRACTION_FAILED,
    REASON_SEED_DISCOVERY_FAILED,
    RunEvent,
    reason_text,
)
from research_explorer.events.projection import RunProjection
from research_explorer.graph.models import Paper
from research_explorer.graph.store import GraphStore
from research_explorer.orchestrator.runner import Orchestrator
from research_explorer.providers.routing import SeedKind, SeedRef
from research_explorer.replay.models import DetailedEvaluation
from research_explorer.replay.trace import RunTracer, RunTraceStore
from research_explorer.tui import text as render

SEED = "arxiv:1905.07697"


class _SeedProvider:
    name = "arxiv"

    async def get_paper(self, native: str, id_type: str = "auto") -> Paper:
        return Paper(id="1905.07697", title="Graph Attention Networks", provider="arxiv")


class _StubColony:
    """Minimal colony that reports an empty frontier and no winner."""

    def __init__(self, reason_code: str) -> None:
        self.init_reason = reason_code
        self.agents: list = []
        self.best_agent = None
        self.best_quality = 0.0
        self.best_snapshot_agent = ""
        self.best_snapshot_oleada = 0
        self.best_narrative = ""
        self.shared_frontier = SimpleNamespace(papers={})
        self.tracer_attached = False

    @property
    def init_reason_text(self) -> str:
        return reason_text(self.init_reason) if self.init_reason else ""

    async def initialize(self, seed_id: str, seed_query: str, tracer=None) -> None:
        self.tracer_attached = tracer is not None

    def active_candidates(self) -> list:
        return []

    def pheromone_concentration(self) -> float:
        return 0.0


def _orchestrator(tmp_path, reason_code: str):
    graph = GraphStore(str(tmp_path / "g.db"))
    store = RunTraceStore(str(tmp_path / "replay.db"))
    run_id = store.create_run(SEED, "graph attention")
    tracer = RunTracer(store, run_id)
    orch = Orchestrator.__new__(Orchestrator)
    orch.cfg = Config()
    orch.graph = graph
    orch.trace = store
    orch.run_id = run_id
    orch.tracer = tracer
    orch.colony = _StubColony(reason_code)
    orch.scheduler = SimpleNamespace(total_fetches=4, oleada_count=1, history=[])
    orch.convergence = SimpleNamespace(
        should_stop=lambda *args, **kwargs: True,
        state=SimpleNamespace(quality_history=[]),
    )
    orch.outcome = "completed"
    orch.terminal_reason = ""
    return orch, store, run_id, tracer, graph


def _expected(code: str) -> tuple[str, str]:
    effective = code or REASON_NO_WINNER
    return effective, reason_text(effective)


def _persisted_events(store: RunTraceStore, run_id: str) -> list[RunEvent]:
    return [
        RunEvent(seq=e["seq"], type=e["type"], payload=e["payload"], ts=e["ts"])
        for e in store.list_events(run_id)
    ]


@pytest.mark.parametrize(
    "reason_code",
    [
        REASON_NO_NEIGHBORS_DISCOVERED,
        REASON_NO_TRAVERSABLE_IDENTIFIERS,
        REASON_REFERENCE_EXTRACTION_FAILED,
        REASON_SEED_DISCOVERY_FAILED,
        "",
    ],
)
async def test_no_winner_emits_structured_degraded_completion(tmp_path, reason_code) -> None:
    orch, store, run_id, tracer, graph = _orchestrator(tmp_path, reason_code)
    seed_ref = SeedRef(
        kind=SeedKind.ARXIV,
        value="1905.07697",
        raw=SEED,
        fetch_value="1905.07697",
    )

    await orch._run_impl(
        SEED,
        "graph attention",
        time.monotonic(),
        run_id,
        tracer,
        _SeedProvider(),
        seed_ref,
    )

    effective_code, expected_text = _expected(reason_code)
    assert orch.outcome == OUTCOME_DEGRADED
    assert orch.terminal_reason == expected_text
    # The tracer reaches the colony before seed discovery (TUI-REL-4).
    assert orch.colony.tracer_attached is True

    events = store.list_events(run_id)
    types = [e["type"] for e in events]
    assert types.index("warning") < types.index("no_winner")
    warning = next(e for e in events if e["type"] == "warning")
    no_winner = next(e for e in events if e["type"] == "no_winner")
    for payload in (warning["payload"], no_winner["payload"]):
        assert payload["outcome"] == OUTCOME_DEGRADED
        assert payload["reason_code"] == effective_code
        assert payload["reason"] == expected_text
    assert warning["payload"]["classification"] == "warning"
    assert no_winner["payload"]["status"] == "completed"
    assert no_winner["payload"]["total_fetches"] == 4
    assert no_winner["payload"]["total_waves"] == 1

    # The durable run is a completed run, not a failed one.
    assert store.get_run(run_id)["status"] == "completed"

    state = RunProjection.from_events(_persisted_events(store, run_id)).state
    assert state.status == "completed"
    assert state.outcome == OUTCOME_DEGRADED
    assert state.winner_agent == ""
    assert len(state.warnings) == 1
    assert state.reason_code == effective_code
    assert state.terminal_reason == expected_text
    assert state.total_waves == 1

    # Terminal reason reaches the timeline, metadata, and header (TUI-REL-6).
    assert expected_text in render.render_timeline(state, state.selected_entry_id)
    assert expected_text in render.render_metadata(state)
    assert "degraded" in render.render_header(state)
    # Exactly one warning timeline node, so the stop is explained once.
    warning_nodes = [e for e in state.timeline if e.kind == "warning"]
    assert len(warning_nodes) == 1
    assert warning_nodes[0].label == expected_text

    # The generated report carries the outcome and terminal reason.
    report = orch.generate_report(SEED, "graph attention")
    assert "**Outcome:** degraded" in report
    assert f"**Terminal reason:** {expected_text}" in report
    assert "No winning narrative was produced" in report
    assert f"Reason: {expected_text}." in report

    store.close()
    graph.close()


async def test_completed_winner_run_keeps_completed_outcome(tmp_path) -> None:
    orch, store, run_id, tracer, graph = _orchestrator(tmp_path, "")

    class _Winner:
        state = SimpleNamespace(quality=0.9)

    orch.colony.best_agent = _Winner()
    orch.colony.best_snapshot_agent = "a0"
    orch.colony.best_quality = 0.9
    orch.colony.best_narrative = "winning narrative"
    orch.scheduler.evaluations = [
        DetailedEvaluation(
            agent_id="a0",
            oleada=1,
            turn=0,
            q=0.9,
            new_papers=["openalex:W1"],
            status="complete",
        )
    ]
    seed_ref = SeedRef(
        kind=SeedKind.ARXIV,
        value="1905.07697",
        raw=SEED,
        fetch_value="1905.07697",
    )

    await orch._run_impl(
        SEED,
        "graph attention",
        time.monotonic(),
        run_id,
        tracer,
        _SeedProvider(),
        seed_ref,
    )

    assert orch.outcome == "completed"
    assert orch.terminal_reason == ""
    no_winner = [e for e in store.list_events(run_id) if e["type"] == "no_winner"]
    assert not no_winner
    complete = next(e for e in store.list_events(run_id) if e["type"] == "orchestrator_complete")
    assert complete["payload"]["outcome"] == "completed"
    graph.close()
    store.close()
