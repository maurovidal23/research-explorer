"""Honest terminal-outcome invariants (WAVE-5) for completed winner runs."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from research_explorer.config import Config
from research_explorer.events.models import (
    OUTCOME_COMPLETED,
    OUTCOME_DEGRADED,
    REASON_EMPTY_WINNER_NARRATIVE,
    REASON_NO_EVALUATED_EVIDENCE,
    REASON_WINNER_EVALUATION_MISSING,
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
    def __init__(self) -> None:
        self.agents: list = []
        self.best_agent = SimpleNamespace(state=SimpleNamespace(quality=0.0809))
        self.best_quality = 0.0809
        self.best_snapshot_agent = "a0"
        self.best_snapshot_oleada = 1
        self.best_narrative = ""
        self.shared_frontier = SimpleNamespace(papers={})
        self.init_reason = ""
        self.tracer_attached = False

    @property
    def init_reason_text(self) -> str:
        return ""

    async def initialize(self, seed_id: str, seed_query: str, tracer=None) -> None:
        self.tracer_attached = tracer is not None

    def active_candidates(self) -> list:
        return [SimpleNamespace(state=SimpleNamespace(quality=0.08))]

    def pheromone_concentration(self) -> float:
        return 0.0


def _evaluation(agent_id: str, new_papers: list[str], q: float = 0.0809) -> DetailedEvaluation:
    return DetailedEvaluation(
        agent_id=agent_id,
        oleada=1,
        turn=0,
        q=q,
        new_papers=new_papers,
        status="complete",
        unavailable={"S": "empty_narrative", "P": "no_peers", "J": "empty_narrative"},
    )


def _orchestrator(tmp_path, narrative: str, evaluations: list[DetailedEvaluation]):
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
    orch.colony = _StubColony()
    orch.colony.best_narrative = narrative
    orch.scheduler = SimpleNamespace(
        total_fetches=4, oleada_count=1, history=[], evaluations=evaluations
    )
    orch.convergence = SimpleNamespace(
        should_stop=lambda *args, **kwargs: True,
        last_reason="budget_exhausted",
        state=SimpleNamespace(quality_history=[]),
    )
    orch.outcome = OUTCOME_COMPLETED
    orch.terminal_reason = ""
    orch.stop_reason = ""
    return orch, store, run_id, graph


async def _run(orch, store, run_id):
    seed_ref = SeedRef(
        kind=SeedKind.ARXIV,
        value="1905.07697",
        raw=SEED,
        fetch_value="1905.07697",
    )
    await orch._run_impl(
        SEED, "graph attention", time.monotonic(), run_id, orch.tracer, _SeedProvider(), seed_ref
    )


@pytest.mark.parametrize(
    ("narrative", "evaluations", "expected"),
    [
        (
            "",
            [_evaluation("a0", ["openalex:W1"])],
            REASON_EMPTY_WINNER_NARRATIVE,
        ),
        (
            "a real narrative",
            [],
            REASON_NO_EVALUATED_EVIDENCE,
        ),
        (
            "a real narrative",
            [_evaluation("a1", ["openalex:W1"])],
            REASON_WINNER_EVALUATION_MISSING,
        ),
    ],
)
async def test_completed_but_degraded_outcomes(tmp_path, narrative, evaluations, expected) -> None:
    orch, store, run_id, graph = _orchestrator(tmp_path, narrative, evaluations)
    await _run(orch, store, run_id)

    assert orch.outcome == OUTCOME_DEGRADED
    assert orch.terminal_reason == reason_text(expected)
    events = store.list_events(run_id)
    complete = next(e for e in events if e["type"] == "orchestrator_complete")
    assert complete["payload"]["status"] == "completed"
    assert complete["payload"]["outcome"] == OUTCOME_DEGRADED
    assert complete["payload"]["reason_code"] == expected
    assert complete["payload"]["stop_reason"] == "budget_exhausted"

    state = RunProjection.from_events(
        [
            RunEvent(seq=e["seq"], type=e["type"], payload=e["payload"], ts=e["ts"])
            for e in events
        ]
    ).state
    assert state.status == "completed"
    assert state.outcome == OUTCOME_DEGRADED
    assert state.reason_code == expected
    assert expected in render.render_metadata(state)
    assert "degraded" in render.render_header(state)

    report = orch.generate_report(SEED, "graph attention")
    assert "**Outcome:** degraded" in report
    assert f"**Terminal reason:** {reason_text(expected)}" in report
    graph.close()
    store.close()


async def test_valid_narrative_and_terminal_evaluation_stay_completed(tmp_path) -> None:
    evaluations = [_evaluation("a0", ["openalex:W1"])]
    orch, store, run_id, graph = _orchestrator(tmp_path, "a valid narrative", evaluations)
    # The winner must be the agent that actually has the terminal evaluation.
    orch.colony.best_snapshot_agent = "a0"
    await _run(orch, store, run_id)

    assert orch.outcome == OUTCOME_COMPLETED
    assert orch.terminal_reason == ""
    complete = next(
        e for e in store.list_events(run_id) if e["type"] == "orchestrator_complete"
    )
    assert complete["payload"]["outcome"] == OUTCOME_COMPLETED
    assert complete["payload"]["stop_reason"] == "budget_exhausted"
    graph.close()
    store.close()
