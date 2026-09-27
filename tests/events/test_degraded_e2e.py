"""Frozen end-to-end regression for degraded no-winner run ``7a67d30b7f39``.

Three agents initialize from an arXiv seed, no traversable neighbors survive,
the backend emits a completed degraded outcome, the TUI renders the reason, and
the report is preserved for inspection.
"""

from __future__ import annotations

from types import SimpleNamespace

from research_explorer.aco.colony import Colony
from research_explorer.aco.frontier import SharedFrontier
from research_explorer.config import Config
from research_explorer.events.models import (
    REASON_LABELS,
    REASON_NO_TRAVERSABLE_IDENTIFIERS,
    RunEvent,
)
from research_explorer.events.projection import RunProjection
from research_explorer.graph.store import GraphStore
from research_explorer.orchestrator.report import build_report
from research_explorer.tui import text as render

RUN_ID = "7a67d30b7f39"
SEED = "arxiv:2401.12345"
REASON = REASON_LABELS[REASON_NO_TRAVERSABLE_IDENTIFIERS]


def _frozen_stream() -> list[RunEvent]:
    events = [
        RunEvent(
            seq=1,
            type="orchestrator_start",
            payload={
                "run_id": RUN_ID,
                "seed": SEED,
                "query": "recent advances",
                "pipeline": "aco",
                "colony_size": 3,
                "K": 2,
                "k_per_turn": 4,
                "max_fetches": 40,
                "explorer_model": "explorer-x",
                "judge_model": "judge-y",
            },
        ),
        RunEvent(seq=2, type="seed_routing_started", payload={"seed": SEED}),
        RunEvent(
            seq=3,
            type="seed_routed",
            payload={"seed": SEED, "kind": "arxiv", "normalized": SEED, "provider": "arxiv"},
        ),
        RunEvent(seq=4, type="colony_init_started", payload={"seed": SEED}),
        RunEvent(
            seq=5,
            type="colony_initialized",
            payload={
                "size": 3,
                "seed": SEED,
                "agents": ["agent-000", "agent-001", "agent-002"],
                "empty_frontier_reason": REASON_NO_TRAVERSABLE_IDENTIFIERS,
            },
        ),
    ]
    seq = 6
    for agent_id in ("agent-000", "agent-001", "agent-002"):
        events.append(
            RunEvent(
                seq=seq,
                type="neighbor_discovery_started",
                payload={"agent_id": agent_id, "paper_id": SEED, "seed": SEED, "oleada": 0, "turn": 0},
            )
        )
        seq += 1
        events.append(
            RunEvent(
                seq=seq,
                type="neighbor_discovery_completed",
                payload={
                    "agent_id": agent_id,
                    "paper_id": SEED,
                    "seed": SEED,
                    "oleada": 0,
                    "turn": 0,
                    "refs": 2,
                    "cits": 0,
                    "traversable": 0,
                },
            )
        )
        seq += 1
    events.append(
        RunEvent(
            seq=seq,
            type="warning",
            payload={
                "reason": REASON,
                "reason_code": REASON_NO_TRAVERSABLE_IDENTIFIERS,
                "classification": "degraded",
                "phase": "seed_discovery",
            },
        )
    )
    seq += 1
    events.append(
        RunEvent(
            seq=seq,
            type="no_winner",
            payload={
                "run_id": RUN_ID,
                "status": "completed",
                "outcome": "degraded",
                "reason_code": REASON_NO_TRAVERSABLE_IDENTIFIERS,
                "reason": REASON,
                "elapsed": 23.0,
                "total_fetches": 3,
                "waves": 1,
            },
        )
    )
    return events


def test_frozen_degraded_run_projects_and_renders_the_reason() -> None:
    state = RunProjection.from_events(_frozen_stream()).state
    assert state.status == "completed"
    assert state.outcome == "degraded"
    assert state.winner_agent == ""
    assert state.terminal_reason_code == REASON_NO_TRAVERSABLE_IDENTIFIERS
    assert state.terminal_reason == REASON
    assert len(state.agents) == 3
    assert len(state.warnings) == 1
    assert state.fetches_used == 3
    assert state.elapsed_seconds == 23.0

    timeline = render.render_timeline(state, state.selected_entry_id)
    assert "traversable" in timeline
    assert "A01" in render.render_tabs(state, 0)
    event_view = render.render_events(state, outcome="warning")
    assert "traversable" in event_view
    assert "no_winner" not in event_view
    metadata = render.render_metadata(state)
    assert "outcome: degraded" in metadata
    assert "terminal_reason_code: no_traversable_identifiers" in metadata
    assert "traversable" in render.render_footer(state)


def test_frozen_degraded_run_report_is_preserved(tmp_path) -> None:
    graph = GraphStore(tmp_path / "graph.db")
    colony = Colony.__new__(Colony)
    colony.graph = graph
    colony.agents = []
    colony.shared_visited = set()
    colony.shared_frontier = SharedFrontier()
    colony._best_agent = None
    colony._best_quality = 0.0
    colony._best_narrative = ""
    colony._best_snapshot_agent = ""
    colony._best_snapshot_oleada = 0
    scheduler = SimpleNamespace(history=[], oleada_count=1, total_fetches=3)
    convergence = SimpleNamespace(state=SimpleNamespace(quality_history=[]))

    report = build_report(
        config=Config(),
        colony=colony,
        scheduler=scheduler,
        convergence=convergence,
        graph=graph,
        seed_paper_id=SEED,
        seed_query="recent advances",
        elapsed=23.0,
        outcome="degraded",
        terminal_reason=REASON,
    )

    assert f"`{SEED}`" in report
    assert "**Outcome:** degraded" in report
    assert REASON in report
    assert "No winning narrative was produced" in report
    assert "**Elapsed:** 23.0s" in report
    assert "**Total fetches:** 3" in report
    graph.close()
