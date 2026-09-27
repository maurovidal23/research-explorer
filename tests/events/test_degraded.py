"""Degraded (no-winner) run projection and replay compatibility."""

from __future__ import annotations

from research_explorer.events.models import (
    GENERIC_NO_WINNER_REASON,
    REASON_NO_TRAVERSABLE_IDENTIFIERS,
    RunEvent,
)
from research_explorer.events.projection import RunProjection
from research_explorer.events.sink import CallbackSink
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
