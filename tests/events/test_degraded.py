"""Degraded ``no_winner`` projection and live/replay equivalence (TUI-REL-5/6)."""

from __future__ import annotations

from research_explorer.events.models import (
    OUTCOME_COMPLETED,
    OUTCOME_DEGRADED,
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


def _start() -> RunEvent:
    return RunEvent(
        seq=1,
        type="orchestrator_start",
        payload={"run_id": "run-degraded", "seed": "arxiv:1", "query": "q", "colony_size": 3},
    )


def _new_degraded_stream() -> list[RunEvent]:
    return [
        _start(),
        RunEvent(
            seq=2,
            type="colony_initialized",
            payload={"size": 3, "agents": ["a0", "a1", "a2"]},
        ),
        RunEvent(
            seq=3,
            type="warning",
            payload={
                "classification": "warning",
                "outcome": "degraded",
                "reason_code": REASON_NO_TRAVERSABLE_IDENTIFIERS,
                "reason": "discovered neighbors carried no traversable DOI or arXiv identifier",
            },
        ),
        RunEvent(
            seq=4,
            type="no_winner",
            payload={
                "run_id": "run-degraded",
                "status": "completed",
                "outcome": "degraded",
                "reason_code": REASON_NO_TRAVERSABLE_IDENTIFIERS,
                "reason": "discovered neighbors carried no traversable DOI or arXiv identifier",
                "elapsed": 23.0,
                "total_fetches": 4,
                "total_waves": 1,
            },
        ),
    ]


def test_new_degraded_trace_is_completed_with_one_warning() -> None:
    state = RunProjection.from_events(_new_degraded_stream()).state
    assert state.status == "completed"
    assert state.outcome == OUTCOME_DEGRADED
    assert state.winner_agent == ""
    assert len(state.warnings) == 1
    assert state.reason_code == REASON_NO_TRAVERSABLE_IDENTIFIERS
    assert "no traversable" in state.terminal_reason
    assert state.total_waves == 1
    assert state.elapsed_seconds == 23.0
    assert all(a.is_winner is False for a in state.agents.values())
    assert all(a.status == "completed" for a in state.agents.values())


def test_old_unenriched_no_winner_gets_one_generic_warning() -> None:
    state = RunProjection.from_events(
        [
            _start(),
            RunEvent(seq=2, type="no_winner", payload={"run_id": "legacy", "elapsed": 5.0}),
        ]
    ).state
    assert state.status == "completed"
    assert state.outcome == OUTCOME_DEGRADED
    assert state.winner_agent == ""
    assert len(state.warnings) == 1
    assert "without a winning narrative" in state.terminal_reason
    assert state.warnings[0] == state.terminal_reason
    # A single warning node is also present in the timeline.
    warnings = [e for e in state.timeline if e.kind == "warning"]
    assert len(warnings) == 1


def test_completed_run_sets_completed_outcome() -> None:
    state = RunProjection.from_events(
        [
            _start(),
            RunEvent(
                seq=2,
                type="orchestrator_complete",
                payload={"status": "completed", "winner": "a0", "total_waves": 2},
            ),
        ]
    ).state
    assert state.outcome == OUTCOME_COMPLETED
    assert state.terminal_reason == ""
    assert state.total_waves == 2


def _drive_degraded(tracer: RunTracer, run_id: str) -> None:
    tracer.emit(
        "orchestrator_start",
        run_id=run_id,
        seed="arxiv:1",
        query="q",
        colony_size=3,
    )
    tracer.emit("colony_initialized", size=3, agents=["a0", "a1", "a2"])
    tracer.emit(
        "warning",
        classification="warning",
        outcome="degraded",
        reason_code=REASON_NO_TRAVERSABLE_IDENTIFIERS,
        reason="discovered neighbors carried no traversable DOI or arXiv identifier",
    )
    tracer.emit(
        "no_winner",
        run_id=run_id,
        status="completed",
        outcome="degraded",
        reason_code=REASON_NO_TRAVERSABLE_IDENTIFIERS,
        reason="discovered neighbors carried no traversable DOI or arXiv identifier",
        elapsed=23.0,
        total_fetches=4,
        total_waves=1,
    )


def test_live_and_replay_degraded_projections_are_equivalent(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("arxiv:1", "q")
    live_events: list[RunEvent] = []
    tracer = RunTracer(store, run_id, sink=CallbackSink(live_events.append))
    _drive_degraded(tracer, run_id)

    live = RunProjection.from_events(live_events)
    replayed = RunProjection.from_events(
        [
            RunEvent(seq=e["seq"], type=e["type"], payload=e["payload"], ts=e["ts"])
            for e in store.list_events(run_id)
        ]
    )
    assert _stable_state(live.state) == _stable_state(replayed.state)
    assert replayed.state.status == "completed"
    assert replayed.state.outcome == OUTCOME_DEGRADED
    assert replayed.state.winner_agent == ""
    assert len(replayed.state.warnings) == 1
    assert render.render_metadata(replayed.state).count("degraded") >= 1
    store.close()
