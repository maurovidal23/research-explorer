"""Bounded live-projection windows and durable-trace completeness."""

from __future__ import annotations

from research_explorer.events.limits import LIVE_CANDIDATE_WINDOW, LIVE_EVENT_WINDOW
from research_explorer.events.models import EventType, RunEvent
from research_explorer.events.projection import RunProjection
from research_explorer.events.sink import ChannelSink
from research_explorer.replay.trace import RunTracer, RunTraceStore
from research_explorer.tui import text as render


def _candidate(seq: int) -> RunEvent:
    return RunEvent(
        seq=seq,
        type="candidate_score",
        payload={
            "paper_id": f"paper:{seq}",
            "agent_id": "a0",
            "eta": 0.5,
            "components": {"sim": 0.5},
        },
    )


def test_candidate_burst_is_bounded_with_exact_counters() -> None:
    projection = RunProjection()
    for seq in range(1, 10_001):
        projection.apply(_candidate(seq))
    state = projection.state
    assert len(state.candidate_scores) == LIVE_CANDIDATE_WINDOW
    assert state.candidate_scores_seen_total == 10_000
    assert state.candidate_scores_dropped == 10_000 - LIVE_CANDIDATE_WINDOW
    assert state.candidate_scores[-1].paper_id == "paper:10000"
    assert state.candidate_scores[0].paper_id == f"paper:{10_000 - LIVE_CANDIDATE_WINDOW + 1}"


def test_fifty_thousand_mixed_events_stay_bounded() -> None:
    projection = RunProjection()
    event_types = ("candidate_score", "paper_integration_completed", "status")
    for seq in range(1, 50_001):
        kind = event_types[seq % len(event_types)]
        if kind == "candidate_score":
            event = _candidate(seq)
        else:
            event = RunEvent(seq=seq, type=kind, payload={"agent_id": "a0"})
        projection.apply(event)
    state = projection.state
    assert len(state.events) == LIVE_EVENT_WINDOW
    assert state.events_seen_total == 50_000
    assert state.events_dropped == 50_000 - LIVE_EVENT_WINDOW
    assert len(state.candidate_scores) == LIVE_CANDIDATE_WINDOW
    assert state.candidate_scores_seen_total > 0
    assert state.events[-1].seq == 50_000


def test_smaller_windows_are_honored_and_deterministic() -> None:
    events = [_candidate(seq) for seq in range(1, 201)]
    first = RunProjection(event_window=10, candidate_window=5)
    second = RunProjection(event_window=10, candidate_window=5)
    first.apply_many(events)
    second.apply_many(events)
    assert len(first.state.events) == 10
    assert len(first.state.candidate_scores) == 5
    assert first.state.events_seen_total == second.state.events_seen_total == 200
    assert first.state.candidate_scores_dropped == second.state.candidate_scores_dropped == 195
    assert [e.seq for e in first.state.events] == [e.seq for e in second.state.events]


def test_terminal_event_survives_compaction() -> None:
    projection = RunProjection(event_window=8)
    projection.apply(RunEvent(seq=1, type=EventType.RUN_STARTED, payload={"run_id": "r"}))
    for seq in range(2, 100):
        projection.apply(_candidate(seq))
    projection.apply(
        RunEvent(seq=100, type=EventType.RUN_COMPLETED, payload={"run_id": "r", "status": "completed"})
    )
    state = projection.state
    assert state.status == "completed"
    assert state.events[-1].canonical_type() == EventType.RUN_COMPLETED


def test_durable_trace_complete_despite_window_and_channel_overflow(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "q")
    channel = ChannelSink(maxsize=64)
    tracer = RunTracer(store, run_id, sink=channel)
    projection = RunProjection(event_window=50, candidate_window=25)
    total = 2_000
    for i in range(total):
        tracer.emit("candidate_score", **_candidate(i).payload)
    while not channel.empty():
        projection.apply(channel.get_nowait())
    store.finish_run(run_id, "completed")
    durable = store.list_events(run_id)
    assert len(durable) == total
    assert [e["seq"] for e in durable] == list(range(1, total + 1))
    assert channel.dropped > 0
    delivered = projection.state.events_seen_total
    assert delivered <= 64
    assert delivered + channel.dropped == total
    assert len(projection.state.candidate_scores) <= 25
    store.close()


def test_events_tab_renders_one_bounded_page() -> None:
    projection = RunProjection()
    for seq in range(1, 10_001):
        projection.apply(_candidate(seq))
    state = projection.state
    body = render.render_events_tab(state, page=0)
    bullets = [line for line in body.splitlines() if line.startswith("- `[")]
    assert len(bullets) == 200
    assert "Showing 1-200 of 1000" in body
    assert "9000 evicted" in body
    last = render.render_events_tab(state, page=4)
    assert "Showing 801-1000 of 1000" in last
    clamped = render.render_events_tab(state, page=99)
    assert "Showing 801-1000 of 1000" in clamped


def test_event_filter_and_page_navigation_are_deterministic() -> None:
    projection = RunProjection()
    for seq in range(1, 2_001):
        projection.apply(_candidate(seq))
    state = projection.state
    first = render.render_events_tab(state, agent_id="a0", page=1)
    second = render.render_events_tab(state, agent_id="a0", page=1)
    assert first == second
    assert "Showing 201-400 of 1000" in first
    none = render.render_events_tab(state, agent_id="nobody")
    assert "No events recorded" in none
