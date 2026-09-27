"""Ambient trace context enriches candidate/LLM events with agent turn fields."""

from __future__ import annotations

from research_explorer.events.models import RunEvent
from research_explorer.events.projection import RunProjection
from research_explorer.replay.models import CandidateScore
from research_explorer.replay.trace import RunTracer, RunTraceStore


def test_tracer_context_enriches_events_and_replays(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("s", "q")
    tracer = RunTracer(store, run_id)

    tracer.set_context(agent_id="a0", oleada=2, turn=3)
    tracer.record_candidate_score(
        CandidateScore(paper_id="p1", agent_id="a0", eta=0.5)
    )
    tracer.emit("llm_operation_started", purpose="paper_integration", model="explorer-x")
    tracer.emit("paper_fetch_started", paper_id="p1")
    tracer.clear_context()
    tracer.emit("llm_operation_completed", purpose="unattributed")

    payloads = {e["type"]: e["payload"] for e in store.list_events(run_id)}
    candidate = payloads["candidate_score"]
    assert candidate["oleada"] == 2
    assert candidate["turn"] == 3
    assert candidate["agent_id"] == "a0"
    llm = payloads["llm_operation_started"]
    assert llm["agent_id"] == "a0"
    assert llm["oleada"] == 2
    assert llm["turn"] == 3
    paper = payloads["paper_fetch_started"]
    assert paper["agent_id"] == "a0"
    assert "agent_id" not in payloads["llm_operation_completed"]

    persisted = [
        RunEvent(seq=e["seq"], type=e["type"], payload=e["payload"], ts=e["ts"])
        for e in store.list_events(run_id)
    ]
    state = RunProjection.from_events(persisted).state
    assert state.current_operation_agent == "a0"
    store.close()
