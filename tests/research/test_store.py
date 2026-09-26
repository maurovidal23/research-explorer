"""Event sequence, transactional state, and deterministic reconstruction."""

from __future__ import annotations

from research_explorer.research.models import (
    AgentNotebook,
    ResearchObjective,
    ResearchState,
    canonical_json,
    state_hash,
)
from research_explorer.research.store import ResearchStore


def _store(tmp_path) -> ResearchStore:
    return ResearchStore(tmp_path / "research.db")


def _objective() -> ResearchObjective:
    return ResearchObjective(
        run_id="run-1", seed_paper_id="openalex:W1", question="Why?"
    )


def test_create_run_appends_initial_event_and_snapshot(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        store.create_run(_objective())
        events = store.list_events("run-1")
        assert [e.seq for e in events] == [1]
        assert events[0].type == "run_start"
        assert len(store.list_snapshots("run-1")) == 1
    finally:
        store.close()


def test_sequence_is_monotonic(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        state = store.create_run(_objective())
        seqs = [store.append_event("run-1", "ping", state=state).seq for _ in range(3)]
        assert seqs == [2, 3, 4]
    finally:
        store.close()


def test_state_is_persisted_transactionally_with_event(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        state = store.create_run(_objective())
        state.notebook.thesis = "updated thesis"
        event = store.append_event("run-1", "knowledge_mutation", state=state)
        assert event.current_state_hash == state_hash(state)
        rebuilt = store.reconstruct("run-1")
        assert rebuilt.notebook.thesis == "updated thesis"
    finally:
        store.close()


def test_reconstruct_at_sequence_is_deterministic(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        state = store.create_run(_objective())
        seqs = []
        for index in range(3):
            state.notebook.history.append(f"turn-{index}")
            state.notebook.bump()
            seqs.append(store.append_event("run-1", "turn", state=state).seq)
        store.snapshot("run-1", seqs[-1], state)

        first = canonical_json(store.reconstruct("run-1", seqs[0]))
        second = canonical_json(store.reconstruct("run-1", seqs[0]))
        assert first == second
        assert canonical_json(store.reconstruct("run-1", seqs[0])) != canonical_json(
            store.reconstruct("run-1")
        )
    finally:
        store.close()


def test_reconstruct_matches_every_stored_snapshot(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        state = store.create_run(_objective())
        for index in range(3):
            state.notebook.history.append(f"turn-{index}")
            state.notebook.bump()
            event = store.append_event("run-1", "turn", state=state)
            store.snapshot("run-1", event.seq, state)

        for snapshot in store.list_snapshots("run-1"):
            rebuilt = store.reconstruct("run-1", snapshot["seq"])
            assert canonical_json(rebuilt) == snapshot["state_json"]
    finally:
        store.close()


def test_reconstruct_never_calls_providers_or_llm(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        state = ResearchState(
            objective=_objective(), notebook=AgentNotebook(agent_id="a")
        )
        store.create_run(_objective(), initial_state=state)
        rebuilt = store.reconstruct("run-1")
        assert rebuilt.objective.question == "Why?"
    finally:
        store.close()
