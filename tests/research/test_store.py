"""Event sequence, transactional state, and deterministic reconstruction."""

from __future__ import annotations

import sqlite3

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


def test_event_costs_and_artifact_refs_round_trip(tmp_path) -> None:
    store = _store(tmp_path)
    try:
        state = store.create_run(_objective())
        store.append_event(
            "run-1",
            "evidence_acquired",
            state=state,
            turn=3,
            tokens=42,
            fetches=2,
            seconds=1.25,
            input_refs=["in-1"],
            output_refs=["openalex:W1"],
        )
        event = store.list_events("run-1")[-1]
        assert event.tokens == 42
        assert event.fetches == 2
        assert event.seconds == 1.25
        assert event.input_refs == ["in-1"]
        assert event.output_refs == ["openalex:W1"]
    finally:
        store.close()


def test_event_payload_and_artifact_redact_secrets(tmp_path) -> None:
    sentinel = "SENTINEL-SECRET-STORE"
    store = _store(tmp_path)
    try:
        store.create_run(_objective())
        store.append_event(
            "run-1",
            "provider_failure",
            payload={"url": f"https://api.x.test?api_key={sentinel}"},
        )
        event = store.list_events("run-1")[-1]
        assert sentinel not in str(event.payload)

        artifact_id = store.save_artifact(
            "run-1", "error.txt", "error", f"boom token={sentinel}"
        )
        artifact = store.get_artifact(artifact_id)
        assert artifact is not None
        assert sentinel not in artifact["content"]
    finally:
        store.close()


def test_existing_database_is_additively_migrated(tmp_path) -> None:
    db = tmp_path / "legacy.db"
    legacy = sqlite3.connect(str(db))
    legacy.executescript(
        """
        CREATE TABLE research_runs (
            run_id TEXT PRIMARY KEY, seed_paper_id TEXT NOT NULL,
            question TEXT NOT NULL, status TEXT DEFAULT 'running',
            terminal_reason TEXT, objective_json TEXT NOT NULL,
            config_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE research_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
            seq INTEGER NOT NULL, type TEXT NOT NULL, ts TEXT NOT NULL,
            actor TEXT DEFAULT 'controller', wave INTEGER, turn INTEGER,
            payload_json TEXT NOT NULL, state_json TEXT,
            prev_state_hash TEXT, state_hash TEXT, schema_version TEXT NOT NULL
        );
        CREATE UNIQUE INDEX idx_research_events_run_seq ON research_events(run_id, seq);
        """
    )
    legacy.execute(
        "INSERT INTO research_runs VALUES ('run-1', 'openalex:W1', 'Why?', "
        "'running', NULL, '{}', NULL, 't0', 't0')"
    )
    legacy.commit()
    legacy.close()

    store = ResearchStore(db)
    try:
        columns = {
            row["name"]
            for row in store._conn.execute("PRAGMA table_info(research_events)")
        }
        assert {"input_refs_json", "output_refs_json", "tokens", "fetches", "seconds"} <= columns
        store.append_event(
            "run-1", "run_start", tokens=1, fetches=1, output_refs=["artifact-1"]
        )
        event = store.list_events("run-1")[-1]
        assert event.tokens == 1
        assert event.fetches == 1
        assert event.output_refs == ["artifact-1"]
    finally:
        store.close()
