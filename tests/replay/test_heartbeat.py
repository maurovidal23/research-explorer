"""Durable heartbeat and stale-run reconciliation tests (TUI-MEM-5/6)."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from research_explorer.events.models import STATUS_INTERRUPTED, RunEvent
from research_explorer.events.projection import RunProjection
from research_explorer.replay.trace import RunTraceStore, local_host_id
from research_explorer.tui import text as render
from research_explorer.tui.replay import apply_reconciled_status


@pytest.fixture
def store(tmp_path) -> RunTraceStore:
    s = RunTraceStore(tmp_path / "replay.db")
    yield s
    s.close()


def test_active_heartbeat_remains_active(store: RunTraceStore) -> None:
    run_id = store.create_run("seed", "q")
    store.start_heartbeat(run_id, process_id=os.getpid(), host_id=local_host_id())
    assert store.reconcile_stale_runs() == []
    run = store.get_run(run_id)
    assert run["status"] == "running"
    assert run["heartbeat_state"] == "active"


def test_clean_completion_finalizes_heartbeat(store: RunTraceStore) -> None:
    run_id = store.create_run("seed", "q")
    store.start_heartbeat(run_id, process_id=os.getpid(), host_id=local_host_id())
    store.finish_run(run_id, "completed", best_quality=0.5)
    run = store.get_run(run_id)
    assert run["status"] == "completed"
    assert run["heartbeat_state"] == "finalized"
    assert run["heartbeat_at"] is None
    assert store.reconcile_stale_runs() == []


def test_stale_local_heartbeat_becomes_interrupted_exactly_once(store: RunTraceStore) -> None:
    run_id = store.create_run("seed", "q")
    store.append_event(run_id, "orchestrator_start", {"run_id": run_id})
    store.append_event(run_id, "candidate_score", {"paper_id": "p1", "agent_id": "a0"})
    store.start_heartbeat(run_id, process_id=999_999, host_id=local_host_id())

    first = store.reconcile_stale_runs(pid_alive=lambda pid: False)
    assert first == [run_id]
    run = store.get_run(run_id, reconcile=False)
    assert run["status"] == STATUS_INTERRUPTED
    assert run["interrupt_reason"]
    assert run["best_quality"] is None
    assert run["event_count"] == 2

    second = store.reconcile_stale_runs(pid_alive=lambda pid: False)
    assert second == []
    assert store.get_run(run_id, reconcile=False)["status"] == STATUS_INTERRUPTED


def test_stale_heartbeat_beyond_threshold_is_interrupted(store: RunTraceStore) -> None:
    run_id = store.create_run("seed", "q")
    store.start_heartbeat(run_id, process_id=os.getpid(), host_id=local_host_id())
    future = datetime.now(timezone.utc) + timedelta(seconds=10_000)
    assert store.reconcile_stale_runs(now=future, stale_after=120) == [run_id]
    assert store.get_run(run_id, reconcile=False)["status"] == STATUS_INTERRUPTED


def test_remote_heartbeat_is_not_falsely_relabeled(store: RunTraceStore) -> None:
    run_id = store.create_run("seed", "q")
    store.start_heartbeat(run_id, process_id=999_999, host_id="some-other-host")
    assert store.reconcile_stale_runs(local_host=local_host_id(), pid_alive=lambda pid: False) == []
    assert store.get_run(run_id, reconcile=False)["status"] == "running"


def test_unverifiable_legacy_run_is_not_relabeled(store: RunTraceStore) -> None:
    run_id = store.create_run("seed", "q")
    assert store.reconcile_stale_runs(pid_alive=lambda pid: False) == []
    assert store.get_run(run_id, reconcile=False)["status"] == "running"


def test_old_database_migrates_additively(tmp_path) -> None:
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            seed_paper_id TEXT,
            seed_query TEXT,
            started_at TEXT,
            completed_at TEXT,
            status TEXT DEFAULT 'running',
            config_json TEXT,
            best_quality REAL
        );
        CREATE TABLE events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            ts TEXT NOT NULL,
            type TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO runs (run_id, seed_paper_id, seed_query, started_at, status) VALUES (?,?,?,?,?)",
        ("legacy", "seed", "q", "2020-01-01T00:00:00+00:00", "completed"),
    )
    conn.execute(
        "INSERT INTO events (run_id, seq, ts, type, payload) VALUES (?,?,?,?,?)",
        ("legacy", 1, "2020-01-01T00:00:00+00:00", "orchestrator_complete", "{}"),
    )
    conn.commit()
    conn.close()

    store = RunTraceStore(db)
    run = store.get_run("legacy")
    assert run is not None
    assert run["status"] == "completed"
    assert run["host_id"] is None
    assert run["heartbeat_state"] is None
    assert [e["type"] for e in store.list_events("legacy")] == ["orchestrator_complete"]
    store.close()


def test_reconciled_interrupted_run_projects_without_fabricating_winner(store: RunTraceStore) -> None:
    run_id = store.create_run("seed", "q")
    store.append_event(run_id, "orchestrator_start", {"run_id": run_id})
    store.append_event(run_id, "candidate_score", {"paper_id": "p1", "agent_id": "a0"})
    store.start_heartbeat(run_id, process_id=999_999, host_id=local_host_id())
    store.reconcile_stale_runs(pid_alive=lambda pid: False)

    run = store.get_run(run_id, reconcile=False)
    projection = RunProjection.from_events(store.list_events(run_id))
    apply_reconciled_status(projection, run)
    state = projection.state
    assert state.status == STATUS_INTERRUPTED
    assert state.winner_agent == ""
    last = render.last_durable_event(state)
    assert last is not None and last.canonical_type() == "candidate_score"
    assert run["interrupt_reason"] in state.terminal_reason


def test_durable_trace_preserved_after_reconciliation(store: RunTraceStore) -> None:
    run_id = store.create_run("seed", "q")
    for i in range(1, 21):
        store.append_event(run_id, "candidate_score", {"paper_id": f"p{i}", "agent_id": "a0"})
    store.start_heartbeat(run_id, process_id=999_999, host_id=local_host_id())
    store.reconcile_stale_runs(pid_alive=lambda pid: False)
    events = store.list_events(run_id)
    assert len(events) == 20
    assert [e["seq"] for e in events] == list(range(1, 21))
    assert "candidate_score" in {e["type"] for e in events}


def test_run_event_supports_interrupted_type() -> None:
    projection = RunProjection.from_events(
        [
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r"}),
            RunEvent(seq=0, type="run_interrupted", payload={"reason": "killed"}),
        ]
    )
    assert projection.state.status == STATUS_INTERRUPTED
    assert projection.state.terminal_reason == "killed"
