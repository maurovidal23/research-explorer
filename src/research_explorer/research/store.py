"""Append-only research event store with transactional state and reconstruction.

Every material transition appends a ``ResearchEvent`` and, in the same
transaction, persists the logical ``ResearchState`` that resulted from it. A
snapshot row is written at run initialization and after each evaluated turn.
``reconstruct(run_id, sequence)`` is a pure projection over stored state: it
never calls providers or LLMs and is deterministic across repeated calls.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from research_explorer.logging_setup import get_logger
from research_explorer.redaction import redact_obj, redact_secrets
from research_explorer.research.models import (
    SCHEMA_VERSION,
    AgentNotebook,
    ResearchEvent,
    ResearchObjective,
    ResearchState,
    canonical_json,
    state_hash,
    utc_now,
)

log = get_logger("research.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY,
    seed_paper_id TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT DEFAULT 'running',
    terminal_reason TEXT,
    objective_json TEXT NOT NULL,
    config_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    ts TEXT NOT NULL,
    actor TEXT DEFAULT 'controller',
    wave INTEGER,
    turn INTEGER,
    payload_json TEXT NOT NULL,
    state_json TEXT,
    prev_state_hash TEXT,
    state_hash TEXT,
    schema_version TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_research_events_run_seq
    ON research_events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_research_events_run ON research_events(run_id);

CREATE TABLE IF NOT EXISTS research_snapshots (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS research_artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_artifacts_run ON research_artifacts(run_id);
"""


class ResearchStore:
    """SQLite-backed append-only store for the research kernel."""

    def __init__(self, db_path: str | Path = "data/research.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---- Runs ------------------------------------------------------------

    def create_run(
        self,
        objective: ResearchObjective,
        config_json: str | None = None,
        initial_state: ResearchState | None = None,
    ) -> ResearchState:
        """Create the run and persist its initial state + snapshot atomically."""
        state = initial_state or ResearchState(
            objective=objective,
            notebook=self._initial_notebook(objective),
        )
        if state.objective.run_id != objective.run_id:
            state = state.model_copy(update={"objective": objective})
        now = utc_now()
        with self._conn:
            self._conn.execute(
                """INSERT INTO research_runs
                   (run_id, seed_paper_id, question, status, objective_json,
                    config_json, created_at, updated_at)
                   VALUES (?, ?, ?, 'running', ?, ?, ?, ?)""",
                (
                    objective.run_id,
                    objective.seed_paper_id,
                    objective.question,
                    canonical_json(objective),
                    redact_secrets(config_json) if config_json else None,
                    now,
                    now,
                ),
            )
            seq = self._next_seq(self._conn, objective.run_id)
            self._insert_event(
                self._conn,
                run_id=objective.run_id,
                seq=seq,
                type="run_start",
                actor="controller",
                payload={"seed": objective.seed_paper_id, "question": objective.question},
                state=state,
            )
            self._insert_snapshot(self._conn, objective.run_id, seq, state)
        return state

    @staticmethod
    def _initial_notebook(objective: ResearchObjective) -> AgentNotebook:
        return AgentNotebook(
            agent_id="agent-1",
            role="explorer",
            thesis=objective.question,
        )

    def finish_run(self, run_id: str, status: str, terminal_reason: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE research_runs SET status = ?, terminal_reason = ?, updated_at = ? "
                "WHERE run_id = ?",
                (status, terminal_reason, utc_now(), run_id),
            )

    def get_run(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM research_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    # ---- Events ----------------------------------------------------------

    def append_event(
        self,
        run_id: str,
        type: str,
        *,
        state: ResearchState | None = None,
        payload: dict | None = None,
        actor: str = "controller",
        wave: int | None = None,
        turn: int | None = None,
        input_refs: list[str] | None = None,
        output_refs: list[str] | None = None,
        tokens: int | None = None,
        fetches: int | None = None,
        seconds: float | None = None,
    ) -> ResearchEvent:
        """Append an event and, when given, persist the resulting state atomically."""
        with self._conn:
            seq = self._next_seq(self._conn, run_id)
            return self._insert_event(
                self._conn,
                run_id=run_id,
                seq=seq,
                type=type,
                actor=actor,
                payload=payload,
                state=state,
                wave=wave,
                turn=turn,
                input_refs=input_refs,
                output_refs=output_refs,
                tokens=tokens,
                fetches=fetches,
                seconds=seconds,
            )

    def snapshot(self, run_id: str, seq: int, state: ResearchState) -> None:
        """Persist a named snapshot row for the given sequence."""
        with self._conn:
            self._insert_snapshot(self._conn, run_id, seq, state)

    def list_events(self, run_id: str, until_seq: int | None = None) -> list[ResearchEvent]:
        query = "SELECT * FROM research_events WHERE run_id = ?"
        params: list[object] = [run_id]
        if until_seq is not None:
            query += " AND seq <= ?"
            params.append(until_seq)
        query += " ORDER BY seq"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    def reconstruct(self, run_id: str, sequence: int | None = None) -> ResearchState:
        """Return the logical state as of ``sequence`` (latest when None).

        Reads only from stored state — no provider or LLM calls — and is
        deterministic, so reconstructing the same range twice is equal.
        """
        query = (
            "SELECT state_json FROM research_events "
            "WHERE run_id = ? AND state_json IS NOT NULL"
        )
        params: list[object] = [run_id]
        if sequence is not None:
            query += " AND seq <= ?"
            params.append(sequence)
        query += " ORDER BY seq DESC LIMIT 1"
        row = self._conn.execute(query, params).fetchone()
        if row is not None:
            return ResearchState.model_validate(json.loads(row["state_json"]))
        snap_query = (
            "SELECT state_json FROM research_snapshots WHERE run_id = ?"
        )
        snap_params: list[object] = [run_id]
        if sequence is not None:
            snap_query += " AND seq <= ?"
            snap_params.append(sequence)
        snap_query += " ORDER BY seq DESC LIMIT 1"
        snap = self._conn.execute(snap_query, snap_params).fetchone()
        if snap is None:
            raise KeyError(f"no state stored for run {run_id!r}")
        return ResearchState.model_validate(json.loads(snap["state_json"]))

    def latest_evaluation(self, run_id: str, sequence: int | None = None):
        return self.reconstruct(run_id, sequence).latest_evaluation

    def list_snapshots(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT seq, state_json, state_hash FROM research_snapshots "
            "WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ---- Artifacts -------------------------------------------------------

    def save_artifact(self, run_id: str, name: str, kind: str, content: str) -> str:
        artifact_id = uuid.uuid4().hex[:12]
        with self._conn:
            self._conn.execute(
                """INSERT INTO research_artifacts
                   (artifact_id, run_id, name, kind, content, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (artifact_id, run_id, name, kind, redact_secrets(content), utc_now()),
            )
        return artifact_id

    def get_artifact(self, artifact_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM research_artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def latest_artifact(self, run_id: str, kind: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM research_artifacts WHERE run_id = ? AND kind = ? "
            "ORDER BY created_at DESC, artifact_id DESC LIMIT 1",
            (run_id, kind),
        ).fetchone()
        return dict(row) if row is not None else None

    # ---- Internal helpers ------------------------------------------------

    @staticmethod
    def _next_seq(conn: sqlite3.Connection, run_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM research_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return int(row[0]) + 1

    @staticmethod
    def _last_hash(conn: sqlite3.Connection, run_id: str) -> str | None:
        row = conn.execute(
            "SELECT state_hash FROM research_events WHERE run_id = ? "
            "ORDER BY seq DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return row["state_hash"] if row is not None else None

    def _insert_event(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        seq: int,
        type: str,
        actor: str,
        payload: dict | None,
        state: ResearchState | None,
        wave: int | None = None,
        turn: int | None = None,
        input_refs: list[str] | None = None,
        output_refs: list[str] | None = None,
        tokens: int | None = None,
        fetches: int | None = None,
        seconds: float | None = None,
    ) -> ResearchEvent:
        prev_hash = self._last_hash(conn, run_id)
        state_json = canonical_json(state) if state is not None else None
        current_hash = state_hash(state) if state is not None else prev_hash
        event = ResearchEvent(
            run_id=run_id,
            seq=seq,
            type=type,
            actor=actor,
            wave=wave,
            turn=turn,
            input_refs=input_refs or [],
            output_refs=output_refs or [],
            payload=redact_obj(payload or {}),
            tokens=tokens,
            fetches=fetches,
            seconds=seconds,
            previous_state_hash=prev_hash,
            current_state_hash=current_hash,
            schema_version=SCHEMA_VERSION,
        )
        conn.execute(
            """INSERT INTO research_events
               (run_id, seq, type, ts, actor, wave, turn, payload_json, state_json,
                prev_state_hash, state_hash, schema_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                seq,
                type,
                event.timestamp,
                actor,
                wave,
                turn,
                canonical_json(event.payload),
                state_json,
                prev_hash,
                current_hash,
                SCHEMA_VERSION,
            ),
        )
        conn.execute(
            "UPDATE research_runs SET updated_at = ? WHERE run_id = ?",
            (utc_now(), run_id),
        )
        return event

    @staticmethod
    def _insert_snapshot(
        conn: sqlite3.Connection, run_id: str, seq: int, state: ResearchState
    ) -> None:
        state_json = canonical_json(state)
        conn.execute(
            """INSERT OR REPLACE INTO research_snapshots
               (run_id, seq, state_json, state_hash, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (run_id, seq, state_json, state_hash(state), utc_now()),
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> ResearchEvent:
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            payload = {}
        return ResearchEvent(
            run_id=row["run_id"],
            seq=row["seq"],
            type=row["type"],
            actor=row["actor"] or "controller",
            wave=row["wave"],
            turn=row["turn"],
            payload=payload,
            previous_state_hash=row["prev_state_hash"],
            current_state_hash=row["state_hash"],
            timestamp=row["ts"],
            schema_version=row["schema_version"],
        )
