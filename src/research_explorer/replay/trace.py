"""SQLite-backed run trace store for ordered events, evaluation results, and
narrative snapshots.

The store is append-only in practice: events are assigned a monotonically
increasing per-run sequence number on write, evaluation records are stored as
typed JSON, and narrative artifacts are snapshotted as content in the database.
Writing artifacts to disk is opt-in and names are sanitized to prevent path
traversal.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from research_explorer.events.models import RunEvent
from research_explorer.events.sink import EventSink
from research_explorer.logging_setup import get_logger
from research_explorer.redaction import redact_obj, redact_secrets
from research_explorer.replay.models import (
    CandidateScore,
    CandidateSelection,
    DetailedEvaluation,
)

log = get_logger("replay.trace")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    seed_paper_id TEXT,
    seed_query TEXT,
    started_at TEXT,
    completed_at TEXT,
    status TEXT DEFAULT 'running',
    config_json TEXT,
    best_quality REAL
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    oleada INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_run ON evaluation_results(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_run_agent ON evaluation_results(run_id, agent_id);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
"""

_UNSAFE_CHARS = re.compile(r"[/\\\x00-\x1f]")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(name: str) -> str:
    """Reduce an arbitrary name to a safe single-path filename.

    Rejects names containing a '..' path component (traversal) or ending in an
    empty component. Directory separators and control characters are replaced
    so the result is always a single filename inside the target directory.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("artifact name must be a non-empty string")
    parts = name.replace("\\", "/").split("/")
    if ".." in parts:
        raise ValueError(f"unsafe artifact filename: {name!r}")
    last = parts[-1]
    if not last:
        raise ValueError(f"unsafe artifact filename: {name!r}")
    cleaned = _UNSAFE_CHARS.sub("_", last)
    if not cleaned:
        raise ValueError(f"unsafe artifact filename: {name!r}")
    return cleaned


def safe_artifact_path(out_dir: str | Path, name: str) -> Path:
    """Resolve an artifact name inside out_dir, refusing path traversal."""
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    target = (base / safe_filename(name)).resolve()
    root = base.resolve()
    if target.parent != root:
        raise ValueError(f"artifact path escapes out_dir: {name!r}")
    return target


class RunTraceStore:
    """SQLite store for runs, ordered events, evaluation results, and artifacts."""

    def __init__(self, db_path: str | Path = "data/replay.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def create_run(
        self,
        seed_paper_id: str,
        seed_query: str,
        config_json: str | None = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            """INSERT INTO runs (run_id, seed_paper_id, seed_query, started_at, status, config_json)
               VALUES (?, ?, ?, ?, 'running', ?)""",
            (run_id, seed_paper_id, seed_query, utc_now(), config_json),
        )
        self._conn.commit()
        return run_id

    def finish_run(self, run_id: str, status: str = "completed", best_quality: float | None = None) -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, completed_at = ?, best_quality = COALESCE(?, best_quality) WHERE run_id = ?",
            (status, utc_now(), best_quality, run_id),
        )
        self._conn.commit()

    def list_runs(self) -> list[dict]:
        rows = self._conn.execute(
            """SELECT r.run_id, r.seed_paper_id, r.seed_query, r.started_at, r.completed_at,
                      r.status, r.best_quality,
                      (SELECT COUNT(*) FROM events e WHERE e.run_id = r.run_id) AS event_count,
                      (SELECT COUNT(*) FROM evaluation_results er WHERE er.run_id = r.run_id) AS evaluation_count
               FROM runs r ORDER BY r.started_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            """SELECT r.*,
                      (SELECT COUNT(*) FROM events e WHERE e.run_id = r.run_id) AS event_count,
                      (SELECT COUNT(*) FROM evaluation_results er WHERE er.run_id = r.run_id) AS evaluation_count
               FROM runs r WHERE r.run_id = ?""",
            (run_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def append_event(self, run_id: str, type: str, payload: dict | None = None) -> int:
        """Append an event with the next per-run sequence number."""
        payload = redact_obj(payload or {})
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM events WHERE run_id = ?", (run_id,)
        ).fetchone()
        seq = int(row[0]) + 1
        self._conn.execute(
            "INSERT INTO events (run_id, seq, ts, type, payload) VALUES (?, ?, ?, ?, ?)",
            (run_id, seq, utc_now(), type, json.dumps(payload, default=str)),
        )
        self._conn.commit()
        return seq

    def list_events(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT event_id, seq, ts, type, payload FROM events WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
        events = []
        for r in rows:
            try:
                payload = json.loads(r["payload"])
            except json.JSONDecodeError:
                payload = {}
            events.append(
                {
                    "event_id": r["event_id"],
                    "seq": r["seq"],
                    "ts": r["ts"],
                    "type": r["type"],
                    "payload": payload,
                }
            )
        return events

    def save_evaluation(self, run_id: str, detail: DetailedEvaluation) -> int:
        self._conn.execute(
            """INSERT INTO evaluation_results (run_id, oleada, agent_id, ts, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (
                run_id,
                detail.oleada,
                detail.agent_id,
                detail.timestamp,
                detail.model_dump_json(),
            ),
        )
        self._conn.commit()
        return self._conn.execute("SELECT LAST_INSERT_ROWID()").fetchone()[0]

    def list_evaluations(self, run_id: str) -> list[DetailedEvaluation]:
        rows = self._conn.execute(
            "SELECT payload FROM evaluation_results WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        results = []
        for r in rows:
            try:
                results.append(DetailedEvaluation.model_validate_json(r["payload"]))
            except Exception as e:
                log.warning("invalid_evaluation_record", error=str(e))
        return results

    def evaluations_summary(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT payload FROM evaluation_results WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        summary = []
        for r in rows:
            try:
                d = json.loads(r["payload"])
            except json.JSONDecodeError:
                continue
            summary.append(
                {
                    "agent_id": d.get("agent_id"),
                    "oleada": d.get("oleada"),
                    "turn": d.get("turn"),
                    "q": d.get("q"),
                    "delta_q": d.get("delta_q"),
                    "S": d.get("self_assessment", {}).get("score"),
                    "P": d.get("peers", {}).get("aggregated_score"),
                    "J": d.get("virgin_judge", {}).get("score"),
                    "R": d.get("structural", {}).get("r"),
                }
            )
        return summary

    def save_artifact(self, run_id: str, name: str, kind: str, content: str) -> str:
        artifact_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            """INSERT INTO artifacts (artifact_id, run_id, name, kind, content, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                artifact_id,
                run_id,
                safe_filename(name),
                kind,
                redact_secrets(content),
                utc_now(),
            ),
        )
        self._conn.commit()
        return artifact_id

    def list_artifacts(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT artifact_id, name, kind, created_at FROM artifacts WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_artifact(self, artifact_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def get_latest_artifact(self, run_id: str, kind: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE run_id = ? AND kind = ? ORDER BY created_at DESC, artifact_id DESC LIMIT 1",
            (run_id, kind),
        ).fetchone()
        return dict(row) if row is not None else None

    def export_artifact(self, artifact_id: str, out_dir: str | Path) -> Path:
        artifact = self.get_artifact(artifact_id)
        if artifact is None:
            raise KeyError(f"artifact not found: {artifact_id}")
        target = safe_artifact_path(out_dir, artifact["name"])
        target.write_text(artifact["content"], encoding="utf-8")
        return target

    def export_narrative(self, run_id: str, out_dir: str | Path, filename: str | None = None) -> Path:
        artifact = self.get_latest_artifact(run_id, "narrative")
        if artifact is None:
            raise KeyError(f"no narrative artifact for run: {run_id}")
        name = filename or artifact["name"]
        target = safe_artifact_path(out_dir, name)
        target.write_text(artifact["content"], encoding="utf-8")
        return target


class RunTracer:
    """Thin writer facade that scopes all trace writes to a single run.

    It also keeps the event stream lean: rationales live in the evaluation
    records; event payloads carry only compact fields.

    When an :class:`~research_explorer.events.sink.EventSink` is attached, every
    durable event is republished as a normalized
    :class:`~research_explorer.events.models.RunEvent` carrying the durable
    per-run sequence number, so the live TUI and replay share one ordering.
    """

    def __init__(self, store: RunTraceStore, run_id: str, sink: EventSink | None = None):
        self.store = store
        self.run_id = run_id
        self._sink = sink
        self.durability_failed = False

    def emit(self, type: str, **payload) -> int:
        try:
            seq = self.store.append_event(self.run_id, type, payload)
        except Exception as exc:
            self.durability_failed = True
            self._publish(0, "trace_write_failed", {"error": str(exc)})
            raise
        self._publish(seq, type, payload)
        return seq

    def _publish(self, seq: int, type: str, payload: dict) -> None:
        if self._sink is None:
            return
        try:
            normalized = json.loads(json.dumps(redact_obj(payload), default=str))
        except (TypeError, ValueError):
            normalized = {}
        self._sink.publish(RunEvent(seq=seq, type=type, payload=normalized, ts=utc_now()))

    def record_evaluation(self, detail: DetailedEvaluation) -> None:
        self.store.save_evaluation(self.run_id, detail)
        self.emit(
            "evaluation_complete",
            agent_id=detail.agent_id,
            oleada=detail.oleada,
            Q=round(detail.q, 4),
            S=round(detail.self_assessment.score, 4),
            P=round(detail.peers.aggregated_score, 4),
            J=round(detail.virgin_judge.score, 4),
            R=round(detail.structural.r, 4),
            delta_q=round(detail.delta_q, 4),
            num_votes=detail.peers.num_votes,
        )
        try:
            full = json.loads(detail.model_dump_json())
        except (TypeError, ValueError):
            full = None
        if full is not None:
            self.emit("evaluation_detail", detail=full)

    def record_artifact(self, name: str, kind: str, content: str) -> str:
        artifact_id = self.store.save_artifact(self.run_id, name, kind, content)
        self.emit(
            "artifact_saved",
            artifact_id=artifact_id,
            name=name,
            kind=kind,
            content=redact_secrets(content),
        )
        return artifact_id

    def record_candidate_score(self, score: CandidateScore) -> int:
        return self.emit("candidate_score", **score.model_dump_payload())

    def record_candidate_selected(self, selection: CandidateSelection) -> int:
        return self.emit("candidate_selected", **selection.model_dump_payload())
