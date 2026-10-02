"""FastAPI replay server: run selector, room replay, and narrative view.

Plain REST endpoints (no SSE, no WebSockets). The frontend is static
HTML/CSS/JS served from the packaged static directory.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from research_explorer.replay.trace import RunTraceStore

STATIC_DIR = Path(__file__).parent / "static"

_HEARTBEAT_FIELDS = frozenset(
    {"process_id", "host_id", "heartbeat_at", "heartbeat_state", "interrupt_reason"}
)

_PRIVATE_ARTIFACT_KINDS = frozenset({"answer_key"})


def _public_run(run: dict) -> dict:
    """Strip local process/host heartbeat metadata from an HTTP run payload.

    These columns exist only to reconcile stale local runs. Exposing the host
    name, PID, and timestamps over the unauthenticated replay API leaks host
    identity without benefiting the UI, which reads only status and metrics.
    """
    return {key: value for key, value in run.items() if key not in _HEARTBEAT_FIELDS}


def build_app(
    db_path: str | Path = "data/replay.db",
    default_run_id: str | None = None,
) -> FastAPI:
    """Build the FastAPI app bound to the given replay database.

    FastAPI routes close over the store, so each app is wired to exactly one
    database file. Tests can pass a tmp_path.
    """
    store = RunTraceStore(db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        store.close()

    app = FastAPI(title="Research Explorer Replay", version="0.1.0", lifespan=lifespan)

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        return [_public_run(run) for run in store.list_runs()]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return _public_run(run)

    @app.get("/api/runs/{run_id}/events")
    def get_events(run_id: str) -> list[dict]:
        if store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return store.list_events(run_id)

    @app.get("/api/runs/{run_id}/evaluations")
    def get_evaluations(run_id: str) -> list[dict]:
        if store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        evaluations = store.list_evaluations(run_id)
        return [e.model_dump() for e in evaluations]

    @app.get("/api/runs/{run_id}/artifacts")
    def get_artifacts(run_id: str) -> list[dict]:
        if store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return [
            artifact
            for artifact in store.list_artifacts(run_id)
            if artifact.get("kind") not in _PRIVATE_ARTIFACT_KINDS
        ]

    @app.get("/api/artifacts/{artifact_id}")
    def get_artifact(artifact_id: str) -> dict:
        artifact = store.get_artifact(artifact_id)
        if artifact is None or artifact.get("kind") in _PRIVATE_ARTIFACT_KINDS:
            raise HTTPException(status_code=404, detail="artifact not found")
        return artifact

    @app.get("/api/config")
    def get_config() -> dict[str, str | None]:
        return {"default_run_id": default_run_id}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
