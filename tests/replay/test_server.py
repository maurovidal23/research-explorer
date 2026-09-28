"""FastAPI replay server endpoint tests."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from research_explorer.replay.models import (
    DetailedEvaluation,
    PeerVoteDetail,
    PeerVotesDetail,
    SelfAssessmentDetail,
    StructuralComponentsDetail,
    VirginJudgeDetail,
)
from research_explorer.replay.server import build_app
from research_explorer.replay.trace import RunTraceStore


def _seed(db_path) -> str:
    store = RunTraceStore(db_path)
    run_id = store.create_run("10.1038/nrn3241", "neuroscience", config_json="{}")
    store.append_event(run_id, "orchestrator_start", {"seed": "x"})
    store.append_event(run_id, "oleada_start", {"oleada": 1})
    store.append_event(run_id, "oleada_complete", {"oleada": 1, "best_Q": 0.7})
    store.save_evaluation(
        run_id,
        DetailedEvaluation(
            agent_id="agent-000-abc",
            oleada=1,
            turn=1,
            q=0.55,
            self_assessment=SelfAssessmentDetail(score=0.6, reasoning="self r"),
            peers=PeerVotesDetail(
                votes=[PeerVoteDetail(voter_id="p", score=0.5, reasoning="peer r")],
                aggregated_score=0.5,
                num_votes=1,
            ),
            virgin_judge=VirginJudgeDetail(score=0.6, coverage="cov", gaps="gap"),
            structural=StructuralComponentsDetail(coverage=0.5, r=0.5),
        ),
    )
    store.save_artifact(run_id, "narrative_agent-000-abc.md", "narrative", "the narrative")
    store.finish_run(run_id, "completed", best_quality=0.7)
    return run_id


@pytest.fixture
async def client(tmp_path):
    db = tmp_path / "replay.db"
    run_id = _seed(db)
    app = build_app(db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, run_id


async def test_index_served(client):
    c, _ = client
    res = await c.get("/")
    assert res.status_code == 200
    assert "Research Explorer Replay" in res.text


async def test_static_assets_served(client):
    c, _ = client
    for path in ("/static/style.css", "/static/app.js"):
        res = await c.get(path)
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/")


async def test_list_runs(client):
    c, _ = client
    res = await c.get("/api/runs")
    assert res.status_code == 200
    runs = res.json()
    assert len(runs) == 1
    assert runs[0]["seed_paper_id"] == "10.1038/nrn3241"
    assert runs[0]["event_count"] == 3
    assert runs[0]["evaluation_count"] == 1


async def test_get_run(client):
    c, run_id = client
    res = await c.get(f"/api/runs/{run_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "completed"


async def test_run_endpoints_omit_local_heartbeat_identity(tmp_path):
    heartbeat_fields = {
        "process_id",
        "host_id",
        "heartbeat_at",
        "heartbeat_state",
        "interrupt_reason",
    }
    db = tmp_path / "replay.db"
    store = RunTraceStore(db)
    run_id = store.create_run("seed", "q")
    store.start_heartbeat(run_id, process_id=os.getpid(), host_id="some-host")
    store.close()

    app = build_app(db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        detail = (await c.get(f"/api/runs/{run_id}")).json()
        listing = (await c.get("/api/runs")).json()[0]

    for payload in (detail, listing):
        assert payload["status"] == "running"
        assert heartbeat_fields.isdisjoint(payload.keys())


async def test_events_ordered(client):
    c, run_id = client
    res = await c.get(f"/api/runs/{run_id}/events")
    assert res.status_code == 200
    events = res.json()
    assert [e["type"] for e in events] == [
        "orchestrator_start",
        "oleada_start",
        "oleada_complete",
    ]
    assert [e["seq"] for e in events] == [1, 2, 3]


async def test_evaluations_include_rationales(client):
    c, run_id = client
    res = await c.get(f"/api/runs/{run_id}/evaluations")
    assert res.status_code == 200
    recs = res.json()
    assert len(recs) == 1
    rec = recs[0]
    assert rec["self_assessment"]["reasoning"] == "self r"
    assert rec["peers"]["votes"][0]["reasoning"] == "peer r"
    assert rec["virgin_judge"]["coverage"] == "cov"
    assert rec["virgin_judge"]["gaps"] == "gap"
    assert rec["structural"]["coverage"] == 0.5


async def test_artifacts_endpoints(client):
    c, run_id = client
    res = await c.get(f"/api/runs/{run_id}/artifacts")
    artifacts = res.json()
    assert len(artifacts) == 1
    artifact_id = artifacts[0]["artifact_id"]
    res = await c.get(f"/api/artifacts/{artifact_id}")
    assert res.status_code == 200
    assert res.json()["content"] == "the narrative"


async def test_unknown_run_404(client):
    c, _ = client
    res = await c.get("/api/runs/missing/events")
    assert res.status_code == 404
    res = await c.get("/api/runs/missing")
    assert res.status_code == 404


async def test_config_endpoint_no_default(client):
    c, _ = client
    res = await c.get("/api/config")
    assert res.status_code == 200
    assert res.json() == {"default_run_id": None}


async def test_config_endpoint_with_default(tmp_path):
    db = tmp_path / "replay.db"
    run_id = _seed(db)
    app = build_app(db, default_run_id=run_id)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        res = await c.get("/api/config")
        assert res.status_code == 200
        assert res.json() == {"default_run_id": run_id}


async def test_lifespan_closes_store(tmp_path):
    db = tmp_path / "replay.db"
    app = build_app(db)

    assert app.router.lifespan_context is not None

    async with (
        httpx.ASGITransport(app=app) as transport,
        httpx.AsyncClient(transport=transport, base_url="http://test") as c,
    ):
        res = await c.get("/api/runs")
        assert res.status_code == 200

    store = RunTraceStore(db)
    store.close()


def test_cli_replay_serve_validates_run_id(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from research_explorer.cli import app

    runner = CliRunner()
    db = tmp_path / "replay.db"

    result = runner.invoke(app, ["replay", "serve", "--db", str(db), "--run-id", "nonexistent"])
    assert result.exit_code == 1
    assert "Run not found" in result.output

    store = RunTraceStore(db)
    run_id = store.create_run("seed", "q")
    store.close()

    calls = []

    def fake_run(app, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = runner.invoke(app, ["replay", "serve", "--db", str(db), "--run-id", run_id])
    assert result.exit_code == 0
    assert f"run: {run_id}" in result.output
    assert len(calls) == 1


def test_cli_replay_serve_passes_run_id_to_build_app(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import research_explorer.replay.server as srv_mod
    from research_explorer.cli import app

    runner = CliRunner()
    db = tmp_path / "replay.db"
    store = RunTraceStore(db)
    run_id = store.create_run("seed", "q")
    store.close()

    build_calls: list[dict] = []
    real_build = srv_mod.build_app

    def capturing_build(*args, **kwargs):
        build_calls.append(kwargs)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(srv_mod, "build_app", capturing_build)
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: None)
    result = runner.invoke(app, ["replay", "serve", "--db", str(db), "--run-id", run_id])
    assert result.exit_code == 0
    assert len(build_calls) == 1
    assert build_calls[0].get("default_run_id") == run_id


def test_app_js_init_app_is_top_level():
    js_path = Path(__file__).resolve().parents[2] / "src" / "research_explorer" / "replay" / "static" / "app.js"
    text = js_path.read_text(encoding="utf-8")
    assert "async function initApp()" in text
    lines = text.splitlines()
    def_line = next(i for i, line in enumerate(lines) if "async function initApp()" in line)
    indent = len(lines[def_line]) - len(lines[def_line].lstrip())
    assert indent == 0, f"initApp must be top-level, found indent={indent}"
    back_btn_line = next(i for i, line in enumerate(lines) if "#back-btn" in line)
    assert def_line > back_btn_line, "initApp must appear after the back-btn listener"
