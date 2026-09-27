"""CLI regression tests for TUI selection, rejection, and non-TUI behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from research_explorer.cli import app
from research_explorer.orchestrator.runner import Orchestrator
from research_explorer.replay.trace import RunTracer, RunTraceStore

runner = CliRunner()


def _config(tmp_path: Path, body: str = 'pipeline = "aco"\n') -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


class _FakeOrch:
    events: ClassVar[list[str]] = []

    def __init__(self, config, event_sink=None):
        self.config = config
        self.event_sink = event_sink
        _FakeOrch.events.append("init")

    async def run(self, seed_paper_id, seed_query):
        _FakeOrch.events.append("run")
        return "narrative"

    def generate_report(self, seed_paper_id, seed_query):
        return "FAKE REPORT BODY"

    def generate_obsidian(self, seed_query, output_dir="obsidian"):
        return None

    async def aclose(self):
        _FakeOrch.events.append("close")


def test_tui_rejects_unsupported_pipeline_before_resources(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, 'pipeline = "research-kernel"\n[providers]\nactive = []\n')

    def _explode(*args, **kwargs):
        raise AssertionError("no pipeline runner may be constructed for rejected --tui")

    monkeypatch.setattr("research_explorer.cli._run_research_kernel", _explode)
    monkeypatch.setattr(
        "research_explorer.orchestrator.runner.Orchestrator", _explode
    )

    result = runner.invoke(
        app,
        ["explore", "10.1/x", "q", "--tui", "--config", str(config)],
    )
    assert result.exit_code != 0
    combined = result.output + str(result.exception or "")
    assert "--tui is not supported" in combined


def test_tui_help_documents_the_flag() -> None:
    result = runner.invoke(app, ["explore", "--help"])
    assert result.exit_code == 0
    assert "--tui" in result.output


def test_tui_flag_selects_tui_path(tmp_path, monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_tui(cfg, seed, query, output):
        calls.append((seed, query, output))

    monkeypatch.setattr("research_explorer.cli._run_aco_tui", fake_tui)
    result = runner.invoke(
        app,
        ["explore", "10.1/x", "question", "--tui", "--config", str(_config(tmp_path))],
    )
    assert result.exit_code == 0, result.output
    assert calls == [("10.1/x", "question", None)]


def test_non_tui_behavior_unchanged(tmp_path, monkeypatch) -> None:
    _FakeOrch.events.clear()
    monkeypatch.setattr(
        "research_explorer.orchestrator.runner.Orchestrator", _FakeOrch
    )
    result = runner.invoke(
        app,
        ["explore", "10.1/x", "question", "--config", str(_config(tmp_path))],
    )
    assert result.exit_code == 0, result.output
    assert "FAKE REPORT BODY" in result.output
    assert "init" in _FakeOrch.events and "run" in _FakeOrch.events
    assert "close" in _FakeOrch.events


def test_explicit_no_tui_runs_non_tui(tmp_path, monkeypatch) -> None:
    _FakeOrch.events.clear()
    monkeypatch.setattr("research_explorer.orchestrator.runner.Orchestrator", _FakeOrch)

    def _explode(*args, **kwargs):
        raise AssertionError("--no-tui must not launch the TUI path")

    monkeypatch.setattr("research_explorer.cli._run_aco_tui", _explode)
    result = runner.invoke(
        app,
        ["explore", "10.1/x", "question", "--no-tui", "--config", str(_config(tmp_path))],
    )
    assert result.exit_code == 0, result.output
    assert "FAKE REPORT BODY" in result.output
    assert "run" in _FakeOrch.events


class _FakeApp:
    def __init__(self, projection, queue, runner, can_detach=False):
        self.projection = projection
        self.queue = queue
        self.runner = runner
        self.can_detach = can_detach
        self.cancelled = False
        self.state = SimpleNamespace(status="completed")
        self.run_result = None

    def run(self):
        self.run_result = ("TUI REPORT BODY", None)


def test_tui_output_still_writes_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("research_explorer.orchestrator.runner.Orchestrator", _FakeOrch)
    monkeypatch.setattr("research_explorer.tui.build_app", _FakeApp)
    out = tmp_path / "report.md"
    result = runner.invoke(
        app,
        [
            "explore",
            "10.1/x",
            "question",
            "--tui",
            "--output",
            str(out),
            "--config",
            str(_config(tmp_path)),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == "TUI REPORT BODY"


def test_tui_cancelled_run_is_reported(tmp_path, monkeypatch) -> None:
    class _CancelledApp(_FakeApp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cancelled = True
            self.state = SimpleNamespace(status="cancelled")

    monkeypatch.setattr("research_explorer.orchestrator.runner.Orchestrator", _FakeOrch)
    monkeypatch.setattr("research_explorer.tui.build_app", _CancelledApp)
    result = runner.invoke(
        app,
        ["explore", "10.1/x", "question", "--tui", "--config", str(_config(tmp_path))],
    )
    assert result.exit_code == 0, result.output
    assert "Run cancelled." in result.output


def test_mark_cancelled_persists_distinct_status(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "q")
    orch = Orchestrator.__new__(Orchestrator)
    orch.run_id = run_id
    orch.tracer = RunTracer(store, run_id)
    orch.trace = store
    orch.mark_cancelled()
    assert store.get_run(run_id)["status"] == "cancelled"
    types = [e["type"] for e in store.list_events(run_id)]
    assert "run_cancelled" in types
    store.close()


def test_mark_cancelled_never_relabels_completed_run(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "q")
    store.finish_run(run_id, "completed", best_quality=0.9)
    orch = Orchestrator.__new__(Orchestrator)
    orch.run_id = run_id
    orch.tracer = RunTracer(store, run_id)
    orch.trace = store
    orch.mark_cancelled()
    assert store.get_run(run_id)["status"] == "completed"
    assert "run_cancelled" not in [e["type"] for e in store.list_events(run_id)]
    store.close()


def test_tui_cancelled_run_does_not_write_report(tmp_path, monkeypatch) -> None:
    class _CancelledApp(_FakeApp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cancelled = True
            self.state = SimpleNamespace(status="cancelled")

    monkeypatch.setattr("research_explorer.orchestrator.runner.Orchestrator", _FakeOrch)
    monkeypatch.setattr("research_explorer.tui.build_app", _CancelledApp)
    out = tmp_path / "report.md"
    result = runner.invoke(
        app,
        [
            "explore",
            "10.1/x",
            "question",
            "--tui",
            "--output",
            str(out),
            "--config",
            str(_config(tmp_path)),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Run cancelled." in result.output
    assert not out.exists()


def test_tui_incomplete_run_exits_nonzero(tmp_path, monkeypatch) -> None:
    class _IncompleteApp(_FakeApp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.state = SimpleNamespace(status="failed")

        def run(self):
            self.run_result = None

    monkeypatch.setattr("research_explorer.orchestrator.runner.Orchestrator", _FakeOrch)
    monkeypatch.setattr("research_explorer.tui.build_app", _IncompleteApp)
    result = runner.invoke(
        app,
        ["explore", "10.1/x", "question", "--tui", "--config", str(_config(tmp_path))],
    )
    assert result.exit_code != 0
    combined = result.output + str(result.exception or "")
    assert "Run did not complete" in combined


class _RunningApp:
    """Test app that runs the captured runner synchronously, like Textual."""

    review_checks: ClassVar[list] = []

    def __init__(self, projection, queue, runner, can_detach=False):
        self.projection = projection
        self.queue = queue
        self.runner = runner
        self.can_detach = can_detach
        self.cancelled = False
        self.state = SimpleNamespace(status="completed")
        self.run_result = None

    def run(self):
        self.run_result = asyncio.run(self.runner())
        # The app stays open for review after the runner completes; the report
        # must already be durable at that point.
        for check in _RunningApp.review_checks:
            check()


def test_tui_output_is_durable_before_review_and_written_once(tmp_path, monkeypatch) -> None:
    import research_explorer.cli as cli_mod

    out = tmp_path / "report.md"
    writes: list[str] = []
    real_write = cli_mod._atomic_write_report

    def counting_write(path, report):
        writes.append(report)
        real_write(path, report)

    def assert_durable():
        assert out.exists()
        assert out.read_text(encoding="utf-8") == "FAKE REPORT BODY"

    _RunningApp.review_checks = [assert_durable]
    monkeypatch.setattr(cli_mod, "_atomic_write_report", counting_write)
    monkeypatch.setattr("research_explorer.orchestrator.runner.Orchestrator", _FakeOrch)
    monkeypatch.setattr("research_explorer.tui.build_app", _RunningApp)
    try:
        result = runner.invoke(
            app,
            [
                "explore",
                "10.1/x",
                "question",
                "--tui",
                "--output",
                str(out),
                "--config",
                str(_config(tmp_path)),
            ],
        )
    finally:
        _RunningApp.review_checks = []
    assert result.exit_code == 0, result.output
    assert writes == ["FAKE REPORT BODY"]
    assert out.read_text(encoding="utf-8") == "FAKE REPORT BODY"
    assert "Report written to" in result.output


def test_tui_atomic_write_failure_redacted_nonzero_and_run_completed(
    tmp_path, monkeypatch
) -> None:
    import research_explorer.cli as cli_mod

    db = tmp_path / "replay.db"

    class _PersistingOrch:
        def __init__(self, config, event_sink=None):
            self.event_sink = event_sink

        async def run(self, seed_paper_id, seed_query):
            store = RunTraceStore(db)
            run_id = store.create_run(seed_paper_id, seed_query)
            store.finish_run(run_id, "completed", best_quality=0.0)
            store.close()

        def generate_report(self, seed_paper_id, seed_query):
            return "REPORT BODY"

        def generate_obsidian(self, seed_query, output_dir="obsidian"):
            return None

        async def aclose(self):
            return None

    def boom(path, report):
        raise OSError("disk full api_key=sk-sentinel-zzz")

    _RunningApp.review_checks = []
    monkeypatch.setattr(cli_mod, "_atomic_write_report", boom)
    monkeypatch.setattr("research_explorer.orchestrator.runner.Orchestrator", _PersistingOrch)
    monkeypatch.setattr("research_explorer.tui.build_app", _RunningApp)
    out = tmp_path / "report.md"
    result = runner.invoke(
        app,
        [
            "explore",
            "10.1/x",
            "question",
            "--tui",
            "--output",
            str(out),
            "--config",
            str(_config(tmp_path)),
        ],
    )
    assert result.exit_code != 0
    combined = result.output + str(result.exception or "")
    assert "sk-sentinel-zzz" not in combined
    assert "failed to write report" in combined
    assert not out.exists()

    store = RunTraceStore(db)
    runs = store.list_runs()
    store.close()
    assert runs and runs[0]["status"] == "completed"


def test_tui_runner_cancels_run_and_closes_resources(tmp_path, monkeypatch) -> None:
    closed: list[str] = []

    class _CancellingOrch:
        def __init__(self, config, event_sink=None):
            self.event_sink = event_sink

        async def run(self, seed_paper_id, seed_query):
            raise asyncio.CancelledError()

        def mark_cancelled(self) -> None:
            closed.append("cancelled")

        def generate_report(self, seed_paper_id, seed_query) -> str:
            return ""

        def generate_obsidian(self, seed_query, output_dir="obsidian"):
            return None

        async def aclose(self) -> None:
            closed.append("closed")

    captured: dict = {}

    class _CapturingApp:
        def __init__(self, projection, queue, runner, can_detach=False):
            captured["runner"] = runner
            self.cancelled = False
            self.state = SimpleNamespace(status="running")
            self.run_result = None

        def run(self) -> None:
            return None

    monkeypatch.setattr("research_explorer.orchestrator.runner.Orchestrator", _CancellingOrch)
    monkeypatch.setattr("research_explorer.tui.build_app", _CapturingApp)
    runner.invoke(
        app,
        ["explore", "10.1/x", "question", "--tui", "--config", str(_config(tmp_path))],
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(captured["runner"]())
    assert closed == ["cancelled", "closed"]
