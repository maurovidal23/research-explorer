"""OOM-incident observability: last activity, interrupt notice, token wording."""

from __future__ import annotations

from research_explorer.events.models import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    RunEvent,
    RunViewState,
)
from research_explorer.events.projection import RunProjection
from research_explorer.tui import text as render


def test_metadata_reports_last_durable_activity_and_operation() -> None:
    state = RunProjection.from_events(
        [
            RunEvent(
                seq=1,
                type="orchestrator_start",
                payload={"run_id": "r"},
                ts="2026-01-01T00:00:00+00:00",
            ),
            RunEvent(
                seq=2,
                type="candidate_score",
                payload={"paper_id": "p", "agent_id": "a0"},
                ts="2026-01-01T00:05:00+00:00",
            ),
        ]
    ).state
    body = render.render_metadata(state)
    assert "last_activity: 2026-01-01T00:05:00+00:00" in body
    assert "last_operation: candidate_score" in body
    assert "API-reported historical" in body
    assert "interrupted:" not in body


def test_metadata_interrupt_notice_skips_synthetic_event() -> None:
    projection = RunProjection.from_events(
        [
            RunEvent(
                seq=1,
                type="orchestrator_start",
                payload={"run_id": "r"},
                ts="2026-01-01T00:00:00+00:00",
            ),
            RunEvent(
                seq=0,
                type="run_interrupted",
                payload={"reason": "stale heartbeat"},
                ts="2030-01-01T00:00:00+00:00",
            ),
        ]
    )
    state = projection.state
    assert state.status == STATUS_INTERRUPTED
    last = render.last_durable_event(state)
    assert last is not None and last.canonical_type() == "run_started"
    body = render.render_metadata(state)
    assert "last_operation: run_started" in body
    assert "interrupted: final in-memory work may have been lost" in body


def test_status_banner_distinguishes_interrupted_from_failed() -> None:
    interrupted = RunViewState(
        run_id="r", status=STATUS_INTERRUPTED, terminal_reason="host lost"
    )
    interrupted_text = render.render_status_banner(interrupted).plain
    assert "interrupted" in interrupted_text
    assert "host lost" in interrupted_text
    assert "durable trace preserved" in interrupted_text

    failed = RunViewState(run_id="r", status=STATUS_FAILED, terminal_reason="boom")
    failed_text = render.render_status_banner(failed).plain
    assert "failed" in failed_text
    assert "durable trace preserved" not in failed_text

    completed = RunViewState(run_id="r", status=STATUS_COMPLETED)
    completed_text = render.render_status_banner(completed).plain
    assert "completed" in completed_text
    assert "run finished" in completed_text
