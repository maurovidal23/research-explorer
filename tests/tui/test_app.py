"""Textual pilot tests for the live research TUI."""

from __future__ import annotations

import asyncio

from research_explorer.events.models import RunEvent
from research_explorer.events.projection import RunProjection
from research_explorer.tui import TUIController, build_app
from research_explorer.tui.app import (
    AgentTabs,
    ConfirmQuitScreen,
    DetailPane,
    HeaderBar,
    ResearchTUIApp,
    TextViewer,
)


def _analysis() -> dict:
    return {
        "summary": "A summary",
        "key_concepts": ["concept-a", "concept-b"],
        "methods": "Methods",
        "findings": "Findings",
        "relevance": "Relevance",
        "limitations": "Limitations",
        "key_references": ["ref-a"],
    }


def _events() -> list[RunEvent]:
    return [
        RunEvent(
            seq=1,
            type="orchestrator_start",
            payload={
                "run_id": "run-abc",
                "seed": "10.1/seed",
                "query": "How do things work?",
                "colony_size": 3,
                "K": 2,
                "k_per_turn": 4,
                "max_fetches": 100,
                "explorer_model": "explorer-x",
                "judge_model": "judge-y",
            },
        ),
        RunEvent(
            seq=2,
            type="oleada_start",
            payload={"oleada": 1, "active": ["agent-0", "agent-1"], "max_fetches": 100},
        ),
        RunEvent(
            seq=3,
            type="agent_turn_start",
            payload={"agent_id": "agent-0", "caste": "fundaciones", "oleada": 1, "turn": 0},
        ),
        RunEvent(
            seq=4,
            type="paper_fetch_started",
            payload={
                "agent_id": "agent-0",
                "paper_id": "openalex:W1",
                "mode": "ref",
                "src": "10.1/seed",
                "turn": 0,
            },
        ),
        RunEvent(
            seq=5,
            type="paper_integration_completed",
            payload={
                "agent_id": "agent-0",
                "paper_id": "openalex:W1",
                "title": "Paper One",
                "year": 2020,
                "authors": ["Ada", "Bob"],
                "mode": "ref",
                "src": "10.1/seed",
                "turn": 0,
                "provider": "openalex",
                "analysis": _analysis(),
            },
        ),
        RunEvent(
            seq=6,
            type="candidate_score",
            payload={
                "paper_id": "openalex:W1",
                "agent_id": "agent-0",
                "provider": "openalex",
                "components": {"sim": 0.5, "citations": 0.4},
                "weights": {"w_sim": 0.6},
                "eta": 0.6,
                "source": "10.1/seed",
                "mode": "ref",
            },
        ),
        RunEvent(
            seq=7,
            type="candidate_selected",
            payload={
                "agent_id": "agent-0",
                "paper_id": "openalex:W1",
                "src": "10.1/seed",
                "mode": "ref",
                "caste": "fundaciones",
                "dir_modifier": 0.7,
                "tau": 1.0,
                "alpha": 1.0,
                "beta": 3.0,
                "eta": 0.6,
                "final_weight": 0.216,
                "probability": 0.8,
                "epsilon_branch": False,
                "chosen": True,
            },
        ),
        RunEvent(
            seq=8,
            type="evaluation_detail",
            payload={
                "detail": {
                    "agent_id": "agent-0",
                    "oleada": 1,
                    "turn": 0,
                    "q": 0.7,
                    "delta_q": 0.7,
                    "weights": {"S": 0.25, "P": 0.25, "J": 0.25, "R": 0.25},
                    "self_assessment": {"score": 0.7, "reasoning": "self good"},
                    "peers": {
                        "votes": [{"voter_id": "agent-1", "score": 0.6, "reasoning": "peer ok"}],
                        "aggregated_score": 0.6,
                        "num_votes": 1,
                    },
                    "virgin_judge": {"score": 0.7, "coverage": "covers", "gaps": "gaps"},
                    "structural": {
                        "coverage": 0.5,
                        "diversity": 0.5,
                        "depth": 0.5,
                        "coherence": 0.5,
                        "r": 0.5,
                    },
                }
            },
        ),
        RunEvent(
            seq=9,
            type="agent_turn_complete",
            payload={"agent": "agent-0", "oleada": 1, "turn": 0, "Q": 0.7, "delta_q": 0.7, "budget": 10},
        ),
        RunEvent(
            seq=10,
            type="artifact_saved",
            payload={
                "artifact_id": "a1",
                "name": "narrative_agent-0_t0.md",
                "kind": "narrative",
                "content": "# Narrative\nBody text",
            },
        ),
        RunEvent(
            seq=13,
            type="neighbor_discovery_started",
            payload={"agent_id": "agent-0", "paper_id": "openalex:W1", "turn": 0},
        ),
        RunEvent(
            seq=14,
            type="neighbor_discovery_completed",
            payload={"agent_id": "agent-0", "paper_id": "openalex:W1", "turn": 0, "refs": 3, "cits": 1},
        ),
        RunEvent(
            seq=15,
            type="frontier_reference_evaluation_started",
            payload={"agent_id": "agent-0", "count": 3, "turn": 0},
        ),
        RunEvent(
            seq=16,
            type="frontier_reference_evaluation_completed",
            payload={"agent_id": "agent-0", "count": 3, "turn": 0},
        ),
        RunEvent(
            seq=17,
            type="llm_operation_started",
            payload={"purpose": "paper_integration", "model": "explorer-x"},
        ),
        RunEvent(
            seq=18,
            type="llm_operation_completed",
            payload={"purpose": "paper_integration", "model": "explorer-x", "elapsed": 1.25},
        ),
        RunEvent(
            seq=19,
            type="provider_failure",
            payload={
                "agent_id": "agent-0",
                "paper_id": "openalex:W9",
                "reason": "429 timeout",
                "classification": "transient",
            },
        ),
    ]


def _projection() -> RunProjection:
    return RunProjection.from_events(_events())


def _app() -> ResearchTUIApp:
    return build_app(_projection())


async def test_initial_layout() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        header = app.query_one("#header", HeaderBar).plain_text
        assert "run-abc" in header
        assert "explorer-x" in header
        tabs = app.query_one("#tabs", AgentTabs).plain_text
        assert "A01" in tabs
        assert "foundations" in tabs
        timeline = app.query_one("#timeline").plain_text
        assert "Wave 1" in timeline
        assert "Paper One" in timeline
        detail = app.query_one("#detail", DetailPane).plain_text
        assert "Current activity" in detail
        assert not app.narrow


async def test_agent_navigation_changes_detail() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.selected_agent_id == "agent-0"
        await pilot.press("right")
        assert app.selected_agent_index == 1
        assert app.selected_agent_id == "agent-1"
        await pilot.press("left")
        assert app.selected_agent_id == "agent-0"


async def test_timeline_navigation_and_follow_live() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.state.follow_live
        await pilot.press("up")
        assert not app.state.follow_live
        assert app.state.selected_entry_id is not None
        await pilot.press("home")
        assert app.state.follow_live
        assert app.state.selected_entry_id == app.state.timeline[-1].entry_id


async def test_focused_views_and_restoration() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        selected_before = app.state.selected_entry_id
        for key, expected in (
            ("e", "Evaluation"),
            ("f", "Frontier decision record"),
            ("p", "Paper analysis"),
            ("n", "Agent narrative"),
            ("l", "Events"),
            ("r", "Metadata"),
            ("question_mark", "Key help"),
        ):
            await pilot.press(key)
            await pilot.pause()
            assert isinstance(app.screen, TextViewer), key
            assert app.screen.viewer_title == expected
            await pilot.press("escape")
            await pilot.pause()
            assert app.state.selected_entry_id == selected_before


async def test_details_overlay_and_restoration() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("up")
        selected = app.state.selected_entry_id
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, TextViewer)
        await pilot.press("escape")
        await pilot.pause()
        assert app.state.selected_entry_id == selected


async def test_narrow_terminal_fallback() -> None:
    app = _app()
    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        assert app.narrow
        await pilot.press("t")
        await pilot.pause()
        assert app.active_pane in ("timeline", "detail")
        timeline = app.query_one("#timeline-pane")
        detail = app.query_one("#detail-pane")
        assert not (timeline.display and detail.display)


async def test_completed_and_failed_screens() -> None:
    projection = _projection()
    projection.apply(
        RunEvent(
            seq=11,
            type="orchestrator_complete",
            payload={
                "run_id": "run-abc",
                "winner": "agent-0",
                "peak_Q": 0.7,
                "status": "completed",
                "total_fetches": 5,
                "elapsed": 12.0,
            },
        )
    )
    app = build_app(projection)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.state.status == "completed"
        footer = app.query_one("#footer").plain_text
        assert "completed" in footer
        await pilot.press("q")
        await pilot.pause()

    failed = RunProjection.from_events(
        [
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r2"}),
            RunEvent(seq=2, type="run_failed", payload={"error": "boom"}),
        ]
    )
    app2 = build_app(failed)
    async with app2.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app2.state.status == "failed"
        assert any("boom" in f for f in app2.state.failures)


async def test_quit_confirmation_when_active() -> None:
    sentinel = asyncio.Event()

    async def runner() -> str:
        await sentinel.wait()
        return "finished"

    projection = RunProjection.from_events(
        [RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r3"})]
    )
    app = build_app(projection, runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmQuitScreen)
        assert not app.screen.can_detach
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is app.screen_stack[0]
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.cancelled
        assert "settling" in app.query_one("#footer").plain_text
        sentinel.set()
        await pilot.pause()


async def test_live_event_channel_updates_view() -> None:
    app = _app()
    projection = app.projection
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        projection.apply(
            RunEvent(
                seq=12,
                type="agent_turn_start",
                payload={"agent_id": "agent-2", "caste": "mixto", "oleada": 2, "turn": 0},
            )
        )
        app.refresh_view()
        await pilot.pause()
        assert "A03" in app.query_one("#tabs", AgentTabs).plain_text


async def test_terminal_safe_shutdown() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
    assert app.state.status in ("running", "completed")


async def test_live_channel_queue_updates_projection_and_view() -> None:
    controller = TUIController()
    app = build_app(controller.projection, queue=controller.queue)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        controller.event_sink.publish(
            RunEvent(
                seq=1,
                type="orchestrator_start",
                payload={"run_id": "live-run", "colony_size": 1, "K": 1},
            )
        )
        controller.event_sink.publish(
            RunEvent(
                seq=2,
                type="agent_turn_start",
                payload={"agent_id": "z0", "caste": "mixto", "oleada": 1, "turn": 0},
            )
        )
        for _ in range(20):
            await pilot.pause()
            if app.state.run_id == "live-run" and "A01" in app.query_one(
                "#tabs", AgentTabs
            ).plain_text:
                break
        assert app.state.run_id == "live-run"
        assert "A01" in app.query_one("#tabs", AgentTabs).plain_text
        await controller.queue.put(None)


def test_controller_drain_matches_direct_projection() -> None:
    events = _events()
    controller = TUIController()
    for event in events:
        controller.event_sink.publish(event)
    assert not controller.channel.empty()
    controller.drain()
    assert controller.channel.empty()
    drained = controller.projection.state
    direct = RunProjection.from_events(events).state
    assert drained.run_id == direct.run_id
    assert drained.status == direct.status
    assert drained.agents == direct.agents
    assert drained.timeline == direct.timeline
    assert drained.narratives == direct.narratives
    assert [s.paper_id for s in drained.selections] == [s.paper_id for s in direct.selections]
    assert [c.paper_id for c in drained.candidate_scores] == [
        c.paper_id for c in direct.candidate_scores
    ]
    assert drained.evaluations["agent-0"][-1].q == direct.evaluations["agent-0"][-1].q


async def test_runner_failure_marks_failed_and_redacts() -> None:
    async def runner() -> str:
        raise RuntimeError("boom api_key=sk-sentinel-4242")

    app = build_app(_projection(), runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if app.state.status == "failed":
                break
    assert app.state.status == "failed"
    assert any("boom" in failure for failure in app.state.failures)
    assert "sk-sentinel-4242" not in " ".join(app.state.failures)


async def test_runner_success_captures_result() -> None:
    async def runner() -> tuple[str, None]:
        return "report body", None

    app = build_app(_projection(), runner=runner)
    await app._run_runner()
    assert app.run_result == ("report body", None)


async def test_quit_confirmation_offers_detach_when_allowed() -> None:
    async def runner() -> str:
        return "done"

    app = build_app(_projection(), runner=runner, can_detach=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmQuitScreen)
        assert app.screen.can_detach


async def test_cancel_choice_cancels_active_run() -> None:
    gate = asyncio.Event()

    async def runner() -> str:
        await gate.wait()
        return "done"

    app = build_app(_projection(), runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert app.cancelled
        gate.set()
        await pilot.pause()


async def test_down_navigation_then_home_restores_follow() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        newest = app.state.selected_entry_id
        await pilot.press("up")
        assert not app.state.follow_live
        assert app.state.selected_entry_id != newest
        await pilot.press("down")
        assert app.state.selected_entry_id == newest
        await pilot.press("home")
        assert app.state.follow_live
        assert app.state.selected_entry_id == newest


async def test_focused_views_render_projected_content() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for key, expected in (
            ("e", "Q=0.7000"),
            ("f", "openalex:W1"),
            ("n", "Body text"),
            ("l", "Events"),
        ):
            await pilot.press(key)
            await pilot.pause()
            assert isinstance(app.screen, TextViewer), key
            assert expected in app.screen.viewer_body, key
            await pilot.press("escape")
            await pilot.pause()


async def test_narrow_toggle_switches_visible_pane() -> None:
    app = _app()
    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        assert app.narrow
        assert app.active_pane == "timeline"
        await pilot.press("t")
        await pilot.pause()
        assert app.active_pane == "detail"
        assert app.query_one("#detail-pane").display
        assert not app.query_one("#timeline-pane").display
        await pilot.press("t")
        await pilot.pause()
        assert app.active_pane == "timeline"


async def test_resize_preserves_selection() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("right")
        agent = app.selected_agent_id
        selected = app.state.selected_entry_id
        await pilot.resize_terminal(70, 24)
        await pilot.pause()
        assert app.narrow
        await pilot.resize_terminal(120, 40)
        await pilot.pause()
        assert not app.narrow
        assert app.selected_agent_id == agent
        assert app.state.selected_entry_id == selected


async def test_timeline_and_detail_project_discovery_and_telemetry() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        timeline = app.query_one("#timeline").plain_text
        assert "discover openalex:W1" in timeline
        assert "refs=3" in timeline
        assert "frontier evaluation" in timeline
        detail = app.query_one("#detail", DetailPane).plain_text
        assert "operation paper_integration" in detail
        assert "model explorer-x" in detail
        assert "elapsed 1s" in detail
        assert "year 2020" in detail
        assert "authors Ada, Bob" in detail
        assert "source 10.1/seed" in detail


async def test_render_failure_requests_orderly_shutdown(monkeypatch) -> None:
    from research_explorer.tui import text as render

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        def boom(state):
            raise RuntimeError("render exploded")

        monkeypatch.setattr(render, "render_header", boom)
        app.refresh_view()
        await pilot.pause()
        assert any("ui render failed" in w for w in app.state.warnings)
        assert app._exit


async def test_consumer_and_runner_workers_are_independent() -> None:
    controller = TUIController()
    gate = asyncio.Event()
    started = asyncio.Event()

    async def runner() -> tuple[str, None]:
        started.set()
        await gate.wait()
        return "late report", None

    app = build_app(controller.projection, queue=controller.queue, runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        await started.wait()
        controller.event_sink.publish(
            RunEvent(
                seq=1,
                type="orchestrator_start",
                payload={"run_id": "live-run", "colony_size": 1, "K": 1},
            )
        )
        controller.event_sink.publish(
            RunEvent(
                seq=2,
                type="agent_turn_start",
                payload={"agent_id": "z0", "caste": "mixto", "oleada": 1, "turn": 0},
            )
        )
        for _ in range(40):
            await pilot.pause()
            if app.state.run_id == "live-run" and "A01" in app.query_one(
                "#tabs", AgentTabs
            ).plain_text:
                break
        assert app.state.run_id == "live-run"
        assert not app._runner_finished
        assert not app._runner_worker.is_finished
        assert not app._consumer_worker.is_finished

        gate.set()
        for _ in range(40):
            await pilot.pause()
            if app._runner_finished:
                break
        assert app._runner_finished
        assert app.run_result == ("late report", None)


async def test_runner_completion_waits_for_queued_events() -> None:
    controller = TUIController()

    async def runner() -> tuple[str, None]:
        controller.event_sink.publish(
            RunEvent(
                seq=1,
                type="orchestrator_start",
                payload={"run_id": "sync-run", "colony_size": 1, "K": 1},
            )
        )
        for i in range(40):
            controller.event_sink.publish(
                RunEvent(
                    seq=2 + i,
                    type="agent_turn_start",
                    payload={"agent_id": "a0", "caste": "mixto", "oleada": 1, "turn": i},
                )
            )
        controller.event_sink.publish(
            RunEvent(
                seq=100,
                type="warning",
                payload={
                    "reason": "no traversable identifiers",
                    "reason_code": "no_traversable_identifiers",
                    "classification": "degraded",
                },
            )
        )
        controller.event_sink.publish(
            RunEvent(
                seq=101,
                type="no_winner",
                payload={
                    "run_id": "sync-run",
                    "status": "completed",
                    "outcome": "degraded",
                    "reason_code": "no_traversable_identifiers",
                    "reason": "no traversable identifiers",
                    "elapsed": 23.0,
                },
            )
        )
        return "report body", None

    app = build_app(controller.projection, queue=controller.queue, runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(60):
            await pilot.pause()
            if app._runner_finished and app.run_result is not None:
                break
        assert app._runner_finished
        assert app.state.status == "completed"
        assert app.state.status != "initializing"
        assert app.state.terminal_reason_code == "no_traversable_identifiers"
        assert "a0" in app.state.agents
        assert app.run_result == ("report body", None)


async def test_q_after_completion_exits_without_confirmation() -> None:
    projection = RunProjection.from_events(
        [
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "done"}),
            RunEvent(
                seq=2,
                type="orchestrator_complete",
                payload={"run_id": "done", "status": "completed", "winner": "a0"},
            ),
        ]
    )
    gate = asyncio.Event()

    async def runner() -> str:
        await gate.wait()
        return "done"

    app = build_app(projection, runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.state.status == "completed"
        await pilot.press("q")
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmQuitScreen)
        assert app._exit
        gate.set()


async def test_cancellation_projects_durable_run_cancelled() -> None:
    controller = TUIController()
    gate = asyncio.Event()

    async def runner() -> tuple[str, None]:
        controller.event_sink.publish(
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "cancel-run"})
        )
        try:
            await gate.wait()
        except asyncio.CancelledError:
            controller.event_sink.publish(
                RunEvent(seq=2, type="run_cancelled", payload={"run_id": "cancel-run"})
            )
            raise
        return "never", None

    app = build_app(controller.projection, queue=controller.queue, runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+c")
        for _ in range(40):
            await pilot.pause()
            if app.state.status == "cancelled":
                break
        assert app.cancelled
        assert app.state.status == "cancelled"


async def test_cancel_cannot_relabel_a_completed_run() -> None:
    controller = TUIController()
    controller.event_sink.publish(
        RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "done"})
    )
    controller.event_sink.publish(
        RunEvent(
            seq=2,
            type="orchestrator_complete",
            payload={"run_id": "done", "status": "completed", "winner": "a0"},
        )
    )
    never = asyncio.Event()

    async def runner() -> str:
        await never.wait()
        return "done"

    app = build_app(controller.projection, queue=controller.queue, runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.state.status == "completed"
        app.action_cancel_flow()
        await pilot.pause()
        assert app.state.status == "completed"
        assert app.state.winner_agent == "a0"
        assert not any(e.type == "run_cancelled" for e in app.state.events)
        never.set()


async def test_report_persisted_before_review_and_not_rewritten(tmp_path) -> None:
    from research_explorer.cli import _write_report_atomic

    controller = TUIController()
    target = tmp_path / "report.md"
    writes: list[str] = []

    def persist(result) -> None:
        writes.append(result[0])
        _write_report_atomic(result[0], str(target))

    async def runner() -> tuple[str, None]:
        controller.event_sink.publish(
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "persist"})
        )
        controller.event_sink.publish(
            RunEvent(
                seq=2,
                type="no_winner",
                payload={
                    "run_id": "persist",
                    "status": "completed",
                    "outcome": "degraded",
                    "reason_code": "no_traversable_identifiers",
                    "reason": "no traversable identifiers",
                },
            )
        )
        return "REPORT BODY", None

    app = build_app(
        controller.projection, queue=controller.queue, runner=runner, on_result=persist
    )
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._runner_finished:
                break
        assert app.state.status == "completed"
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "REPORT BODY"
        assert writes == ["REPORT BODY"]
        assert app.run_result == ("REPORT BODY", None)


async def test_report_write_failure_keeps_run_completed() -> None:
    controller = TUIController()

    def boom(result) -> None:
        raise RuntimeError("disk full api_key=sk-sentinel-4242")

    async def runner() -> tuple[str, None]:
        controller.event_sink.publish(
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "writer"})
        )
        controller.event_sink.publish(
            RunEvent(
                seq=2,
                type="orchestrator_complete",
                payload={"run_id": "writer", "status": "completed", "winner": "a0"},
            )
        )
        return "body", None

    app = build_app(
        controller.projection, queue=controller.queue, runner=runner, on_result=boom
    )
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._runner_finished:
                break
        assert app.state.status == "completed"
        assert app.state.winner_agent == "a0"
        assert app.report_error is not None
        assert "sk-sentinel-4242" not in app.report_error


async def test_event_outcome_filter_cycles_in_pilot() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("o")
        await pilot.pause()
        assert isinstance(app.screen, TextViewer)
        assert app.screen.viewer_title == "Events (error)"
        assert "no events recorded for this filter" in app.screen.viewer_body
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("o")
        await pilot.pause()
        assert app.screen.viewer_title == "Events (warning)"
        assert "429 timeout" in app.screen.viewer_body
        assert "run_failed" not in app.screen.viewer_body
        await pilot.press("escape")
        await pilot.pause()
