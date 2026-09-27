"""Live-reliability worker, ordering, and cancellation tests (TUI-REL-1/2/3)."""

from __future__ import annotations

import asyncio

from research_explorer.events.models import RunEvent
from research_explorer.events.projection import RunProjection
from research_explorer.tui import TUIController, build_app
from research_explorer.tui.app import AgentTabs, ConfirmQuitScreen


async def test_worker_groups_keep_both_workers_alive_before_runner_finishes() -> None:
    controller = TUIController()
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner() -> str:
        controller.event_sink.publish(
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "live", "colony_size": 1, "K": 1})
        )
        controller.event_sink.publish(
            RunEvent(seq=2, type="colony_initialized", payload={"size": 1, "agents": ["z0"]})
        )
        controller.event_sink.publish(
            RunEvent(
                seq=3,
                type="agent_turn_start",
                payload={"agent_id": "z0", "caste": "mixto", "oleada": 1, "turn": 0},
            )
        )
        started.set()
        await release.wait()
        return "done"

    app = build_app(controller.projection, queue=controller.queue, runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(30):
            await pilot.pause()
            if started.is_set() and "A01" in app.query_one("#tabs", AgentTabs).plain_text:
                break
        assert started.is_set()
        assert app.state.run_id == "live"
        assert "A01" in app.query_one("#tabs", AgentTabs).plain_text
        # The exclusive research runner must not have cancelled the consumer.
        assert app._consumer_worker is not None
        assert not app._consumer_worker.is_finished
        assert app._runner_worker is not None and not app._runner_worker.is_finished
        release.set()
        for _ in range(30):
            await pilot.pause()
            if app.run_finished:
                break
        assert app.run_finished
        assert app.run_result == "done"


async def test_runner_completion_waits_for_queued_events() -> None:
    controller = TUIController()

    async def runner() -> str:
        controller.event_sink.publish(
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "wait", "colony_size": 1, "K": 1})
        )
        controller.event_sink.publish(
            RunEvent(seq=2, type="colony_initialized", payload={"size": 1, "agents": ["a0"]})
        )
        controller.event_sink.publish(
            RunEvent(
                seq=3,
                type="agent_turn_start",
                payload={"agent_id": "a0", "caste": "mixto", "oleada": 1, "turn": 0},
            )
        )
        controller.event_sink.publish(
            RunEvent(
                seq=4,
                type="orchestrator_complete",
                payload={"run_id": "wait", "status": "completed", "winner": "a0", "peak_Q": 0.5},
            )
        )
        return "report"

    app = build_app(controller.projection, queue=controller.queue, runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app.run_finished:
                break
    assert app.run_finished
    # Every event published before completion was applied before the result.
    assert app.state.run_id == "wait"
    assert app.state.status == "completed"
    assert app.state.agent_order == ["a0"]
    assert app.state.agents["a0"].status == "completed"


async def test_q_after_completion_exits_without_confirmation() -> None:
    projection = RunProjection.from_events(
        [
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r"}),
            RunEvent(seq=2, type="orchestrator_complete", payload={"status": "completed"}),
        ]
    )

    async def runner() -> str:
        return "done"

    app = build_app(projection, runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.state.status == "completed"
        await pilot.press("q")
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmQuitScreen)
        assert app._exit


async def test_cancellation_projects_run_cancelled_not_failed() -> None:
    controller = TUIController()
    gate = asyncio.Event()

    async def runner() -> str:
        controller.event_sink.publish(
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "c", "colony_size": 1, "K": 1})
        )
        try:
            await gate.wait()
        except asyncio.CancelledError:
            controller.event_sink.publish(
                RunEvent(seq=2, type="run_cancelled", payload={"run_id": "c"})
            )
            raise
        return "never"

    app = build_app(controller.projection, queue=controller.queue, runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(30):
            await pilot.pause()
            if app.state.run_id == "c":
                break
        await pilot.press("ctrl+c")
        for _ in range(30):
            await pilot.pause()
            if app.state.status == "cancelled":
                break
        assert app.cancelled
        assert app.state.status == "cancelled"
        assert not app.state.failures
        assert not app.state.warnings
        gate.set()


async def test_exception_with_durable_failure_is_not_duplicated() -> None:
    controller = TUIController()

    async def runner() -> str:
        controller.event_sink.publish(
            RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "f"})
        )
        controller.event_sink.publish(
            RunEvent(seq=2, type="run_failed", payload={"error": "durable boom"})
        )
        raise RuntimeError("secondary wrapper error")

    app = build_app(controller.projection, queue=controller.queue, runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app.run_finished:
                break
    assert app.state.status == "failed"
    # Only the durable failure is recorded; the synthetic fallback is suppressed.
    assert app.state.failures == ["durable boom"]
    assert sum(1 for e in app.state.events if e.type == "run_failed") == 1
