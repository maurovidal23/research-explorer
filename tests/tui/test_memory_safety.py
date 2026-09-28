"""Coalesced rendering, widget lifecycle, and bounded live consumer stress."""

from __future__ import annotations

from research_explorer.events.limits import LIVE_CANDIDATE_WINDOW, LIVE_EVENT_WINDOW
from research_explorer.events.models import RunEvent
from research_explorer.events.projection import RunProjection
from research_explorer.tui import TUIController, build_app

MAX_REFRESHES = 100


def _base_events() -> list[RunEvent]:
    return [
        RunEvent(
            seq=1,
            type="orchestrator_start",
            payload={"run_id": "mem", "colony_size": 1, "K": 1, "max_fetches": 10},
        ),
        RunEvent(seq=2, type="colony_initialized", payload={"size": 1, "agents": ["a0"]}),
        RunEvent(seq=3, type="oleada_start", payload={"oleada": 1, "active": ["a0"]}),
        RunEvent(
            seq=4,
            type="agent_turn_start",
            payload={"agent_id": "a0", "oleada": 1, "turn": 0},
        ),
    ]


def _score(seq: int) -> RunEvent:
    return RunEvent(
        seq=seq,
        type="candidate_score",
        payload={
            "paper_id": f"paper:{seq}",
            "agent_id": "a0",
            "eta": 0.4,
            "components": {"sim": 0.4},
        },
    )


async def _drain(pilot, app, controller, predicate=None) -> None:
    for _ in range(400):
        await pilot.pause()
        if predicate is not None and predicate():
            return
        if controller.queue.empty() and app._refresh_timer is None:
            return


async def test_event_burst_coalesces_into_bounded_refreshes(monkeypatch) -> None:
    controller = TUIController(maxsize=20_000)
    app = build_app(RunProjection.from_events(_base_events()), queue=controller.queue)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        calls = {"count": 0}
        original = app._refresh

        def counting() -> None:
            calls["count"] += 1
            original()

        monkeypatch.setattr(app, "_refresh", counting)
        for seq in range(10_001, 20_001):
            controller.event_sink.publish(_score(seq))
        await _drain(pilot, app, controller)
        assert calls["count"] < MAX_REFRESHES
        assert app.state.candidate_scores_seen_total == 10_000
        assert len(app.state.candidate_scores) == LIVE_CANDIDATE_WINDOW
        assert app.state.candidate_scores[-1].paper_id == "paper:20000"


async def test_terminal_delivery_flushes_and_leaves_no_scheduled_refresh() -> None:
    controller = TUIController()
    app = build_app(RunProjection.from_events(_base_events()), queue=controller.queue)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for seq in range(101, 151):
            controller.event_sink.publish(_score(seq))
        controller.event_sink.publish(
            RunEvent(
                seq=200,
                type="run_completed",
                payload={"run_id": "mem", "status": "completed"},
            )
        )
        await _drain(
            pilot,
            app,
            controller,
            lambda: app.state.status == "completed" and app._refresh_timer is None,
        )
        assert app.state.status == "completed"
        assert app._refresh_timer is None
        assert app.state.events[-1].canonical_type() == "run_completed"


async def test_candidate_scores_do_not_rebuild_the_tree(monkeypatch) -> None:
    app = build_app(RunProjection.from_events(_base_events()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        builds = {"count": 0}
        original = app._build_tree_items

        def counting(*args, **kwargs) -> None:
            builds["count"] += 1
            original(*args, **kwargs)

        monkeypatch.setattr(app, "_build_tree_items", counting)
        for seq in range(500, 1_500):
            app.projection.apply(_score(seq))
        app.refresh_view()
        await pilot.pause()
        assert builds["count"] == 0
        assert len(list(app.query("#tree ListItem"))) == len(app._rows)

        app.projection.apply(
            RunEvent(
                seq=2_000,
                type="agent_turn_start",
                payload={"agent_id": "a1", "oleada": 1, "turn": 0},
            )
        )
        app.refresh_view()
        await pilot.pause()
        assert builds["count"] >= 1
        assert len(list(app.query("#tree ListItem"))) == len(app._rows)


async def test_pilot_stress_keeps_queue_rows_and_markdown_bounded(monkeypatch) -> None:
    controller = TUIController(maxsize=20_000)
    app = build_app(RunProjection.from_events(_base_events()), queue=controller.queue)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        calls = {"count": 0}
        original = app._refresh

        def counting() -> None:
            calls["count"] += 1
            original()

        monkeypatch.setattr(app, "_refresh", counting)
        for seq in range(1_000, 11_000):
            controller.event_sink.publish(_score(seq))
        await _drain(pilot, app, controller)
        await pilot.press("4")
        for _ in range(20):
            await pilot.pause()
        assert calls["count"] < MAX_REFRESHES
        assert controller.queue.qsize() <= controller.channel._queue.maxsize
        assert app.state.events_seen_total == 10_000 + len(_base_events())
        assert len(app.state.events) == LIVE_EVENT_WINDOW
        assert len(app._rows) < 50
        body = str(app.query_one("#content").source)
        assert len(body) < 200_000
        assert body.count("- `[") <= 200
