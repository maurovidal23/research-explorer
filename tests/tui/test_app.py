"""Textual pilot tests for the Convoy-style research dashboard."""

from __future__ import annotations

import asyncio

from research_explorer.events.models import RunEvent
from research_explorer.events.projection import RunProjection
from research_explorer.tui import TUIController, build_app
from research_explorer.tui.app import (
    CommandPaletteScreen,
    ConfirmQuitScreen,
    ResearchTUIApp,
    TextViewer,
)


def _analysis() -> dict:
    return {
        "summary": "A summary",
        "key_concepts": ["concept-a"],
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
                "k_per_turn": 2,
                "max_fetches": 100,
                "explorer_model": "explorer-x",
                "judge_model": "judge-y",
            },
        ),
        RunEvent(
            seq=2,
            type="colony_initialized",
            payload={"size": 3, "agents": ["agent-0", "agent-1", "agent-2"]},
        ),
        RunEvent(seq=3, type="oleada_start", payload={"oleada": 1, "active": ["agent-0", "agent-1"]}),
        RunEvent(
            seq=4,
            type="agent_turn_start",
            payload={"agent_id": "agent-0", "caste": "fundaciones", "oleada": 1, "turn": 0},
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
                "oleada": 1,
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
        RunEvent(seq=10, type="new_best", payload={"agent": "agent-0", "Q": 0.7, "oleada": 1}),
        RunEvent(
            seq=11,
            type="artifact_saved",
            payload={
                "artifact_id": "a1",
                "name": "narrative_agent-0_t0.md",
                "kind": "narrative",
                "content": "# Narrative\nBody text",
            },
        ),
        RunEvent(
            seq=12,
            type="provider_failure",
            payload={
                "agent_id": "agent-0",
                "paper_id": "openalex:W9",
                "reason": "429 timeout",
                "classification": "transient",
            },
        ),
    ]


def _app() -> ResearchTUIApp:
    return build_app(RunProjection.from_events(_events()))


async def test_initial_dashboard_layout() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        header = app.query_one("#header").plain_text
        assert "Research Explorer" in header
        assert "run-abc" in header
        assert "explorer-x" in header
        rows = app.tree_row_texts()
        assert any("A01" in row and "foundations" in row for row in rows)
        assert any("Paper One" in row for row in rows)
        activity = app.query_one("#activity").plain_text
        assert "Agent" in activity and "Paper One" in activity
        assert app.query_one("#tabbar").active == "research"
        assert "Research Explorer" in app.query_one("#footer").plain_text
        assert not app.session.narrowed


async def test_tree_navigation_is_agent_first_and_home_restores_follow() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.session.follow_live
        await pilot.press("left")
        await pilot.pause()
        assert app.session.focus == "tree"
        await pilot.press("up")
        await pilot.pause()
        assert not app.session.follow_live
        selected = app.session.selected_node_id
        await pilot.press("down")
        await pilot.pause()
        assert app.session.selected_node_id != selected
        await pilot.press("home")
        await pilot.pause()
        assert app.session.follow_live
        assert app.session.selected_agent_id == "agent-0"


async def test_tabs_select_with_number_and_cycle_keys() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for key, tab in (("2", "paper"), ("3", "evaluation"), ("4", "events"), ("1", "research")):
            await pilot.press(key)
            await pilot.pause()
            assert app.session.active_tab == tab
            assert app.query_one("#tabbar").active == tab
        await pilot.press("tab")
        await pilot.pause()
        assert app.session.active_tab == "paper"
        await pilot.press("shift+tab")
        await pilot.pause()
        assert app.session.active_tab == "research"


async def test_content_tabs_render_projected_facts() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert "Research line" in str(app.query_one("#content").source)
        await pilot.press("2")
        await pilot.pause()
        assert "Selection rationale" in str(app.query_one("#content").source)
        await pilot.press("3")
        await pilot.pause()
        content = str(app.query_one("#content").source)
        assert "Q=0.7000" in content and "self rationale" in content
        await pilot.press("4")
        await pilot.pause()
        assert "429 timeout" in str(app.query_one("#content").source)


async def test_enter_toggles_tree_expansion() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("left")
        await pilot.pause()
        # Select the first agent root explicitly.
        app._on_tree_select(app._tree_nodes[0])
        await pilot.pause()
        assert app._tree_nodes[0].node_id in app.session.expanded
        await pilot.press("enter")
        await pilot.pause()
        assert app._tree_nodes[0].node_id not in app.session.expanded


async def test_reader_and_help_overlays() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("v")
        await pilot.pause()
        assert isinstance(app.screen, TextViewer)
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, TextViewer)
        assert "keyboard shortcuts" in app.screen.viewer_body
        await pilot.press("escape")
        await pilot.pause()


async def test_command_palette_runs_an_action() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert isinstance(app.screen, CommandPaletteScreen)
        commands = app.screen._actions
        assert "tab_events" in commands
        app.screen._choose(commands.index("tab_events"))
        await pilot.pause()
        assert app.session.active_tab == "events"


async def test_event_outcome_and_agent_filters() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("o")
        await pilot.pause()
        assert app.session.event_outcome == "error"
        assert app.session.active_tab == "events"
        assert "outcome=error" in str(app.query_one("#content").source)
        await pilot.press("o")
        await pilot.pause()
        assert app.session.event_outcome == "warning"
        assert "429 timeout" in str(app.query_one("#content").source)
        await pilot.press("a")
        await pilot.pause()
        assert app.session.event_agent == "all"
        assert "agent=" not in str(app.query_one("#content").source)


async def test_focused_views_open_and_restore_selection() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        before = app.session.selected_node_id
        for key, expected in (
            ("e", "Evaluation"),
            ("f", "Frontier"),
            ("p", "Paper"),
            ("n", "Narrative"),
            ("l", "Events"),
            ("r", "Metadata"),
        ):
            await pilot.press(key)
            await pilot.pause()
            assert isinstance(app.screen, TextViewer), key
            assert app.screen.viewer_title == expected, key
            await pilot.press("escape")
            await pilot.pause()
            assert app.session.selected_node_id == before


async def test_mouse_selection_on_tree_row() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        items = list(app.query("#tree ListItem"))
        await pilot.click(items[1])
        await pilot.pause()
        assert not app.session.follow_live


async def test_mouse_click_selects_tab() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tabs = list(app.query("#tabbar Tab"))
        await pilot.click(tabs[3])
        await pilot.pause()
        assert app.session.active_tab == "events"


async def test_wheel_scrolls_the_focused_panel() -> None:
    from textual.containers import VerticalScroll
    from textual.events import MouseScrollDown

    events = _events()
    events.append(
        RunEvent(
            seq=20,
            type="artifact_saved",
            payload={
                "artifact_id": "long",
                "name": "narrative_agent-0_t0.md",
                "kind": "narrative",
                "content": "\n".join(f"paragraph {i}" for i in range(200)),
            },
        )
    )
    app = build_app(RunProjection.from_events(events))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        scroll = app.query_one("#content-scroll", VerticalScroll)
        scroll.scroll_y = 0
        assert scroll.max_scroll_y > 0
        event = MouseScrollDown(scroll, 10, 5, 0, 3, 0, False, False, False, 10, 5)
        scroll._on_mouse_scroll_down(event)
        await pilot.pause()
        assert scroll.scroll_y > 0


async def test_compact_layout_stacks_and_footer_adapts() -> None:
    app = _app()
    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        assert app.session.narrowed
        assert app.query_one("#main").has_class("narrow")
        assert app.query_one("#left").has_class("narrow")
        assert "toggle panes" in app.query_one("#footer").plain_text
        await pilot.press("t")
        await pilot.pause()
        assert app.session.focus == "content"
        await pilot.press("t")
        await pilot.pause()
        assert app.session.focus == "tree"


async def test_resize_preserves_selection_and_tab() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("3")
        await pilot.pause()
        agent = app.session.selected_agent_id
        await pilot.resize_terminal(70, 24)
        await pilot.pause()
        assert app.session.narrowed
        await pilot.resize_terminal(120, 40)
        await pilot.pause()
        assert not app.session.narrowed
        assert app.session.active_tab == "evaluation"
        assert app.session.selected_agent_id == agent


async def test_completed_failed_and_cancelled_states_are_distinct() -> None:
    completed = _app()
    completed.projection.apply(
        RunEvent(
            seq=13,
            type="orchestrator_complete",
            payload={"run_id": "run-abc", "winner": "agent-0", "peak_Q": 0.7, "status": "completed"},
        )
    )
    async with completed.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert completed.state.status == "completed"
        assert "completed" in completed.query_one("#footer").plain_text
        assert completed.query_one("#status-banner").display
        await pilot.press("q")
        await pilot.pause()

    failed = build_app(
        RunProjection.from_events(
            [
                RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r2"}),
                RunEvent(seq=2, type="run_failed", payload={"error": "boom"}),
            ]
        )
    )
    async with failed.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert failed.state.status == "failed"
        assert "boom" in failed.query_one("#status-banner").plain_text

    cancelled = build_app(
        RunProjection.from_events(
            [
                RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r3"}),
                RunEvent(seq=2, type="run_cancelled", payload={"run_id": "r3"}),
            ]
        )
    )
    async with cancelled.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert cancelled.state.status == "cancelled"


async def test_quit_confirmation_and_cancel_flow() -> None:
    gate = asyncio.Event()

    async def runner() -> str:
        await gate.wait()
        return "done"

    projection = RunProjection.from_events(
        [RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r4"})]
    )
    app = build_app(projection, runner=runner)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmQuitScreen)
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.cancelled
        gate.set()
        await pilot.pause()


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


def _projection() -> RunProjection:
    return RunProjection.from_events(_events())


async def test_live_channel_updates_projection_and_tree() -> None:
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
            RunEvent(seq=2, type="colony_initialized", payload={"size": 1, "agents": ["z0"]})
        )
        for _ in range(20):
            await pilot.pause()
            if app.state.run_id == "live-run" and "A01" in app.agent_roster_text:
                break
        assert app.state.run_id == "live-run"
        assert "A01" in app.agent_roster_text
        await controller.queue.put(None)


async def test_read_only_replay_freezes_elapsed() -> None:
    projection = RunProjection.from_events(_events())
    projection.apply(
        RunEvent(
            seq=13,
            type="orchestrator_complete",
            payload={"run_id": "run-abc", "winner": "agent-0", "status": "completed", "elapsed": 42.0},
        )
    )
    before = projection.state.elapsed_seconds
    app = build_app(projection, read_only=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._tick_elapsed()
        assert app.state.elapsed_seconds == before
        assert app.session.follow_live
        assert app.session.selected_agent_id == "agent-0"


async def test_replay_without_winner_selects_first_agent_and_research_tab() -> None:
    projection = RunProjection.from_events(
        [
            RunEvent(
                seq=1,
                type="orchestrator_start",
                payload={"run_id": "degraded", "colony_size": 2, "K": 1},
            ),
            RunEvent(
                seq=2,
                type="colony_initialized",
                payload={"size": 2, "agents": ["a0", "a1"]},
            ),
        ]
    )
    app = build_app(projection, read_only=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.session.active_tab == "research"
        assert app.session.selected_agent_id == "a0"
        assert app.session.follow_live


async def test_read_only_replay_has_no_cancel_or_detach() -> None:
    projection = RunProjection.from_events(
        [RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r"})]
    )
    app = build_app(projection, read_only=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.can_detach is False
        assert app._runner is None
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.cancelled
        assert app.state.status == "running"


async def test_quit_confirmation_keep_then_detach() -> None:
    gate = asyncio.Event()

    async def runner() -> str:
        await gate.wait()
        return "done"

    projection = RunProjection.from_events(
        [RunEvent(seq=1, type="orchestrator_start", payload={"run_id": "r5"})]
    )
    app = build_app(projection, runner=runner, can_detach=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmQuitScreen)
        assert app.screen.can_detach is True
        await pilot.press("k")
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmQuitScreen)
        assert not app._exit
        await pilot.press("q")
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        assert app._exit
        gate.set()
        await pilot.pause()


async def test_command_palette_filters_commands() -> None:
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert isinstance(app.screen, CommandPaletteScreen)
        app.screen._populate("events")
        await pilot.pause()
        assert "tab_events" in app.screen._actions
        assert "quit" not in app.screen._actions


async def test_render_failure_requests_orderly_shutdown(monkeypatch) -> None:
    from research_explorer.tui import text as render

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        def boom(*args, **kwargs):
            raise RuntimeError("render exploded")

        monkeypatch.setattr(render, "render_dashboard_header", boom)
        app.refresh_view()
        await pilot.pause()
        assert any("ui render failed" in w for w in app.state.warnings)
        assert app._exit


async def test_footer_and_help_come_from_action_registry() -> None:
    from research_explorer.tui.actions import ACTIONS, help_text

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        help_body = help_text()
        for action in ACTIONS:
            key = action.key_display or action.key or ""
            assert key in help_body, action.id
        footer = app.query_one("#footer").plain_text
        for action in ACTIONS:
            if action.footer and not action.narrow:
                assert (action.key_display or action.key) in footer, action.id
