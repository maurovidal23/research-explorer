"""Textual application for the live ACO research pipeline.

The app renders a :class:`~research_explorer.events.models.RunViewState`
produced by the UI-independent projection. It never reads ``Orchestrator``,
``Colony``, ``ExplorerAgent``, or ``GraphStore`` internals; exploration runs in
an asynchronous worker and publishes through an event sink.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, HorizontalScroll, VerticalScroll
from textual.events import Resize
from textual.screen import ModalScreen
from textual.widgets import Static

from research_explorer.events.models import (
    AgentSummary,
    RunEvent,
    RunViewState,
    TimelineEntry,
)
from research_explorer.events.projection import RunProjection
from research_explorer.redaction import redact_secrets
from research_explorer.tui import text as render

NARROW_BREAKPOINT = render.NARROW_BREAKPOINT


class TextPane(Static):
    """A Static widget that retains the last rendered plain text."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("markup", False)
        super().__init__("", *args, **kwargs)
        self.plain_text = ""

    def set_text(self, value: str) -> None:
        self.plain_text = value
        self.update(value)


class HeaderBar(TextPane):
    pass


class AgentTabs(TextPane):
    pass


class TimelinePane(TextPane):
    pass


class DetailPane(TextPane):
    pass


class FooterBar(TextPane):
    pass


class TextViewer(ModalScreen):
    """Scrollable overlay showing a titled body of text."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self.viewer_title = title
        self.viewer_body = body

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="viewer-scroll"):
            yield Static(self.viewer_body, id="viewer-body", markup=False)

    def action_close(self) -> None:
        self.dismiss(None)


class ConfirmQuitScreen(ModalScreen):
    """Quit confirmation; options depend on whether the run can be detached."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("d", "detach", "Detach view"),
        Binding("k", "keep", "Keep running"),
        Binding("c", "cancel_run", "Cancel run"),
        Binding("escape", "close", "Close"),
    ]

    def __init__(self, can_detach: bool) -> None:
        super().__init__()
        self.can_detach = can_detach

    def compose(self) -> ComposeResult:
        if self.can_detach:
            options = "d = detach view    c = cancel run    esc = close"
        else:
            options = "k = keep running    c = cancel run    esc = close"
        yield Static(
            "A run is active. Choose an action:\n\n" + options,
            id="confirm-body",
            markup=False,
        )

    def action_detach(self) -> None:
        self.dismiss("detach")

    def action_keep(self) -> None:
        self.dismiss("keep")

    def action_cancel_run(self) -> None:
        self.dismiss("cancel")

    def action_close(self) -> None:
        self.dismiss(None)


class ResearchTUIApp(App[None]):
    """Live terminal interface for an ACO exploration run."""

    CSS = """
    Screen { layout: vertical; }
    #header { height: auto; padding: 0 1; background: $panel; }
    #tabs-strip { height: 2; background: $boost; }
    #tabs { width: auto; height: 1; padding: 0 1; }
    #body { height: 1fr; }
    #timeline-pane { width: 45%; border: round $primary; }
    #detail-pane { width: 1fr; border: round $secondary; }
    #timeline, #detail { width: 1fr; height: auto; padding: 0 1; }
    #footer { height: auto; min-height: 2; padding: 0 1; background: $panel; }
    #viewer-scroll { padding: 1 2; }
    #confirm-body { padding: 2; }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("left", "prev_agent", "Prev agent"),
        Binding("right", "next_agent", "Next agent"),
        Binding("up", "prev_item", "Prev item"),
        Binding("down", "next_item", "Next item"),
        Binding("enter", "open_details", "Details"),
        Binding("home", "follow_live", "Follow live"),
        Binding("e", "view_evaluation", "Evaluation"),
        Binding("f", "view_frontier", "Frontier"),
        Binding("p", "view_paper", "Paper"),
        Binding("n", "view_narrative", "Narrative"),
        Binding("l", "view_events", "Events"),
        Binding("o", "cycle_event_outcome", "Events by outcome"),
        Binding("r", "view_metadata", "Metadata"),
        Binding("t", "toggle_panes", "Toggle panes"),
        Binding("question_mark", "view_help", "Help"),
        Binding("q", "quit_flow", "Quit"),
        Binding("ctrl+c", "cancel_flow", "Cancel"),
    ]

    def __init__(
        self,
        projection: RunProjection | RunViewState | None = None,
        queue: asyncio.Queue[RunEvent] | None = None,
        runner: Callable[[], Awaitable[Any]] | None = None,
        can_detach: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(projection, RunProjection):
            self.projection = projection
        elif isinstance(projection, RunViewState):
            self.projection = RunProjection(projection)
        else:
            self.projection = RunProjection()
        self._queue = queue
        self._runner = runner
        self.can_detach = can_detach
        self.run_result: Any = None
        self.cancelled = False
        self.narrow = False
        self.active_pane = "both"
        self.selected_agent_index = 0
        self.event_outcome_filter = ""
        self._runner_worker: Any = None
        self._consumer_worker: Any = None
        self.run_finished = False

    # ---- lifecycle --------------------------------------------------------

    @property
    def state(self) -> RunViewState:
        return self.projection.state

    @property
    def selected_agent_id(self) -> str:
        agents = self.state.ordered_agents()
        if not agents:
            return ""
        index = max(0, min(self.selected_agent_index, len(agents) - 1))
        return agents[index].agent_id

    @property
    def selected_entry(self) -> TimelineEntry | None:
        entry = self.projection.selected_entry()
        if entry is not None:
            return entry
        return self.state.default_entry()

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        with HorizontalScroll(id="tabs-strip"):
            yield AgentTabs(id="tabs")
        with Horizontal(id="body"):
            with VerticalScroll(id="timeline-pane"):
                yield TimelinePane(id="timeline")
            with VerticalScroll(id="detail-pane"):
                yield DetailPane(id="detail")
        yield FooterBar(id="footer")

    def on_mount(self) -> None:
        # The consumer and the runner live in distinct Textual worker groups so
        # the exclusive runner never cancels the live-event consumer (TUI-REL-1).
        if self._queue is not None:
            self._consumer_worker = self.run_worker(
                self._consume(), name="live-consumer", group="live-consumer", exclusive=False
            )
        if self._runner is not None:
            self._runner_worker = self.run_worker(
                self._run_runner(), name="research-runner", group="research-runner", exclusive=True
            )
        self.set_interval(1.0, self._tick_elapsed)
        self._apply_narrow(self.size.width < NARROW_BREAKPOINT)
        self.refresh_view()

    def _tick_elapsed(self) -> None:
        """Advance the wall-clock elapsed time while a run is active."""
        if self.state.status not in ("running", "evaluating") or not self.state.started_at:
            return
        try:
            started = datetime.fromisoformat(self.state.started_at)
        except ValueError:
            return
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        if elapsed > self.state.elapsed_seconds:
            self.state.elapsed_seconds = elapsed
            self.refresh_view()

    def on_resize(self, event: Resize) -> None:
        self._apply_narrow(event.size.width < NARROW_BREAKPOINT)
        self.refresh_view()

    async def _consume(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    break
                self.projection.apply(event)
                self.refresh_view()
            finally:
                # Acknowledge every dequeued event so the runner can await
                # ``queue.join()`` as a bounded terminal-state barrier.
                with contextlib.suppress(ValueError):
                    self._queue.task_done()

    async def _run_runner(self) -> None:
        if self._runner is None:
            return
        error: Exception | None = None
        result: Any = None
        try:
            result = await self._runner()
        except asyncio.CancelledError:
            self.cancelled = True
        except Exception as exc:
            error = exc
        finally:
            if self._queue is not None and not self.cancelled:
                # Runner completion waits for every event published before it,
                # so the projected terminal state agrees with the durable trace.
                with contextlib.suppress(Exception):
                    await self._queue.join()
            # The result only becomes ready once the queued events are applied.
            self.run_result = result
            if error is not None and not self._durable_terminal_received():
                self.projection.apply(
                    RunEvent(
                        seq=0,
                        type="run_failed",
                        payload={"error": redact_secrets(str(error))},
                    )
                )
            self.run_finished = True
            self.refresh_view()

    def _durable_terminal_received(self) -> bool:
        """Whether the event stream already delivered a terminal event."""
        if self._terminal():
            return True
        return any(
            event.type in ("run_failed", "run_cancelled", "orchestrator_complete", "no_winner")
            for event in self.state.events
        )

    # ---- rendering --------------------------------------------------------

    def refresh_view(self) -> None:
        try:
            self._render_widgets()
        except Exception as exc:
            self._request_render_shutdown(exc)

    def _render_widgets(self) -> None:
        self.query_one("#header", HeaderBar).set_text(
            render.render_header(self.state)
        )
        self.query_one("#tabs", AgentTabs).set_text(
            render.render_tabs(self.state, self.selected_agent_index)
        )
        self.query_one("#timeline", TimelinePane).set_text(
            render.render_timeline(self.state, self.state.selected_entry_id)
        )
        self.query_one("#detail", DetailPane).set_text(
            render.render_detail(self.state, self.selected_entry)
        )
        footer = render.render_footer(self.state)
        if self.cancelled and not self._terminal():
            footer = (
                "cancellation requested; the current operation is still settling\u2026  "
                + footer
            )
        self.query_one("#footer", FooterBar).set_text(footer)
        self._apply_pane_visibility()

    def _request_render_shutdown(self, exc: Exception) -> None:
        """A rendering failure must request orderly shutdown, not crash the loop."""
        with contextlib.suppress(Exception):
            self.projection.state.warnings.append(
                redact_secrets(f"ui render failed: {exc}")
            )
        with contextlib.suppress(Exception):
            self.exit()

    def _apply_narrow(self, narrow: bool) -> None:
        if narrow and not self.narrow:
            self.active_pane = "timeline"
        elif not narrow:
            self.active_pane = "both"
        self.narrow = narrow

    def _apply_pane_visibility(self) -> None:
        timeline = self.query("#timeline-pane").first()
        detail = self.query("#detail-pane").first()
        if timeline is None or detail is None:
            return
        if not self.narrow:
            timeline.display = True
            detail.display = True
            timeline.styles.width = "45%"
            detail.styles.width = "1fr"
        elif self.active_pane == "detail":
            timeline.display = False
            detail.display = True
            detail.styles.width = "100%"
        else:
            timeline.display = True
            detail.display = False
            timeline.styles.width = "100%"

    # ---- navigation -------------------------------------------------------

    def _agents(self) -> list[AgentSummary]:
        return self.state.ordered_agents()

    def _timeline_index(self) -> int:
        entry = self.selected_entry
        if entry is None:
            return 0
        for index, candidate in enumerate(self.state.timeline):
            if candidate.entry_id == entry.entry_id:
                return index
        return 0

    def action_prev_agent(self) -> None:
        if self._agents():
            self.selected_agent_index = max(0, self.selected_agent_index - 1)
            self.refresh_view()

    def action_next_agent(self) -> None:
        agents = self._agents()
        if agents:
            self.selected_agent_index = min(len(agents) - 1, self.selected_agent_index + 1)
            self.refresh_view()

    def action_prev_item(self) -> None:
        index = self._timeline_index()
        self.projection.select_by_index(max(0, index - 1))
        self.refresh_view()

    def action_next_item(self) -> None:
        index = self._timeline_index()
        self.projection.select_by_index(min(len(self.state.timeline) - 1, index + 1))
        self.refresh_view()

    def action_follow_live(self) -> None:
        self.projection.restore_follow()
        self.refresh_view()

    def action_toggle_panes(self) -> None:
        if not self.narrow:
            return
        self.active_pane = "detail" if self.active_pane != "detail" else "timeline"
        self.refresh_view()

    # ---- focused views ----------------------------------------------------

    def _select_agent_from_entry(self) -> None:
        entry = self.selected_entry
        if entry and entry.agent_id:
            agents = self._agents()
            for index, agent in enumerate(agents):
                if agent.agent_id == entry.agent_id:
                    self.selected_agent_index = index
                    return

    def action_open_details(self) -> None:
        entry = self.selected_entry
        self.push_screen(
            TextViewer("Selected detail", render.render_detail(self.state, entry))
        )

    def action_view_evaluation(self) -> None:
        agent_id = self.selected_agent_id or (self.selected_entry.agent_id if self.selected_entry else "")
        self.push_screen(
            TextViewer("Evaluation", render.render_evaluation(self.state, agent_id))
        )

    def action_view_frontier(self) -> None:
        agent_id = self.selected_agent_id or (self.selected_entry.agent_id if self.selected_entry else "")
        self.push_screen(
            TextViewer("Frontier decision record", render.render_frontier(self.state, agent_id))
        )

    def action_view_paper(self) -> None:
        entry = self.selected_entry
        agent_id = entry.agent_id if entry else self.selected_agent_id
        paper_id = entry.paper_id if entry else ""
        self.push_screen(
            TextViewer("Paper analysis", render.render_paper_full(self.state, agent_id, paper_id))
        )

    def action_view_narrative(self) -> None:
        entry = self.selected_entry
        agent_id = entry.agent_id if entry else self.selected_agent_id
        if not agent_id:
            agent_id = self.state.winner_agent
        self.push_screen(
            TextViewer("Agent narrative", render.render_narrative_full(self.state, agent_id))
        )

    def action_view_events(self) -> None:
        entry = self.selected_entry
        agent_id = entry.agent_id if entry else self.selected_agent_id
        title = "Events"
        if self.event_outcome_filter:
            title = f"Events ({self.event_outcome_filter})"
        self.push_screen(
            TextViewer(
                title,
                render.render_events(self.state, agent_id, self.event_outcome_filter),
            )
        )

    def action_cycle_event_outcome(self) -> None:
        self.event_outcome_filter = render.next_event_outcome(self.event_outcome_filter)
        self.action_view_events()

    def action_view_metadata(self) -> None:
        self.push_screen(TextViewer("Metadata", render.render_metadata(self.state)))

    def action_view_help(self) -> None:
        self.push_screen(TextViewer("Key help", render.render_help()))

    # ---- quit / cancellation ---------------------------------------------

    def _terminal(self) -> bool:
        return self.state.status in ("completed", "cancelled", "failed")

    def action_quit_flow(self) -> None:
        if self.screen is not self.screen_stack[0]:
            self.screen.dismiss(None)
            return
        if self._terminal() or self._runner is None:
            self.exit()
            return
        self.push_screen(ConfirmQuitScreen(self.can_detach), self._on_quit_choice)

    def _on_quit_choice(self, choice: str | None) -> None:
        if choice == "detach":
            self.exit()
        elif choice == "keep":
            self.refresh_view()
        elif choice == "cancel":
            self.action_cancel_flow()

    def action_cancel_flow(self) -> None:
        # A run already terminal in the projection/trace is never cancelled or
        # relabelled; ``q``/Ctrl+C in a terminal state simply exits.
        if self._terminal():
            self.exit()
            return
        if self.run_finished:
            return
        if self._runner_worker is not None and not self._runner_worker.is_finished:
            # Target only the runner; the consumer stays alive to project the
            # durable ``run_cancelled`` event and the persisted cancelled status.
            self.cancelled = True
            self._runner_worker.cancel()
            self.refresh_view()


def build_app(
    projection: RunProjection | RunViewState | None = None,
    queue: asyncio.Queue[RunEvent] | None = None,
    runner: Callable[[], Awaitable[Any]] | None = None,
    can_detach: bool = False,
) -> ResearchTUIApp:
    return ResearchTUIApp(projection=projection, queue=queue, runner=runner, can_detach=can_detach)
