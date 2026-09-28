"""Textual dashboard for the live and replayed ACO research pipeline.

The app renders a :class:`~research_explorer.events.models.RunViewState`
produced by the UI-independent projection. It never reads ``Orchestrator``,
``Colony``, ``ExplorerAgent``, or ``GraphStore`` internals: exploration runs in
an asynchronous worker and publishes through an event sink, while replay
hydrates the same projection from the durable trace store.

The visual grammar is modeled on Convoy: border-only panels on the terminal
background, dim borders with a single accent focus border, semantic status
icons alongside color, and a compact context/header stack.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Input, ListItem, ListView, Markdown, Static, Tab, Tabs

from research_explorer.events.limits import TUI_REFRESH_INTERVAL_SECONDS
from research_explorer.events.models import (
    STATUS_EVALUATING,
    STATUS_RUNNING,
    TERMINAL_STATUSES,
    EventType,
    RunEvent,
    RunViewState,
    TimelineEntry,
)
from research_explorer.events.navigation import (
    NavNode,
    active_agent_id,
    active_leaf_entry,
    build_agent_navigation,
    find_node,
    flatten_navigation,
    parent_ids,
)
from research_explorer.events.projection import RunProjection
from research_explorer.redaction import redact_secrets
from research_explorer.tui import text as render
from research_explorer.tui.actions import ACTIONS, action_by_id, help_text, palette_commands
from research_explorer.tui.session import (
    CONTENT_FOCUS,
    EVENT_AGENT_SCOPE,
    TAB_EVALUATION,
    TAB_EVENTS,
    TAB_PAPER,
    TAB_RESEARCH,
    TREE_FOCUS,
    UISession,
    next_event_agent,
    next_event_outcome,
    next_tab,
)
from research_explorer.tui.theme import (
    BORDER_ACCENT,
    BORDER_DIM,
    COLOR_ACCENT,
    COLOR_BAD,
    COLOR_BG,
    COLOR_CYAN,
    COLOR_GOOD,
    COLOR_MUTED,
    COLOR_TEXT,
    COLOR_WARN,
)

COMPACT_BREAKPOINT = 84

TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.RUN_COMPLETED,
        EventType.RUN_FAILED,
        EventType.RUN_CANCELLED,
        EventType.RUN_INTERRUPTED,
    }
)

RESEARCH_THEME_NAME = "research-explorer"

RESEARCH_THEME = Theme(
    name=RESEARCH_THEME_NAME,
    primary=COLOR_ACCENT,
    secondary=COLOR_CYAN,
    accent=COLOR_ACCENT,
    foreground=COLOR_TEXT,
    background=COLOR_BG,
    surface=COLOR_BG,
    panel=COLOR_BG,
    success=COLOR_GOOD,
    warning=COLOR_WARN,
    error=COLOR_BAD,
    variables={
        "footer-key-foreground": COLOR_ACCENT,
        "footer-description-foreground": COLOR_MUTED,
        "markdown-h1-color": COLOR_ACCENT,
        "markdown-h2-color": COLOR_ACCENT,
        "markdown-h3-color": COLOR_ACCENT,
        "input-selection-background": f"{COLOR_ACCENT} 40%",
    },
)


class TextPane(Static):
    """A Static widget that retains the last rendered text."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("markup", False)
        super().__init__("", *args, **kwargs)
        self.plain_text = ""

    def set_text(self, value: str | Text) -> None:
        self.plain_text = str(value)
        self.update(value)


class HeaderBar(TextPane):
    pass


class ActivityCard(TextPane):
    pass


class StatusBanner(TextPane):
    pass


class FooterBar(TextPane):
    pass


class AgentTree(ListView):
    """Selectable agent-first execution tree."""


class ContentTabBar(Tabs):
    BINDINGS: ClassVar[list[Any]] = []
    can_focus = False


class TextViewer(ModalScreen):
    """Scrollable overlay showing a titled Markdown body."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self.viewer_title = title
        self.viewer_body = body

    def compose(self) -> ComposeResult:
        yield Static(self.viewer_title, id="viewer-title")
        with VerticalScroll(id="viewer-scroll"):
            yield Markdown(self.viewer_body, id="viewer-body")

    def action_close(self) -> None:
        self.dismiss(None)


class CommandPaletteScreen(ModalScreen):
    """Filterable command palette generated from the action registry."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type a command…", id="palette-input")
        yield ListView(id="palette-list")

    def on_mount(self) -> None:
        self._populate("")
        self.query_one("#palette-input", Input).focus()

    def _populate(self, query: str) -> None:
        needle = query.strip().lower()
        listing = self.query_one("#palette-list", ListView)
        listing.clear()
        self._actions: list[str] = []
        for action_id, title, subtitle in palette_commands():
            if needle and needle not in title.lower() and needle not in subtitle.lower():
                continue
            self._actions.append(action_id)
            listing.append(ListItem(Static(f"{title}  ·  {subtitle}", markup=False)))

    def on_input_changed(self, event: Input.Changed) -> None:
        self._populate(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._choose(0)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self._choose(event.list_view.index or 0)

    def _choose(self, index: int) -> None:
        if 0 <= index < getattr(self, "_actions", []).__len__():
            self.dismiss(self._actions[index])
        else:
            self.dismiss(None)

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
    """Convoy-style dashboard for an ACO exploration run."""

    CSS = f"""
    Screen {{ layout: vertical; background: $background; color: {COLOR_TEXT}; }}
    #header {{ height: auto; max-height: 5; padding: 0 1; }}
    #main {{ height: 1fr; }}
    #main.narrow {{ layout: vertical; }}
    #left {{ width: 42%; height: 1fr; border: round {BORDER_DIM}; }}
    #left.narrow {{ width: 100%; height: 34%; max-height: 16; }}
    #left.focused {{ border: round {BORDER_ACCENT}; }}
    #tree-title {{ height: 1; padding: 0 1; color: {COLOR_MUTED}; }}
    #tree {{ height: 1fr; background: transparent; }}
    #tree > ListItem {{ background: transparent; color: {COLOR_TEXT}; }}
    #tree > ListItem.-highlight {{ background: {COLOR_ACCENT} 18%; color: {COLOR_TEXT}; }}
    #tree:focus > ListItem.-highlight {{ background: {COLOR_ACCENT} 28%; color: {COLOR_TEXT}; }}
    #right {{ width: 1fr; height: 1fr; }}
    #activity {{ height: 7; border: round {BORDER_DIM}; padding: 0 1; }}
    #activity.narrow {{ height: 5; }}
    #status-banner {{ height: auto; padding: 0 1; color: {COLOR_MUTED}; }}
    #tabbar {{ height: 2; background: transparent; }}
    Tab {{ background: transparent; color: {COLOR_MUTED}; }}
    Tab.-active {{ background: transparent; color: {COLOR_ACCENT}; text-style: bold; }}
    Tab:hover {{ color: {COLOR_TEXT}; }}
    #content-scroll {{ height: 1fr; border: round {BORDER_DIM}; padding: 0 1; }}
    #content-scroll.focused {{ border: round {BORDER_ACCENT}; }}
    #footer {{ height: auto; max-height: 4; border: round {BORDER_DIM}; padding: 0 1; }}
    TextViewer {{ align: center middle; }}
    #viewer-title {{ height: 1; padding: 0 1; color: {COLOR_ACCENT}; }}
    #viewer-scroll {{ width: 90%; height: 85%; border: round {BORDER_ACCENT}; padding: 1 2; }}
    CommandPaletteScreen {{ align: center middle; }}
    #palette-input {{ width: 70%; }}
    #palette-list {{ width: 70%; height: 12; border: round {BORDER_ACCENT}; }}
    #confirm-body {{ padding: 1 2; border: round {BORDER_ACCENT}; }}
    """

    BINDINGS: ClassVar[list[Any]] = [
        Binding(
            action.key,
            action.method,
            action.label,
            show=False,
            priority=action.id in ("tab_next", "tab_prev"),
        )
        for action in ACTIONS
        if action.key
    ]

    # The dashboard ships its own registry-driven palette; disable Textual's
    # built-in Ctrl+P palette so one action registry remains authoritative.
    ENABLE_COMMAND_PALETTE: ClassVar[bool] = False
    COMMAND_PALETTE_BINDING: ClassVar[str] = ""

    def __init__(
        self,
        projection: RunProjection | RunViewState | None = None,
        queue: asyncio.Queue[RunEvent] | None = None,
        runner: Callable[[], Awaitable[Any]] | None = None,
        can_detach: bool = False,
        read_only: bool = False,
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
        self.read_only = read_only
        self.session = UISession()
        self.run_result: Any = None
        self.cancelled = False
        self.run_finished = False
        self._runner_worker: Any = None
        self._consumer_worker: Any = None
        self._syncing = False
        self._settling = False
        self._tree_built = False
        self._tree_nodes: list[NavNode] = []
        self._rows: list[tuple[int, NavNode]] = []
        self._tree_width = 120
        self._render_interval = TUI_REFRESH_INTERVAL_SECONDS
        self._refresh_timer: Any = None

    # ---- lifecycle --------------------------------------------------------

    @property
    def state(self) -> RunViewState:
        return self.projection.state

    @property
    def narrow(self) -> bool:
        return self.session.narrowed

    @property
    def selected_agent_id(self) -> str:
        return self.session.selected_agent_id

    @property
    def selected_entry(self) -> TimelineEntry | None:
        node = find_node(self._tree_nodes, self.session.selected_node_id)
        if node is not None and node.entry_id:
            return self.state.entry_by_id(node.entry_id)
        return None

    @property
    def agent_roster_text(self) -> str:
        return str(render.render_agent_roster(self.state))

    def tree_row_texts(self) -> list[str]:
        rows: list[str] = []
        for item in self.query("#tree ListItem"):
            static = next(iter(item.query(Static)), None)
            rows.append(str(static.content) if static is not None else "")
        return rows

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Static("Execution tree", id="tree-title")
                yield AgentTree(id="tree")
            with Vertical(id="right"):
                yield ActivityCard(id="activity")
                yield StatusBanner(id="status-banner")
                yield ContentTabBar(
                    Tab("Research", id=TAB_RESEARCH),
                    Tab("Paper", id=TAB_PAPER),
                    Tab("Evaluation", id=TAB_EVALUATION),
                    Tab("Events", id=TAB_EVENTS),
                    id="tabbar",
                )
                with VerticalScroll(id="content-scroll"):
                    yield Markdown("", id="content")
        yield FooterBar(id="footer")

    def on_mount(self) -> None:
        self.register_theme(RESEARCH_THEME)
        self.theme = RESEARCH_THEME_NAME
        if self._queue is not None:
            self._consumer_worker = self.run_worker(
                self._consume(), name="live-consumer", group="live-consumer", exclusive=False
            )
        if self._runner is not None:
            self._runner_worker = self.run_worker(
                self._run_runner(), name="research-runner", group="research-runner", exclusive=True
            )
        self.set_interval(1.0, self._tick_elapsed)
        self._apply_narrow(self.size.width <= COMPACT_BREAKPOINT)
        self.refresh_view()

    def on_unmount(self) -> None:
        self._cancel_scheduled_refresh()
        for worker in (self._consumer_worker, self._runner_worker):
            if worker is not None and not worker.is_finished:
                with contextlib.suppress(Exception):
                    worker.cancel()
        self._consumer_worker = None
        self._runner_worker = None

    def _tick_elapsed(self) -> None:
        if self.read_only:
            return
        if self.state.status not in (STATUS_RUNNING, STATUS_EVALUATING) or not self.state.started_at:
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
        self._apply_narrow(event.size.width <= COMPACT_BREAKPOINT)
        self.refresh_view()

    def _apply_narrow(self, narrow: bool) -> None:
        if narrow == self.session.narrowed:
            return
        self.session.narrowed = narrow
        main = self.query("#main").first()
        left = self.query("#left").first()
        activity = self.query("#activity").first()
        if main is not None:
            main.set_class(narrow, "narrow")
        if left is not None:
            left.set_class(narrow, "narrow")
        if activity is not None:
            activity.set_class(narrow, "narrow")

    async def _consume(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    self.refresh_view()
                    break
                self.projection.apply(event)
                if event.canonical_type() in TERMINAL_EVENT_TYPES:
                    self.refresh_view()
                else:
                    self._schedule_refresh()
            finally:
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
                with contextlib.suppress(Exception):
                    await self._queue.join()
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
        if self._terminal():
            return True
        return any(
            event.canonical_type() in TERMINAL_EVENT_TYPES
            for event in self.state.events
        )

    # ---- rendering --------------------------------------------------------

    def refresh_view(self) -> None:
        """Render the newest fully applied projection immediately.

        Any pending scheduled refresh is collapsed into this call so a burst
        never renders an intermediate snapshot after an explicit refresh.
        """
        self._cancel_scheduled_refresh()
        try:
            self._refresh()
        except Exception as exc:
            self._request_render_shutdown(exc)

    def _schedule_refresh(self) -> None:
        if self._refresh_timer is not None:
            return
        self._refresh_timer = self.set_timer(
            self._render_interval, self._flush_scheduled_refresh
        )

    def _cancel_scheduled_refresh(self) -> None:
        timer = self._refresh_timer
        self._refresh_timer = None
        if timer is not None:
            with contextlib.suppress(Exception):
                timer.stop()

    def _flush_scheduled_refresh(self) -> None:
        self._refresh_timer = None
        try:
            self._refresh()
        except Exception as exc:
            self._request_render_shutdown(exc)

    def _refresh(self) -> None:
        state = self.state
        width = self.size.width or 120
        self.query_one("#header", HeaderBar).set_text(
            render.render_dashboard_header(
                state, max(20, width - 2), self.session.follow_live
            )
        )

        roots = build_agent_navigation(state)
        self._tree_nodes = roots
        self._ensure_selection(roots)
        self._render_tree(roots)
        self.query_one("#tree-title", Static).update(
            render.render_tree_title(len(self._rows))
        )

        info_width = width if self.narrow else max(30, width // 2)
        self.query_one("#activity", ActivityCard).set_text(
            render.render_activity_card(
                state,
                self.session.selected_agent_id,
                self.selected_entry,
                info_width,
                self.session.follow_live,
                compact=self.narrow,
            )
        )
        self._render_status_banner()
        self._sync_tabbar()
        self.query_one("#content", Markdown).update(
            render.render_tab_body(state, self.session, self.selected_entry)
        )
        self.query_one("#footer", FooterBar).set_text(
            render.render_footer_text(state, self.session, self._settling)
        )
        self._apply_focus_classes()

    def _render_status_banner(self) -> None:
        state = self.state
        banner = self.query_one("#status-banner", StatusBanner)
        if self._terminal():
            banner.set_text(render.render_status_banner(state))
            banner.display = True
        else:
            banner.display = False

    def _ensure_selection(self, roots: list[NavNode]) -> None:
        if not roots:
            self.session.selected_agent_id = ""
            self.session.selected_node_id = None
            return
        agent_ids = {node.agent_id for node in roots}
        if self.session.selected_agent_id not in agent_ids:
            self.session.selected_agent_id = active_agent_id(self.state) or roots[0].agent_id
        if self.session.follow_live:
            self._follow(roots)
        elif find_node(roots, self.session.selected_node_id) is None:
            self.session.selected_node_id = f"agent:{self.session.selected_agent_id}"

    def _follow(self, roots: list[NavNode]) -> None:
        agent_id = active_agent_id(self.state)
        if agent_id:
            self.session.selected_agent_id = agent_id
        node_id = active_leaf_entry(self.state, agent_id) or f"agent:{agent_id}"
        self.session.selected_node_id = node_id
        parents = parent_ids(roots)
        cursor = node_id
        guard = 0
        while cursor in parents and guard < 32:
            cursor = parents[cursor]
            self.session.expanded.add(cursor)
            guard += 1

    def _render_tree(self, roots: list[NavNode]) -> None:
        self._rows = flatten_navigation(roots, self.session.expanded)
        tree = self.query_one("#tree", AgentTree)
        self._tree_width = max(20, tree.content_size.width or self.size.width)
        content = tuple((n.node_id, n.label, n.status) for _, n in self._rows)
        if content != self.session.tree_signature or not self._tree_built:
            self.session.tree_signature = content
            self._tree_built = True
            self._build_tree_items(tree)
        else:
            self._update_tree_items(tree)
        self._select_tree_row(tree)

    def _build_tree_items(self, tree: AgentTree) -> None:
        items = [
            ListItem(Static(self._tree_label(node, depth), markup=False))
            for depth, node in self._rows
        ]
        tree.clear()
        tree.extend(items)

    def _update_tree_items(self, tree: AgentTree) -> None:
        for index, (depth, node) in enumerate(self._rows):
            if index >= len(tree.children):
                break
            item = tree.children[index]
            static = next(iter(item.query(Static)), None)
            if static is not None:
                static.update(self._tree_label(node, depth))

    def _tree_label(self, node: NavNode, depth: int) -> Text:
        return render.render_tree_label(
            self.state,
            node,
            selected=node.node_id == self.session.selected_node_id,
            expanded=node.node_id in self.session.expanded,
            depth=depth,
            width=self._tree_width,
        )

    def _select_tree_row(self, tree: AgentTree) -> None:
        index = 0
        for candidate, (_, node) in enumerate(self._rows):
            if node.node_id == self.session.selected_node_id:
                index = candidate
                break
        if not self._rows:
            return
        self._syncing = True
        try:
            tree.index = index
        except Exception:
            pass
        finally:
            self._syncing = False

    def _sync_tabbar(self) -> None:
        tabbar = self.query_one("#tabbar", ContentTabBar)
        if tabbar.active != self.session.active_tab:
            tabbar.active = self.session.active_tab

    def _apply_focus_classes(self) -> None:
        tree_focused = self.session.focus == TREE_FOCUS
        left = self.query("#left").first()
        content = self.query("#content-scroll").first()
        if left is not None:
            left.set_class(tree_focused, "focused")
        if content is not None:
            content.set_class(not tree_focused, "focused")

    def _request_render_shutdown(self, exc: Exception) -> None:
        with contextlib.suppress(Exception):
            self.projection.state.warnings.append(
                redact_secrets(f"ui render failed: {exc}")
            )
        with contextlib.suppress(Exception):
            self.exit()

    # ---- events -----------------------------------------------------------

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if self._syncing:
            return
        index = event.list_view.index
        if index is None or index >= len(self._rows):
            return
        _, node = self._rows[index]
        if node.node_id == self.session.selected_node_id:
            return
        self._on_tree_select(node)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = event.list_view.index
        if index is None or index >= len(self._rows):
            return
        _, node = self._rows[index]
        if node.children:
            if node.node_id in self.session.expanded:
                self.session.expanded.discard(node.node_id)
            else:
                self.session.expanded.add(node.node_id)
            self.refresh_view()

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        tab_id = event.tab.id
        if tab_id and tab_id != self.session.active_tab:
            self.session.select_tab(tab_id)
            self.refresh_view()

    def _on_tree_select(self, node: NavNode) -> None:
        self.session.selected_node_id = node.node_id
        if node.agent_id:
            self.session.selected_agent_id = node.agent_id
        self.session.follow_live = False
        self.refresh_view()

    # ---- navigation -------------------------------------------------------

    def action_focus_left(self) -> None:
        self.session.focus = TREE_FOCUS
        with contextlib.suppress(Exception):
            self.query_one("#tree", AgentTree).focus()
        self.refresh_view()

    def action_focus_right(self) -> None:
        self.session.focus = CONTENT_FOCUS
        with contextlib.suppress(Exception):
            self.query_one("#content-scroll", VerticalScroll).focus()
        self.refresh_view()

    def action_follow_live(self) -> None:
        self.session.follow_live = True
        self.session.event_agent = EVENT_AGENT_SCOPE
        self.refresh_view()

    def action_toggle_panes(self) -> None:
        if not self.session.narrowed:
            return
        if self.session.focus == TREE_FOCUS:
            self.action_focus_right()
        else:
            self.action_focus_left()

    def action_select_tab(self, tab: str) -> None:
        self.session.select_tab(tab)
        self.refresh_view()

    def action_tab_research(self) -> None:
        self.action_select_tab(TAB_RESEARCH)

    def action_tab_paper(self) -> None:
        self.action_select_tab(TAB_PAPER)

    def action_tab_evaluation(self) -> None:
        self.action_select_tab(TAB_EVALUATION)

    def action_tab_events(self) -> None:
        self.action_select_tab(TAB_EVENTS)

    def action_tab_next(self) -> None:
        if self.screen is not self.screen_stack[0]:
            return
        self.action_select_tab(next_tab(self.session.active_tab, 1))

    def action_tab_prev(self) -> None:
        if self.screen is not self.screen_stack[0]:
            return
        self.action_select_tab(next_tab(self.session.active_tab, -1))

    # ---- focused views ----------------------------------------------------

    def _open_viewer(self, title: str, body: str) -> None:
        self.push_screen(TextViewer(title, body))

    def action_open_reader(self) -> None:
        title, body = render.render_reader(self.state, self.session, self.selected_entry)
        self._open_viewer(title, body)

    def action_view_evaluation(self) -> None:
        agent_id = self._scope_agent()
        self._open_viewer("Evaluation", render.render_evaluation_tab(self.state, agent_id))

    def action_view_frontier(self) -> None:
        agent_id = self._scope_agent()
        self._open_viewer("Frontier", render.render_frontier(self.state, agent_id))

    def action_view_paper(self) -> None:
        entry = self.selected_entry
        agent_id = entry.agent_id if entry else self._scope_agent()
        self._open_viewer(
            "Paper", render.render_paper_full(self.state, agent_id, entry.paper_id if entry else "")
        )

    def action_view_narrative(self) -> None:
        agent_id = self._scope_agent()
        self._open_viewer("Narrative", render.render_narrative_full(self.state, agent_id))

    def action_view_events(self) -> None:
        scope = self._events_scope_agent()
        body = render.render_events_tab(self.state, scope, self.session.event_outcome)
        self._open_viewer("Events", body)

    def action_view_metadata(self) -> None:
        self._open_viewer("Metadata", render.render_metadata(self.state))

    def action_cycle_event_outcome(self) -> None:
        self.session.event_outcome = next_event_outcome(self.session.event_outcome)
        self.session.event_page = 0
        self.action_select_tab(TAB_EVENTS)

    def action_cycle_event_agent(self) -> None:
        self.session.event_agent = next_event_agent(self.session.event_agent)
        self.session.event_page = 0
        self.action_select_tab(TAB_EVENTS)

    def action_events_older(self) -> None:
        self.session.event_page += 1
        self.action_select_tab(TAB_EVENTS)

    def action_events_newer(self) -> None:
        self.session.event_page = max(0, self.session.event_page - 1)
        self.action_select_tab(TAB_EVENTS)

    def _scope_agent(self) -> str:
        entry = self.selected_entry
        if entry is not None and entry.agent_id:
            return entry.agent_id
        return self.session.selected_agent_id or self.state.winner_agent

    def _events_scope_agent(self) -> str:
        return render.events_scope_agent(self.state, self.session)

    def action_show_palette(self) -> None:
        self.push_screen(CommandPaletteScreen(), self._run_palette_action)

    def _run_palette_action(self, action_id: str | None) -> None:
        if action_id is None:
            return
        action = action_by_id(action_id)
        if action is None:
            return
        method = getattr(self, f"action_{action.method}", None)
        if callable(method):
            method()

    def action_show_help(self) -> None:
        self._open_viewer("Help", help_text())

    # ---- quit / cancellation ---------------------------------------------

    def _terminal(self) -> bool:
        return self.state.status in TERMINAL_STATUSES

    def action_close_view(self) -> None:
        if self.screen is not self.screen_stack[0]:
            self.screen.dismiss(None)
            return
        self.action_focus_left()

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
        elif choice == "cancel":
            self.action_cancel_flow()
        else:
            self.refresh_view()

    def action_cancel_flow(self) -> None:
        if self._terminal():
            self.exit()
            return
        if self.run_finished:
            return
        if self._runner_worker is not None and not self._runner_worker.is_finished:
            self.cancelled = True
            self._settling = True
            self._runner_worker.cancel()
            self.refresh_view()


def build_app(
    projection: RunProjection | RunViewState | None = None,
    queue: asyncio.Queue[RunEvent] | None = None,
    runner: Callable[[], Awaitable[Any]] | None = None,
    can_detach: bool = False,
    read_only: bool = False,
) -> ResearchTUIApp:
    return ResearchTUIApp(
        projection=projection,
        queue=queue,
        runner=runner,
        can_detach=can_detach,
        read_only=read_only,
    )
