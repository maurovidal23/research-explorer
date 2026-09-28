"""Pure text renderers for the run view state.

These helpers contain no Textual imports so projection and rendering can be
unit-tested without a terminal. The Textual widgets in
:mod:`research_explorer.tui.app` only place these strings on screen.
"""

from __future__ import annotations

from rich.text import Text

from research_explorer.events.limits import CANDIDATE_TOP_ALTERNATIVES, EVENT_PAGE_SIZE
from research_explorer.events.models import (
    AGENT_ACTIVE,
    AGENT_COMPLETED,
    AGENT_EVALUATING,
    AGENT_EXHAUSTED,
    AGENT_FAILED,
    EVAL_FAILED,
    EVAL_PENDING,
    EVAL_SKIPPED,
    NODE_COMPLETED,
    NODE_FAILED,
    NODE_PENDING,
    NODE_SKIPPED,
    OUTCOME_DEGRADED,
    PHASE_DECISION,
    PHASE_EVALUATION,
    PHASE_RESEARCH,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_EVALUATING,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    TERMINAL_STATUSES,
    AgentSummary,
    EventType,
    RunEvent,
    RunViewState,
    TimelineEntry,
)
from research_explorer.events.navigation import (
    AGENT_NODE,
    DEBUG_NODE,
    FINAL_NODE,
    PHASE_NODE,
    SETUP_NODE,
    WAVE_NODE,
    NavNode,
)
from research_explorer.redaction import redact_secrets
from research_explorer.tui import theme
from research_explorer.tui.actions import footer_hints
from research_explorer.tui.session import (
    EVENT_AGENT_ALL,
    EVENT_AGENT_WINNER,
    PRIMARY_TABS,
    TAB_EVALUATION,
    TAB_EVENTS,
    TAB_PAPER,
    TAB_RESEARCH,
    UISession,
    tab_label,
)

UNAVAILABLE = "unavailable"
DASH = "\u2014"


def ellipsize(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "\u2026"


def format_duration(seconds: float) -> str:
    total = int(max(0.0, seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


_STATUS_MARK = {
    "initializing": "init",
    AGENT_ACTIVE: "active",
    AGENT_EVALUATING: "evaluating",
    "waiting": "waiting",
    AGENT_EXHAUSTED: "exhausted",
    AGENT_COMPLETED: "done",
    AGENT_FAILED: "failed",
    "running": "running",
    STATUS_EVALUATING: "evaluating",
    "converged": "converged",
    STATUS_COMPLETED: "completed",
    STATUS_CANCELLED: "cancelled",
    STATUS_FAILED: "failed",
    STATUS_INTERRUPTED: "interrupted",
}

_CASTE_LABEL = {
    "fundaciones": "foundations",
    "fundamentos": "foundations",
    "impacto": "impact",
    "mixto": "mixed",
}


def caste_label(caste: str) -> str:
    return _CASTE_LABEL.get(caste, caste or "mixed")


def status_mark(status: str) -> str:
    return _STATUS_MARK.get(status, status or "unknown")


def status_label(state: RunViewState) -> str:
    """One status token shared by the header and footer.

    A completed-but-degraded run keeps its terminal status and annotates the
    outcome in parentheses, rather than concatenating with ``/`` which this UI
    reserves for ratios (``fetch 3/100``, ``5s/1m00s``).
    """
    label = status_mark(state.status)
    if state.outcome == OUTCOME_DEGRADED:
        return f"{label} ({OUTCOME_DEGRADED})"
    return label


def phase_progress_text(state: RunViewState) -> str:
    """Compact ``done/total agents`` progress for the current wave phase."""
    wave = state.current_wave
    phase = state.current_phase
    if not wave or not phase:
        return DASH
    record = state.phase(wave, phase)
    if record is None or not record.selected:
        return DASH if not record else f"0/{len(record.selected)} agents"
    terminal = set(record.completed) | set(record.failed) | set(record.skipped)
    if phase == PHASE_RESEARCH and not terminal:
        terminal = {
            agent_id
            for agent_id in record.selected
            if state.agents.get(agent_id) is not None
            and state.agents[agent_id].research_status
            in (NODE_COMPLETED, NODE_FAILED, NODE_SKIPPED)
        }
    return f"{len(terminal)}/{len(record.selected)} agents"


def phase_summary_text(state: RunViewState, wave: int, phase: str) -> str:
    record = state.phase(wave, phase)
    if record is None:
        return DASH
    if phase == PHASE_RESEARCH:
        return f"{len(record.completed)} done · {record.papers_integrated} papers"
    if phase == PHASE_EVALUATION:
        best = f"best Q {record.best_q:.3f}" if record.best_q is not None else "best Q --"
        return (
            f"{len(record.completed)} complete / {len(record.skipped)} skipped / "
            f"{len(record.failed)} failed · {best}"
        )
    return f"leader {record.leader or DASH} · Δ{record.q_delta:+.3f}"


_ERROR_OUTCOMES = frozenset({"fatal", "error", "failed"})
_WARNING_OUTCOMES = frozenset({"transient", "warning", "skipped", "degraded"})


def classify_event_outcome(event: RunEvent) -> str:
    """Bucket an event into a severity-like outcome for filtering."""
    payload = event.payload or {}
    classification = str(payload.get("classification", "")).lower()
    if classification in _ERROR_OUTCOMES:
        return "error"
    if classification in _WARNING_OUTCOMES:
        return "warning"
    name = str(event.type).lower()
    if "failed" in name or name in ("run_failed", "id_title_mismatch", "trace_write_failed"):
        return "error"
    if "skipped" in name or name == "warning" or name in ("provider_failure", "metadata_transit"):
        return "warning"
    return "info"


def render_header(state: RunViewState) -> str:
    tokens = UNAVAILABLE
    if state.token_usage is not None:
        tokens = str(state.token_usage)
    elif state.cost is not None:
        tokens = f"cost {state.cost:.4f}"
    winner_agent = state.agents.get(state.winner_agent) if state.winner_agent else None
    winner = winner_agent.label if winner_agent is not None else (state.winner_agent or DASH)
    status_text = status_label(state)
    row1 = (
        f"run {state.run_id or DASH}  [{status_text}]  "
        f"elapsed {format_duration(state.elapsed_seconds)}  "
        f"fetch {state.fetches_used}/{state.max_fetches}  "
        f"wave {state.current_wave}  turn {state.current_turn}  "
        f"best {state.best_quality:.3f}  winner {winner}  tokens {tokens}"
    )
    row2 = (
        f"seed {state.seed_paper_id or DASH}  "
        f"pipeline {state.pipeline}  "
        f"explorer {state.explorer_model or UNAVAILABLE}  "
        f"judge {state.judge_model or UNAVAILABLE}  "
        f"colony {state.colony_size}  slots {state.max_concurrent}  "
        f"papers/eval {state.k_per_turn}"
    )
    query = f"Q: {state.query}" if state.query else f"Q: {DASH}"
    return row1 + "\n" + row2 + "\n" + query


def render_budget(state: RunViewState) -> str:
    max_fetches = max(1, state.max_fetches)
    filled = round(20 * min(1.0, state.fetches_used / max_fetches))
    bar = "=" * filled + "." * (20 - filled)
    fetch_line = f"[{bar}] {state.fetches_used}/{state.max_fetches} fetches"
    if state.budget_type == "time":
        primary = (
            f"time {format_duration(state.elapsed_seconds)}"
            f"/{format_duration(float(state.max_time_seconds))}"
        )
        return f"time stopping rule: {primary}  (fetch work {state.fetches_used}/{state.max_fetches})"
    if state.budget_type == "convergence":
        return (
            f"convergence stopping rule: best Q {state.best_quality:.3f}  "
            f"(fetch work {state.fetches_used}/{state.max_fetches})"
        )
    return fetch_line


_TIMELINE_GLYPH = {
    "wave": "",
    "turn": "\u251c\u2500 ",
    "paper": "\u2502  ",
    "discovery": "\u2502  ",
    "frontier": "\u2502  ",
    "reference_mapping": "\u2502  ",
    "evaluation": "\u2502  ",
    "warning": "! ",
}


def render_timeline(state: RunViewState, selected_entry_id: str | None) -> str:
    lines = [render_budget(state), ""]
    if not state.timeline:
        lines.append("timeline: waiting for the first wave")
        return "\n".join(lines)
    for entry in state.timeline:
        depth = 0
        parent = entry.parent_id
        guard = 0
        while parent and guard < 8:
            depth += 1
            parent_entry = state.entry_by_id(parent)
            parent = parent_entry.parent_id if parent_entry else None
            guard += 1
        marker = ">" if entry.entry_id == selected_entry_id else " "
        indent = "  " * depth
        glyph = _TIMELINE_GLYPH.get(entry.kind, "  ")
        if entry.kind == "wave" and lines[-1] != "":
            lines.append("")
        lines.append(
            f"{marker}{indent}{glyph}{status_mark(entry.status):<10} {entry.label}"
        )
    if state.terminal_reason:
        lines.append("")
        lines.append(f"terminal: {status_label(state)} {DASH} {state.terminal_reason}")
    return "\n".join(lines)


def _agent_for(state: RunViewState, entry: TimelineEntry | None) -> AgentSummary | None:
    if entry is None:
        return None
    return state.agents.get(entry.agent_id)


def _paper_entry_for(state: RunViewState, agent_id: str) -> TimelineEntry | None:
    """Most recent recorded paper step for an agent, used to fill compact views."""
    for entry in reversed(state.timeline):
        if entry.kind != "paper" or not entry.paper_id:
            continue
        if agent_id and entry.agent_id != agent_id:
            continue
        return entry
    return None


def _activity_meta(
    entry: TimelineEntry | None,
    paper_entry: TimelineEntry | None,
    agent: AgentSummary | None,
) -> dict:
    """Merge persisted paper metadata for the current-activity section."""
    meta: dict = {}
    if paper_entry is not None:
        meta.update(paper_entry.detail)
    if entry is not None and entry.kind == "paper":
        meta.update(entry.detail)
    if agent is not None:
        meta.setdefault("paper_id", agent.current_paper_id)
        meta.setdefault("title", agent.current_paper_title)
        meta.setdefault("year", agent.current_paper_year)
        meta.setdefault("authors", agent.current_paper_authors)
        meta.setdefault("source", agent.current_paper_source)
    return meta


def _format_optional_duration(value: float | None) -> str:
    if value is None:
        return UNAVAILABLE
    return format_duration(float(value))


def render_detail(state: RunViewState, entry: TimelineEntry | None) -> str:
    if state.follow_live:
        head = "live head"
    else:
        head = f"historical snapshot (seq {entry.seq if entry else DASH})"
    lines = [f"=== Selected agent detail ({head}) ==="]

    agent = _agent_for(state, entry)
    lines.append("")
    lines.append("-- Current activity --")
    if agent is None:
        lines.append("no agent selected (waiting for colony initialization)")
    else:
        lines.append(
            f"agent {agent.label} ({agent.agent_id})  caste {caste_label(agent.caste)}  "
            f"state {status_mark(agent.status)}"
        )
        lines.append(f"action {entry.kind if entry else 'idle'}: {entry.label if entry else DASH}")
        paper_entry = _paper_entry_for(state, agent.agent_id)
        meta = _activity_meta(entry, paper_entry, agent)
        title = str(meta.get("title") or agent.current_paper_title or "")
        paper_id = str(meta.get("paper_id") or agent.current_paper_id or "")
        if title or paper_id:
            lines.append(f"paper {title or UNAVAILABLE}")
            lines.append(f"id {paper_id or UNAVAILABLE}")
        else:
            lines.append("paper unavailable (no paper read yet)")
        year = meta.get("year")
        lines.append(f"year {year if year is not None else UNAVAILABLE}")
        authors = meta.get("authors") or []
        lines.append(f"authors {_format_field(authors) or UNAVAILABLE}")
        source = str(meta.get("source") or "")
        lines.append(f"source {source or UNAVAILABLE}")
        mode = str(meta.get("mode") or "")
        provider = str(meta.get("provider") or "")
        lines.append(f"mode {mode or UNAVAILABLE}")
        lines.append(f"provider {provider or UNAVAILABLE}")
        model = state.current_operation_model or (
            state.explorer_model if state.current_operation else ""
        )
        lines.append(
            f"operation {state.current_operation or UNAVAILABLE}  "
            f"model {model or UNAVAILABLE}  "
            f"elapsed {_format_optional_duration(state.operation_elapsed_seconds)}"
        )
        lines.append(f"budget remaining {agent.budget}  frontier {agent.frontier}")

    lines.append("")
    lines.append("-- Selection rationale --")
    lines.extend(_render_rationale(state, entry))

    lines.append("")
    lines.append("-- Paper contribution --")
    lines.extend(_render_paper(state, entry, agent.agent_id if agent else ""))

    lines.append("")
    lines.append("-- Research line --")
    lines.extend(_render_narrative(state, entry))
    return "\n".join(lines)


def _render_rationale(state: RunViewState, entry: TimelineEntry | None) -> list[str]:
    agent_id = entry.agent_id if entry else ""
    selection = _latest_selection(state, agent_id, entry)
    if selection is None:
        candidates = [
            c for c in state.candidate_scores if entry is None or c.paper_id == entry.paper_id
        ]
        if not candidates:
            return ["no recorded decision for the selected step"]
        lines = ["no selection record; scored candidates:"]
        for candidate in sorted(candidates, key=lambda c: c.eta, reverse=True)[:10]:
            lines.append(
                f"  {ellipsize(candidate.paper_id, 40)} eta={candidate.eta:.4f} "
                f"mode={candidate.mode or UNAVAILABLE}"
            )
        return lines
    comps = selection.eta
    score = _eta_for(state, selection.paper_id)
    lines = [
        f"chosen {selection.paper_id}  probability={selection.probability:.4f}  "
        f"branch={'explore' if selection.epsilon_branch else 'exploit'}",
        f"final weight {selection.final_weight:.5f}  "
        f"tau^alpha term base={selection.tau:.4f}  eta={comps:.4f}  dir={selection.dir_modifier:.4f}",
        f"alpha={selection.alpha}  beta={selection.beta}  caste={caste_label(selection.caste)}",
        f"pheromone {selection.tau:.4f}  direction weight {selection.dir_modifier:.4f}",
    ]
    if score is not None:
        lines.append(
            "eta components: "
            f"sim={score.components.get('sim', UNAVAILABLE)} "
            f"citations={score.components.get('citations', UNAVAILABLE)} "
            f"recency={score.components.get('recency', UNAVAILABLE)} "
            f"confidence={score.components.get('confidence', UNAVAILABLE)} "
            f"llm={score.components.get('llm', UNAVAILABLE)}"
        )
    else:
        lines.append("eta components: unavailable")
    alternatives = sorted(state.selections, key=lambda s: s.probability, reverse=True)[:5]
    if len(alternatives) > 1:
        lines.append("considered alternatives:")
        for alt in alternatives:
            chosen = "chosen" if alt.chosen else "rejected"
            lines.append(
                f"  {ellipsize(alt.paper_id, 40)} p={alt.probability:.4f} "
                f"eta={alt.eta:.4f} [{chosen}]"
            )
    return lines


def _latest_selection(state: RunViewState, agent_id: str, entry: TimelineEntry | None):
    relevant = [s for s in state.selections if not agent_id or s.agent_id == agent_id]
    if entry is not None and entry.paper_id:
        for selection in reversed(relevant):
            if selection.paper_id == entry.paper_id:
                return selection
    return relevant[-1] if relevant else None


def _eta_for(state: RunViewState, paper_id: str):
    for candidate in reversed(state.candidate_scores):
        if candidate.paper_id == paper_id:
            return candidate
    return None


def _render_paper(
    state: RunViewState, entry: TimelineEntry | None, agent_id: str = ""
) -> list[str]:
    resolved_agent = agent_id or (entry.agent_id if entry else "")
    paper_id = entry.paper_id if entry else ""
    if not paper_id:
        paper_entry = _paper_entry_for(state, resolved_agent)
        if paper_entry is not None:
            paper_id = paper_entry.paper_id
            resolved_agent = resolved_agent or paper_entry.agent_id
    if not paper_id:
        return ["no paper selected"]
    analysis = state.paper_analyses.get(resolved_agent, {}).get(paper_id)
    if not analysis:
        return [f"no structured analysis recorded for {paper_id}"]
    lines: list[str] = []
    fields = (
        ("summary", "summary"),
        ("key_concepts", "key concepts"),
        ("methods", "methods"),
        ("findings", "findings"),
        ("relevance", "relevance"),
        ("limitations", "limitations"),
        ("key_references", "key references"),
    )
    for key, label in fields:
        value = _format_field(analysis.get(key))
        if value is None:
            lines.append(f"{label}: unavailable")
        else:
            lines.append(f"{label}: {value}")
    return lines


def _format_field(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value if v is not None]
        return ", ".join(items) if items else None
    if isinstance(value, dict):
        parts = [f"{k}={_format_field(v) or UNAVAILABLE}" for k, v in value.items()]
        return "; ".join(parts) if parts else None
    return str(value)


def _render_narrative(state: RunViewState, entry: TimelineEntry | None) -> list[str]:
    agent_id = entry.agent_id if entry else ""
    if not agent_id and state.winner_agent:
        agent_id = state.winner_agent
    narrative = state.narratives.get(agent_id, "")
    if not narrative:
        return ["no narrative recorded yet"]
    lines: list[str] = []
    detail = entry.detail if entry else {}
    delta = detail.get("delta_narrative") or detail.get("changed_understanding")
    if delta:
        lines.append(f"delta: {delta}")
        lines.append("")
    for paragraph in narrative.splitlines():
        lines.append(paragraph)
    return lines


def render_frontier(state: RunViewState, agent_id: str) -> str:
    candidates = [c for c in state.candidate_scores if not agent_id or c.agent_id == agent_id]
    if not candidates:
        return (
            "=== Frontier decision record ===\n\n"
            f"no recorded frontier candidates for {agent_id or 'the selected agent'}"
        )
    chosen = {
        s.paper_id: s for s in state.selections if not agent_id or s.agent_id == agent_id
    }
    count = len(candidates)
    lines = [f"=== Frontier decision record ({count} candidate{'s' if count != 1 else ''}) ==="]
    for candidate in sorted(candidates, key=lambda c: c.eta, reverse=True):
        selection = chosen.get(candidate.paper_id)
        status = "chosen" if selection is not None else "candidate"
        lines.append(
            f"{ellipsize(candidate.paper_id, 44)} eta={candidate.eta:.4f} "
            f"mode={candidate.mode or DASH} source={ellipsize(candidate.source, 24) or DASH} "
            f"provider={candidate.provider or DASH} [{status}]"
        )
        components = candidate.components
        if components:
            rendered = "  ".join(
                f"{key}={components[key]}"
                for key in ("sim", "citations", "recency", "confidence", "llm")
                if key in components
            )
            if rendered:
                lines.append(f"    eta components: {rendered}")
        if selection is not None:
            branch = "explore" if selection.epsilon_branch else "exploit"
            lines.append(
                f"    pheromone={selection.tau:.4f} weight={selection.final_weight:.5f} "
                f"dir={selection.dir_modifier:.4f} probability={selection.probability:.4f} "
                f"branch={branch}"
            )
    return "\n".join(lines)


def render_paper_full(state: RunViewState, agent_id: str, paper_id: str) -> str:
    analyses = state.paper_analyses.get(agent_id, {})
    analysis = analyses.get(paper_id)
    if analysis is None:
        return (
            "=== Paper analysis ===\n\n"
            f"no structured analysis recorded for {paper_id or 'the selected paper'}"
        )
    lines = [f"=== Paper analysis: {paper_id} ==="]
    lines.extend(_render_paper_fields(analysis))
    return "\n".join(lines)


def _render_paper_fields(analysis: dict) -> list[str]:
    lines: list[str] = []
    for key in (
        "summary",
        "key_concepts",
        "methods",
        "findings",
        "relevance",
        "limitations",
        "key_references",
    ):
        lines.append(f"{key}: {_format_field(analysis.get(key)) or UNAVAILABLE}")
    for key, value in analysis.items():
        if key in {
            "summary",
            "key_concepts",
            "methods",
            "findings",
            "relevance",
            "limitations",
            "key_references",
        }:
            continue
        lines.append(f"{key}: {_format_field(value) or UNAVAILABLE}")
    return lines


def render_narrative_full(state: RunViewState, agent_id: str) -> str:
    narrative = state.narratives.get(agent_id, "")
    if not narrative:
        return (
            "=== Agent narrative ===\n\n"
            f"no research line recorded yet for {agent_id or 'the selected agent'}"
        )
    return f"=== Narrative: {agent_id} ===\n\n{narrative}"


def render_events(state: RunViewState, agent_id: str = "", outcome: str = "") -> str:
    events = [
        event for event in state.events if not agent_id or agent_id in str(event.payload)
    ]
    if outcome:
        events = [event for event in events if classify_event_outcome(event) == outcome]
    filters: list[str] = []
    if agent_id:
        filters.append(f"filtered to {agent_id}")
    if outcome:
        filters.append(f"outcome={outcome}")
    scope = f" {', '.join(filters)}" if filters else ""
    lines: list[str] = [f"=== Events ({len(events)}{scope}) ==="]
    if not events:
        lines.append("no events recorded for this filter")
        return "\n".join(lines)
    for event in events:
        summary = " ".join(f"{k}={v}" for k, v in event.payload.items() if k != "content")[:160]
        lines.append(f"[{event.seq:>4}] {event.type:<32} {redact_secrets(summary)}")
    return "\n".join(lines)


def last_durable_event(state: RunViewState) -> RunEvent | None:
    """Newest durable event in the bounded window, skipping a synthetic interrupt."""
    for event in reversed(state.events):
        if event.canonical_type() == EventType.RUN_INTERRUPTED:
            continue
        return event
    return None


def render_metadata(state: RunViewState) -> str:
    last = last_durable_event(state)
    last_activity = last.ts if last is not None else ""
    last_operation = last.canonical_type() if last is not None else ""
    lines = [
        "=== Run metadata and resolved configuration ===",
        f"run_id: {state.run_id or UNAVAILABLE}",
        f"status: {status_mark(state.status)}",
        f"seed: {state.seed_paper_id or UNAVAILABLE}",
        f"pipeline: {state.pipeline}",
        f"outcome: {state.outcome or UNAVAILABLE}",
        f"terminal_reason: {state.terminal_reason or UNAVAILABLE}",
        f"reason_code: {state.reason_code or UNAVAILABLE}",
        f"explorer_model: {state.explorer_model or UNAVAILABLE}",
        f"judge_model: {state.judge_model or UNAVAILABLE}",
        f"colony_size: {state.colony_size}",
        f"max_concurrent: {state.max_concurrent}",
        f"k_per_turn (papers per evaluation): {state.k_per_turn}",
        f"budget_type: {state.budget_type}",
        f"fetches: {state.fetches_used}/{state.max_fetches}",
        f"waves: {state.total_waves or state.current_wave}",
        f"elapsed: {format_duration(state.elapsed_seconds)}",
        f"last_activity: {last_activity or UNAVAILABLE}",
        f"last_operation: {last_operation or UNAVAILABLE}",
        f"tokens: {state.token_usage if state.token_usage is not None else UNAVAILABLE} "
        "(API-reported historical; not inferred to keep accumulating)",
        f"question: {state.query or UNAVAILABLE}",
    ]
    if state.status == STATUS_INTERRUPTED:
        lines.append(
            "interrupted: final in-memory work may have been lost; the durable "
            "trace and graph data remain available."
        )
    if state.failures:
        lines.append("failures:")
        lines.extend(f"  {redact_secrets(f)}" for f in state.failures)
    if state.warnings:
        lines.append("warnings:")
        lines.extend(f"  {redact_secrets(w)}" for w in state.warnings)
    return "\n".join(lines)


def agent_label(state: RunViewState, agent_id: str) -> str:
    summary = state.agents.get(agent_id)
    return summary.label if summary is not None else (agent_id or DASH)


def winner_label(state: RunViewState) -> str:
    if not state.winner_agent:
        return DASH
    return agent_label(state, state.winner_agent)


def _fit_segments(segments: list[tuple[str, str | None]], width: int) -> Text:
    """Join styled segments, ellipsizing the segment that crosses ``width``.

    Used for the context and query rows and for tree labels, where a wrapped or
    overflowing value would break the row rhythm. Segments after the crossing
    one are dropped.
    """
    text = Text()
    for segment, style in segments:
        remaining = width - text.cell_len
        if remaining <= 0:
            break
        if len(segment) <= remaining:
            text.append(segment, style=style)
        else:
            text.append(ellipsize(segment, remaining), style=style)
            break
    return text


def render_context_line(state: RunViewState, width: int = 120) -> Text:
    return _fit_segments(
        [
            (" Research Explorer ", f"bold {theme.COLOR_ACCENT}"),
            (" ACO colony ", theme.COLOR_MUTED),
            (f" run {state.run_id or DASH}", None),
            (f"  seed {state.seed_paper_id or DASH}", theme.COLOR_MUTED),
            (f"  pipeline {state.pipeline}", theme.COLOR_MUTED),
            (
                f"  explorer {state.explorer_model or UNAVAILABLE}"
                f"  judge {state.judge_model or UNAVAILABLE}",
                theme.COLOR_MUTED,
            ),
        ],
        width,
    )


def render_dashboard_header(
    state: RunViewState, width: int = 120, follow_live: bool = True
) -> Text:
    """Context line plus two run-fact rows.

    The context and query rows are ellipsized so they stay single-line. The
    run-fact row may wrap instead, because dropping the winner or the budget
    would hide a run fact; the header caps its own height.
    """
    glyph, color = theme.status_style(state.status)
    tokens = UNAVAILABLE
    if state.token_usage is not None:
        tokens = str(state.token_usage)
    elif state.cost is not None:
        tokens = f"cost {state.cost:.4f}"

    row1 = Text()
    row1.append(f"{glyph} ", style=color)
    row1.append(status_label(state), style=f"bold {color}")
    row1.append(f"  elapsed {format_duration(state.elapsed_seconds)}")
    row1.append(f"  fetch {state.fetches_used}/{state.max_fetches}")
    row1.append(f"  wave {state.current_wave}  turn {state.current_turn}")
    row1.append(f"  phase {state.current_phase or DASH}")
    row1.append(f"  {phase_progress_text(state)}")
    row1.append(f"  best {state.best_quality:.3f}", style=theme.COLOR_GOOD)
    row1.append(f"  leader {winner_label(state)}", style=theme.COLOR_WARN)
    row1.append(f"  tokens {tokens}", style=theme.COLOR_MUTED)
    if state.warnings or state.failures:
        row1.append(
            f"  warn {len(state.warnings)} fail {len(state.failures)}",
            style=theme.COLOR_BAD if state.failures else theme.COLOR_WARN,
        )

    row2_segments: list[tuple[str, str | None]] = [
        (f"Q {state.query or DASH}", None),
        (
            f"  colony {state.colony_size}  slots {state.max_concurrent}  "
            f"papers/eval {state.k_per_turn}",
            theme.COLOR_MUTED,
        ),
    ]
    if not follow_live:
        row2_segments.append(("  historical snapshot", f"bold {theme.COLOR_WARN}"))
    row2 = _fit_segments(row2_segments, width)

    header = Text()
    header.append_text(render_context_line(state, width))
    header.append("\n")
    header.append_text(row1)
    header.append("\n")
    header.append_text(row2)
    return header


def render_tree_label(
    state: RunViewState,
    node: NavNode,
    *,
    selected: bool,
    expanded: bool,
    depth: int,
    width: int = 120,
) -> Text:
    """One tree row, pruned to ``width`` so the tree keeps a one-row rhythm.

    The lifecycle glyph and label are reserved first; descriptive fields are
    then ellipsized into what is left, so a narrow pane never hides the state.
    """
    indent = "  " * depth
    marker = f"{theme.SELECTION_MARK} " if selected else "  "
    prefix = indent + marker
    if node.kind == AGENT_NODE:
        summary = state.agents.get(node.agent_id)
        if summary is None:
            return _fit_segments([(prefix + node.label, None)], width)
        glyph, color = theme.status_style(summary.status)
        status = f"  {glyph} {status_mark(summary.status)}"
        winner = "  ★" if summary.is_winner else ""
        budget = max(1, width - len(prefix) - len(status) - len(winner))
        head = _fit_segments(
            [
                (node.label, f"bold {theme.COLOR_TEXT}"),
                (f" {caste_label(summary.caste)}", theme.COLOR_MUTED),
                (f"  Q={summary.quality:.3f}", None),
                (
                    f"  Δ{summary.delta_q:+.3f}",
                    theme.COLOR_GOOD if summary.delta_q >= 0 else theme.COLOR_BAD,
                ),
            ],
            budget,
        )
        text = Text(prefix)
        text.append_text(head)
        if winner:
            text.append(winner, style=theme.COLOR_WARN)
        text.append(status, style=color)
        return text

    geometry = "\u25be" if (node.children and expanded) else (
        "\u25b8" if node.children else "\u00b7"
    )
    glyph, color = theme.status_style(node.status)
    status = f"  {glyph} {status_mark(node.status)}"
    detail = ""
    if node.kind == WAVE_NODE:
        research = state.phase(node.wave, PHASE_RESEARCH)
        evaluation = state.phase(node.wave, PHASE_EVALUATION)
        decision = state.phase(node.wave, PHASE_DECISION)
        parts = []
        if research is not None:
            parts.append(
                f"{len(research.completed)}/{len(research.selected)} research"
            )
        if evaluation is not None:
            best = (
                f"Q {evaluation.best_q:.2f}"
                if evaluation.best_q is not None
                else "Q --"
            )
            parts.append(
                f"eval {len(evaluation.completed)}/{len(evaluation.skipped)}/"
                f"{len(evaluation.failed)} {best}"
            )
        if decision is not None and (decision.leader or decision.stop_reason):
            reason = decision.stop_reason or decision.continue_reason or "continue"
            parts.append(
                f"leader {decision.leader or DASH} "
                f"Δ{decision.q_delta:+.2f} {reason}"
            )
        detail = " · " + " · ".join(parts) if parts else ""
    elif node.kind == PHASE_NODE:
        detail = " · " + phase_summary_text(state, node.wave, node.node_id.split(":")[-1])
    elif node.kind == FINAL_NODE:
        detail = f" · {state.outcome or UNAVAILABLE}"
        if state.stop_reason:
            detail += f" · {state.stop_reason}"
    elif node.kind == DEBUG_NODE:
        detail = f" · {state.events_seen_total} events · {len(state.warnings)} warnings"
    elif node.kind == SETUP_NODE:
        detail = f" · seed {state.seed_paper_id or DASH}"
    body = _fit_segments(
        [(node.label, None), (detail, theme.COLOR_MUTED)],
        max(1, width - len(prefix) - len(geometry) - 1 - len(status)),
    )
    text = Text(prefix + geometry + " ")
    text.append_text(body)
    text.append(status, style=color)
    return text


def render_tree_title(row_count: int) -> str:
    """Left-pane heading that states whether the colony has reported yet."""
    if row_count:
        return "Execution tree"
    return "Execution tree · waiting for the first wave"


def render_agent_roster(state: RunViewState) -> Text:
    """Compact agent roster used by tests and narrow layouts."""
    text = Text()
    for summary in state.ordered_agents():
        glyph, color = theme.status_style(summary.status)
        text.append(f" {summary.label}", style=f"bold {theme.COLOR_TEXT}")
        text.append(f" {caste_label(summary.caste)}", style=theme.COLOR_MUTED)
        text.append(f" Q={summary.quality:.3f}")
        text.append(f" {glyph}", style=color)
        if summary.is_winner:
            text.append(" ★", style=theme.COLOR_WARN)
    return text


def render_activity_card(
    state: RunViewState,
    agent_id: str,
    entry: TimelineEntry | None,
    width: int = 60,
    follow_live: bool = True,
    compact: bool = False,
) -> Text:
    summary = state.agents.get(agent_id)
    text = Text()
    text.append("Agent", style=f"bold {theme.COLOR_ACCENT}")
    if summary is None:
        text.append("\n  waiting for colony initialization", style=theme.COLOR_MUTED)
        return text
    glyph, color = theme.status_style(summary.status)
    text.append(f" {summary.label}")
    text.append(f"  {caste_label(summary.caste)}", style=theme.COLOR_MUTED)
    text.append(f"  {glyph} {status_mark(summary.status)}", style=color)
    if not compact:
        text.append("\n")

    action = entry.label if entry is not None else "idle"
    paper_entry = _paper_entry_for(state, agent_id)
    meta = _activity_meta(entry, paper_entry, summary)
    mode = "live" if follow_live else "historical"
    title = str(meta.get("title") or summary.current_paper_title or "")
    paper_id = str(meta.get("paper_id") or summary.current_paper_id or "")
    operation = state.current_operation or UNAVAILABLE
    model = state.current_operation_model or (
        state.explorer_model if state.current_operation else ""
    )
    elapsed = _format_optional_duration(state.operation_elapsed_seconds)

    if compact:
        mode_style = theme.COLOR_WARN if mode == "historical" else theme.COLOR_MUTED
        text.append(f"  [{mode}]", style=mode_style)
        text.append("\n  ")
        text.append(ellipsize(str(action), max(10, width - 18)))
        text.append(" · ", style=theme.COLOR_MUTED)
        text.append(ellipsize(title or paper_id or UNAVAILABLE, max(10, width // 2)))
        operation_bits = []
        if model:
            operation_bits.append(model)
        if state.operation_elapsed_seconds is not None:
            operation_bits.append(elapsed)
        if operation_bits:
            text.append("  " + "  ".join(operation_bits), style=theme.COLOR_MUTED)
        text.append("\n  ")
        text.append("Q ", style=theme.COLOR_MUTED)
        text.append(f"{summary.quality:.3f}")
        delta_style = theme.COLOR_GOOD if summary.delta_q >= 0 else theme.COLOR_BAD
        text.append(f"  Δ{summary.delta_q:+.3f}", style=delta_style)
        text.append(f"  budget {summary.budget}  frontier {summary.frontier}")
        return text

    text.append("  action ", style=theme.COLOR_MUTED)
    text.append(ellipsize(str(action), max(10, width - 12)))
    text.append(f"  [{mode}]", style=theme.COLOR_WARN if mode == "historical" else theme.COLOR_MUTED)
    text.append("\n")

    text.append("  paper ", style=theme.COLOR_MUTED)
    text.append(ellipsize(title or paper_id or UNAVAILABLE, max(10, width - 8)))
    text.append("\n")

    text.append("  op ", style=theme.COLOR_MUTED)
    text.append(f"{operation}  model {model or UNAVAILABLE}  elapsed {elapsed}")
    text.append("\n")

    text.append("  Q ", style=theme.COLOR_MUTED)
    text.append(f"{summary.quality:.3f}")
    delta_style = theme.COLOR_GOOD if summary.delta_q >= 0 else theme.COLOR_BAD
    text.append(f"  Δ{summary.delta_q:+.3f}", style=delta_style)
    text.append(f"  budget {summary.budget}  frontier {summary.frontier}")
    return text


def render_tab_bar(session: UISession) -> Text:
    text = Text()
    for index, tab in enumerate(PRIMARY_TABS, start=1):
        active = tab == session.active_tab
        label = f"{index}:{tab_label(tab)}"
        if active:
            text.append(f" {theme.SELECTION_MARK} {label} ", style=f"bold {theme.COLOR_ACCENT}")
        else:
            text.append(f"   {label} ", style=theme.COLOR_MUTED)
    return text


def render_footer_text(state: RunViewState, session: UISession, settling: bool = False) -> Text:
    text = Text()
    if settling and state.status not in TERMINAL_STATUSES:
        text.append(" cancellation requested; settling… ", style=f"bold {theme.COLOR_WARN}")
    if state.status in TERMINAL_STATUSES:
        text.append(f" {status_label(state)} ", style=f"bold {theme.COLOR_ACCENT}")
        text.append("press "
                    "q to exit  ? for help  Ctrl+P for commands", style=theme.COLOR_MUTED)
        return text
    text.append("Research Explorer", style=f"bold {theme.COLOR_ACCENT}")
    text.append("  ")
    text.append(footer_hints(session.narrowed), style=theme.COLOR_MUTED)
    return text


def render_scope_caption(
    state: RunViewState,
    agent_id: str,
    entry: TimelineEntry | None,
    follow_live: bool = True,
) -> str:
    parts = [f"agent {agent_label(state, agent_id)}"]
    if entry is not None:
        parts.append(f"wave {entry.wave or DASH}")
        parts.append(f"turn {entry.turn}")
        parts.append(f"{entry.kind} · {entry.status}")
    if not follow_live:
        parts.append("historical snapshot")
    return "  ".join(parts)


def render_research_tab(
    state: RunViewState,
    agent_id: str,
    entry: TimelineEntry | None,
    node: NavNode | None = None,
) -> str:
    lines: list[str] = []
    if node is not None and node.kind == FINAL_NODE:
        return render_final_result(state)
    if node is not None and node.kind in (WAVE_NODE, PHASE_NODE):
        wave = node.wave or state.current_wave
        lines.extend(render_wave_summary(state, wave))
        lines.append("")
        lines.extend(render_phase_agent_table(state, wave, node.node_id))
        lines.append("")
    lines.append(f"## Research line · {agent_label(state, agent_id)}")
    lines.append("")
    detail = entry.detail if entry is not None else {}
    delta = detail.get("delta_narrative") or detail.get("changed_understanding")
    if delta is None:
        records = state.evaluations.get(agent_id, [])
        if records:
            delta = f"Q {records[-1].q:.4f} (Δ{records[-1].delta_q:+.4f})"
    narrative = state.narratives.get(agent_id) or state.narratives.get("__latest__", "")
    if delta:
        lines.append("### Understanding delta")
        lines.append(str(delta))
        lines.append("")
    if not narrative:
        lines.append("_No narrative recorded yet._")
    else:
        lines.append(narrative)
    return "\n".join(lines)


def render_wave_summary(state: RunViewState, wave: int) -> list[str]:
    """Compact, always-available wave rows used by the phase detail view."""
    record = state.entry_by_id(f"wave:{wave}")
    status = record.status if record is not None else NODE_PENDING
    lines = [f"### Wave {wave} · {status_mark(status)}"]
    research = state.phase(wave, PHASE_RESEARCH)
    if research is not None:
        lines.append(
            f"- Research: {len(research.completed)}/{len(research.selected)} done · "
            f"{research.papers_integrated} papers · {research.evidence_added} evidence"
        )
    evaluation = state.phase(wave, PHASE_EVALUATION)
    if evaluation is not None:
        best = f"{evaluation.best_q:.3f}" if evaluation.best_q is not None else DASH
        lines.append(
            f"- Evaluation: {len(evaluation.completed)} complete / "
            f"{len(evaluation.skipped)} skipped / {len(evaluation.failed)} failed · best Q {best}"
        )
    decision = state.phase(wave, PHASE_DECISION)
    if decision is not None:
        reason = decision.stop_reason or decision.continue_reason or "continue"
        lines.append(
            f"- Decision: leader {decision.leader or DASH} · "
            f"Δ{decision.q_delta:+.3f} · {reason}"
        )
    return lines


def render_phase_agent_table(state: RunViewState, wave: int, node_id: str = "") -> list[str]:
    """Compact per-agent table shown for a selected wave or phase."""
    phase = PHASE_RESEARCH
    if node_id.startswith("wave:") and node_id.count(":") == 2:
        phase = node_id.split(":")[2]
    selected = state.selected_agents.get(wave, [])
    if not selected:
        selected = state.agent_order
    lines = ["Agent  Research     Evaluation   Q       Delta    Work"]
    for agent_id in selected:
        summary = state.agents.get(agent_id)
        if summary is None:
            continue
        evaluation = state.evaluation_state(wave, agent_id)
        if evaluation is not None:
            eval_label = evaluation.status
            q_text = f"{evaluation.q:.3f}" if evaluation.q is not None else DASH
            delta_text = (
                f"{evaluation.q_delta:+.3f}" if evaluation.q_delta is not None else DASH
            )
        else:
            eval_label = EVAL_PENDING
            q_text = DASH
            delta_text = DASH
        if eval_label == EVAL_SKIPPED and evaluation is not None and evaluation.reason:
            eval_label = f"skipped:{evaluation.reason}"
        work = summary.current_paper_title or "no recent paper"
        lines.append(
            f"{summary.label:<6} {status_mark(summary.research_status):<12} "
            f"{eval_label:<12} {q_text:<7} {delta_text:<8} {ellipsize(work, 32)}"
        )
    if phase == PHASE_EVALUATION and len(lines) == 1:
        lines.append("no evaluation recorded for this wave")
    return lines


def render_evaluation_state(state: RunViewState, agent_id: str, wave: int) -> list[str]:
    evaluation = state.evaluation_state(wave, agent_id)
    lines = [f"### Evaluation · {agent_label(state, agent_id)} (wave {wave})"]
    if evaluation is None:
        lines.append("no terminal evaluation recorded for this agent and wave")
        return lines
    lines.append(f"status: {evaluation.status}" + (f" ({evaluation.reason})" if evaluation.reason else ""))
    for key in ("S", "P", "J", "R"):
        value = evaluation.components.get(key)
        if value is None:
            reason = evaluation.unavailable.get(key, "unavailable")
            lines.append(f"{key}: unavailable ({reason})")
        else:
            lines.append(f"{key}: {value:.4f}")
    if evaluation.q is not None:
        delta = f"{evaluation.q_delta:+.4f}" if evaluation.q_delta is not None else DASH
        lines.append(f"Q: {evaluation.q:.4f}  Δ{delta}")
    return lines


def render_final_result(state: RunViewState) -> str:
    lines = ["## Final result", ""]
    lines.append(f"- Outcome: {state.outcome or UNAVAILABLE}")
    lines.append(f"- Status: {status_label(state)}")
    lines.append(
        f"- Stop reason: {state.stop_reason or state.terminal_reason or UNAVAILABLE}"
    )
    lines.append(f"- Winner: {winner_label(state)}")
    lines.append(f"- Best Q: {state.best_quality:.4f}")
    lines.append(
        f"- Waves: {state.total_waves or state.current_wave}  "
        f"fetches: {state.fetches_used}/{state.max_fetches}  "
        f"elapsed: {format_duration(state.elapsed_seconds)}  "
        f"tokens: {state.token_usage if state.token_usage is not None else UNAVAILABLE}"
    )
    lines.append("")
    lines.append("### Winning narrative")
    narrative = state.narratives.get(state.winner_agent, "")
    if narrative.strip():
        lines.append(narrative)
    else:
        lines.append(
            "_No winning narrative was produced;_ "
            f"outcome is completed-degraded because {state.terminal_reason or 'the winner has no usable result'}."
        )
    lines.append("")
    lines.append("### Winning evaluation")
    latest = state.latest_evaluation(state.winner_agent) if state.winner_agent else None
    if latest is not None:
        lines.extend(render_evaluation_state(state, state.winner_agent, latest.wave))
    else:
        lines.append("no winner evaluation recorded")
    failed = sum(
        1 for record in state.evaluation_states.values() if record.status == EVAL_FAILED
    )
    skipped = sum(
        1 for record in state.evaluation_states.values() if record.status == EVAL_SKIPPED
    )
    lines.append("")
    lines.append(
        f"### Quality gaps\n- failed evaluations: {failed}\n- skipped evaluations: {skipped}\n"
        f"- warnings: {len(state.warnings)}  failures: {len(state.failures)}"
    )
    return "\n".join(lines)


def render_paper_tab(state: RunViewState, agent_id: str, entry: TimelineEntry | None) -> str:
    entry = _paper_entry_for(state, agent_id) if (
        entry is None or not entry.paper_id
    ) else entry
    if entry is None or (not entry.paper_id and not entry.detail.get("title")):
        return "_No paper selected yet._"
    paper_id = entry.paper_id
    meta = entry.detail
    lines = [f"## Paper · {ellipsize(paper_id or DASH, 60)}", ""]
    lines.append(f"- title: {meta.get('title') or UNAVAILABLE}")
    lines.append(f"- year: {meta.get('year') if meta.get('year') is not None else UNAVAILABLE}")
    authors = ", ".join(str(a) for a in (meta.get("authors") or [])) or UNAVAILABLE
    lines.append(f"- authors: {authors}")
    lines.append(f"- source paper: {meta.get('source') or UNAVAILABLE}")
    lines.append(f"- traversal mode: {meta.get('mode') or UNAVAILABLE}")
    lines.append(f"- provider: {meta.get('provider') or UNAVAILABLE}")
    lines.append("")
    lines.append("### Selection rationale")
    lines.extend(_render_rationale(state, entry))
    lines.append("")
    lines.append("### Structured analysis")
    analysis = state.paper_analyses.get(agent_id or entry.agent_id, {}).get(paper_id)
    lines.extend(_render_paper_fields(analysis) if analysis else ["no structured analysis recorded"])
    lines.append("")
    lines.append("### Decision-time frontier")
    frontier = [
        c for c in state.candidate_scores
        if not entry.agent_id or c.agent_id == entry.agent_id
    ]
    if not frontier:
        lines.append("no recorded frontier candidates")
    else:
        for candidate in sorted(frontier, key=lambda c: c.eta, reverse=True)[
            :CANDIDATE_TOP_ALTERNATIVES
        ]:
            marker = "chosen" if any(
                s.paper_id == candidate.paper_id for s in state.selections
            ) else "candidate"
            lines.append(
                f"- {ellipsize(candidate.paper_id, 48)} eta={candidate.eta:.4f} "
                f"mode={candidate.mode or DASH} [{marker}]"
            )
    return "\n".join(lines)


def render_evaluation_tab(state: RunViewState, agent_id: str) -> str:
    records = state.evaluations.get(agent_id, [])
    lines = [f"## Evaluation · {agent_label(state, agent_id)}", ""]
    if not records:
        lines.append("_No detailed evaluation recorded for this agent yet._")
        lines.append("")
        lines.append("A skipped evaluation means no new evidence was produced; it is not a zero score.")
        return "\n".join(lines)
    if len(records) > 1:
        lines.append("### Q history")
        for record in records:
            lines.append(
                f"- wave {record.oleada} turn {record.turn}: "
                f"Q={record.q:.4f} (Δ{record.delta_q:+.4f})"
            )
        lines.append("")
    record = records[-1]
    lines.append(
        f"Q={record.q:.4f}  old={record.old_quality:.4f}  delta={record.delta_q:+.4f}"
    )
    lines.append(f"weights {_format_field(record.weights) or UNAVAILABLE}")
    lines.append(
        f"S={record.self_assessment.score:.4f}  P={record.peers.aggregated_score:.4f}  "
        f"J={record.virgin_judge.score:.4f}  R={record.structural.r:.4f}"
    )
    lines.append("")
    lines.append(f"self rationale: {record.self_assessment.reasoning or UNAVAILABLE}")
    if record.peers.votes:
        for vote in record.peers.votes:
            lines.append(
                f"peer {vote.voter_id}: {vote.score:.4f} — {vote.reasoning or UNAVAILABLE}"
            )
    else:
        lines.append("peer votes: none recorded")
    lines.append(
        f"judge coverage: {record.virgin_judge.coverage or UNAVAILABLE}  "
        f"gaps: {record.virgin_judge.gaps or UNAVAILABLE}"
    )
    structural = record.structural
    lines.append(
        f"structural coverage={structural.coverage:.3f} diversity={structural.diversity:.3f} "
        f"depth={structural.depth:.3f} coherence={structural.coherence:.3f} R={structural.r:.3f}"
    )
    return "\n".join(lines)


def filter_events(
    state: RunViewState, agent_id: str = "", outcome: str = ""
) -> list[RunEvent]:
    """Filter the bounded live event window without building unbounded lists."""
    events = [e for e in state.events if not agent_id or agent_id in str(e.payload)]
    if outcome:
        events = [e for e in events if classify_event_outcome(e) == outcome]
    return events


def event_page_count(total: int, page_size: int = EVENT_PAGE_SIZE) -> int:
    if total <= 0:
        return 1
    return (total + page_size - 1) // page_size


def render_events_tab(
    state: RunViewState,
    agent_id: str = "",
    outcome: str = "",
    newest_first: bool = True,
    page: int = 0,
    page_size: int = EVENT_PAGE_SIZE,
) -> str:
    events = filter_events(state, agent_id, outcome)
    if newest_first:
        events = list(reversed(events))
    filters: list[str] = []
    if agent_id:
        filters.append(f"agent={agent_label(state, agent_id)}")
    if outcome:
        filters.append(f"outcome={outcome}")
    scope = f" ({', '.join(filters)})" if filters else ""
    total = len(events)
    page_count = event_page_count(total, page_size)
    current = max(0, min(page, page_count - 1))
    start = current * page_size
    window = events[start : start + page_size]
    seen_total = state.events_seen_total or total
    dropped = state.events_dropped
    lines = [f"## Events{scope}", ""]
    if total:
        lines.append(
            f"_Showing {start + 1}-{start + len(window)} of {total} "
            f"(newest first; page {current + 1}/{page_count})_"
        )
        lines.append("")
    else:
        lines.append("_No events recorded for this filter._")
        if seen_total and dropped:
            lines.append(
                f"_{seen_total} durable events were seen, but none match the "
                "current filter in the bounded live window._"
            )
        return "\n".join(lines)
    omitted = max(0, seen_total - total)
    if dropped or omitted:
        lines.extend(
            [
                f"_{seen_total} durable events seen; {dropped} evicted from the "
                "bounded live window. Older history lives in the durable trace "
                "(use `replay tui` once the run is finished)._",
                "",
            ]
        )
    for event in window:
        summary = " ".join(
            f"{k}={v}" for k, v in event.payload.items() if k != "content"
        )[:160]
        lines.append(f"- `[{event.seq:>4}] {event.type}` {redact_secrets(summary)}")
    return "\n".join(lines)


def events_scope_agent(state: RunViewState, session: UISession) -> str:
    """Resolve the Events filter, shared by the tab and the fullscreen reader."""
    if session.event_agent == EVENT_AGENT_ALL:
        return ""
    if session.event_agent == EVENT_AGENT_WINNER:
        return state.winner_agent
    return session.selected_agent_id


def render_tab_body(
    state: RunViewState,
    session: UISession,
    entry: TimelineEntry | None,
    node: NavNode | None = None,
) -> str:
    if session.active_tab == TAB_RESEARCH:
        return render_research_tab(state, session.selected_agent_id, entry, node)
    if session.active_tab == TAB_PAPER:
        return render_paper_tab(state, session.selected_agent_id, entry)
    if session.active_tab == TAB_EVALUATION:
        if (
            node is not None
            and node.kind == PHASE_NODE
            and node.wave
            and state.evaluation_state(node.wave, session.selected_agent_id) is not None
        ):
            return "\n".join(
                render_evaluation_state(state, session.selected_agent_id, node.wave)
            )
        return render_evaluation_tab(state, session.selected_agent_id)
    if session.active_tab == TAB_EVENTS:
        return render_events_tab(
            state,
            events_scope_agent(state, session),
            session.event_outcome,
            page=session.event_page,
        )
    return ""


def render_reader(
    state: RunViewState,
    session: UISession,
    entry: TimelineEntry | None,
    node: NavNode | None = None,
) -> tuple[str, str]:
    title = f"{tab_label(session.active_tab)} · {agent_label(state, session.selected_agent_id)}"
    body = render_tab_body(state, session, entry, node)
    if session.active_tab == TAB_EVENTS:
        return title, body
    caption = render_scope_caption(
        state, session.selected_agent_id, entry, session.follow_live
    )
    return title, body + "\n\n" + caption


def render_status_banner(state: RunViewState) -> Text:
    """Terminal banner colored by outcome rather than a single failure color."""
    glyph, color = theme.status_style(state.status)
    text = Text()
    text.append(f"{glyph} ", style=color)
    if state.status == STATUS_FAILED:
        text.append(render_failure_summary(state), style=color)
        return text
    if state.status == STATUS_INTERRUPTED:
        text.append(status_label(state), style=f"bold {color}")
        reason = state.terminal_reason or "the run was interrupted"
        text.append(f" {DASH} {reason}", style=theme.COLOR_MUTED)
        text.append(
            " (durable trace preserved; final in-memory work may be lost)",
            style=theme.COLOR_MUTED,
        )
        return text
    text.append(status_label(state), style=f"bold {color}")
    text.append(f" {DASH} {state.terminal_reason or 'run finished'}", style=theme.COLOR_MUTED)
    return text


def render_failure_summary(state: RunViewState) -> str:
    lines = [f"Run {state.run_id or DASH} ended with status {status_label(state)}."]
    if state.terminal_reason:
        lines.append(state.terminal_reason)
    for failure in state.failures:
        lines.append(redact_secrets(str(failure)))
    return "\n".join(lines)
