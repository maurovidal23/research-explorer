"""Pure text renderers for the run view state.

These helpers contain no Textual imports so projection and rendering can be
unit-tested without a terminal. The Textual widgets in
:mod:`research_explorer.tui.app` only place these strings on screen.
"""

from __future__ import annotations

from research_explorer.events.models import (
    AGENT_ACTIVE,
    AGENT_COMPLETED,
    AGENT_EVALUATING,
    AGENT_EXHAUSTED,
    AGENT_FAILED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_EVALUATING,
    STATUS_FAILED,
    AgentSummary,
    RunEvent,
    RunViewState,
    TimelineEntry,
)
from research_explorer.redaction import redact_secrets

UNAVAILABLE = "unavailable"
NARROW_BREAKPOINT = 100
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


EVENT_OUTCOMES: tuple[str, ...] = ("", "error", "warning", "info")

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


def next_event_outcome(current: str) -> str:
    index = EVENT_OUTCOMES.index(current) if current in EVENT_OUTCOMES else 0
    return EVENT_OUTCOMES[(index + 1) % len(EVENT_OUTCOMES)]


def render_header(state: RunViewState) -> str:
    tokens = UNAVAILABLE
    if state.token_usage is not None:
        tokens = str(state.token_usage)
    elif state.cost is not None:
        tokens = f"cost {state.cost:.4f}"
    winner_agent = state.agents.get(state.winner_agent) if state.winner_agent else None
    winner = winner_agent.label if winner_agent is not None else (state.winner_agent or DASH)
    status_text = status_mark(state.status)
    if state.outcome and state.outcome != "ok":
        status_text = f"{status_text}/{state.outcome}"
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


def render_tabs(state: RunViewState, selected_index: int) -> str:
    agents = state.ordered_agents()
    if not agents:
        return "agents: waiting for colony initialization"
    parts: list[str] = []
    for index, agent in enumerate(agents):
        winner = "*" if agent.is_winner else " "
        marker = ">" if index == selected_index else " "
        delta = f"{agent.delta_q:+.3f}"
        parts.append(
            f"{marker} [{agent.label}{winner} {caste_label(agent.caste)} "
            f"Q={agent.quality:.3f} d={delta} {status_mark(agent.status)}]"
        )
    return " ".join(parts)


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


def render_footer(state: RunViewState) -> str:
    if state.status in (STATUS_COMPLETED, STATUS_CANCELLED, STATUS_FAILED):
        line = f"[{status_mark(state.status)}]"
        if state.terminal_reason:
            line += f" {state.terminal_reason}"
        return f"{line} press q to exit, ? for help"
    return (
        "left/right agent  up/down step  enter details  home live\n"
        "e eval  f frontier  p paper  n narrative  l events  r metadata  t panes  ? help  q quit"
    )


def render_evaluation(state: RunViewState, agent_id: str) -> str:
    records = state.evaluations.get(agent_id, [])
    if not records:
        return (
            "=== Evaluation ===\n\n"
            f"no detailed evaluation recorded for {agent_id or 'the selected agent'}"
        )
    record = records[-1]
    lines = [
        f"=== Evaluation: {record.agent_id} wave={record.oleada} turn={record.turn} ===",
        f"Q={record.q:.4f}  old={record.old_quality:.4f}  delta={record.delta_q:+.4f}",
        f"weights {_format_field(record.weights) or UNAVAILABLE}",
        f"S={record.self_assessment.score:.4f}  P={record.peers.aggregated_score:.4f}  "
        f"J={record.virgin_judge.score:.4f}  R={record.structural.r:.4f}",
        "",
        f"self rationale: {record.self_assessment.reasoning or UNAVAILABLE}",
    ]
    if record.peers.votes:
        for vote in record.peers.votes:
            lines.append(
                f"peer {vote.voter_id}: {vote.score:.4f} -- {vote.reasoning or UNAVAILABLE}"
            )
    else:
        lines.append("peer votes: none recorded")
    lines.append(f"virgin coverage: {record.virgin_judge.coverage or UNAVAILABLE}")
    lines.append(f"virgin gaps: {record.virgin_judge.gaps or UNAVAILABLE}")
    structural = record.structural
    lines.append(
        f"structural coverage={structural.coverage:.3f} diversity={structural.diversity:.3f} "
        f"depth={structural.depth:.3f} coherence={structural.coherence:.3f} R={structural.r:.3f}"
    )
    return "\n".join(lines)


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


def render_metadata(state: RunViewState) -> str:
    lines = [
        "=== Run metadata and resolved configuration ===",
        f"run_id: {state.run_id or UNAVAILABLE}",
        f"status: {status_mark(state.status)}",
        f"outcome: {state.outcome or UNAVAILABLE}",
        f"terminal_reason: {state.terminal_reason or UNAVAILABLE}",
        f"terminal_reason_code: {state.terminal_reason_code or UNAVAILABLE}",
        f"seed: {state.seed_paper_id or UNAVAILABLE}",
        f"pipeline: {state.pipeline}",
        f"explorer_model: {state.explorer_model or UNAVAILABLE}",
        f"judge_model: {state.judge_model or UNAVAILABLE}",
        f"colony_size: {state.colony_size}",
        f"max_concurrent: {state.max_concurrent}",
        f"k_per_turn (papers per evaluation): {state.k_per_turn}",
        f"budget_type: {state.budget_type}",
        f"fetches: {state.fetches_used}/{state.max_fetches}",
        f"elapsed: {format_duration(state.elapsed_seconds)}",
        f"question: {state.query or UNAVAILABLE}",
    ]
    if state.failures:
        lines.append("failures:")
        lines.extend(f"  {redact_secrets(f)}" for f in state.failures)
    if state.warnings:
        lines.append("warnings:")
        lines.extend(f"  {redact_secrets(w)}" for w in state.warnings)
    return "\n".join(lines)


def render_help() -> str:
    return (
        "=== Key help ===\n"
        "Left/Right  select previous/next agent tab\n"
        "Up/Down     select timeline item\n"
        "Enter       open details for selected item\n"
        "Home        restore follow-live mode and jump to newest\n"
        "e           detailed evaluation for selected agent\n"
        "f           recorded frontier / candidate ranking\n"
        "p           full selected-paper analysis\n"
        "n           complete agent narrative\n"
        "l           run events filtered to selected agent\n"
        "o           run events filtered by outcome (error/warning/info)\n"
        "r           run metadata and resolved configuration\n"
        "t           toggle timeline/detail panes (narrow terminals)\n"
        "?           this help\n"
        "q           exit flow (confirm when a run is active)\n"
        "Ctrl+C      graceful cancellation"
    )
