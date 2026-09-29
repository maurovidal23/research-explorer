"""Pure event -> :class:`RunViewState` projection.

The reducer tolerates duplicate delivery (idempotent by durable sequence),
unknown/future event types, malformed artifacts, and partial payloads. It is
the single source of truth shared by live TUI rendering and replay.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterable
from typing import Any

from research_explorer.events.limits import LIVE_CANDIDATE_WINDOW, LIVE_EVENT_WINDOW
from research_explorer.events.models import (
    AGENT_ACTIVE,
    AGENT_COMPLETED,
    AGENT_EVALUATING,
    AGENT_EXHAUSTED,
    AGENT_FAILED,
    AGENT_WAITING,
    EVAL_COMPLETE,
    EVAL_FAILED,
    NODE_ACTIVE,
    NODE_COMPLETED,
    NODE_FAILED,
    NODE_PENDING,
    NODE_SKIPPED,
    OUTCOME_COMPLETED,
    OUTCOME_DEGRADED,
    PHASE_BENCHMARK,
    PHASE_DECISION,
    PHASE_EVALUATION,
    PHASE_EXAM_BUILD,
    PHASE_RESEARCH,
    PHASE_SELECTION,
    PHASE_SURVIVOR,
    REASON_NO_WINNER,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_EVALUATING,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    AgentSummary,
    EvaluationState,
    EventType,
    PhaseState,
    RunEvent,
    RunViewState,
    TimelineEntry,
    reason_text,
)
from research_explorer.redaction import redact_obj, redact_secrets
from research_explorer.replay.models import (
    CandidateScore,
    CandidateSelection,
    DetailedEvaluation,
)

_NARRATIVE_RE = re.compile(r"^narrative_(?P<agent>.+?)(?:_t\d+)?\.md$")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class RunProjection:
    """Reduces ordered events into a :class:`RunViewState`."""

    def __init__(
        self,
        state: RunViewState | None = None,
        event_window: int = LIVE_EVENT_WINDOW,
        candidate_window: int = LIVE_CANDIDATE_WINDOW,
    ) -> None:
        self.state = state if state is not None else RunViewState()
        self._max_seq: int = 0
        self._event_window = max(1, event_window)
        self._candidate_window = max(1, candidate_window)
        self._open_frontier: dict[tuple[str, int, int], str] = {}
        self._current_agent: str = ""
        self._saw_phase_event: bool = False

    @classmethod
    def from_events(cls, events: Iterable[RunEvent | dict]) -> RunProjection:
        projection = cls()
        for raw in events:
            event = raw if isinstance(raw, RunEvent) else RunEvent.model_validate(raw)
            projection.apply(event)
        return projection

    # ---- selection --------------------------------------------------------

    def select_entry(self, entry_id: str | None) -> None:
        self.state.follow_live = False
        self.state.selected_entry_id = entry_id

    def select_by_index(self, index: int) -> None:
        if not self.state.timeline:
            return
        index = max(0, min(index, len(self.state.timeline) - 1))
        self.select_entry(self.state.timeline[index].entry_id)

    def restore_follow(self) -> None:
        self.state.follow_live = True
        self._sync_follow()

    def _sync_follow(self) -> None:
        if self.state.follow_live and self.state.timeline:
            self.state.selected_entry_id = self.state.timeline[-1].entry_id

    def selected_entry(self) -> TimelineEntry | None:
        return self.state.entry_by_id(self.state.selected_entry_id)

    # ---- reducer ----------------------------------------------------------

    def apply(self, event: RunEvent) -> None:
        if event.seq > 0:
            if event.seq <= self._max_seq:
                return
            self._max_seq = event.seq
        if event.payload:
            event = event.model_copy(update={"payload": redact_obj(event.payload)})
        self._append_event(event)
        # ``no_winner`` is a durable legacy event that must stay readable but
        # carries richer semantics than the generic completed handler.
        handler: Any
        if event.type == "no_winner":
            handler = RunProjection._on_no_winner
        else:
            handler = _HANDLERS.get(event.canonical_type())
        if handler is not None:
            try:
                handler(self, event)
            except Exception:  # pragma: no cover - projection must never crash
                self.state.warnings.append(
                    redact_secrets(f"projection_error type={event.type}")
                )
        self._sync_follow()

    def _append_event(self, event: RunEvent) -> None:
        events = self.state.events
        events.append(event)
        self.state.events_seen_total += 1
        overflow = len(events) - self._event_window
        if overflow > 0:
            del events[:overflow]
            self.state.events_dropped += overflow

    def _append_candidate_score(self, score: CandidateScore) -> None:
        scores = self.state.candidate_scores
        scores.append(score)
        self.state.candidate_scores_seen_total += 1
        overflow = len(scores) - self._candidate_window
        if overflow > 0:
            del scores[:overflow]
            self.state.candidate_scores_dropped += overflow

    # ---- helpers ----------------------------------------------------------

    def _agent_id(self, event: RunEvent) -> str:
        raw = (
            event.payload.get("agent_id")
            or event.payload.get("agent")
            or event.payload.get("winner")
        )
        if raw:
            return str(raw)
        # Old traces omit ``agent_id`` on some paper/telemetry events; derive the
        # surrounding turn's agent so agent-first navigation stays truthful.
        return self._current_agent

    def _ensure_agent(self, agent_id: str, caste: str = "") -> AgentSummary:
        summary = self.state.agents.get(agent_id)
        if summary is None:
            label = f"A{len(self.state.agent_order) + 1:02d}"
            summary = AgentSummary(agent_id=agent_id, label=label, caste=caste or "mixto")
            self.state.agents[agent_id] = summary
            self.state.agent_order.append(agent_id)
        elif caste:
            summary.caste = caste
        return summary

    def _add_entry(self, entry: TimelineEntry) -> TimelineEntry:
        self.state.timeline.append(entry)
        return entry

    def _wave_entry(self, wave: int) -> TimelineEntry | None:
        for entry in self.state.timeline:
            if entry.kind == "wave" and entry.wave == wave:
                return entry
        return None

    def _turn_entry(self, wave: int, turn: int, agent_id: str) -> TimelineEntry | None:
        for entry in self.state.timeline:
            if (
                entry.kind == "turn"
                and entry.wave == wave
                and entry.turn == turn
                and entry.agent_id == agent_id
            ):
                return entry
        return None

    def _phase_entry(self, wave: int, phase: str) -> TimelineEntry | None:
        entry_id = f"phase:{wave}:{phase}"
        return self.state.entry_by_id(entry_id)

    def _ensure_phase(self, wave: int, phase: str, status: str = NODE_ACTIVE) -> PhaseState:
        record = self.state.phase(wave, phase)
        if record is None:
            record = PhaseState(wave=wave, phase=phase, status=status)
            self.state.phases[record.key] = record
        else:
            record.status = status
        self.state.current_wave = max(self.state.current_wave, wave)
        self.state.current_phase = phase
        label = phase.title()
        entry = self._phase_entry(wave, phase)
        if entry is None:
            self._add_entry(
                TimelineEntry(
                    entry_id=f"phase:{wave}:{phase}",
                    kind="phase",
                    label=f"{label}",
                    status=status,
                    wave=wave,
                    parent_id=f"wave:{wave}",
                )
            )
        else:
            entry.status = status
        return record

    def _set_wave_status(self, wave: int, status: str) -> None:
        for entry in self.state.timeline:
            if entry.kind == "wave" and entry.wave == wave:
                entry.status = status
            elif entry.wave == wave and entry.status in (NODE_PENDING, NODE_ACTIVE):
                entry.status = status if status == NODE_COMPLETED else entry.status

    def _upsert_evaluation(self, detail: DetailedEvaluation) -> None:
        records = self.state.evaluations.setdefault(detail.agent_id, [])
        for i, existing in enumerate(records):
            if existing.oleada == detail.oleada and existing.turn == detail.turn:
                records[i] = detail
                return
        records.append(detail)

    def hydrate_evaluation(self, detail: DetailedEvaluation) -> None:
        """Merge a persisted evaluation record without re-recording an event.

        Replay uses this to recover full rationale detail from the
        ``evaluation_results`` table when a trace only carried the compact
        ``evaluation_complete`` event. Persisted rows are stored verbatim, so the
        detail passes through the redaction boundary here before it can reach a
        renderer. Existing records win to avoid duplication.
        """
        detail = DetailedEvaluation.model_validate(redact_obj(detail.model_dump()))
        existing = self.state.evaluations.get(detail.agent_id, [])
        if any(
            record.oleada == detail.oleada and record.turn == detail.turn
            for record in existing
        ):
            return
        self._upsert_evaluation(detail)
        summary = self._ensure_agent(detail.agent_id)
        summary.quality = detail.q
        summary.delta_q = detail.delta_q

    def hydrate_narrative(self, agent_id: str, content: str) -> None:
        """Recover a narrative artifact that a trace did not inline."""
        if agent_id and not self.state.narratives.get(agent_id):
            self.state.narratives[agent_id] = content
        if content and not self.state.narratives.get("__latest__"):
            self.state.narratives["__latest__"] = content

    # ---- handlers ---------------------------------------------------------

    def _on_run_started(self, event: RunEvent) -> None:
        p = event.payload
        self.state.status = STATUS_RUNNING
        self.state.run_id = str(p.get("run_id") or self.state.run_id)
        self.state.seed_paper_id = str(p.get("seed") or self.state.seed_paper_id)
        self.state.query = str(p.get("query") or self.state.query)
        self.state.pipeline = str(p.get("pipeline") or self.state.pipeline)
        self.state.colony_size = _as_int(p.get("colony_size"), self.state.colony_size)
        self.state.max_concurrent = _as_int(p.get("K", p.get("max_concurrent")), self.state.max_concurrent)
        self.state.max_fetches = _as_int(p.get("max_fetches"), self.state.max_fetches)
        self.state.k_per_turn = _as_int(p.get("k_per_turn"), self.state.k_per_turn)
        self.state.explorer_model = str(p.get("explorer_model") or self.state.explorer_model)
        self.state.judge_model = str(p.get("judge_model") or self.state.judge_model)
        self.state.budget_type = str(p.get("budget_type") or self.state.budget_type)
        self.state.max_time_seconds = _as_int(p.get("max_time_seconds"), self.state.max_time_seconds)
        if not self.state.started_at:
            self.state.started_at = event.ts

    def _on_seed_routing(self, event: RunEvent) -> None:
        self.state.status = STATUS_RUNNING

    def _on_colony_init(self, event: RunEvent) -> None:
        p = event.payload
        self.state.colony_size = _as_int(p.get("size"), self.state.colony_size)
        for agent_id in p.get("agents", []) or []:
            summary = self._ensure_agent(str(agent_id))
            summary.status = AGENT_WAITING

    def _on_wave_started(self, event: RunEvent) -> None:
        p = event.payload
        wave = _as_int(p.get("oleada", p.get("wave")), self.state.current_wave + 1)
        previous = self.state.current_wave
        if previous and previous != wave:
            self._set_wave_status(previous, NODE_COMPLETED)
            for summary in self.state.agents.values():
                if summary.status == AGENT_ACTIVE:
                    summary.status = AGENT_WAITING
        self.state.current_wave = wave
        self.state.status = STATUS_RUNNING
        if self._wave_entry(wave) is None:
            self._add_entry(
                TimelineEntry(
                    entry_id=f"wave:{wave}",
                    kind="wave",
                    label=f"Wave {wave}",
                    status=NODE_ACTIVE,
                    wave=wave,
                    seq=event.seq,
                )
            )
        else:
            entry = self._wave_entry(wave)
            if entry is not None:
                entry.status = NODE_ACTIVE
        for agent_id in p.get("active", []) or []:
            summary = self._ensure_agent(str(agent_id))
            if summary.status in (AGENT_WAITING, AGENT_ACTIVE):
                summary.status = AGENT_WAITING
        selected = [str(a) for a in (p.get("active", []) or [])]
        self.state.selected_agents[wave] = selected
        phase = self._ensure_phase(wave, PHASE_RESEARCH, NODE_ACTIVE)
        phase.selected = selected
        phase.status = NODE_ACTIVE
        self.state.current_phase = PHASE_RESEARCH

    def _on_wave_completed(self, event: RunEvent) -> None:
        p = event.payload
        wave = _as_int(p.get("oleada", p.get("wave")), self.state.current_wave)
        self.current_wave_done(wave)
        self.state.best_quality = _as_float(p.get("best_Q"), self.state.best_quality)
        if "total_fetches" in p:
            self.state.fetches_used = _as_int(p.get("total_fetches"), self.state.fetches_used)
        if "max_fetches" in p:
            self.state.max_fetches = _as_int(p.get("max_fetches"), self.state.max_fetches)
        if "elapsed" in p:
            self.state.elapsed_seconds = _as_float(p.get("elapsed"), self.state.elapsed_seconds)
        decision = self.state.phase(wave, PHASE_DECISION)
        if decision is not None:
            if p.get("leader"):
                decision.leader = str(p.get("leader"))
            if "q_delta" in p:
                decision.q_delta = _as_float(p.get("q_delta"), decision.q_delta)
            if p.get("continue_reason"):
                decision.continue_reason = str(p.get("continue_reason"))
            if p.get("stop_reason"):
                decision.stop_reason = str(p.get("stop_reason"))
                self.state.stop_reason = str(p.get("stop_reason"))
        if self.state.status == STATUS_EVALUATING:
            self.state.status = STATUS_RUNNING

    def current_wave_done(self, wave: int) -> None:
        self._set_wave_status(wave, NODE_COMPLETED)

    def _on_phase_started(self, event: RunEvent) -> None:
        p = event.payload
        self._saw_phase_event = True
        wave = _as_int(p.get("oleada", p.get("wave")), self.state.current_wave)
        phase = str(p.get("phase") or PHASE_RESEARCH)
        record = self._ensure_phase(wave, phase, NODE_ACTIVE)
        record.selected = [str(a) for a in (p.get("selected", []) or [])]
        self.state.selected_agents[wave] = record.selected
        self.state.status = STATUS_EVALUATING if phase == PHASE_EVALUATION else STATUS_RUNNING
        for agent_id in record.selected:
            summary = self._ensure_agent(agent_id)
            if phase == PHASE_EVALUATION and summary.status in (AGENT_WAITING, AGENT_ACTIVE):
                summary.status = AGENT_EVALUATING

    def _on_phase_completed(self, event: RunEvent) -> None:
        p = event.payload
        self._saw_phase_event = True
        wave = _as_int(p.get("oleada", p.get("wave")), self.state.current_wave)
        phase = str(p.get("phase") or PHASE_RESEARCH)
        record = self._ensure_phase(wave, phase, NODE_COMPLETED)
        record.status = NODE_COMPLETED
        for key in ("selected", "completed", "failed", "skipped"):
            if key in p:
                setattr(record, key, [str(a) for a in (p.get(key) or [])])
        if "papers_attempted" in p:
            record.papers_attempted = _as_int(p.get("papers_attempted"), 0)
        if "papers_integrated" in p:
            record.papers_integrated = _as_int(p.get("papers_integrated"), 0)
        if "evidence_added" in p:
            record.evidence_added = _as_int(p.get("evidence_added"), 0)
        if p.get("best_Q") is not None:
            record.best_q = _as_float(p.get("best_Q"), 0.0)
        if p.get("leader"):
            record.leader = str(p.get("leader"))
            self.state.winner_agent = record.leader
            for agent_id, summary in self.state.agents.items():
                summary.is_winner = agent_id == record.leader
        if "q_delta" in p:
            record.q_delta = _as_float(p.get("q_delta"), 0.0)
        if "budget_used" in p:
            record.budget_used = _as_int(p.get("budget_used"), 0)
            self.state.fetches_used = record.budget_used
        if p.get("continue_reason"):
            record.continue_reason = str(p.get("continue_reason"))
        if p.get("stop_reason"):
            record.stop_reason = str(p.get("stop_reason"))
            self.state.stop_reason = str(p.get("stop_reason"))
        if "converged" in p:
            record.converged = bool(p.get("converged"))
        if "pheromone_concentration" in p:
            record.pheromone_concentration = _as_float(
                p.get("pheromone_concentration"), record.pheromone_concentration
            )
        if "elapsed" in p:
            record.elapsed_seconds = _as_float(p.get("elapsed"), 0.0)
        if "top_ranking" in p or "ranking" in p:
            ranking = p.get("ranking") or []
            with contextlib.suppress(Exception):
                record.leader = str(ranking[0][0]) if ranking else record.leader
        entry = self._phase_entry(wave, phase)
        if entry is not None:
            entry.status = NODE_COMPLETED
            entry.label = self._phase_label(record, phase)
        self.state.current_phase = phase
        if phase == PHASE_EVALUATION and "best_Q" in p:
            with contextlib.suppress(Exception):
                self.state.best_quality = max(
                    self.state.best_quality, _as_float(p.get("best_Q"), 0.0)
                )
        if phase == PHASE_DECISION and self.state.status == STATUS_EVALUATING:
            self.state.status = STATUS_RUNNING

    def _on_phase_failed(self, event: RunEvent) -> None:
        p = event.payload
        wave = _as_int(p.get("oleada", p.get("wave")), self.state.current_wave)
        phase = str(p.get("phase") or PHASE_RESEARCH)
        record = self._ensure_phase(wave, phase, NODE_FAILED)
        record.status = NODE_FAILED
        record.reason = redact_secrets(str(p.get("reason") or p.get("error") or ""))
        entry = self._phase_entry(wave, phase)
        if entry is not None:
            entry.status = NODE_FAILED

    def _phase_label(self, record: PhaseState, phase: str) -> str:
        if phase == PHASE_RESEARCH:
            return (
                f"Research · {len(record.completed)}/{len(record.selected)} agents · "
                f"{record.papers_integrated} papers"
            )
        if phase == PHASE_EVALUATION:
            best = f" · best Q {record.best_q:.3f}" if record.best_q is not None else ""
            return (
                f"Evaluation · {len(record.completed)} complete / "
                f"{len(record.skipped)} skipped / {len(record.failed)} failed{best}"
            )
        leader = record.leader or "—"
        return (
            f"Decision · leader {leader} · Δ{record.q_delta:+.3f} · "
            f"{record.stop_reason or record.continue_reason or 'continue'}"
        )

    def _on_eval_settled(self, event: RunEvent) -> None:
        p = event.payload
        self._saw_phase_event = True
        agent_id = self._agent_id(event)
        if not agent_id:
            return
        wave = _as_int(p.get("oleada", p.get("wave")), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        status = str(p.get("status") or EVAL_COMPLETE)
        unavailable = {
            str(k): str(v) for k, v in (p.get("unavailable") or {}).items()
        }
        state = EvaluationState(
            agent_id=agent_id,
            wave=wave,
            turn=turn,
            status=status,
            reason=redact_secrets(str(p.get("reason") or "")),
            q=_as_float(p.get("Q")) if status == EVAL_COMPLETE and "Q" in p else None,
            q_delta=_as_float(p.get("delta_q")) if "delta_q" in p else None,
            components={
                key: (None if key in unavailable else _as_float(p.get(key)) if key in p else None)
                for key in ("S", "P", "J", "R")
            },
            unavailable=unavailable,
            evidence_papers=_as_int(p.get("evidence_papers"), 0),
        )
        self.state.evaluation_states[state.key] = state
        summary = self._ensure_agent(agent_id)
        summary.evaluation_status = status
        summary.evaluation_reason = state.reason
        if status == EVAL_COMPLETE and state.q is not None:
            summary.quality = state.q
            summary.delta_q = state.q_delta or 0.0
        if status == EVAL_FAILED:
            summary.status = AGENT_FAILED
        elif summary.status == AGENT_EVALUATING:
            summary.status = AGENT_WAITING

    def _on_turn_queued(self, event: RunEvent) -> None:
        agent_id = self._agent_id(event)
        if not agent_id:
            return
        summary = self._ensure_agent(agent_id, str(event.payload.get("caste", "")))
        if summary.status in (AGENT_WAITING, ""):
            summary.status = AGENT_WAITING

    def _on_turn_started(self, event: RunEvent) -> None:
        p = event.payload
        agent_id = self._agent_id(event)
        if not agent_id:
            return
        wave = _as_int(p.get("oleada", p.get("wave")), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        summary = self._ensure_agent(agent_id, str(p.get("caste", "")))
        summary.status = AGENT_ACTIVE
        summary.research_status = NODE_ACTIVE
        self._current_agent = agent_id
        self.state.current_turn = turn
        turn_entry = self._turn_entry(wave, turn, agent_id)
        if turn_entry is None:
            turn_entry = self._add_entry(
                TimelineEntry(
                    entry_id=f"turn:{wave}:{agent_id}:{turn}",
                    kind="turn",
                    label=f"{summary.label} {summary.caste}",
                    status=NODE_ACTIVE,
                    wave=wave,
                    turn=turn,
                    agent_id=agent_id,
                    parent_id=f"wave:{wave}",
                    seq=event.seq,
                )
            )
        else:
            turn_entry.status = NODE_ACTIVE

    def _on_turn_completed(self, event: RunEvent) -> None:
        p = event.payload
        agent_id = self._agent_id(event)
        if not agent_id:
            return
        summary = self._ensure_agent(agent_id)
        summary.turns += 1
        for key in ("Q", "quality"):
            if key in p:
                summary.quality = _as_float(p.get(key), summary.quality)
        summary.delta_q = _as_float(p.get("delta_q"), summary.delta_q)
        summary.budget = _as_int(p.get("budget"), summary.budget)
        fetches = _as_int(p.get("fetches"), -1)
        if fetches == 0 and event.canonical_type() == EventType.AGENT_TURN_COMPLETED:
            summary.research_status = NODE_SKIPPED
        else:
            summary.research_status = NODE_COMPLETED
        if fetches > 0:
            summary.papers_attempted += fetches
            summary.evidence_added += fetches
        if "frontier" in p:
            summary.frontier = _as_int(p.get("frontier"), summary.frontier)
            self.state.frontier_size = summary.frontier
        summary.status = AGENT_WAITING if summary.budget > 0 else AGENT_EXHAUSTED
        wave = _as_int(p.get("oleada"), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        turn_entry = self._turn_entry(wave, turn, agent_id)
        if turn_entry is not None:
            turn_entry.status = NODE_COMPLETED
        if self.state.status == STATUS_EVALUATING:
            self.state.status = STATUS_RUNNING

    def _on_turn_failed(self, event: RunEvent) -> None:
        agent_id = self._agent_id(event)
        summary = self._ensure_agent(agent_id) if agent_id else None
        if summary is not None:
            summary.status = AGENT_FAILED
            summary.research_status = NODE_FAILED
        error = redact_secrets(str(event.payload.get("error", "")))
        if error:
            self.state.failures.append(f"{agent_id or 'agent'}: {error}")
        p = event.payload
        wave = _as_int(p.get("oleada"), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        turn_entry = self._turn_entry(wave, turn, agent_id)
        if turn_entry is not None:
            turn_entry.status = NODE_FAILED

    def _on_eval_started(self, event: RunEvent) -> None:
        agent_id = self._agent_id(event)
        if agent_id:
            self._ensure_agent(agent_id).status = AGENT_EVALUATING
        self.state.status = STATUS_EVALUATING

    def _on_eval_skipped(self, event: RunEvent) -> None:
        p = event.payload
        agent_id = self._agent_id(event)
        reason = redact_secrets(str(p.get("reason", "skipped")))
        if agent_id:
            summary = self._ensure_agent(agent_id)
            summary.status = AGENT_WAITING
        wave = _as_int(p.get("oleada"), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        turn_entry = self._turn_entry(wave, turn, agent_id)
        parent = turn_entry.entry_id if turn_entry else None
        self._add_entry(
            TimelineEntry(
                entry_id=f"eval:{wave}:{agent_id}:{turn}",
                kind="evaluation",
                label=f"evaluate skipped ({reason})",
                status=NODE_SKIPPED,
                wave=wave,
                turn=turn,
                agent_id=agent_id,
                parent_id=parent,
                seq=event.seq,
                detail={"skipped": True, "reason": reason},
            )
        )
        if self.state.status == STATUS_EVALUATING:
            self.state.status = STATUS_RUNNING

    def _on_eval_completed(self, event: RunEvent) -> None:
        p = event.payload
        detail = self._parse_detail(p)
        agent_id = detail.agent_id if detail is not None else self._agent_id(event)
        if detail is None:
            return
        self._upsert_evaluation(detail)
        summary = self._ensure_agent(detail.agent_id)
        summary.quality = detail.q
        summary.delta_q = detail.delta_q
        wave = detail.oleada
        turn = detail.turn
        turn_entry = self._turn_entry(wave, turn, detail.agent_id)
        parent = turn_entry.entry_id if turn_entry else None
        entry_id = f"eval:{wave}:{agent_id}:{turn}"
        existing = self.state.entry_by_id(entry_id)
        if existing is None:
            self._add_entry(
                TimelineEntry(
                    entry_id=entry_id,
                    kind="evaluation",
                    label=f"evaluate Q={detail.q:.3f}",
                    status=NODE_COMPLETED,
                    wave=wave,
                    turn=turn,
                    agent_id=agent_id,
                    parent_id=parent,
                    seq=event.seq,
                    detail={"q": detail.q, "delta_q": detail.delta_q},
                )
            )
        else:
            existing.status = NODE_COMPLETED
            existing.label = f"evaluate Q={detail.q:.3f}"
        if summary.status == AGENT_EVALUATING:
            summary.status = AGENT_WAITING
        if self.state.status == STATUS_EVALUATING:
            self.state.status = STATUS_RUNNING

    def _parse_detail(self, payload: dict[str, Any]) -> DetailedEvaluation | None:
        raw = payload.get("detail")
        if isinstance(raw, dict):
            try:
                return DetailedEvaluation.model_validate(raw)
            except Exception:
                return None
        if isinstance(raw, str):
            try:
                return DetailedEvaluation.model_validate_json(raw)
            except Exception:
                return None
        if ("q" in payload or "Q" in payload) and "agent_id" in payload:
            try:
                return DetailedEvaluation(
                    agent_id=str(payload["agent_id"]),
                    oleada=_as_int(payload.get("oleada")),
                    turn=_as_int(payload.get("turn")),
                    q=_as_float(payload.get("Q", payload.get("q"))),
                    delta_q=_as_float(payload.get("delta_q")),
                )
            except Exception:
                return None
        return None

    def _on_eval_failed(self, event: RunEvent) -> None:
        agent_id = self._agent_id(event)
        error = redact_secrets(str(event.payload.get("error", "evaluation failed")))
        self.state.failures.append(f"{agent_id or 'agent'}: {error}")
        if agent_id:
            self._ensure_agent(agent_id).status = AGENT_FAILED
        if self.state.status == STATUS_EVALUATING:
            self.state.status = STATUS_RUNNING

    def _on_paper_step(self, event: RunEvent) -> None:
        p = event.payload
        agent_id = self._agent_id(event)
        paper_id = str(p.get("paper_id") or p.get("paper") or "")
        title = str(p.get("title") or paper_id)
        wave = _as_int(p.get("oleada"), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        summary = self._ensure_agent(agent_id) if agent_id else None
        if summary is not None:
            summary.current_paper_id = paper_id
            summary.current_paper_title = title
            if p.get("year") is not None:
                summary.current_paper_year = _as_int(p.get("year"), summary.current_paper_year or 0)
            if p.get("authors"):
                summary.current_paper_authors = [str(a) for a in p["authors"]]
            source = str(p.get("src") or p.get("source") or "")
            if source:
                summary.current_paper_source = source
            summary.budget = _as_int(p.get("budget"), summary.budget)
        if p.get("analysis") and isinstance(p["analysis"], dict) and agent_id:
            self.state.paper_analyses.setdefault(agent_id, {})[paper_id] = p["analysis"]
        canonical = event.canonical_type()
        phase = "fetch" if canonical in (
            EventType.PAPER_FETCH_STARTED,
            EventType.PAPER_FETCH_COMPLETED,
            EventType.PAPER_FETCH_FAILED,
        ) else "read"
        if canonical in (EventType.PAPER_FETCH_STARTED, EventType.PAPER_INTEGRATION_STARTED):
            status = NODE_ACTIVE
        elif canonical in (EventType.PAPER_FETCH_FAILED, EventType.PAPER_INTEGRATION_FAILED):
            status = NODE_FAILED
        else:
            status = NODE_COMPLETED
        turn_entry = self._turn_entry(wave, turn, agent_id)
        parent = turn_entry.entry_id if turn_entry else (f"wave:{wave}" if wave else None)
        entry_id = f"paper:{phase}:{agent_id}:{wave}:{turn}:{paper_id}"
        label = f"{phase} {title}"
        if phase == "read" and p.get("mode"):
            label = f"{phase} {title} [{p.get('mode')}]"
        existing = self.state.entry_by_id(entry_id)
        if existing is None:
            self._add_entry(
                TimelineEntry(
                    entry_id=entry_id,
                    kind="paper",
                    label=label,
                    status=status,
                    wave=wave,
                    turn=turn,
                    agent_id=agent_id,
                    paper_id=paper_id,
                    parent_id=parent,
                    seq=event.seq,
                    detail={
                        "title": title,
                        "mode": p.get("mode", ""),
                        "provider": p.get("provider", ""),
                        "year": p.get("year"),
                        "authors": list(p.get("authors") or []),
                        "source": p.get("src") or p.get("source") or "",
                        "analysis": p.get("analysis"),
                    },
                )
            )
        else:
            existing.status = status
            existing.label = label
            if p.get("title"):
                existing.detail["title"] = p.get("title")
            if p.get("year") is not None:
                existing.detail["year"] = p.get("year")
            if p.get("authors"):
                existing.detail["authors"] = list(p["authors"])
            source_ref = str(p.get("src") or p.get("source") or "")
            if source_ref:
                existing.detail["source"] = source_ref
            if p.get("analysis"):
                existing.detail["analysis"] = p["analysis"]

    def _on_candidate_score(self, event: RunEvent) -> None:
        p = event.payload
        with contextlib.suppress(Exception):
            self._append_candidate_score(CandidateScore.model_validate(p))

    def _on_candidate_selected(self, event: RunEvent) -> None:
        p = event.payload
        try:
            selection = CandidateSelection.model_validate(p)
        except Exception:
            return
        if event.canonical_type() == EventType.CANDIDATE_SELECTED:
            self.state.selections.append(selection)

    def _on_new_best(self, event: RunEvent) -> None:
        p = event.payload
        winner = str(p.get("agent") or p.get("winner") or "")
        if winner:
            self._ensure_agent(winner)
            self.state.winner_agent = winner
            for agent_id, summary in self.state.agents.items():
                summary.is_winner = agent_id == winner
        self.state.best_quality = _as_float(p.get("Q", p.get("best_Q")), self.state.best_quality)

    def _on_budget(self, event: RunEvent) -> None:
        p = event.payload
        self.state.fetches_used = _as_int(p.get("used", p.get("total_fetches")), self.state.fetches_used)
        self.state.max_fetches = _as_int(p.get("max_fetches"), self.state.max_fetches)
        if p.get("token_usage") is not None:
            self.state.token_usage = _as_int(p.get("token_usage"), 0)
        if p.get("cost") is not None:
            self.state.cost = _as_float(p.get("cost"), 0.0)
        if p.get("frontier") is not None:
            self.state.frontier_size = _as_int(p.get("frontier"), self.state.frontier_size)

    def _on_llm_started(self, event: RunEvent) -> None:
        p = event.payload
        self.state.current_operation = str(p.get("purpose") or p.get("operation") or "")
        self.state.current_operation_model = str(p.get("model") or "")
        self.state.current_operation_agent = self._agent_id(event)
        self.state.operation_elapsed_seconds = None

    def _on_llm_completed(self, event: RunEvent) -> None:
        p = event.payload
        self.state.current_operation = str(p.get("purpose") or self.state.current_operation)
        self.state.current_operation_model = str(p.get("model") or self.state.current_operation_model)
        self.state.current_operation_agent = self._agent_id(event) or self.state.current_operation_agent
        if "elapsed" in p:
            self.state.operation_elapsed_seconds = _as_float(p.get("elapsed"), 0.0)
        tokens = p.get("total_tokens", p.get("token_usage"))
        if tokens is not None:
            self.state.token_usage = (self.state.token_usage or 0) + _as_int(tokens, 0)
        if p.get("cost") is not None:
            self.state.cost = (self.state.cost or 0.0) + _as_float(p.get("cost"), 0.0)

    def _on_llm_failed(self, event: RunEvent) -> None:
        p = event.payload
        purpose = str(p.get("purpose") or "llm_operation")
        error = redact_secrets(str(p.get("error", "llm operation failed")))
        self.state.failures.append(f"{purpose}: {error}")
        self.state.current_operation = purpose
        self.state.current_operation_model = str(
            p.get("model") or self.state.current_operation_model
        )
        if "elapsed" in p:
            self.state.operation_elapsed_seconds = _as_float(p.get("elapsed"), 0.0)

    def _parent_for(self, wave: int, turn: int, agent_id: str) -> str | None:
        turn_entry = self._turn_entry(wave, turn, agent_id)
        if turn_entry is not None:
            return turn_entry.entry_id
        return f"wave:{wave}" if wave else None

    def _on_discovery_started(self, event: RunEvent) -> None:
        p = event.payload
        agent_id = self._agent_id(event)
        paper_id = str(p.get("paper_id") or "")
        wave = _as_int(p.get("oleada"), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        entry_id = f"discovery:{agent_id}:{wave}:{turn}:{paper_id}"
        existing = self.state.entry_by_id(entry_id)
        if existing is None:
            self._add_entry(
                TimelineEntry(
                    entry_id=entry_id,
                    kind="discovery",
                    label=f"discover {paper_id or 'paper'}",
                    status=NODE_ACTIVE,
                    wave=wave,
                    turn=turn,
                    agent_id=agent_id,
                    paper_id=paper_id,
                    parent_id=self._parent_for(wave, turn, agent_id),
                    seq=event.seq,
                )
            )
        else:
            existing.status = NODE_ACTIVE

    def _on_discovery_completed(self, event: RunEvent) -> None:
        p = event.payload
        agent_id = self._agent_id(event)
        paper_id = str(p.get("paper_id") or "")
        wave = _as_int(p.get("oleada"), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        refs = _as_int(p.get("refs"), 0)
        cits = _as_int(p.get("cits"), 0)
        traversable = _as_int(p.get("traversable"), refs + cits)
        entry_id = f"discovery:{agent_id}:{wave}:{turn}:{paper_id}"
        label = f"discover {paper_id or 'paper'} (refs={refs} cits={cits})"
        existing = self.state.entry_by_id(entry_id)
        if existing is None:
            self._add_entry(
                TimelineEntry(
                    entry_id=entry_id,
                    kind="discovery",
                    label=label,
                    status=NODE_COMPLETED,
                    wave=wave,
                    turn=turn,
                    agent_id=agent_id,
                    paper_id=paper_id,
                    parent_id=self._parent_for(wave, turn, agent_id),
                    seq=event.seq,
                    detail={"refs": refs, "cits": cits, "traversable": traversable},
                )
            )
        else:
            existing.status = NODE_COMPLETED
            existing.label = label
            existing.detail.update(
                {"refs": refs, "cits": cits, "traversable": traversable}
            )

    def _on_discovery_failed(self, event: RunEvent) -> None:
        p = event.payload
        agent_id = self._agent_id(event)
        paper_id = str(p.get("paper_id") or "")
        wave = _as_int(p.get("oleada"), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        error = redact_secrets(str(p.get("error", "neighbor discovery failed")))
        self.state.failures.append(f"{agent_id or 'agent'}: {error}")
        entry_id = f"discovery:{agent_id}:{wave}:{turn}:{paper_id}"
        existing = self.state.entry_by_id(entry_id)
        if existing is None:
            self._add_entry(
                TimelineEntry(
                    entry_id=entry_id,
                    kind="discovery",
                    label=f"discover {paper_id or 'paper'} failed",
                    status=NODE_FAILED,
                    wave=wave,
                    turn=turn,
                    agent_id=agent_id,
                    paper_id=paper_id,
                    parent_id=self._parent_for(wave, turn, agent_id),
                    seq=event.seq,
                )
            )
        else:
            existing.status = NODE_FAILED

    def _on_reference_mapping(self, event: RunEvent) -> None:
        """Surface paper-level reference mapping progress and outages."""
        p = event.payload
        paper_id = str(p.get("paper_id") or "")
        job_id = str(p.get("job_id") or "")
        status = str(p.get("status") or "")
        observed = _as_int(p.get("observed"), 0)
        mapped = _as_int(p.get("mapped"), 0)
        resolved = _as_int(p.get("resolved"), 0)
        provisional = _as_int(p.get("provisional"), 0)
        failed = _as_int(p.get("failed"), 0)
        reused = bool(p.get("reused"))
        is_failure = event.canonical_type() == EventType.REFERENCE_MAPPING_FAILED
        degraded = status in ("partial", "incomplete")
        if is_failure or status == "failed":
            error = redact_secrets(str(p.get("error") or p.get("error_code") or "")).strip()
            message = f"{paper_id or 'paper'}: reference mapping failed"
            if error:
                message += f" ({error})"
            if message not in self.state.failures:
                self.state.failures.append(message)
        elif degraded and observed > 0:
            message = (
                f"reference mapping {status} for {paper_id or 'paper'}: "
                f"mapped={mapped} resolved={resolved} provisional={provisional} failed={failed}"
            )
            if message not in self.state.warnings:
                self.state.warnings.append(message)
        if is_failure or status == "failed":
            entry_status = NODE_FAILED
        elif degraded or status == "completed":
            # Partial/incomplete mapping is terminal (no further progress will
            # arrive); the degraded annotation in the label keeps it honest
            # instead of showing an indefinite "active" step or a clean check.
            entry_status = NODE_COMPLETED
        else:
            entry_status = NODE_ACTIVE
        label = (
            f"map references {paper_id or 'paper'} "
            f"(observed={observed} mapped={mapped} resolved={resolved})"
        )
        if degraded:
            label += " [degraded]"
        if reused:
            label += " [reused]"
        agent_id = self._agent_id(event)
        wave = self.state.current_wave
        turn = self.state.current_turn
        entry_id = f"reference:{job_id or paper_id or event.seq}"
        detail = {
            "observed": observed,
            "mapped": mapped,
            "resolved": resolved,
            "provisional": provisional,
            "failed": failed,
            "status": status,
            "reused": reused,
        }
        existing = self.state.entry_by_id(entry_id)
        if existing is None:
            self._add_entry(
                TimelineEntry(
                    entry_id=entry_id,
                    kind="reference_mapping",
                    label=label,
                    status=entry_status,
                    wave=wave,
                    turn=turn,
                    agent_id=agent_id,
                    paper_id=paper_id,
                    parent_id=self._parent_for(wave, turn, agent_id),
                    seq=event.seq,
                    detail=detail,
                )
            )
        else:
            existing.status = entry_status
            existing.label = label
            existing.detail.update(detail)

    def _on_frontier_started(self, event: RunEvent) -> None:
        p = event.payload
        agent_id = self._agent_id(event)
        wave = _as_int(p.get("oleada"), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        count = _as_int(p.get("count"), 0)
        entry_id = f"frontier:{agent_id}:{wave}:{turn}:{event.seq}"
        self._open_frontier[(agent_id, wave, turn)] = entry_id
        self._add_entry(
            TimelineEntry(
                entry_id=entry_id,
                kind="frontier",
                label=f"frontier evaluation ({count} candidates)",
                status=NODE_ACTIVE,
                wave=wave,
                turn=turn,
                agent_id=agent_id,
                parent_id=self._parent_for(wave, turn, agent_id),
                seq=event.seq,
                detail={"count": count},
            )
        )

    def _on_frontier_completed(self, event: RunEvent) -> None:
        p = event.payload
        agent_id = self._agent_id(event)
        wave = _as_int(p.get("oleada"), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        count = _as_int(p.get("count"), 0)
        label = f"frontier evaluation ({count} candidates scored)"
        if p.get("degraded"):
            label += " [degraded]"
        entry_id = self._open_frontier.pop((agent_id, wave, turn), None)
        existing = self.state.entry_by_id(entry_id)
        if existing is None:
            existing = self._add_entry(
                TimelineEntry(
                    entry_id=f"frontier:{agent_id}:{wave}:{turn}:{event.seq}",
                    kind="frontier",
                    label=label,
                    wave=wave,
                    turn=turn,
                    agent_id=agent_id,
                    parent_id=self._parent_for(wave, turn, agent_id),
                    seq=event.seq,
                    detail={"count": count},
                )
            )
        existing.status = NODE_COMPLETED
        existing.label = label
        existing.detail["count"] = count

    def _on_frontier_failed(self, event: RunEvent) -> None:
        p = event.payload
        agent_id = self._agent_id(event)
        wave = _as_int(p.get("oleada"), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        error = redact_secrets(str(p.get("error", "frontier evaluation failed")))
        self.state.failures.append(f"{agent_id or 'agent'}: {error}")
        entry_id = self._open_frontier.pop((agent_id, wave, turn), None)
        existing = self.state.entry_by_id(entry_id)
        if existing is None:
            existing = self._add_entry(
                TimelineEntry(
                    entry_id=f"frontier:{agent_id}:{wave}:{turn}:{event.seq}",
                    kind="frontier",
                    label="frontier evaluation failed",
                    wave=wave,
                    turn=turn,
                    agent_id=agent_id,
                    parent_id=self._parent_for(wave, turn, agent_id),
                    seq=event.seq,
                )
            )
        existing.status = NODE_FAILED

    def _on_artifact(self, event: RunEvent) -> None:
        p = event.payload
        if p.get("kind") != "narrative":
            return
        name = str(p.get("name", ""))
        match = _NARRATIVE_RE.match(name)
        agent_id = match.group("agent") if match else ""
        content = str(p.get("content", ""))
        if agent_id:
            self.state.narratives[agent_id] = content
        if content:
            self.state.narratives.setdefault("__latest__", content)

    def _on_warning(self, event: RunEvent) -> None:
        p = event.payload
        text = redact_secrets(str(p.get("error") or p.get("reason") or p.get("message") or event.type))
        classification = str(p.get("classification", "warning"))
        paper_id = p.get("paper_id")
        label = text if not paper_id else f"{paper_id}: {text}"
        if classification in ("fatal", "error", "failed"):
            self.state.failures.append(label)
        else:
            self.state.warnings.append(label)
        wave = _as_int(p.get("oleada"), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        turn_entry = self._turn_entry(wave, turn, self._agent_id(event))
        self._add_entry(
            TimelineEntry(
                entry_id=f"warning:{event.seq}",
                kind="warning",
                label=label,
                status=NODE_FAILED if classification in ("fatal", "error", "failed") else NODE_SKIPPED,
                wave=wave,
                turn=turn,
                agent_id=self._agent_id(event),
                paper_id=str(paper_id or ""),
                parent_id=turn_entry.entry_id if turn_entry else (f"wave:{wave}" if wave else None),
                seq=event.seq,
            )
        )

    def _on_metadata_transit(self, event: RunEvent) -> None:
        p = event.payload
        agent_id = self._agent_id(event)
        paper_id = str(p.get("paper_id", ""))
        wave = _as_int(p.get("oleada"), self.state.current_wave)
        turn = _as_int(p.get("turn"), self.state.current_turn)
        turn_entry = self._turn_entry(wave, turn, agent_id)
        self._add_entry(
            TimelineEntry(
                entry_id=f"paper:transit:{event.seq}:{paper_id}",
                kind="paper",
                label=f"transit {paper_id}",
                status=NODE_SKIPPED,
                wave=wave,
                turn=turn,
                agent_id=agent_id,
                paper_id=paper_id,
                parent_id=turn_entry.entry_id if turn_entry else (f"wave:{wave}" if wave else None),
                seq=event.seq,
            )
        )

    def _on_run_completed(self, event: RunEvent) -> None:
        p = event.payload
        winner = str(p.get("winner") or self.state.winner_agent)
        if winner:
            self._ensure_agent(winner)
            self.state.winner_agent = winner
            for agent_id, summary in self.state.agents.items():
                summary.is_winner = agent_id == winner
        self.state.best_quality = _as_float(p.get("peak_Q", p.get("best_Q", p.get("Q"))), self.state.best_quality)
        self._apply_terminal_fields(p)
        if not self._saw_phase_event and self.state.current_wave:
            self.state.legacy_projection = True
        if "total_fetches" in p:
            self.state.fetches_used = _as_int(p.get("total_fetches"), self.state.fetches_used)
        if "elapsed" in p:
            self.state.elapsed_seconds = _as_float(p.get("elapsed"), self.state.elapsed_seconds)
        status = str(p.get("status", STATUS_COMPLETED))
        self.state.status = status if status in (
            STATUS_COMPLETED, STATUS_CANCELLED, STATUS_FAILED
        ) else STATUS_COMPLETED
        if self.state.current_wave:
            self._set_wave_status(self.state.current_wave, NODE_COMPLETED)
        self._finalize_agents()

    def _on_no_winner(self, event: RunEvent) -> None:
        """Project a completed-but-degraded run that produced no winner.

        New traces arrive enriched with ``status``/``outcome``/``reason`` plus a
        preceding warning event. Old traces carry only ``run_id``/``elapsed`` and
        still project as completed with exactly one generic, non-duplicated
        explanation.
        """
        p = event.payload
        self._apply_terminal_fields(p, default_outcome=OUTCOME_DEGRADED)
        if not self._saw_phase_event and self.state.current_wave:
            self.state.legacy_projection = True
        enriched = bool(p.get("reason") or p.get("reason_code"))
        if not enriched:
            reason = reason_text(REASON_NO_WINNER)
            self.state.terminal_reason = reason
            if reason not in self.state.warnings:
                self.state.warnings.append(reason)
                self._add_entry(
                    TimelineEntry(
                        entry_id=f"warning:{event.seq or 'legacy'}",
                        kind="warning",
                        label=reason,
                        status=NODE_SKIPPED,
                        seq=event.seq,
                    )
                )
        if "total_fetches" in p:
            self.state.fetches_used = _as_int(p.get("total_fetches"), self.state.fetches_used)
        if "elapsed" in p:
            self.state.elapsed_seconds = _as_float(p.get("elapsed"), self.state.elapsed_seconds)
        self.state.status = STATUS_COMPLETED
        if self.state.current_wave:
            self._set_wave_status(self.state.current_wave, NODE_COMPLETED)
        self._finalize_agents()

    def _apply_terminal_fields(self, p: dict[str, Any], default_outcome: str = OUTCOME_COMPLETED) -> None:
        self.state.outcome = str(p.get("outcome") or default_outcome)
        reason_code = str(p.get("reason_code") or "")
        if reason_code:
            self.state.reason_code = reason_code
        reason = str(p.get("reason") or "")
        if reason:
            self.state.terminal_reason = reason
        elif reason_code:
            self.state.terminal_reason = reason_text(reason_code)
        if p.get("stop_reason"):
            self.state.stop_reason = str(p.get("stop_reason"))
        if "total_waves" in p:
            self.state.total_waves = _as_int(p.get("total_waves"), self.state.total_waves)

    def _on_run_failed(self, event: RunEvent) -> None:
        error = redact_secrets(str(event.payload.get("error", "run failed")))
        self.state.failures.append(error)
        self.state.status = STATUS_FAILED
        self._finalize_agents()

    def _on_run_cancelled(self, event: RunEvent) -> None:
        self.state.status = STATUS_CANCELLED
        self._finalize_agents()

    def _on_run_interrupted(self, event: RunEvent) -> None:
        """Present a stale/heartbeat-reconciled run without fabricating a winner."""
        reason = redact_secrets(str(event.payload.get("reason", ""))).strip()
        if reason:
            self.state.terminal_reason = reason
        self.state.status = STATUS_INTERRUPTED
        self.state.outcome = str(event.payload.get("outcome") or self.state.outcome)
        self._finalize_agents()

    def _on_status(self, event: RunEvent) -> None:
        status = str(event.payload.get("status", ""))
        if status in (
            STATUS_RUNNING, STATUS_EVALUATING, STATUS_COMPLETED,
            STATUS_CANCELLED, STATUS_FAILED, "converged", "exhausted",
        ):
            self.state.status = status

    # ---- examination / survivor (TUI-1) ----------------------------------

    def _exam_phase(self, name: str) -> PhaseState:
        record = self.state.exam_phases.get(name)
        if record is None:
            record = PhaseState(wave=0, phase=name)
            self.state.exam_phases[name] = record
        return record

    def _on_evidence_pack_frozen(self, event: RunEvent) -> None:
        record = self._exam_phase(PHASE_EXAM_BUILD)
        record.status = NODE_ACTIVE
        record.papers_integrated = _as_int(
            event.payload.get("source_count"), record.papers_integrated
        )
        record.evidence_added = _as_int(
            event.payload.get("source_count"), record.evidence_added
        )
        self.state.current_phase = PHASE_EXAM_BUILD

    def _on_exam_generated(self, event: RunEvent) -> None:
        record = self._exam_phase(PHASE_EXAM_BUILD)
        record.status = NODE_ACTIVE
        record.papers_attempted = _as_int(
            event.payload.get("item_count"), record.papers_attempted
        )
        self.state.current_phase = PHASE_EXAM_BUILD

    def _on_exam_validated(self, event: RunEvent) -> None:
        record = self._exam_phase(PHASE_EXAM_BUILD)
        record.status = NODE_COMPLETED
        self.state.exam_rejected_count = _as_int(
            event.payload.get("rejected"), self.state.exam_rejected_count
        )

    def _on_exam_partitioned(self, event: RunEvent) -> None:
        self.state.exam_selection_count = _as_int(
            event.payload.get("selection_count"), self.state.exam_selection_count
        )
        self.state.exam_holdout_count = _as_int(
            event.payload.get("holdout_count"), self.state.exam_holdout_count
        )
        record = self._exam_phase(PHASE_SELECTION)
        record.status = NODE_ACTIVE
        self.state.current_phase = PHASE_SELECTION

    def _on_candidate_test_completed(self, event: RunEvent) -> None:
        record = self._exam_phase(PHASE_SELECTION)
        agent = str(event.payload.get("agent_id") or "")
        if agent and agent not in record.completed:
            record.completed.append(agent)

    def _on_survivor_selected(self, event: RunEvent) -> None:
        p = event.payload
        self.state.survivor_id = str(p.get("survivor_id") or "")
        self.state.survivor_terminal_score = _as_optional_float(p.get("terminal_score"))
        self.state.survivor_selection_score = _as_optional_float(
            p.get("selection_accuracy")
        )
        self.state.survivor_process_score = _as_optional_float(p.get("process_score"))
        self.state.survivor_grounding_score = _as_optional_float(
            p.get("grounding_score")
        )
        ranking = p.get("ranking")
        if isinstance(ranking, list):
            self.state.survivor_ranking = [str(item) for item in ranking]
        selection = self.state.exam_phases.get(PHASE_SELECTION)
        if selection is not None:
            selection.status = NODE_COMPLETED
        record = self._exam_phase(PHASE_SURVIVOR)
        record.status = NODE_COMPLETED
        record.leader = self.state.survivor_id
        self.state.current_phase = PHASE_BENCHMARK

    def _on_baseline_completed(self, event: RunEvent) -> None:
        p = event.payload
        self.state.survivor_accuracy = _as_optional_float(p.get("survivor_accuracy"))
        self.state.naive_accuracy = _as_optional_float(p.get("naive_accuracy"))
        self.state.uplift = _as_optional_float(p.get("uplift"))

    def _on_benchmark_completed(self, event: RunEvent) -> None:
        p = event.payload
        self.state.benchmark_outcome = str(p.get("outcome") or "")
        self.state.benchmark_reason_code = str(p.get("reason_code") or "")
        self.state.benchmark_reason = str(p.get("reason") or "")
        if "survivor_accuracy" in p:
            self.state.survivor_accuracy = _as_optional_float(p.get("survivor_accuracy"))
        if "naive_accuracy" in p:
            self.state.naive_accuracy = _as_optional_float(p.get("naive_accuracy"))
        if "uplift" in p:
            self.state.uplift = _as_optional_float(p.get("uplift"))
        record = self._exam_phase(PHASE_BENCHMARK)
        record.status = NODE_COMPLETED
        self.state.current_phase = PHASE_BENCHMARK

    def _finalize_agents(self) -> None:
        for summary in self.state.agents.values():
            if summary.status == AGENT_FAILED:
                continue
            if summary.status == AGENT_ACTIVE or summary.status in (AGENT_WAITING, AGENT_EVALUATING) or summary.status == AGENT_EXHAUSTED:
                summary.status = AGENT_COMPLETED

    def apply_many(self, events: Iterable[RunEvent | dict]) -> None:
        for raw in events:
            event = raw if isinstance(raw, RunEvent) else RunEvent.model_validate(raw)
            self.apply(event)

    def to_dict(self) -> dict:
        return json.loads(self.state.model_dump_json())


_HANDLERS: dict[str, Any] = {
    EventType.RUN_STARTED: RunProjection._on_run_started,
    EventType.SEED_ROUTING_STARTED: RunProjection._on_seed_routing,
    EventType.SEED_ROUTING_COMPLETED: RunProjection._on_seed_routing,
    EventType.SEED_ROUTING_FAILED: RunProjection._on_run_failed,
    EventType.COLONY_INIT_STARTED: RunProjection._on_colony_init,
    EventType.COLONY_INIT_COMPLETED: RunProjection._on_colony_init,
    EventType.WAVE_STARTED: RunProjection._on_wave_started,
    EventType.WAVE_COMPLETED: RunProjection._on_wave_completed,
    EventType.WAVE_PHASE_STARTED: RunProjection._on_phase_started,
    EventType.WAVE_PHASE_COMPLETED: RunProjection._on_phase_completed,
    EventType.WAVE_PHASE_FAILED: RunProjection._on_phase_failed,
    EventType.EVALUATION_SETTLED: RunProjection._on_eval_settled,
    EventType.AGENT_TURN_QUEUED: RunProjection._on_turn_queued,
    EventType.AGENT_TURN_STARTED: RunProjection._on_turn_started,
    EventType.AGENT_TURN_COMPLETED: RunProjection._on_turn_completed,
    EventType.AGENT_TURN_FAILED: RunProjection._on_turn_failed,
    EventType.CANDIDATE_SET_SCORED: RunProjection._on_candidate_score,
    EventType.CANDIDATE_SELECTED: RunProjection._on_candidate_selected,
    EventType.PAPER_FETCH_STARTED: RunProjection._on_paper_step,
    EventType.PAPER_FETCH_COMPLETED: RunProjection._on_paper_step,
    EventType.PAPER_FETCH_FAILED: RunProjection._on_paper_step,
    EventType.PAPER_INTEGRATION_STARTED: RunProjection._on_paper_step,
    EventType.PAPER_INTEGRATION_COMPLETED: RunProjection._on_paper_step,
    EventType.PAPER_INTEGRATION_FAILED: RunProjection._on_paper_step,
    EventType.QUALITY_EVAL_STARTED: RunProjection._on_eval_started,
    EventType.QUALITY_EVAL_COMPLETED: RunProjection._on_eval_completed,
    EventType.QUALITY_EVAL_SKIPPED: RunProjection._on_eval_skipped,
    EventType.QUALITY_EVAL_FAILED: RunProjection._on_eval_failed,
    EventType.NEW_BEST: RunProjection._on_new_best,
    EventType.BUDGET_SNAPSHOT: RunProjection._on_budget,
    EventType.LLM_STARTED: RunProjection._on_llm_started,
    EventType.LLM_COMPLETED: RunProjection._on_llm_completed,
    EventType.LLM_FAILED: RunProjection._on_llm_failed,
    EventType.NEIGHBOR_DISCOVERY_STARTED: RunProjection._on_discovery_started,
    EventType.NEIGHBOR_DISCOVERY_STARTED: RunProjection._on_discovery_started,
    EventType.NEIGHBOR_DISCOVERY_COMPLETED: RunProjection._on_discovery_completed,
    EventType.NEIGHBOR_DISCOVERY_FAILED: RunProjection._on_discovery_failed,
    EventType.FRONTIER_EVAL_STARTED: RunProjection._on_frontier_started,
    EventType.FRONTIER_EVAL_COMPLETED: RunProjection._on_frontier_completed,
    EventType.FRONTIER_EVAL_FAILED: RunProjection._on_frontier_failed,
    EventType.REFERENCE_MAPPING_STARTED: RunProjection._on_reference_mapping,
    EventType.REFERENCE_MAPPING_COMPLETED: RunProjection._on_reference_mapping,
    EventType.REFERENCE_MAPPING_REUSED: RunProjection._on_reference_mapping,
    EventType.REFERENCE_MAPPING_FAILED: RunProjection._on_reference_mapping,
    EventType.RUN_COMPLETED: RunProjection._on_run_completed,
    EventType.RUN_FAILED: RunProjection._on_run_failed,
    EventType.RUN_CANCELLED: RunProjection._on_run_cancelled,
    EventType.RUN_INTERRUPTED: RunProjection._on_run_interrupted,
    EventType.ARTIFACT_SAVED: RunProjection._on_artifact,
    EventType.WARNING: RunProjection._on_warning,
    EventType.EVIDENCE_PACK_FROZEN: RunProjection._on_evidence_pack_frozen,
    EventType.EXAM_GENERATED: RunProjection._on_exam_generated,
    EventType.EXAM_VALIDATED: RunProjection._on_exam_validated,
    EventType.EXAM_PARTITIONED: RunProjection._on_exam_partitioned,
    EventType.CANDIDATE_TEST_COMPLETED: RunProjection._on_candidate_test_completed,
    EventType.SURVIVOR_SELECTED: RunProjection._on_survivor_selected,
    EventType.BASELINE_COMPLETED: RunProjection._on_baseline_completed,
    EventType.BENCHMARK_COMPLETED: RunProjection._on_benchmark_completed,
    "status": RunProjection._on_status,
    "provider_failure": RunProjection._on_warning,
    "id_title_mismatch": RunProjection._on_warning,
    "metadata_transit": RunProjection._on_metadata_transit,
    "agent_step": RunProjection._on_paper_step,
    "candidate_score": RunProjection._on_candidate_score,
    "trace_write_failed": RunProjection._on_warning,
}
