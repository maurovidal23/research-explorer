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

from research_explorer.events.models import (
    AGENT_ACTIVE,
    AGENT_COMPLETED,
    AGENT_EVALUATING,
    AGENT_EXHAUSTED,
    AGENT_FAILED,
    AGENT_WAITING,
    GENERIC_NO_WINNER_REASON,
    NODE_ACTIVE,
    NODE_COMPLETED,
    NODE_FAILED,
    NODE_PENDING,
    NODE_SKIPPED,
    OUTCOME_DEGRADED,
    OUTCOME_OK,
    REASON_LABELS,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_EVALUATING,
    STATUS_FAILED,
    STATUS_RUNNING,
    AgentSummary,
    EventType,
    RunEvent,
    RunViewState,
    TimelineEntry,
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


class RunProjection:
    """Reduces ordered events into a :class:`RunViewState`."""

    def __init__(self, state: RunViewState | None = None) -> None:
        self.state = state if state is not None else RunViewState()
        self._seen_seq: set[int] = set()
        self._open_frontier: dict[tuple[str, int, int], str] = {}

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
            if event.seq in self._seen_seq:
                return
            self._seen_seq.add(event.seq)
        if event.payload:
            event = event.model_copy(update={"payload": redact_obj(event.payload)})
        self.state.events.append(event)
        handler = _HANDLERS.get(event.canonical_type())
        if handler is not None:
            try:
                handler(self, event)
            except Exception:  # pragma: no cover - projection must never crash
                self.state.warnings.append(
                    redact_secrets(f"projection_error type={event.type}")
                )
        self._sync_follow()

    # ---- helpers ----------------------------------------------------------

    def _agent_id(self, event: RunEvent) -> str:
        return str(
            event.payload.get("agent_id")
            or event.payload.get("agent")
            or event.payload.get("winner")
            or ""
        )

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
        if self.state.status == STATUS_EVALUATING:
            self.state.status = STATUS_RUNNING

    def current_wave_done(self, wave: int) -> None:
        self._set_wave_status(wave, NODE_COMPLETED)

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
                    },
                )
            )
        else:
            existing.status = status
            existing.label = label
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
            self.state.candidate_scores.append(CandidateScore.model_validate(p))

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
        entry_id = f"discovery:{agent_id}:{wave}:{turn}:{paper_id}"
        label = f"discover {paper_id or 'paper'} (refs={refs} cits={cits}"
        detail: dict[str, Any] = {"refs": refs, "cits": cits}
        if "traversable" in p:
            traversable = _as_int(p.get("traversable"), 0)
            label += f" traversable={traversable}"
            detail["traversable"] = traversable
        label += ")"
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
                    detail=detail,
                )
            )
        else:
            existing.status = NODE_COMPLETED
            existing.label = label
            existing.detail.update(detail)

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
        if "total_fetches" in p:
            self.state.fetches_used = _as_int(p.get("total_fetches"), self.state.fetches_used)
        if "elapsed" in p:
            self.state.elapsed_seconds = _as_float(p.get("elapsed"), self.state.elapsed_seconds)
        status = str(p.get("status", STATUS_COMPLETED))
        self.state.status = status if status in (
            STATUS_COMPLETED, STATUS_CANCELLED, STATUS_FAILED
        ) else STATUS_COMPLETED
        self._apply_degraded_outcome(event, winner)
        if self.state.current_wave:
            self._set_wave_status(self.state.current_wave, NODE_COMPLETED)
        self._finalize_agents()

    def _apply_degraded_outcome(self, event: RunEvent, winner: str) -> None:
        """Project the terminal reason for a completed but degraded run.

        Enriched ``no_winner`` events carry ``outcome``/``reason_code``/``reason``
        and are preceded by a ``warning`` event with the same reason. Legacy
        unenriched ``no_winner`` events still project as completed and receive a
        single generic, non-duplicated explanation.
        """
        p = event.payload
        outcome = str(p.get("outcome") or "").strip()
        reason_code = str(p.get("reason_code") or "").strip()
        reason = str(p.get("reason") or "").strip()
        if not reason and reason_code:
            reason = REASON_LABELS.get(reason_code, reason_code)
        legacy = event.type == "no_winner"
        if not reason and not winner and (legacy or outcome == OUTCOME_DEGRADED):
            reason = GENERIC_NO_WINNER_REASON
        if outcome:
            self.state.outcome = outcome
        elif legacy and not winner:
            self.state.outcome = OUTCOME_DEGRADED
        elif winner:
            self.state.outcome = OUTCOME_OK
        if reason_code and not self.state.terminal_reason_code:
            self.state.terminal_reason_code = reason_code
        if reason and not self.state.terminal_reason:
            self.state.terminal_reason = reason
        if reason and not winner:
            self._ensure_warning(reason)

    def _ensure_warning(self, text: str) -> None:
        if not text:
            return
        for existing in self.state.warnings:
            if text in existing or existing in text:
                return
        self.state.warnings.append(text)

    def _on_run_failed(self, event: RunEvent) -> None:
        error = redact_secrets(str(event.payload.get("error", "run failed")))
        self.state.failures.append(error)
        self.state.status = STATUS_FAILED
        self._finalize_agents()

    def _on_run_cancelled(self, event: RunEvent) -> None:
        self.state.status = STATUS_CANCELLED
        self._finalize_agents()

    def _on_status(self, event: RunEvent) -> None:
        status = str(event.payload.get("status", ""))
        if status in (
            STATUS_RUNNING, STATUS_EVALUATING, STATUS_COMPLETED,
            STATUS_CANCELLED, STATUS_FAILED, "converged", "exhausted",
        ):
            self.state.status = status

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
    EventType.NEIGHBOR_DISCOVERY_COMPLETED: RunProjection._on_discovery_completed,
    EventType.NEIGHBOR_DISCOVERY_FAILED: RunProjection._on_discovery_failed,
    EventType.FRONTIER_EVAL_STARTED: RunProjection._on_frontier_started,
    EventType.FRONTIER_EVAL_COMPLETED: RunProjection._on_frontier_completed,
    EventType.FRONTIER_EVAL_FAILED: RunProjection._on_frontier_failed,
    EventType.RUN_COMPLETED: RunProjection._on_run_completed,
    EventType.RUN_FAILED: RunProjection._on_run_failed,
    EventType.RUN_CANCELLED: RunProjection._on_run_cancelled,
    EventType.ARTIFACT_SAVED: RunProjection._on_artifact,
    EventType.WARNING: RunProjection._on_warning,
    "status": RunProjection._on_status,
    "provider_failure": RunProjection._on_warning,
    "id_title_mismatch": RunProjection._on_warning,
    "metadata_transit": RunProjection._on_metadata_transit,
    "agent_step": RunProjection._on_paper_step,
    "candidate_score": RunProjection._on_candidate_score,
    "trace_write_failed": RunProjection._on_warning,
}
