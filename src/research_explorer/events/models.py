"""Typed run events and the projected run view state.

The event contract is intentionally UI-independent: orchestration publishes
:class:`RunEvent` values through an :class:`~research_explorer.events.sink.EventSink`
and a pure projection reduces them into :class:`RunViewState`. The Textual layer
renders the projected state and never reads orchestrator internals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from research_explorer.replay.models import (
    CandidateScore,
    CandidateSelection,
    DetailedEvaluation,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventType:
    """Canonical event type names used by the projection.

    Legacy trace names (``orchestrator_start``, ``oleada_start``, ...) remain
    valid durable events for backward compatibility; the projection normalizes
    them onto these canonical semantics.
    """

    RUN_STARTED = "run_started"
    SEED_ROUTING_STARTED = "seed_routing_started"
    SEED_ROUTING_COMPLETED = "seed_routing_completed"
    SEED_ROUTING_FAILED = "seed_routing_failed"
    COLONY_INIT_STARTED = "colony_init_started"
    COLONY_INIT_COMPLETED = "colony_init_completed"
    WAVE_STARTED = "wave_started"
    WAVE_COMPLETED = "wave_completed"
    AGENT_TURN_QUEUED = "agent_turn_queued"
    AGENT_TURN_STARTED = "agent_turn_started"
    AGENT_TURN_COMPLETED = "agent_turn_completed"
    AGENT_TURN_FAILED = "agent_turn_failed"
    CANDIDATE_SET_SCORED = "candidate_set_scored"
    CANDIDATE_SELECTED = "candidate_selected"
    PAPER_FETCH_STARTED = "paper_fetch_started"
    PAPER_FETCH_COMPLETED = "paper_fetch_completed"
    PAPER_FETCH_FAILED = "paper_fetch_failed"
    PAPER_INTEGRATION_STARTED = "paper_integration_started"
    PAPER_INTEGRATION_COMPLETED = "paper_integration_completed"
    PAPER_INTEGRATION_FAILED = "paper_integration_failed"
    NEIGHBOR_DISCOVERY_STARTED = "neighbor_discovery_started"
    NEIGHBOR_DISCOVERY_COMPLETED = "neighbor_discovery_completed"
    NEIGHBOR_DISCOVERY_FAILED = "neighbor_discovery_failed"
    FRONTIER_EVAL_STARTED = "frontier_reference_evaluation_started"
    FRONTIER_EVAL_COMPLETED = "frontier_reference_evaluation_completed"
    FRONTIER_EVAL_FAILED = "frontier_reference_evaluation_failed"
    QUALITY_EVAL_STARTED = "quality_evaluation_started"
    QUALITY_EVAL_COMPLETED = "quality_evaluation_completed"
    QUALITY_EVAL_SKIPPED = "quality_evaluation_skipped"
    QUALITY_EVAL_FAILED = "quality_evaluation_failed"
    NEW_BEST = "new_best"
    BUDGET_SNAPSHOT = "budget_snapshot"
    RUN_COMPLETED = "run_completed"
    RUN_CANCELLED = "run_cancelled"
    RUN_FAILED = "run_failed"
    LLM_STARTED = "llm_operation_started"
    LLM_COMPLETED = "llm_operation_completed"
    LLM_FAILED = "llm_operation_failed"
    ARTIFACT_SAVED = "artifact_saved"
    WARNING = "warning"


LEGACY_TYPES: dict[str, str] = {
    "orchestrator_start": EventType.RUN_STARTED,
    "seed_routed": EventType.SEED_ROUTING_COMPLETED,
    "colony_initialized": EventType.COLONY_INIT_COMPLETED,
    "oleada_start": EventType.WAVE_STARTED,
    "oleada_complete": EventType.WAVE_COMPLETED,
    "agent_turn_start": EventType.AGENT_TURN_STARTED,
    "agent_turn_complete": EventType.AGENT_TURN_COMPLETED,
    "agent_turn_skipped": EventType.AGENT_TURN_COMPLETED,
    "evaluation_skipped": EventType.QUALITY_EVAL_SKIPPED,
    "evaluation_complete": EventType.QUALITY_EVAL_COMPLETED,
    "evaluation_detail": EventType.QUALITY_EVAL_COMPLETED,
    "agent_step": EventType.PAPER_INTEGRATION_COMPLETED,
    "orchestrator_complete": EventType.RUN_COMPLETED,
    "no_winner": EventType.RUN_COMPLETED,
}

STATUS_INITIALIZING = "initializing"
STATUS_RUNNING = "running"
STATUS_EVALUATING = "evaluating"
STATUS_CONVERGED = "converged"
STATUS_EXHAUSTED = "exhausted"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"

OUTCOME_COMPLETED = "completed"
OUTCOME_DEGRADED = "degraded"

# Exactly one primary reason is attached to an empty initial frontier so the
# terminal state is actionable instead of an opaque ``no_winner``.
REASON_NO_NEIGHBORS_DISCOVERED = "no_neighbors_discovered"
REASON_NO_TRAVERSABLE_IDENTIFIERS = "no_traversable_identifiers"
REASON_REFERENCE_EXTRACTION_FAILED = "reference_extraction_failed"
REASON_SEED_DISCOVERY_FAILED = "seed_discovery_failed"
REASON_NO_WINNER = "no_winner"

REASON_TEXT: dict[str, str] = {
    REASON_NO_NEIGHBORS_DISCOVERED: (
        "the seed paper exposed no reference or citation neighbors"
    ),
    REASON_NO_TRAVERSABLE_IDENTIFIERS: (
        "discovered neighbors carried no traversable DOI or arXiv identifier"
    ),
    REASON_REFERENCE_EXTRACTION_FAILED: (
        "reference extraction from the seed bibliography produced no usable entries"
    ),
    REASON_SEED_DISCOVERY_FAILED: (
        "seed neighbor discovery failed during colony initialization"
    ),
    REASON_NO_WINNER: (
        "the run finished without a winning narrative"
    ),
}


def reason_text(code: str) -> str:
    """Human-readable explanation for a terminal reason code."""
    return REASON_TEXT.get(code, REASON_TEXT[REASON_NO_WINNER])


AGENT_ACTIVE = "active"
AGENT_EVALUATING = "evaluating"
AGENT_WAITING = "waiting"
AGENT_EXHAUSTED = "exhausted"
AGENT_COMPLETED = "completed"
AGENT_FAILED = "failed"

NODE_PENDING = "pending"
NODE_ACTIVE = "active"
NODE_COMPLETED = "completed"
NODE_FAILED = "failed"
NODE_SKIPPED = "skipped"


class RunEvent(BaseModel):
    """One normalized, ordered run event."""

    seq: int = 0
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: str = Field(default_factory=utc_now)

    def canonical_type(self) -> str:
        return LEGACY_TYPES.get(self.type, self.type)


class AgentSummary(BaseModel):
    """Projected per-agent summary for tabs and navigation."""

    agent_id: str
    label: str
    caste: str = "mixto"
    quality: float = 0.0
    delta_q: float = 0.0
    status: str = AGENT_WAITING
    is_winner: bool = False
    turns: int = 0
    current_paper_id: str = ""
    current_paper_title: str = ""
    current_paper_year: int | None = None
    current_paper_authors: list[str] = Field(default_factory=list)
    current_paper_source: str = ""
    budget: int = 0
    frontier: int = 0


class TimelineEntry(BaseModel):
    """One selectable node in the hierarchical execution timeline."""

    entry_id: str
    kind: str
    label: str
    status: str = NODE_PENDING
    wave: int = 0
    turn: int = 0
    agent_id: str = ""
    paper_id: str = ""
    seq: int = 0
    parent_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class RunViewState(BaseModel):
    """The complete display state projected from ordered run events."""

    run_id: str = ""
    seed_paper_id: str = ""
    query: str = ""
    pipeline: str = "aco"
    status: str = STATUS_INITIALIZING
    outcome: str = OUTCOME_COMPLETED
    reason_code: str = ""
    terminal_reason: str = ""
    total_waves: int = 0
    started_at: str = ""
    elapsed_seconds: float = 0.0
    fetches_used: int = 0
    max_fetches: int = 0
    budget_type: str = "fetches"
    max_time_seconds: int = 0
    current_wave: int = 0
    current_turn: int = 0
    best_quality: float = 0.0
    winner_agent: str = ""
    current_operation: str = ""
    current_operation_model: str = ""
    current_operation_agent: str = ""
    operation_elapsed_seconds: float | None = None
    frontier_size: int = 0
    explorer_model: str = ""
    judge_model: str = ""
    colony_size: int = 0
    max_concurrent: int = 0
    k_per_turn: int = 0
    agents: dict[str, AgentSummary] = Field(default_factory=dict)
    agent_order: list[str] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    evaluations: dict[str, list[DetailedEvaluation]] = Field(default_factory=dict)
    candidate_scores: list[CandidateScore] = Field(default_factory=list)
    selections: list[CandidateSelection] = Field(default_factory=list)
    narratives: dict[str, str] = Field(default_factory=dict)
    paper_analyses: dict[str, dict[str, dict]] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    events: list[RunEvent] = Field(default_factory=list)
    token_usage: int | None = None
    cost: float | None = None
    follow_live: bool = True
    selected_entry_id: str | None = None

    def ordered_agents(self) -> list[AgentSummary]:
        return [self.agents[a] for a in self.agent_order if a in self.agents]

    def timeline_children(self, parent_id: str | None) -> list[TimelineEntry]:
        return [e for e in self.timeline if e.parent_id == parent_id]

    def entry_by_id(self, entry_id: str | None) -> TimelineEntry | None:
        if entry_id is None:
            return None
        for entry in self.timeline:
            if entry.entry_id == entry_id:
                return entry
        return None

    def default_entry(self) -> TimelineEntry | None:
        """The newest timeline node, used by follow-live mode."""
        return self.timeline[-1] if self.timeline else None

    def narrative_for(self, agent_id: str) -> str:
        return self.narratives.get(agent_id, "")
