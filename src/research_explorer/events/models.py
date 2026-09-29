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
    RESEARCH_SCOPE_RESOLVED = "research_scope_resolved"
    SEED_ROUTING_STARTED = "seed_routing_started"
    SEED_ROUTING_COMPLETED = "seed_routing_completed"
    SEED_ROUTING_FAILED = "seed_routing_failed"
    COLONY_INIT_STARTED = "colony_init_started"
    COLONY_INIT_COMPLETED = "colony_init_completed"
    WAVE_STARTED = "wave_started"
    WAVE_COMPLETED = "wave_completed"
    WAVE_PHASE_STARTED = "wave_phase_started"
    WAVE_PHASE_COMPLETED = "wave_phase_completed"
    WAVE_PHASE_FAILED = "wave_phase_failed"
    EVALUATION_SETTLED = "evaluation_settled"
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
    RUN_INTERRUPTED = "run_interrupted"
    LLM_STARTED = "llm_operation_started"
    LLM_COMPLETED = "llm_operation_completed"
    LLM_FAILED = "llm_operation_failed"
    REFERENCE_MAPPING_STARTED = "reference_mapping_started"
    REFERENCE_MAPPING_REUSED = "reference_mapping_reused"
    REFERENCE_BATCH_STARTED = "reference_batch_started"
    REFERENCE_BATCH_COMPLETED = "reference_batch_completed"
    REFERENCE_BATCH_FAILED = "reference_batch_failed"
    REFERENCE_ENTRY_MAPPED = "reference_entry_mapped"
    REFERENCE_ENTRY_UNPARSED = "reference_entry_unparsed"
    REFERENCE_RESOLVED = "reference_resolved"
    REFERENCE_PROVISIONAL = "reference_provisional"
    REFERENCE_GRAPH_COMMITTED = "reference_graph_committed"
    REFERENCE_MAPPING_COMPLETED = "reference_mapping_completed"
    REFERENCE_MAPPING_FAILED = "reference_mapping_failed"
    ARTIFACT_SAVED = "artifact_saved"
    WARNING = "warning"
    # Examination / survivor lifecycle (OBS-1)
    EVIDENCE_PACK_FROZEN = "evidence_pack_frozen"
    EXAM_GENERATED = "exam_generated"
    EXAM_VALIDATED = "exam_validated"
    EXAM_PARTITIONED = "exam_partitioned"
    EXAM_INSUFFICIENT = "exam_insufficient"
    CANDIDATE_TEST_STARTED = "candidate_test_started"
    CANDIDATE_TEST_COMPLETED = "candidate_test_completed"
    SURVIVOR_SELECTED = "survivor_selected"
    SURVIVOR_FROZEN = "survivor_frozen"
    BASELINE_COMPLETED = "baseline_completed"
    BENCHMARK_COMPLETED = "benchmark_completed"


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
STATUS_INTERRUPTED = "interrupted"

TERMINAL_STATUSES = frozenset(
    {STATUS_COMPLETED, STATUS_CANCELLED, STATUS_FAILED, STATUS_INTERRUPTED}
)

OUTCOME_COMPLETED = "completed"
OUTCOME_DEGRADED = "degraded"

# Terminal benchmark outcomes (EXAM-9)
OUTCOME_BENCHMARKED = "completed_benchmarked"
OUTCOME_SURVIVOR_UNBENCHMARKED = "completed_survivor_unbenchmarked"
OUTCOME_DEGRADED_NO_SURVIVOR = "completed_degraded"
OUTCOME_EXAM_INSUFFICIENT = "completed_exam_insufficient"
OUTCOME_FAILED = "failed"

PHASE_SETUP = "setup"
PHASE_RESEARCH = "research"
PHASE_EVALUATION = "evaluation"
PHASE_DECISION = "decision"
PHASE_RESULT = "result"
PHASE_DEBUG = "debug"
# Terminal examination phases after research convergence (TUI-1)
PHASE_EXAM_BUILD = "exam_build"
PHASE_SELECTION = "selection"
PHASE_SURVIVOR = "survivor"
PHASE_BENCHMARK = "benchmark"

WAVE_PHASE_ORDER: tuple[str, ...] = (PHASE_RESEARCH, PHASE_EVALUATION, PHASE_DECISION)
EXAM_PHASE_ORDER: tuple[str, ...] = (
    PHASE_EXAM_BUILD,
    PHASE_SELECTION,
    PHASE_SURVIVOR,
    PHASE_BENCHMARK,
)

EVAL_PENDING = "pending"
EVAL_COMPLETE = "complete"
EVAL_SKIPPED = "skipped"
EVAL_FAILED = "failed"

EVALUATION_TERMINAL_STATES = frozenset({EVAL_COMPLETE, EVAL_SKIPPED, EVAL_FAILED})

REASON_EMPTY_WINNER_NARRATIVE = "empty_winner_narrative"
REASON_NO_EVALUATED_EVIDENCE = "no_evaluated_evidence"
REASON_WINNER_EVALUATION_MISSING = "winner_evaluation_missing"

# Exactly one primary reason is attached to an empty initial frontier so the
# terminal state is actionable instead of an opaque ``no_winner``.
REASON_NO_NEIGHBORS_DISCOVERED = "no_neighbors_discovered"
REASON_NO_TRAVERSABLE_IDENTIFIERS = "no_traversable_identifiers"
REASON_REFERENCE_EXTRACTION_FAILED = "reference_extraction_failed"
REASON_REFERENCE_MAPPING_INCOMPLETE = "reference_mapping_incomplete"
REASON_SEED_DISCOVERY_FAILED = "seed_discovery_failed"
REASON_NO_WINNER = "no_winner"

# Stable benchmark reason codes (EXAM-9)
REASON_EXAMINER_UNAVAILABLE = "examiner_unavailable"
REASON_INSUFFICIENT_QUESTIONS = "insufficient_validated_questions"
REASON_INVALID_CANDIDATE_RESPONSE = "invalid_candidate_response"
REASON_BASELINE_FAILED = "baseline_failed"
REASON_SURVIVOR_UNAVAILABLE = "survivor_unavailable"
REASON_NO_ELIGIBLE_SURVIVOR = "no_eligible_survivor"

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
    REASON_REFERENCE_MAPPING_INCOMPLETE: (
        "bibliography mapping was pending, partial, failed, or incomplete"
    ),
    REASON_SEED_DISCOVERY_FAILED: (
        "seed neighbor discovery failed during colony initialization"
    ),
    REASON_NO_WINNER: (
        "the run finished without a winning narrative"
    ),
    REASON_EMPTY_WINNER_NARRATIVE: (
        "the winning agent produced no usable narrative"
    ),
    REASON_NO_EVALUATED_EVIDENCE: (
        "no evidence-bearing evaluation completed during exploration"
    ),
    REASON_WINNER_EVALUATION_MISSING: (
        "the winning agent has no terminal evaluation"
    ),
    REASON_EXAMINER_UNAVAILABLE: (
        "the configured examiner was unavailable; no exam was generated"
    ),
    REASON_INSUFFICIENT_QUESTIONS: (
        "the validation budget could not produce the minimum validated bank"
    ),
    REASON_INVALID_CANDIDATE_RESPONSE: (
        "a candidate produced an invalid selection response"
    ),
    REASON_BASELINE_FAILED: (
        "the matched naive baseline could not complete"
    ),
    REASON_SURVIVOR_UNAVAILABLE: (
        "no usable survivor bundle was produced"
    ),
    REASON_NO_ELIGIBLE_SURVIVOR: (
        "no candidate satisfied the survivor eligibility gates"
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
    research_status: str = NODE_PENDING
    evaluation_status: str = EVAL_PENDING
    evaluation_reason: str = ""
    papers_attempted: int = 0
    papers_integrated: int = 0
    evidence_added: int = 0


class PhaseState(BaseModel):
    """Durable Research/Evaluation/Decision state for one wave phase."""

    wave: int = 0
    phase: str = PHASE_RESEARCH
    status: str = NODE_PENDING
    reason: str = ""
    selected: list[str] = Field(default_factory=list)
    completed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    papers_attempted: int = 0
    papers_integrated: int = 0
    evidence_added: int = 0
    best_q: float | None = None
    leader: str = ""
    q_delta: float = 0.0
    budget_used: int = 0
    continue_reason: str = ""
    stop_reason: str = ""
    converged: bool = False
    pheromone_concentration: float = 0.0
    elapsed_seconds: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.wave}:{self.phase}"


class EvaluationState(BaseModel):
    """The single terminal evaluation outcome for a selected agent wave turn."""

    agent_id: str
    wave: int = 0
    turn: int = 0
    status: str = EVAL_PENDING
    reason: str = ""
    q: float | None = None
    q_delta: float | None = None
    components: dict[str, float | None] = Field(default_factory=dict)
    unavailable: dict[str, str] = Field(default_factory=dict)
    evidence_papers: int = 0

    @property
    def key(self) -> str:
        return f"{self.wave}:{self.agent_id}"

    @property
    def is_terminal(self) -> bool:
        return self.status in EVALUATION_TERMINAL_STATES


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
    effective_scope: str = ""
    scope_origin: str = "derived"
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
    events_seen_total: int = 0
    events_dropped: int = 0
    candidate_scores_seen_total: int = 0
    candidate_scores_dropped: int = 0
    token_usage: int | None = None
    cost: float | None = None
    follow_live: bool = True
    selected_entry_id: str | None = None
    phases: dict[str, PhaseState] = Field(default_factory=dict)
    evaluation_states: dict[str, EvaluationState] = Field(default_factory=dict)
    selected_agents: dict[int, list[str]] = Field(default_factory=dict)
    current_phase: str = ""
    stop_reason: str = ""
    legacy_projection: bool = False
    # Terminal examination visibility (TUI-1)
    exam_phases: dict[str, PhaseState] = Field(default_factory=dict)
    survivor_id: str = ""
    survivor_accuracy: float | None = None
    naive_accuracy: float | None = None
    uplift: float | None = None
    benchmark_outcome: str = ""
    benchmark_reason_code: str = ""
    benchmark_reason: str = ""
    exam_selection_count: int = 0
    exam_holdout_count: int = 0
    exam_rejected_count: int = 0
    survivor_terminal_score: float | None = None
    survivor_selection_score: float | None = None
    survivor_process_score: float | None = None
    survivor_grounding_score: float | None = None
    survivor_ranking: list[str] = Field(default_factory=list)
    survivor_synthesis: str = ""
    benchmark_item_outcomes: dict[str, bool] = Field(default_factory=dict)
    benchmark_naive_item_outcomes: dict[str, bool] = Field(default_factory=dict)
    benchmark_by_category: dict[str, dict[str, int]] = Field(default_factory=dict)
    benchmark_by_difficulty: dict[str, dict[str, int]] = Field(default_factory=dict)
    benchmark_naive_by_category: dict[str, dict[str, int]] = Field(default_factory=dict)
    benchmark_naive_by_difficulty: dict[str, dict[str, int]] = Field(default_factory=dict)

    def ordered_agents(self) -> list[AgentSummary]:
        return [self.agents[a] for a in self.agent_order if a in self.agents]

    def phase(self, wave: int, phase: str) -> PhaseState | None:
        return self.phases.get(f"{wave}:{phase}")

    def wave_phases(self, wave: int) -> list[PhaseState]:
        ordered: list[PhaseState] = []
        for name in WAVE_PHASE_ORDER:
            record = self.phase(wave, name)
            if record is not None:
                ordered.append(record)
        return ordered

    def evaluation_state(self, wave: int, agent_id: str) -> EvaluationState | None:
        return self.evaluation_states.get(f"{wave}:{agent_id}")

    def latest_evaluation(self, agent_id: str) -> EvaluationState | None:
        records = [
            record
            for record in self.evaluation_states.values()
            if record.agent_id == agent_id
        ]
        return max(records, key=lambda record: record.wave) if records else None

    def ordered_waves(self) -> list[int]:
        waves = {entry.wave for entry in self.timeline}
        waves.update(phase.wave for phase in self.phases.values())
        return sorted(wave for wave in waves if wave > 0)

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
