"""Typed research-domain contracts for the single-agent research kernel.

These models are deliberately provider- and persistence-agnostic: the kernel
controller coordinates them, the store serializes them, and the policy/agent/
evaluator operate on them without touching HTTP or SQLite directly.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "research-kernel/1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    """Canonical JSON for stable hashing/equality (sorted keys, compact)."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def state_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class ClaimStatus(str, Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    DISPUTED = "disputed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class OpenQuestionStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class ActionKind(str, Enum):
    READ_EVIDENCE = "read_evidence"
    INVESTIGATE_QUESTION = "investigate_question"
    STOP = "stop"


class ProviderOutcome(str, Enum):
    SUCCESS = "success"
    ABSENT = "absent"
    TRANSIENT = "transient"


class BudgetState(BaseModel):
    """Resolved budgets and their consumption."""

    max_fetches: int = 50
    max_tokens: int = 200_000
    max_time_seconds: float = 3600.0
    max_turns: int = 20
    fetches_used: int = 0
    tokens_used: int = 0
    time_used: float = 0.0
    turns_used: int = 0

    @property
    def fetches_remaining(self) -> int:
        return max(0, self.max_fetches - self.fetches_used)

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def exhausted(self) -> bool:
        return (
            self.fetches_remaining <= 0
            or self.tokens_remaining <= 0
            or self.time_used >= self.max_time_seconds
            or self.turns_used >= self.max_turns
        )


class SlotState(BaseModel):
    total_slots: int = 1
    active_slots: int = 0

    @property
    def available(self) -> int:
        return max(0, self.total_slots - self.active_slots)


class EvidenceRef(BaseModel):
    """A pointer to acquired evidence, never a duplicate of the document."""

    paper_id: str
    locator: str | None = None
    content_hash: str | None = None
    acquisition_event: int | None = None


class Claim(BaseModel):
    id: str
    text: str = Field(min_length=1)
    status: ClaimStatus = ClaimStatus.PROPOSED
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting: list[EvidenceRef] = Field(default_factory=list)
    contradicting: list[EvidenceRef] = Field(default_factory=list)
    creator: str = "agent"
    created_seq: int = 0
    updated_seq: int = 0

    @property
    def has_evidence(self) -> bool:
        return bool(self.supporting or self.contradicting)


class OpenQuestion(BaseModel):
    id: str
    text: str = Field(min_length=1)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    status: OpenQuestionStatus = OpenQuestionStatus.OPEN
    related_claims: list[str] = Field(default_factory=list)
    related_papers: list[str] = Field(default_factory=list)
    resolution_evidence: list[EvidenceRef] = Field(default_factory=list)


class PlannedAction(BaseModel):
    action: str
    reason: str = ""
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class AgentNotebook(BaseModel):
    agent_id: str
    role: str = "explorer"
    thesis: str = ""
    accepted_claims: list[str] = Field(default_factory=list)
    disputed_claims: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    planned_actions: list[PlannedAction] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    revision: int = 0

    def bump(self) -> None:
        self.revision += 1


class ClaimMutation(BaseModel):
    """One structured mutation proposed by an agent brief."""

    op: str = Field(
        description="propose | support | dispute | reject | supersede | open_question | answer_question"
    )
    claim_id: str | None = None
    question_id: str | None = None
    text: str = ""
    status: ClaimStatus | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    reason: str = ""


class AgentBrief(BaseModel):
    action_summary: str = ""
    claim_mutations: list[ClaimMutation] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    changed_understanding: str = ""
    contradictions: list[str] = Field(default_factory=list)
    remaining_uncertainty: str = ""
    proposed_next_actions: list[str] = Field(default_factory=list)
    shareable_findings: list[str] = Field(default_factory=list)


class IntegrityResult(BaseModel):
    passed: bool = True
    checks: dict[str, bool] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    duplicate_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    unresolved_citations: list[str] = Field(default_factory=list)


class RubricResult(BaseModel):
    ok: bool = True
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    missing_knowledge: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    recommended_questions: list[str] = Field(default_factory=list)
    error: str | None = None


class ResearchEvaluation(BaseModel):
    rubric_version: str = "v1"
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    overall: float = 0.0
    previous_overall: float = 0.0
    delta_quality: float = 0.0
    missing_knowledge: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    recommended_questions: list[str] = Field(default_factory=list)
    evaluator_id: str = "composite"
    evaluator_version: str = "v1"
    raw_artifact_ref: str | None = None
    integrity: IntegrityResult = Field(default_factory=IntegrityResult)
    rubric: RubricResult | None = None


class CandidateAction(BaseModel):
    paper_id: str
    source: str = ""
    mode: str = "ref"
    score: float = 0.0
    reason: str = ""


class ResearchAction(BaseModel):
    kind: ActionKind = ActionKind.READ_EVIDENCE
    paper_id: str | None = None
    question_id: str | None = None
    reason: str = ""
    predicted_value: float = 0.0
    predicted_cost: int = 1


class ResearchObjective(BaseModel):
    run_id: str
    seed_paper_id: str
    question: str = Field(min_length=1)
    scope: dict[str, Any] = Field(default_factory=dict)
    budget: BudgetState = Field(default_factory=BudgetState)
    random_seed: int = 0

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("research question must not be blank")
        return value


class ResearchState(BaseModel):
    """Logical research state persisted transactionally with each event."""

    objective: ResearchObjective
    claims: dict[str, Claim] = Field(default_factory=dict)
    questions: dict[str, OpenQuestion] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    visited: list[str] = Field(default_factory=list)
    notebook: AgentNotebook
    latest_evaluation: ResearchEvaluation | None = None
    final_answer: str | None = None
    terminal_reason: str | None = None


class ResearchEvent(BaseModel):
    run_id: str
    seq: int
    type: str
    actor: str = "controller"
    wave: int | None = None
    turn: int | None = None
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    tokens: int | None = None
    fetches: int | None = None
    seconds: float | None = None
    previous_state_hash: str | None = None
    current_state_hash: str | None = None
    timestamp: str = Field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION


class FinalAnswer(BaseModel):
    question: str
    supported_conclusions: list[str] = Field(default_factory=list)
    plausible_interpretations: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)

    def render_markdown(self) -> str:
        lines = ["# Research answer\n", f"**Question:** {self.question}\n"]
        self._section(lines, "Supported conclusions", self.supported_conclusions)
        self._section(lines, "Plausible interpretations", self.plausible_interpretations)
        self._section(lines, "Unknowns", self.unknowns)
        self._section(lines, "Contradictions", self.contradictions)
        self._section(lines, "Limitations", self.limitations)
        if self.citations:
            lines.append("## Citations")
            lines.extend(f"- {cid}" for cid in self.citations)
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _section(lines: list[str], title: str, items: list[str]) -> None:
        if not items:
            return
        lines.append(f"\n## {title}")
        lines.extend(f"- {item}" for item in items)
