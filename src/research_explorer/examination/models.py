"""Typed examination contracts (EXAM-1..9).

Public artifacts (questions, options) are separated from private artifacts
(answer keys, examiner rationales, evidence annotations) at the model level so
the leakage boundary is enforced structurally, not by convention.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from research_explorer.memory.models import ContentKind, canonical_json

EXAM_SCHEMA_VERSION = "exam/1"
BUNDLE_SCHEMA_VERSION = "survivor-bundle/1"

CATEGORY_CONCEPTS = "concepts_definitions"
CATEGORY_METHOD = "method_experimental_design"
CATEGORY_RESULTS = "results_interpretation"
CATEGORY_ASSUMPTIONS = "assumptions_limitations"
CATEGORY_CROSS_PAPER = "cross_paper"
CATEGORY_TRANSFER = "transfer_application"

CATEGORY_DISTRIBUTION: dict[str, float] = {
    CATEGORY_CONCEPTS: 0.20,
    CATEGORY_METHOD: 0.20,
    CATEGORY_RESULTS: 0.20,
    CATEGORY_ASSUMPTIONS: 0.15,
    CATEGORY_CROSS_PAPER: 0.15,
    CATEGORY_TRANSFER: 0.10,
}

DIFFICULTY_EASY = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD = "hard"
DIFFICULTIES = (DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD)

SOURCE_DISTANCE_SEED = "seed"
SOURCE_DISTANCE_DIRECT = "direct_reference"
SOURCE_DISTANCE_CITANT = "citant"
SOURCE_DISTANCE_DEEPER = "deeper"
SOURCE_DISTANCES = (
    SOURCE_DISTANCE_SEED,
    SOURCE_DISTANCE_DIRECT,
    SOURCE_DISTANCE_CITANT,
    SOURCE_DISTANCE_DEEPER,
)

STATUS_GENERATED = "generated"
STATUS_VALIDATED = "validated"
STATUS_REJECTED = "rejected"

INSUFFICIENT_EVIDENCE_OPTION = "insufficient_evidence"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class SourceExclusion(BaseModel):
    source_id: str
    reason: str


class EvidenceEntry(BaseModel):
    """One acquired source eligible for examination evidence."""

    source_id: str
    title: str = ""
    content_kind: ContentKind = ContentKind.ABSTRACT
    content_hash: str
    excerpt: str = ""
    source_distance: str = SOURCE_DISTANCE_SEED
    year: int | None = None

    @field_validator("source_distance", mode="before")
    @classmethod
    def _coerce_distance(cls, value: Any) -> Any:
        if value in SOURCE_DISTANCES:
            return value
        return SOURCE_DISTANCE_DEEPER


class EvidencePack(BaseModel):
    """Frozen, hashed exam evidence pack built only from acquired sources."""

    pack_id: str
    seed_paper_id: str
    scope: str = ""
    sources: list[EvidenceEntry] = Field(default_factory=list)
    excluded: list[SourceExclusion] = Field(default_factory=list)
    pack_hash: str = ""
    schema_version: str = EXAM_SCHEMA_VERSION
    created_at: str = Field(default_factory=utc_now)

    def source(self, source_id: str) -> EvidenceEntry | None:
        for entry in self.sources:
            if entry.source_id == source_id:
                return entry
        return None

    def source_ids(self) -> list[str]:
        return [entry.source_id for entry in self.sources]

    def compute_hash(self) -> str:
        return _hash(
            {
                "seed": self.seed_paper_id,
                "sources": [
                    {
                        "id": entry.source_id,
                        "hash": entry.content_hash,
                        "kind": entry.content_kind.value,
                    }
                    for entry in sorted(self.sources, key=lambda e: e.source_id)
                ],
            }
        )

    def freeze(self) -> EvidencePack:
        self.pack_hash = self.compute_hash()
        return self

    def bounded_context(self, max_chars: int = 40_000) -> str:
        parts: list[str] = []
        for entry in self.sources:
            block = f"[{entry.source_id}] {entry.title}\n{entry.excerpt}\n"
            parts.append(block)
        text = "\n".join(parts)
        return text[:max_chars]

    def without_annotations(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "seed_paper_id": self.seed_paper_id,
            "scope": self.scope,
            "sources": [
                {"source_id": e.source_id, "title": e.title, "content_kind": e.content_kind.value}
                for e in self.sources
            ],
            "pack_hash": self.pack_hash,
        }


class ExamOption(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class ExamItem(BaseModel):
    """One grounded multiple-choice item.

    ``correct_option_id``, ``rationale``, and ``evidence_refs`` are private and
    omitted from :meth:`public_payload`.
    """

    question_id: str
    exam_version: str = EXAM_SCHEMA_VERSION
    category: str
    difficulty: str = DIFFICULTY_MEDIUM
    question: str = Field(min_length=1)
    options: list[ExamOption] = Field(default_factory=list)
    correct_option_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str = ""
    source_distance: str = SOURCE_DISTANCE_SEED
    has_insufficient_evidence_option: bool = False
    defensible_option_ids: list[str] = Field(default_factory=list)
    generation_status: str = STATUS_GENERATED
    validation_status: str = STATUS_GENERATED
    rejection_reasons: list[str] = Field(default_factory=list)
    generator_model: str = ""

    def public_payload(self) -> dict[str, Any]:
        """Only the fields a test taker may see."""
        return {
            "question_id": self.question_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "question": self.question,
            "options": [{"id": o.id, "text": o.text} for o in self.options],
        }

    def answer_key(self) -> dict[str, Any]:
        """Private answer-key entry; never part of the public payload."""
        return {
            "question_id": self.question_id,
            "correct_option_id": self.correct_option_id,
            "rationale": self.rationale,
            "evidence_refs": list(self.evidence_refs),
            "defensible_option_ids": list(self.defensible_option_ids),
        }


class AnswerKey(BaseModel):
    """Restricted artifact: the frozen key before any candidate answers."""

    exam_version: str = EXAM_SCHEMA_VERSION
    partition_seed: int
    entries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    key_hash: str = ""

    def freeze(self) -> AnswerKey:
        self.key_hash = _hash(self.entries)
        return self


class ExamBank(BaseModel):
    """The validated bank before partitioning."""

    exam_version: str = EXAM_SCHEMA_VERSION
    items: list[ExamItem] = Field(default_factory=list)
    selection_ids: list[str] = Field(default_factory=list)
    holdout_ids: list[str] = Field(default_factory=list)
    partition_seed: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    rejection_reasons: list[str] = Field(default_factory=list)
    partition_frozen: bool = False

    def accepted_items(self) -> list[ExamItem]:
        return [i for i in self.items if i.validation_status == STATUS_VALIDATED]

    def by_id(self, question_id: str) -> ExamItem | None:
        for item in self.items:
            if item.question_id == question_id:
                return item
        return None

    def selection_items(self) -> list[ExamItem]:
        return [i for i in self.items if i.question_id in self.selection_ids]

    def holdout_items(self) -> list[ExamItem]:
        return [i for i in self.items if i.question_id in self.holdout_ids]

    def public_payload(self, question_ids: list[str] | None = None) -> list[dict[str, Any]]:
        ids = set(question_ids) if question_ids is not None else None
        return [
            item.public_payload()
            for item in self.items
            if item.validation_status == STATUS_VALIDATED
            and (ids is None or item.question_id in ids)
        ]


class AnswerRecord(BaseModel):
    question_id: str
    option_id: str | None = None
    valid: bool = True
    invalid_reason: str = ""
    latency_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class AnswerSet(BaseModel):
    responder: str
    model_id: str = ""
    answers: dict[str, AnswerRecord] = Field(default_factory=dict)
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    def record(self, question_id: str, option_id: str | None, **kwargs: Any) -> AnswerRecord:
        rec = AnswerRecord(question_id=question_id, option_id=option_id, **kwargs)
        self.answers[question_id] = rec
        return rec


class CategoryScore(BaseModel):
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None


class ExamScore(BaseModel):
    correct: int = 0
    total: int = 0
    by_category: dict[str, CategoryScore] = Field(default_factory=dict)
    by_difficulty: dict[str, CategoryScore] = Field(default_factory=dict)
    by_content_kind: dict[str, CategoryScore] = Field(default_factory=dict)
    by_source_distance: dict[str, CategoryScore] = Field(default_factory=dict)
    item_outcomes: dict[str, bool] = Field(default_factory=dict)

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None


class StudentQuestion(BaseModel):
    """Exactly the public question fields plus options."""

    question_id: str
    category: str
    difficulty: str
    question: str
    options: list[ExamOption]


class CandidateMemory(BaseModel):
    """A candidate agent's structured memory presented for selection."""

    agent_id: str
    process_score: float = 0.0
    grounding_score: float = 0.0
    memory: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    has_evidence_bearing_dossier: bool = False
    provenance_integrity: bool = True
    selection_accuracy: float | None = None
    selection_answered: int = 0
    selection_coverage: float = 0.0


class SelectionWeights(BaseModel):
    selection: float = 0.70
    process: float = 0.20
    grounding: float = 0.10

    @property
    def total(self) -> float:
        return self.selection + self.process + self.grounding


class SurvivorSelection(BaseModel):
    survivor_id: str = ""
    eligible: bool = False
    ineligible_reason: str = ""
    terminal_score: float = 0.0
    selection_accuracy: float | None = None
    process_score: float = 0.0
    grounding_score: float = 0.0
    weights: SelectionWeights = Field(default_factory=SelectionWeights)
    ranking: list[str] = Field(default_factory=list)
    process_peak_agent: str = ""


class BenchmarkResult(BaseModel):
    survivor_id: str = ""
    outcome: str = ""
    reason_code: str = ""
    reason: str = ""
    survivor_accuracy: float | None = None
    naive_accuracy: float | None = None
    uplift: float | None = None
    survivor_unavailable: str | None = None
    naive_unavailable: str | None = None
    survivor_score: ExamScore | None = None
    naive_score: ExamScore | None = None
    paired_outcomes: dict[str, dict[str, bool]] = Field(default_factory=dict)
    selection: SurvivorSelection | None = None
    config_fingerprint: str = ""
    model_ids: dict[str, str] = Field(default_factory=dict)
