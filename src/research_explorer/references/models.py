"""Typed models for paper-level bibliography mapping.

These models are the contract between the provider's raw bibliography, the
dedicated LLM mapping call, and the deterministic graph commit. They carry no
hidden reasoning: ``parse_notes`` is a short ambiguity explanation only.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from research_explorer.graph.models import PaperSummary


class EntryType(str, Enum):
    ARTICLE = "article"
    PREPRINT = "preprint"
    BOOK = "book"
    THESIS = "thesis"
    DATASET = "dataset"
    SOFTWARE = "software"
    OTHER = "other"


class MappingStatus(str, Enum):
    PENDING = "pending"
    MAPPED = "mapped"
    UNPARSED = "unparsed"
    FAILED = "mapping_failed"

    @property
    def terminal(self) -> bool:
        return self is not MappingStatus.PENDING


class ResolutionState(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    PROVISIONAL = "provisional"
    RETRYABLE = "retryable"
    UNRESOLVED = "unresolved"


class RawBibliographyEntry(BaseModel):
    """One raw bibliography entry before any LLM mapping."""

    entry_id: str
    ordinal: int
    raw_text: str
    raw_hash: str


class MappedEntry(BaseModel):
    """One LLM-mapped bibliography result (or an explicit unparsed/failed marker)."""

    entry_id: str
    ordinal: int
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    entry_type: EntryType = EntryType.OTHER
    parse_confidence: float = 0.0
    parse_notes: str = ""
    mapping_status: MappingStatus = MappingStatus.MAPPED


class ReferenceAccounting(BaseModel):
    """Structured result returned by the reference graph builder."""

    source_id: str
    job_id: str
    status: str
    reused: bool = False
    observed: int = 0
    processed: int = 0
    mapped: int = 0
    unparsed: int = 0
    failed: int = 0
    resolved: int = 0
    provisional: int = 0
    traversable: int = 0
    resolved_summaries: list[PaperSummary] = Field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return (
            self.observed > 0
            and self.mapped == 0
            and self.status in ("failed", "partial", "incomplete")
        )
