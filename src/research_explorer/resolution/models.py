"""Typed models for cross-provider bibliographic identity resolution."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from research_explorer.graph.models import PaperSummary


class ResolutionMethod(str, Enum):
    ALIAS_CACHE = "alias_cache"
    DOI_LOOKUP = "doi_lookup"
    ARXIV_LOOKUP = "arxiv_lookup"
    PMID_LOOKUP = "pmid_lookup"
    TITLE_SEARCH = "title_search"


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


class RejectReason(str, Enum):
    NO_PROVIDER_MATCH = "no_provider_match"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TITLE_MISMATCH = "title_mismatch"
    AUTHOR_MISMATCH = "author_mismatch"
    YEAR_MISMATCH = "year_mismatch"
    LOW_CONFIDENCE = "low_confidence"
    MULTIPLE_CANDIDATES = "multiple_candidates"
    UNVERIFIABLE_IDENTIFIER = "unverifiable_identifier"


class EntrySource(str, Enum):
    BIBLIOGRAPHY = "bibliography"
    PROVIDER = "provider"


class BibliographicEntry(BaseModel):
    """A bibliographic reference as found in a bibliography or provider record."""

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    source: EntrySource = EntrySource.BIBLIOGRAPHY


class RecordMetadata(BaseModel):
    """Metadata as claimed by the query (expected) or returned by a provider (actual)."""

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None


class CandidateScore(BaseModel):
    """Per-evidence match details for a candidate record."""

    title_similarity: float = 0.0
    author_overlap: float = 0.0
    year_ok: bool = True
    confidence: float = 0.0
    title_checked: bool = False
    authors_checked: bool = False
    year_checked: bool = False


class ResolutionAttempt(BaseModel):
    """Record of one verification attempt against one provider/method."""

    method: ResolutionMethod
    provider: str | None = None
    status: ResolutionStatus
    candidate_count: int = 0
    canonical_id: str | None = None
    reject_reason: RejectReason | None = None
    error: str | None = None


class ResolutionEvidence(BaseModel):
    """Verified match details for one candidate produced by one method/provider."""

    method: ResolutionMethod
    provider: str | None = None
    candidate: PaperSummary
    score: CandidateScore
    accepted: bool = False
    reject_reason: RejectReason | None = None


class ResolutionResult(BaseModel):
    """Deterministic outcome of resolving one bibliographic entry."""

    status: ResolutionStatus
    confidence: float = 0.0
    method: ResolutionMethod | None = None
    provider: str | None = None
    canonical_id: str | None = None
    aliases: list[str] = Field(default_factory=list)
    summary: PaperSummary | None = None
    expected: RecordMetadata = Field(default_factory=RecordMetadata)
    actual: RecordMetadata | None = None
    reject_reason: RejectReason | None = None
    attempts: list[ResolutionAttempt] = Field(default_factory=list)
    evidence: list[ResolutionEvidence] = Field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.status is ResolutionStatus.RESOLVED and self.canonical_id is not None


def record_from_summary(summary: PaperSummary) -> RecordMetadata:
    """Project a candidate summary into the provider-side metadata view."""
    return RecordMetadata(
        title=summary.title,
        authors=summary.authors,
        year=summary.year,
        doi=summary.doi,
        arxiv_id=summary.arxiv_id,
        pmid=summary.pmid,
    )
