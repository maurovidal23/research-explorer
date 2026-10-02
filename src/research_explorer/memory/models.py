"""Typed, persistible structured research memory.

This module is the single source of truth for per-paper understanding, atomic
claims, concept/paper relationships, and explicit knowledge gaps. It replaces
narrative-only integration: the synthesized prose is a *view* over these
models (see :mod:`research_explorer.memory.synthesis`), never the memory itself.

The models are provider- and persistence-agnostic so the ACO agent, the
examination pipeline, and the survivor bundle can all share them.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

MEMORY_SCHEMA_VERSION = "research-memory/1"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def stable_id(prefix: str, *parts: str) -> str:
    joined = "|".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class ContentKind(str, Enum):
    """How much of a paper's content was actually acquired.

    Only ``full_text`` and ``abstract`` are eligible for evidence credit;
    ``metadata`` nodes are navigational only.
    """

    FULL_TEXT = "full_text"
    ABSTRACT = "abstract"
    METADATA = "metadata"

    @property
    def has_evidence(self) -> bool:
        return self in (ContentKind.FULL_TEXT, ContentKind.ABSTRACT)


class ClaimStatus(str, Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"


class RelationKind(str, Enum):
    FOUNDATION = "foundation"
    EXTENSION = "extension"
    REPLICATION = "replication"
    DISAGREEMENT = "disagreement"
    APPLICATION = "application"


class DossierEvidence(BaseModel):
    """A pointer to one acquired, hashable piece of source content.

    It resolves to acquired content and retains paper id, content hash,
    acquisition event, content kind, an optional locator, and a short excerpt.
    """

    evidence_id: str
    paper_id: str
    content_kind: ContentKind
    content_hash: str
    acquisition_event: int | None = None
    locator: str | None = None
    excerpt: str = ""

    @property
    def resolves(self) -> bool:
        return bool(self.paper_id and self.content_hash) and self.content_kind.has_evidence


class LedgerClaim(BaseModel):
    """One atomic, typed claim held by an agent, separate from prose."""

    id: str
    text: str = Field(min_length=1)
    status: ClaimStatus = ClaimStatus.PROPOSED
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    concepts: list[str] = Field(default_factory=list)
    supporting: list[DossierEvidence] = Field(default_factory=list)
    contradicting: list[DossierEvidence] = Field(default_factory=list)
    provenance: str = "agent"
    created_seq: int = 0
    updated_seq: int = 0

    @property
    def has_resolving_support(self) -> bool:
        return any(ref.resolves for ref in self.supporting)


class PaperRelation(BaseModel):
    """A typed relationship between two acquired papers/concepts."""

    id: str
    source_paper_id: str
    target_paper_id: str
    kind: RelationKind
    rationale: str = ""
    evidence: list[DossierEvidence] = Field(default_factory=list)


class Concept(BaseModel):
    id: str
    label: str = Field(min_length=1)
    definition: str = ""
    aliases: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)


class KnowledgeGap(BaseModel):
    """An explicit open knowledge gap (never a negative factual claim)."""

    id: str
    text: str = Field(min_length=1)
    related_claims: list[str] = Field(default_factory=list)
    related_papers: list[str] = Field(default_factory=list)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class PaperDossier(BaseModel):
    """Structured, evidence-backed research memory for one integrated paper."""

    paper_id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    content_kind: ContentKind = ContentKind.METADATA
    research_problem: str = ""
    contribution: str = ""
    definitions: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    method: str = ""
    design: str = ""
    datasets: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    relevance: str = ""
    relationships: list[str] = Field(default_factory=list)
    evidence: list[DossierEvidence] = Field(default_factory=list)
    extraction_status: str = "complete"
    extraction_error: str | None = None
    schema_version: str = MEMORY_SCHEMA_VERSION

    @field_validator("content_kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value: Any) -> Any:
        if isinstance(value, ContentKind):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
            aliases = {
                "fulltext": "full_text",
                "full_text": "full_text",
                "abstract": "abstract",
                "abstract_only": "abstract",
                "metadata": "metadata",
                "metadata_only": "metadata",
                "none": "metadata",
            }
            return aliases.get(normalized, normalized)
        return value

    @property
    def has_evidence_credit(self) -> bool:
        return self.content_kind.has_evidence and any(ref.resolves for ref in self.evidence)

    @property
    def content_hash(self) -> str | None:
        for ref in self.evidence:
            if ref.resolves:
                return ref.content_hash
        return None


class ResearchMemory(BaseModel):
    """Aggregate structured memory for a single agent."""

    agent_id: str = ""
    scope: str = ""
    scope_origin: str = "derived"  # "user" | "derived"
    dossiers: dict[str, PaperDossier] = Field(default_factory=dict)
    claims: dict[str, LedgerClaim] = Field(default_factory=dict)
    concepts: dict[str, Concept] = Field(default_factory=dict)
    relations: list[PaperRelation] = Field(default_factory=list)
    gaps: list[KnowledgeGap] = Field(default_factory=list)
    schema_version: str = MEMORY_SCHEMA_VERSION

    def evidence_index(self) -> dict[str, DossierEvidence]:
        index: dict[str, DossierEvidence] = {}
        for dossier in self.dossiers.values():
            for ref in dossier.evidence:
                index.setdefault(ref.evidence_id, ref)
        for claim in self.claims.values():
            for ref in claim.supporting + claim.contradicting:
                index.setdefault(ref.evidence_id, ref)
        return index

    def evidence_bearing_papers(self) -> list[str]:
        return sorted(
            pid for pid, d in self.dossiers.items() if d.has_evidence_credit
        )

    def supported_claims(self) -> list[LedgerClaim]:
        return [c for c in self.claims.values() if c.status is ClaimStatus.SUPPORTED]

    def disputed_claims(self) -> list[LedgerClaim]:
        return [c for c in self.claims.values() if c.status is ClaimStatus.DISPUTED]

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self).encode()).hexdigest()
