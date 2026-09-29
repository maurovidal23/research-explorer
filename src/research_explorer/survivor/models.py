"""Versioned, frozen survivor bundle (SURV-5).

The bundle stores enough structured evidence to answer later questions without
rerunning exploration. It records model ids and prompt/schema versions used to
construct it, and exposes only canonical source ids.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from research_explorer.examination.models import (
    BUNDLE_SCHEMA_VERSION,
    EvidenceEntry,
    SelectionWeights,
)
from research_explorer.memory.models import (
    Concept,
    DossierEvidence,
    KnowledgeGap,
    LedgerClaim,
    PaperDossier,
    PaperRelation,
    ResearchMemory,
    canonical_json,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SelectionMetadata(BaseModel):
    terminal_score: float = 0.0
    selection_accuracy: float | None = None
    process_score: float = 0.0
    grounding_score: float = 0.0
    weights: SelectionWeights = Field(default_factory=SelectionWeights)
    formula: str = "Q_terminal = 0.70*E_selection + 0.20*Q_process + 0.10*G"
    candidate_ranking: list[str] = Field(default_factory=list)
    process_peak_agent: str = ""


class SurvivorBundle(BaseModel):
    bundle_version: str = BUNDLE_SCHEMA_VERSION
    survivor_id: str
    scope: str = ""
    scope_origin: str = "derived"
    dossiers: dict[str, PaperDossier] = Field(default_factory=dict)
    claims: dict[str, LedgerClaim] = Field(default_factory=dict)
    concepts: dict[str, Concept] = Field(default_factory=dict)
    relations: list[PaperRelation] = Field(default_factory=list)
    gaps: list[KnowledgeGap] = Field(default_factory=list)
    evidence_index: dict[str, DossierEvidence] = Field(default_factory=dict)
    synthesis: str = ""
    source_catalog: list[EvidenceEntry] = Field(default_factory=list)
    config_fingerprint: str = ""
    model_ids: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    schema_versions: dict[str, str] = Field(default_factory=dict)
    selection: SelectionMetadata = Field(default_factory=SelectionMetadata)
    state_hash: str = ""
    created_at: str = Field(default_factory=utc_now)

    def memory(self) -> ResearchMemory:
        return ResearchMemory(
            agent_id=self.survivor_id,
            scope=self.scope,
            scope_origin=self.scope_origin,
            dossiers=self.dossiers,
            claims=self.claims,
            concepts=self.concepts,
            relations=self.relations,
            gaps=self.gaps,
        )

    def compute_state_hash(self) -> str:
        payload = {
            "survivor_id": self.survivor_id,
            "scope": self.scope,
            "dossiers": self.dossiers,
            "claims": self.claims,
            "concepts": self.concepts,
            "relations": self.relations,
            "gaps": self.gaps,
            "evidence_index": self.evidence_index,
        }
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

    def freeze(self) -> SurvivorBundle:
        self.state_hash = self.compute_state_hash()
        return self


class LaterAnswer(BaseModel):
    question: str
    supported: list[str] = Field(default_factory=list)
    plausible: list[str] = Field(default_factory=list)
    disputed: list[str] = Field(default_factory=list)
    unknown: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    assembled_context_chars: int = 0
    answered_with_model: str | None = None
