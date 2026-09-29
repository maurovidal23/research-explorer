"""Deterministic extraction of structured memory from acquired papers.

The explorer's LLM integration may already return a structured ``analysis``
mapping; this module normalizes that mapping into a :class:`PaperDossier` and
typed :class:`LedgerClaim` objects. When no analysis is available it still
produces an evidence-backed dossier so acquired full text is never silently
reduced to its abstract.
"""

from __future__ import annotations

from typing import Any

from research_explorer.graph.models import Paper, PaperSummary
from research_explorer.memory.models import (
    ClaimStatus,
    Concept,
    ContentKind,
    DossierEvidence,
    KnowledgeGap,
    LedgerClaim,
    PaperDossier,
    ResearchMemory,
    content_hash,
    stable_id,
)

DEFAULT_CHUNK_CHARS = 12_000
DEFAULT_MAX_CHUNKS = 8


def derive_content_kind(paper: Paper | PaperSummary) -> ContentKind:
    """Classify acquired content: full text beats abstract beats metadata."""
    fulltext = getattr(paper, "fulltext", None)
    if isinstance(fulltext, str) and fulltext.strip():
        return ContentKind.FULL_TEXT
    abstract = getattr(paper, "abstract", None)
    if isinstance(abstract, str) and abstract.strip():
        return ContentKind.ABSTRACT
    return ContentKind.METADATA


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> list[tuple[str, str]]:
    """Bounded, deterministic chunking of acquired full text.

    Returns ``(locator, chunk)`` pairs; the locator is a stable
    ``chunk:<index>`` marker covering at most ``max_chunks`` chunks. When the
    text exceeds the budget the extra content is dropped deterministically, but
    the caller always sees that full text (not the abstract) was acquired.
    """
    if not text or not text.strip():
        return []
    size = max(1, max_chars)
    chunks: list[tuple[str, str]] = []
    for index in range(max_chunks):
        start = index * size
        if start >= len(text):
            break
        chunk = text[start : start + size]
        if chunk.strip():
            chunks.append((f"chunk:{index}", chunk))
    if not chunks:
        chunks.append(("chunk:0", text[:size]))
    return chunks


def _excerpt(text: str, limit: int = 240) -> str:
    clean = " ".join(text.split())
    return clean[:limit]


def _primary_evidence(paper: Paper, acquisition_event: int | None) -> list[DossierEvidence]:
    kind = derive_content_kind(paper)
    if not kind.has_evidence:
        return []
    if kind is ContentKind.FULL_TEXT:
        refs: list[DossierEvidence] = []
        for locator, chunk in chunk_text(paper.fulltext or ""):
            refs.append(
                DossierEvidence(
                    evidence_id=stable_id("ev", paper.provider, paper.id, locator),
                    paper_id=_normalized(paper),
                    content_kind=kind,
                    content_hash=content_hash(chunk),
                    acquisition_event=acquisition_event,
                    locator=locator,
                    excerpt=_excerpt(chunk),
                )
            )
        return refs
    abstract = paper.abstract or ""
    return [
        DossierEvidence(
            evidence_id=stable_id("ev", paper.provider, paper.id, "abstract"),
            paper_id=_normalized(paper),
            content_kind=kind,
            content_hash=content_hash(abstract),
            acquisition_event=acquisition_event,
            locator="abstract",
            excerpt=_excerpt(abstract),
        )
    ]


def _normalized(paper: Paper) -> str:
    from research_explorer.graph.models import normalize_id

    return normalize_id(paper.provider, paper.id)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _pick(analysis: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in analysis and analysis[key] not in (None, "", [], {}):
            return analysis[key]
    return None


def dossier_from_paper(
    paper: Paper,
    analysis: dict[str, Any] | None = None,
    acquisition_event: int | None = None,
    scope: str = "",
) -> PaperDossier:
    """Build a typed dossier for one acquired paper.

    Full text is preferred and recorded as such; an abstract-only paper is
    clearly marked ``abstract``; a metadata-only node carries no evidence.
    """
    analysis = analysis or {}
    kind = derive_content_kind(paper)
    evidence = _primary_evidence(paper, acquisition_event)
    dossier = PaperDossier(
        paper_id=_normalized(paper),
        title=paper.title or "",
        authors=list(paper.authors or []),
        year=paper.year,
        content_kind=kind,
        research_problem=_as_text(_pick(analysis, "research_problem", "problem")),
        contribution=_as_text(
            _pick(analysis, "contribution", "summary", "tldr") or (paper.tldr or "")
        ),
        definitions=_as_list(_pick(analysis, "definitions")),
        concepts=_as_list(_pick(analysis, "key_concepts", "concepts")),
        method=_as_text(_pick(analysis, "method", "methods")),
        design=_as_text(_pick(analysis, "design", "experimental_design")),
        datasets=_as_list(_pick(analysis, "datasets")),
        baselines=_as_list(_pick(analysis, "baselines")),
        assumptions=_as_list(_pick(analysis, "assumptions")),
        results=_as_list(_pick(analysis, "findings", "results")),
        limitations=_as_list(_pick(analysis, "limitations")),
        relevance=_as_text(_pick(analysis, "relevance")) or scope,
        relationships=_as_list(_pick(analysis, "relationships", "key_references")),
        evidence=evidence,
    )
    return dossier


def claims_from_dossier(
    dossier: PaperDossier,
    analysis: dict[str, Any] | None = None,
    created_seq: int = 0,
) -> list[LedgerClaim]:
    """Create atomic claims from a dossier's findings.

    Each claim carries the dossier's resolving evidence; a claim is only
    ``supported`` when at least one reference resolves to acquired content.
    """
    analysis = analysis or {}
    findings = _as_list(_pick(analysis, "findings", "results")) or dossier.results
    evidence = [ref for ref in dossier.evidence if ref.resolves]
    claims: list[LedgerClaim] = []
    for finding in findings:
        text = finding.strip()
        if not text:
            continue
        claim_id = stable_id("claim", dossier.paper_id, text)
        status = ClaimStatus.SUPPORTED if evidence else ClaimStatus.PROPOSED
        claims.append(
            LedgerClaim(
                id=claim_id,
                text=text,
                status=status,
                confidence=0.7 if evidence else 0.4,
                concepts=list(dossier.concepts),
                supporting=[ref.model_copy() for ref in evidence],
                provenance=dossier.paper_id,
                created_seq=created_seq,
                updated_seq=created_seq,
            )
        )
    return claims


def concepts_from_dossiers(dossiers: list[PaperDossier]) -> dict[str, Concept]:
    concepts: dict[str, Concept] = {}
    for dossier in dossiers:
        for label in dossier.concepts:
            key = stable_id("concept", label.strip().lower())
            concept = concepts.get(key)
            if concept is None:
                concepts[key] = Concept(
                    id=key, label=label.strip(), paper_ids=[dossier.paper_id]
                )
            elif dossier.paper_id not in concept.paper_ids:
                concept.paper_ids.append(dossier.paper_id)
    return concepts


def gaps_from_claims(claims: dict[str, LedgerClaim]) -> list[KnowledgeGap]:
    gaps: list[KnowledgeGap] = []
    for claim in claims.values():
        if claim.status is ClaimStatus.UNKNOWN:
            gaps.append(
                KnowledgeGap(
                    id=stable_id("gap", claim.id),
                    text=claim.text,
                    related_claims=[claim.id],
                )
            )
    return gaps


def merge_claims(existing: LedgerClaim, incoming: LedgerClaim) -> LedgerClaim:
    """Merge an equivalent claim without discarding additional provenance."""
    merged = existing.model_copy(deep=True)
    seen_support = {ref.evidence_id for ref in merged.supporting}
    for ref in incoming.supporting:
        if ref.evidence_id not in seen_support:
            merged.supporting.append(ref)
            seen_support.add(ref.evidence_id)
    seen_against = {ref.evidence_id for ref in merged.contradicting}
    for ref in incoming.contradicting:
        if ref.evidence_id not in seen_against:
            merged.contradicting.append(ref)
            seen_against.add(ref.evidence_id)
    merged.confidence = max(merged.confidence, incoming.confidence)
    for concept_id in incoming.concepts:
        if concept_id not in merged.concepts:
            merged.concepts.append(concept_id)
    if incoming.status is ClaimStatus.SUPPORTED and merged.has_resolving_support:
        merged.status = ClaimStatus.SUPPORTED
    merged.updated_seq = max(merged.updated_seq, incoming.updated_seq)
    return merged


def memory_from_state(state: Any) -> ResearchMemory:
    """Build a :class:`ResearchMemory` view from an ``AgentState`` (lazy import)."""
    from research_explorer.agents.state import AgentState

    assert isinstance(state, AgentState)
    dossiers = dict(state.dossiers)
    return ResearchMemory(
        agent_id=state.id,
        scope=state.research_scope,
        scope_origin=state.scope_origin,
        dossiers=dossiers,
        claims=dict(state.claims),
        concepts=dict(state.concepts),
        relations=list(state.relations),
        gaps=list(state.knowledge_gaps),
    )


def research_memory_from_dossiers(
    agent_id: str,
    scope: str,
    dossiers: dict[str, PaperDossier],
    scope_origin: str = "derived",
) -> ResearchMemory:
    """Build a claim-bearing :class:`ResearchMemory` from dossiers."""
    claims: dict[str, LedgerClaim] = {}
    for dossier in sorted(dossiers.values(), key=lambda d: d.paper_id):
        for claim in claims_from_dossier(
            dossier, {"findings": list(dossier.results)}
        ):
            existing = claims.get(claim.id)
            claims[claim.id] = claim if existing is None else merge_claims(existing, claim)
    return ResearchMemory(
        agent_id=agent_id,
        scope=scope,
        scope_origin=scope_origin,
        dossiers=dict(dossiers),
        claims=claims,
        concepts=concepts_from_dossiers(list(dossiers.values())),
    )
