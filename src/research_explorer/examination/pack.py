"""Frozen examination evidence pack built only from acquired sources (EXAM-1).

The pack never contains agent narratives, candidate identities, memory
synthesis, or process scores. Metadata-only papers, unresolved identities, and
text without durable provenance are excluded with an auditable reason.
"""

from __future__ import annotations

import uuid

from research_explorer.examination.models import (
    EvidenceEntry,
    EvidencePack,
    SourceExclusion,
)
from research_explorer.memory.models import ContentKind, PaperDossier

REASON_METADATA_ONLY = "metadata_only"
REASON_NO_EVIDENCE = "no_resolving_evidence"
REASON_DUPLICATE = "duplicate_content"


def evidence_entry_from_dossier(
    dossier: PaperDossier,
    source_distance: str,
    excerpt_chars: int = 400,
) -> EvidenceEntry | None:
    if not dossier.has_evidence_credit:
        return None
    ref = next(ref for ref in dossier.evidence if ref.resolves)
    excerpt = " ".join(ref.excerpt.split())[:excerpt_chars]
    return EvidenceEntry(
        source_id=dossier.paper_id,
        title=dossier.title,
        content_kind=dossier.content_kind,
        content_hash=ref.content_hash,
        excerpt=excerpt,
        source_distance=source_distance,
        year=dossier.year,
    )


def build_evidence_pack(
    seed_paper_id: str,
    scope: str = "",
    dossiers: dict[str, PaperDossier] | None = None,
    distances: dict[str, str] | None = None,
    extra_exclusions: list[SourceExclusion] | None = None,
    pack_id: str | None = None,
) -> EvidencePack:
    """Build, dedupe, and freeze an evidence pack from acquired dossiers."""
    distances = distances or {}
    dossiers = dossiers or {}
    excluded: list[SourceExclusion] = list(extra_exclusions or [])
    seen_hashes: dict[str, str] = {}
    sources: list[EvidenceEntry] = []

    for paper_id in sorted(dossiers):
        dossier = dossiers[paper_id]
        if dossier.content_kind is ContentKind.METADATA:
            excluded.append(SourceExclusion(source_id=paper_id, reason=REASON_METADATA_ONLY))
            continue
        entry = evidence_entry_from_dossier(
            dossier, distances.get(paper_id, _default_distance(paper_id, seed_paper_id))
        )
        if entry is None:
            excluded.append(SourceExclusion(source_id=paper_id, reason=REASON_NO_EVIDENCE))
            continue
        prior = seen_hashes.get(entry.content_hash)
        if prior is not None:
            excluded.append(SourceExclusion(source_id=paper_id, reason=REASON_DUPLICATE))
            continue
        seen_hashes[entry.content_hash] = paper_id
        sources.append(entry)

    sources.sort(key=lambda e: (e.source_distance, e.source_id))
    pack = EvidencePack(
        pack_id=pack_id or uuid.uuid4().hex[:12],
        seed_paper_id=seed_paper_id,
        scope=scope,
        sources=sources,
        excluded=sorted(excluded, key=lambda x: (x.reason, x.source_id)),
    )
    return pack.freeze()


def _default_distance(paper_id: str, seed_paper_id: str) -> str:
    if paper_id == seed_paper_id:
        return "seed"
    return "direct_reference"
