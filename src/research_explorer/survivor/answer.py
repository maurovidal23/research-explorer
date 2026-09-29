"""Answer later questions from a frozen survivor bundle (SURV-5).

Retrieval may compact context but never invents evidence absent from the bundle.
Answers cite canonical source ids and explicitly distinguish supported,
plausible, disputed, and unknown statements.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from research_explorer.memory.models import ClaimStatus, LedgerClaim
from research_explorer.survivor.models import LaterAnswer, SurvivorBundle

_WORD_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class LaterAnswerClient(Protocol):
    async def answer(self, question: str, context: str) -> str:
        ...


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _claim_score(claim: LedgerClaim, query: set[str]) -> int:
    return len(_tokens(claim.text) & query)


def relevant_claims(bundle: SurvivorBundle, question: str) -> list[LedgerClaim]:
    query = _tokens(question)
    scored = [
        (claim, _claim_score(claim, query))
        for claim in bundle.claims.values()
    ]
    matched = [claim for claim, score in scored if score > 0]
    if not matched:
        matched = [claim for claim, _ in scored]
    status_rank = {
        ClaimStatus.SUPPORTED: 0,
        ClaimStatus.DISPUTED: 1,
        ClaimStatus.PROPOSED: 2,
        ClaimStatus.UNKNOWN: 3,
        ClaimStatus.SUPERSEDED: 4,
    }
    return sorted(matched, key=lambda c: (status_rank.get(c.status, 9), c.id))


def assemble_context(
    bundle: SurvivorBundle, question: str, max_chars: int = 8_000
) -> str:
    query = _tokens(question)
    blocks: list[str] = []
    for paper_id in sorted(bundle.dossiers):
        dossier = bundle.dossiers[paper_id]
        if not dossier.has_evidence_credit:
            continue
        relevance = _tokens(dossier.title + " " + dossier.contribution + " " + " ".join(dossier.concepts))
        if query and not (relevance & query) and len(blocks) >= 4:
            continue
        excerpt = ""
        for ref in dossier.evidence:
            if ref.resolves:
                excerpt = ref.excerpt
                break
        blocks.append(f"[{paper_id}] {dossier.title}\n{excerpt}")
    text = "\n\n".join(blocks)
    return text[:max_chars]


def _cite(claim: LedgerClaim, valid_sources: set[str]) -> str | None:
    refs = sorted(
        {
            ref.paper_id
            for ref in claim.supporting
            if ref.paper_id in valid_sources
        }
    )
    if not refs:
        return claim.text
    return f"{claim.text} [{', '.join(refs)}]"


def answer_later_question(bundle: SurvivorBundle, question: str) -> LaterAnswer:
    """Deterministic answer assembled strictly from frozen bundle memory."""
    valid_sources = {entry.source_id for entry in bundle.source_catalog}
    valid_sources.update(bundle.evidence_index.keys())
    answer = LaterAnswer(question=question)
    answer.assembled_context_chars = len(assemble_context(bundle, question))

    for claim in relevant_claims(bundle, question):
        rendered = _cite(claim, valid_sources)
        if claim.status is ClaimStatus.SUPPORTED and rendered:
            answer.supported.append(rendered)
        elif claim.status is ClaimStatus.DISPUTED and rendered:
            answer.disputed.append(rendered)
        elif claim.status is ClaimStatus.PROPOSED and claim.has_resolving_support and rendered:
            answer.plausible.append(rendered)
        elif claim.status in (ClaimStatus.UNKNOWN, ClaimStatus.SUPERSEDED):
            answer.unknown.append(claim.text)

    answer.unknown.extend(gap.text for gap in bundle.gaps)

    citations: set[str] = set()
    for claim in bundle.claims.values():
        for ref in claim.supporting + claim.contradicting:
            if ref.paper_id in valid_sources:
                citations.add(ref.paper_id)
    for entry in bundle.source_catalog:
        citations.add(entry.source_id)
    answer.citations = sorted(citations)
    return answer


async def answer_later_question_with_model(
    bundle: SurvivorBundle,
    question: str,
    client: LaterAnswerClient,
    max_chars: int = 8_000,
) -> LaterAnswer:
    """LLM-assisted answer using bounded bundle context; citations stay canonical."""
    context = assemble_context(bundle, question, max_chars=max_chars)
    base = answer_later_question(bundle, question)
    if not context.strip():
        return base
    base.assembled_context_chars = len(context)
    try:
        rendered = await client.answer(question, context)
    except Exception:
        return base
    if rendered.strip():
        base.supported.append(rendered.strip())
        base.answered_with_model = getattr(client, "model_id", None)
    return base
