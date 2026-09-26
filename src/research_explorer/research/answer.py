"""Final answer as a view over the structured research state.

The answer is generated from claims/questions/evidence, cites evidence ids that
resolve through the store, distinguishes supported conclusions from plausible
interpretations and unknowns, and never presents an unsupported proposed claim
as established fact.
"""

from __future__ import annotations

from research_explorer.research.models import (
    Claim,
    ClaimStatus,
    FinalAnswer,
    OpenQuestionStatus,
    ResearchObjective,
    ResearchState,
)


def _cite(claim: Claim) -> str:
    ids = sorted({ref.paper_id for ref in claim.supporting if ref.paper_id})
    if not ids:
        return claim.text
    return f"{claim.text} [{', '.join(ids)}]"


def build_final_answer(
    objective: ResearchObjective,
    state: ResearchState,
    paper_titles: dict[str, str] | None = None,
) -> FinalAnswer:
    supported: list[str] = []
    plausible: list[str] = []
    contradictions: list[str] = []
    for claim in sorted(state.claims.values(), key=lambda c: c.id):
        if claim.status is ClaimStatus.SUPPORTED and claim.supporting:
            supported.append(_cite(claim))
        elif claim.status is ClaimStatus.DISPUTED:
            refs = ", ".join(sorted({r.paper_id for r in claim.contradicting}))
            contradictions.append(f"{claim.text} (contradicted by {refs or 'sources'})")
        elif claim.status is ClaimStatus.PROPOSED and claim.has_evidence:
            plausible.append(_cite(claim))
        # Unsupported proposed claims are intentionally omitted from the answer.

    unknowns = [
        q.text
        for q in state.questions.values()
        if q.status in (OpenQuestionStatus.OPEN, OpenQuestionStatus.INVESTIGATING)
    ]
    unknowns.extend(
        q.text for q in state.questions.values() if q.status is OpenQuestionStatus.ABANDONED
    )

    limitations: list[str] = []
    if not supported:
        limitations.append("No claim reached supported status with valid evidence.")
    if state.latest_evaluation is not None:
        limitations.extend(state.latest_evaluation.missing_knowledge[:5])
        contradictions.extend(state.latest_evaluation.contradictions[:5])
    if len(state.evidence) < 3:
        limitations.append("Small evidence base; conclusions are provisional.")

    citations = sorted(
        {
            ref.paper_id
            for claim in state.claims.values()
            if claim.status in (ClaimStatus.SUPPORTED, ClaimStatus.DISPUTED)
            for ref in claim.supporting + claim.contradicting
            if ref.paper_id
        }
    )
    return FinalAnswer(
        question=objective.question,
        supported_conclusions=sorted(supported),
        plausible_interpretations=sorted(plausible),
        unknowns=sorted(set(unknowns)),
        contradictions=sorted(set(contradictions)),
        limitations=limitations,
        citations=citations,
    )
