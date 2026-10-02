"""Final answer as a view over the structured research state.

The answer is generated from claims/questions/evidence, cites evidence ids that
resolve through the store, distinguishes supported conclusions from plausible
interpretations and unknowns, and never presents an unsupported proposed claim
as established fact. The latest successful evaluator verdicts are also honored:
a claim the evaluator names unsupported or contradicted is never rendered as a
supported conclusion.
"""

from __future__ import annotations

from research_explorer.research.models import (
    Claim,
    ClaimStatus,
    FinalAnswer,
    OpenQuestionStatus,
    ResearchObjective,
    ResearchState,
    VerdictAssessment,
    normalize_claim_text,
)

PROVISIONAL_EVIDENCE_THRESHOLD = 3
PROVISIONAL_SETTING_THRESHOLD = 2


def _cite(claim: Claim) -> str:
    tokens: list[str] = []
    for ref in claim.supporting:
        if not ref.paper_id:
            continue
        if ref.locator:
            tokens.append(f"{ref.paper_id} ({ref.locator})")
        else:
            tokens.append(ref.paper_id)
    if not tokens:
        return claim.text
    return f"{claim.text} [{', '.join(sorted(set(tokens)))}]"


def _dedupe_limitations(
    limitations: list[str], state: ResearchState
) -> list[str]:
    """Deduplicate limitations and drop ones disproven by stored provenance."""
    metadata = normalize_claim_text(
        " ".join(ref.locator for ref in state.evidence if ref.locator)
    )
    seen: set[str] = set()
    out: list[str] = []
    for limitation in limitations:
        text = limitation.strip()
        if not text:
            continue
        key = normalize_claim_text(text)
        if key in seen:
            continue
        if metadata and key and key in metadata:
            continue
        seen.add(key)
        out.append(text)
    return out


def build_final_answer(
    objective: ResearchObjective,
    state: ResearchState,
) -> FinalAnswer:
    evaluation = state.latest_evaluation
    verdicts = {}
    if evaluation is not None and (evaluation.rubric is None or evaluation.rubric.ok):
        verdicts = {verdict.claim_id: verdict for verdict in evaluation.claim_verdicts}
    negated = {
        normalize_claim_text(item)
        for item in (
            evaluation.missing_knowledge if evaluation is not None else []
        )
        if item.strip()
    }

    supported: list[str] = []
    plausible: list[str] = []
    contradictions: list[str] = []
    for claim in sorted(state.claims.values(), key=lambda c: c.id):
        verdict = verdicts.get(claim.id)
        rejected = verdict is not None and verdict.assessment in (
            VerdictAssessment.UNSUPPORTED,
            VerdictAssessment.CONTRADICTED,
        )
        if claim.status is ClaimStatus.DISPUTED:
            refs = ", ".join(sorted({r.paper_id for r in claim.contradicting}))
            contradictions.append(f"{claim.text} (contradicted by {refs or 'sources'})")
        elif (
            claim.status is ClaimStatus.SUPPORTED
            and claim.supporting
            and not rejected
            and normalize_claim_text(claim.text) not in negated
        ):
            supported.append(_cite(claim))
        elif claim.status is ClaimStatus.PROPOSED and claim.has_evidence and not rejected:
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
        limitations.append(
            "No claim reached supported status with acquired evidence; the evidence "
            "is insufficient to state a supported conclusion."
        )
    if evaluation is not None:
        limitations.extend(evaluation.missing_knowledge[:5])
        contradictions.extend(evaluation.contradictions[:5])

    settings = {ref.setting for ref in state.evidence if ref.setting}
    provisional = (
        len(state.evidence) < PROVISIONAL_EVIDENCE_THRESHOLD
        or len(settings) < PROVISIONAL_SETTING_THRESHOLD
    )
    if provisional:
        if len(state.evidence) < PROVISIONAL_EVIDENCE_THRESHOLD:
            limitations.append(
                f"Small evidence base ({len(state.evidence)} acquired source(s)); "
                "conclusions are provisional."
            )
        else:
            limitations.append(
                f"Evidence covers {len(settings)} setting(s), below the documented "
                f"cross-setting threshold of {PROVISIONAL_SETTING_THRESHOLD}; "
                "conclusions are provisional."
            )

    citations = sorted(
        {ref.paper_id for ref in state.evidence if ref.paper_id}
        | {
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
        limitations=_dedupe_limitations(limitations, state),
        citations=citations,
        terminal_reason=state.terminal_reason,
        evidence_sufficient=bool(supported),
        provisional=provisional,
    )
