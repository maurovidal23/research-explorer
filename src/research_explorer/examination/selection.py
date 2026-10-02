"""Terminal survivor selection (EXAM-7).

Existing ``S + P + J + R`` becomes ``Q_process``, a navigation signal.
``Q_terminal = w_sel * E_selection + w_proc * Q_process + w_g * G`` with typed,
validated weights.
"""

from __future__ import annotations

from research_explorer.examination.models import (
    CandidateMemory,
    SelectionWeights,
    SurvivorSelection,
)
from research_explorer.memory.ledger import validate_claim
from research_explorer.memory.models import ClaimStatus, ContentKind, ResearchMemory

REASON_NO_EVIDENCE_BEARING_DOSSIER = "no_evidence_bearing_dossier"
REASON_PROVENANCE_INTEGRITY_FAILED = "provenance_integrity_failed"
REASON_NO_SELECTION_RESPONSE = "no_selection_response"
REASON_BELOW_MIN_COVERAGE = "below_min_examination_coverage"
REASON_NO_ELIGIBLE_CANDIDATE = "no_eligible_candidate"


def grounding_score(memory: ResearchMemory, acquired: dict[str, ContentKind]) -> float:
    """Deterministic grounding/integrity score over a candidate's memory.

    Combines evidence-bearing dossier coverage, the share of supported claims
    that resolve to acquired content, and provenance integrity.
    """
    dossiers = list(memory.dossiers.values())
    if not dossiers:
        return 0.0
    coverage = sum(1 for d in dossiers if d.has_evidence_credit) / len(dossiers)

    claims = list(memory.claims.values())
    supported = [c for c in claims if c.status is ClaimStatus.SUPPORTED]
    if supported:
        provenance = sum(1 for c in supported if not validate_claim(c, acquired)) / len(
            supported
        )
    else:
        provenance = 0.0

    integrity = 0.0 if any(validate_claim(c, acquired) for c in claims) else 1.0
    return round(0.5 * coverage + 0.3 * provenance + 0.2 * integrity, 6)


def candidate_eligibility(
    candidate: CandidateMemory,
    min_coverage: float,
) -> str:
    if not candidate.has_evidence_bearing_dossier:
        return REASON_NO_EVIDENCE_BEARING_DOSSIER
    if not candidate.provenance_integrity:
        return REASON_PROVENANCE_INTEGRITY_FAILED
    if candidate.selection_accuracy is None or candidate.selection_answered <= 0:
        return REASON_NO_SELECTION_RESPONSE
    if candidate.selection_coverage < min_coverage:
        return REASON_BELOW_MIN_COVERAGE
    return ""


def _terminal_score(candidate: CandidateMemory, weights: SelectionWeights) -> float:
    selection = candidate.selection_accuracy or 0.0
    return round(
        weights.selection * selection
        + weights.process * candidate.process_score
        + weights.grounding * candidate.grounding_score,
        6,
    )


def select_survivor(
    candidates: list[CandidateMemory],
    weights: SelectionWeights,
    min_coverage: float = 0.5,
) -> SurvivorSelection:
    """Rank eligible candidates and freeze exactly one survivor."""
    eligible: list[tuple[CandidateMemory, float]] = []
    for candidate in candidates:
        if not candidate_eligibility(candidate, min_coverage):
            eligible.append((candidate, _terminal_score(candidate, weights)))
    eligible.sort(
        key=lambda pair: (
            -pair[1],
            -(pair[0].selection_accuracy or 0.0),
            -pair[0].grounding_score,
            -pair[0].process_score,
            pair[0].agent_id,
        )
    )
    process_peak = max(
        (c.agent_id for c in candidates),
        key=lambda aid: next(
            (c.process_score for c in candidates if c.agent_id == aid), 0.0
        ),
        default="",
    )
    if not eligible:
        return SurvivorSelection(
            eligible=False,
            ineligible_reason=REASON_NO_ELIGIBLE_CANDIDATE,
            weights=weights,
            process_peak_agent=process_peak,
        )
    survivor, score = eligible[0]
    return SurvivorSelection(
        survivor_id=survivor.agent_id,
        eligible=True,
        terminal_score=score,
        selection_accuracy=survivor.selection_accuracy,
        process_score=survivor.process_score,
        grounding_score=survivor.grounding_score,
        weights=weights,
        ranking=[c.agent_id for c, _ in eligible],
        process_peak_agent=process_peak,
    )


def compute_selection_accuracy(
    correct: int, total: int
) -> tuple[float | None, float]:
    if total <= 0:
        return None, 0.0
    accuracy = correct / total
    return accuracy, accuracy


def terminal_formula(weights: SelectionWeights) -> str:
    """Render the effective terminal-score formula from configured weights."""
    return (
        f"Q_terminal = {weights.selection:.2f}*E_selection + "
        f"{weights.process:.2f}*Q_process + {weights.grounding:.2f}*G"
    )
