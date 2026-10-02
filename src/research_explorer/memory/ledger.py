"""Claim-ledger integrity: evidence resolution, deterministic dedupe.

A claim may only be ``supported`` when at least one of its references resolves
to *acquired* content. A graph-only (metadata-only or unacquired) paper can
never back a supported claim.
"""

from __future__ import annotations

from research_explorer.memory.extract import merge_claims
from research_explorer.memory.models import (
    ClaimStatus,
    ContentKind,
    DossierEvidence,
    LedgerClaim,
)

AcquiredIndex = dict[str, ContentKind]


def evidence_resolves(ref: DossierEvidence, acquired: AcquiredIndex) -> bool:
    """True only when the referenced paper has acquired, hashable content."""
    kind = acquired.get(ref.paper_id)
    if kind is None or not kind.has_evidence:
        return False
    return ref.resolves


def resolving_support(
    claim: LedgerClaim, acquired: AcquiredIndex
) -> list[DossierEvidence]:
    return [ref for ref in claim.supporting if evidence_resolves(ref, acquired)]


def validate_claim(claim: LedgerClaim, acquired: AcquiredIndex) -> list[str]:
    """Return integrity issues; an empty list means the claim is well-formed."""
    issues: list[str] = []
    if claim.status is ClaimStatus.SUPPORTED and not resolving_support(claim, acquired):
        issues.append("supported_without_acquired_evidence")
    if claim.status is ClaimStatus.DISPUTED and not any(
        evidence_resolves(ref, acquired) for ref in claim.contradicting
    ):
        issues.append("disputed_without_acquired_evidence")
    return issues


def enforce_claim_ledger(
    claims: dict[str, LedgerClaim], acquired: AcquiredIndex
) -> tuple[dict[str, LedgerClaim], list[str]]:
    """Demote claims that violate the evidence contract, deterministically.

    Returns the corrected ledger and the list of demoted claim ids. Earlier
    valid evidence is always preserved; only the illegal status is downgraded.
    """
    corrected: dict[str, LedgerClaim] = {}
    demoted: list[str] = []
    for claim_id in sorted(claims):
        claim = claims[claim_id]
        if validate_claim(claim, acquired):
            corrected_claim = claim.model_copy(deep=True)
            corrected_claim.status = ClaimStatus.PROPOSED
            corrected[claim_id] = corrected_claim
            demoted.append(claim_id)
        else:
            corrected[claim_id] = claim
    return corrected, demoted


def dedupe_claims(claims: list[LedgerClaim]) -> dict[str, LedgerClaim]:
    """Merge text-equivalent claims deterministically, preserving provenance."""
    merged: dict[str, LedgerClaim] = {}
    by_text: dict[str, str] = {}
    for claim in sorted(claims, key=lambda c: c.id):
        key = " ".join(claim.text.strip().lower().split())
        existing_id = by_text.get(key)
        if existing_id is None:
            merged[claim.id] = claim.model_copy(deep=True)
            by_text[key] = claim.id
        else:
            merged[existing_id] = merge_claims(merged[existing_id], claim)
    return merged
