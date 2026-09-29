"""Independent item validation (EXAM-4).

Validation is isolated from generation: it receives only the item (with its
private rationale) and the frozen pack. Ambiguous items are rejected rather
than repaired, and rejections are persisted with stable reason codes.
"""

from __future__ import annotations

import re

from research_explorer.examination.models import (
    CATEGORIES,
    DIFFICULTIES,
    EXAM_SCHEMA_VERSION,
    STATUS_REJECTED,
    STATUS_VALIDATED,
    EvidencePack,
    ExamItem,
)

REASON_INVALID_SCHEMA = "invalid_schema"
REASON_MISSING_KEY = "missing_key"
REASON_KEY_NOT_IN_OPTIONS = "key_not_in_options"
REASON_MULTIPLE_DEFENSIBLE = "multiple_defensible_options"
REASON_MISSING_EVIDENCE = "missing_evidence"
REASON_UNRESOLVED_EVIDENCE = "unresolved_evidence"
REASON_ANSWER_LEAKAGE = "answer_leakage"
REASON_DUPLICATE = "duplicate_question"
REASON_INVALID_CATEGORY = "invalid_category"
REASON_INVALID_DIFFICULTY = "invalid_difficulty"
REASON_CRITIC_REJECTED = "critic_rejected"

_LEAK_PHRASES = (
    "all of the above",
    "none of the above",
    "both a and b",
    "always correct",
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def validate_item(
    item: ExamItem,
    pack: EvidencePack,
    existing: list[ExamItem] | None = None,
) -> list[str]:
    """Return stable reason codes; an empty list means the item is accepted."""
    reasons: list[str] = []
    option_ids = [o.id for o in item.options]

    if (
        len(item.options) != 4
        or len(set(option_ids)) != len(option_ids)
        or any(not o.text.strip() for o in item.options)
        or not item.question.strip()
    ):
        reasons.append(REASON_INVALID_SCHEMA)

    if not item.correct_option_id:
        reasons.append(REASON_MISSING_KEY)
    elif item.correct_option_id not in option_ids:
        reasons.append(REASON_KEY_NOT_IN_OPTIONS)

    defensible = [oid for oid in item.defensible_option_ids if oid in option_ids]
    if len(defensible) > 1:
        reasons.append(REASON_MULTIPLE_DEFENSIBLE)

    if not item.evidence_refs:
        reasons.append(REASON_MISSING_EVIDENCE)
    elif pack is not None:
        known = set(pack.source_ids())
        if any(ref not in known for ref in item.evidence_refs):
            reasons.append(REASON_UNRESOLVED_EVIDENCE)

    if item.category not in CATEGORIES:
        reasons.append(REASON_INVALID_CATEGORY)
    if item.difficulty not in DIFFICULTIES:
        reasons.append(REASON_INVALID_DIFFICULTY)

    if _leakage(item):
        reasons.append(REASON_ANSWER_LEAKAGE)

    if existing and _is_duplicate(item, existing):
        reasons.append(REASON_DUPLICATE)

    if item.critic_status == STATUS_REJECTED:
        reasons.append(REASON_CRITIC_REJECTED)

    return reasons


def _leakage(item: ExamItem) -> bool:
    if not item.correct_option_id:
        return True
    texts = {o.id: o.text.strip() for o in item.options}
    key_text = texts.get(item.correct_option_id, "")
    lowered = key_text.lower()
    if any(phrase in lowered for phrase in _LEAK_PHRASES):
        return True
    question_lower = item.question.lower()
    if any(phrase in question_lower for phrase in _LEAK_PHRASES):
        return True
    lengths = {oid: len(text) for oid, text in texts.items()}
    if lengths.get(item.correct_option_id, 0) > 0:
        longest = max(lengths.values())
        shortest = min(lengths.values())
        key_len = lengths[item.correct_option_id]
        if longest > 0 and shortest > 0 and key_len == longest and longest > shortest * 1.6:
            return True
    return False


def _is_duplicate(item: ExamItem, existing: list[ExamItem]) -> bool:
    item_tokens = _tokens(item.question)
    return any(_jaccard(item_tokens, _tokens(other.question)) >= 0.85 for other in existing)


def validate_bank(
    items: list[ExamItem],
    pack: EvidencePack,
) -> tuple[list[ExamItem], list[ExamItem], list[str]]:
    """Validate every item before partitioning.

    Returns ``(accepted, rejected, rejection_reasons)``. Accepted items are
    marked ``validated``; rejected items keep their reason codes.
    """
    accepted: list[ExamItem] = []
    rejected: list[ExamItem] = []
    rejection_reasons: list[str] = []
    for item in items:
        reasons = validate_item(item, pack, existing=accepted)
        if reasons:
            item.validation_status = STATUS_REJECTED
            item.rejection_reasons = reasons
            rejected.append(item)
            rejection_reasons.extend(reasons)
        else:
            item.validation_status = STATUS_VALIDATED
            item.exam_version = item.exam_version or EXAM_SCHEMA_VERSION
            accepted.append(item)
    return accepted, rejected, sorted(set(rejection_reasons))
