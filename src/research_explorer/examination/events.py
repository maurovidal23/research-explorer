"""Typed, redacted event payloads for the examination lifecycle (OBS-1).

These payloads never contain examiner hidden reasoning, answer keys, option
keys, or evidence annotations. The private key is persisted only through a
separate restricted artifact.
"""

from __future__ import annotations

from typing import Any

from research_explorer.events.models import EventType
from research_explorer.examination.models import BenchmarkResult, EvidencePack, ExamBank


def evidence_pack_frozen_payload(pack: EvidencePack) -> dict[str, Any]:
    return {
        "pack_id": pack.pack_id,
        "pack_hash": pack.pack_hash,
        "source_count": len(pack.sources),
        "excluded_count": len(pack.excluded),
        "exclusion_reasons": sorted({x.reason for x in pack.excluded}),
    }


def exam_payloads(
    bank: ExamBank,
    accepted: int,
    rejected: int,
    rejection_reasons: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = [
        (
            EventType.EXAM_GENERATED,
            {"item_count": len(bank.items), "exam_version": bank.exam_version},
        ),
        (
            EventType.EXAM_VALIDATED,
            {
                "accepted": accepted,
                "rejected": rejected,
                "rejection_reasons": list(rejection_reasons),
            },
        ),
        (
            EventType.EXAM_PARTITIONED,
            {
                "partition_seed": bank.partition_seed,
                "selection_count": len(bank.selection_ids),
                "holdout_count": len(bank.holdout_ids),
            },
        ),
    ]
    for _event_type, payload in payloads:
        assert_no_private_key(payload)
    return payloads


def _bucket_payload(buckets: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        str(name): {"correct": int(bucket.correct), "total": int(bucket.total)}
        for name, bucket in buckets.items()
    }


def _public_outcome_payload(result: BenchmarkResult) -> dict[str, Any]:
    """Public post-benchmark outcomes: per-item correctness and bucket totals.

    These are booleans and aggregate counts only; no option key, rationale, or
    evidence annotation is ever included.
    """
    survivor = result.survivor_score
    naive = result.naive_score
    if survivor is None and naive is None:
        return {}
    payload: dict[str, Any] = {
        "item_outcomes": dict(survivor.item_outcomes) if survivor else {},
        "naive_item_outcomes": dict(naive.item_outcomes) if naive else {},
    }
    if survivor is not None:
        payload["by_category"] = _bucket_payload(survivor.by_category)
        payload["by_difficulty"] = _bucket_payload(survivor.by_difficulty)
    if naive is not None:
        payload["naive_by_category"] = _bucket_payload(naive.by_category)
        payload["naive_by_difficulty"] = _bucket_payload(naive.by_difficulty)
    return payload


def survivor_payloads(
    result: BenchmarkResult, synthesis: str = ""
) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    selection = result.selection
    if selection is not None:
        payloads.append(
            (
                EventType.SURVIVOR_SELECTED,
                {
                    "survivor_id": selection.survivor_id,
                    "terminal_score": selection.terminal_score,
                    "selection_accuracy": selection.selection_accuracy,
                    "process_score": selection.process_score,
                    "grounding_score": selection.grounding_score,
                    "ranking": list(selection.ranking),
                },
            )
        )
    payloads.append(
        (
            EventType.SURVIVOR_FROZEN,
            {"survivor_id": result.survivor_id, "state_hash": result.state_hash},
        )
    )
    payloads.append(
        (
            EventType.BASELINE_COMPLETED,
            {
                "survivor_accuracy": result.survivor_accuracy,
                "naive_accuracy": result.naive_accuracy,
                "uplift": result.uplift,
                "naive_unavailable": result.naive_unavailable,
                "survivor_unavailable": result.survivor_unavailable,
            },
        )
    )
    payloads.append(
        (
            EventType.BENCHMARK_COMPLETED,
            {
                "outcome": result.outcome,
                "reason_code": result.reason_code,
                "reason": result.reason,
                "survivor_synthesis": synthesis,
                **_public_outcome_payload(result),
            },
        )
    )
    for _event_type, payload in payloads:
        assert_no_private_key(payload)
    return payloads


def assert_no_private_key(payload: dict[str, Any]) -> None:
    """Raise when a payload leaks a private answer-key field."""
    forbidden = {"correct_option_id", "rationale", "answer_key", "evidence_refs"}
    leaked = forbidden & set(payload)
    if leaked:
        raise ValueError(f"private answer-key fields leaked into event: {sorted(leaked)}")
