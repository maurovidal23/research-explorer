"""Deterministic selection/holdout partitioning (EXAM-5).

Question ids are partitioned from a recorded random seed. Selection and holdout
ids can never overlap, and the holdout partition remains inaccessible until one
survivor has been frozen.
"""

from __future__ import annotations

import random

from research_explorer.examination.models import ExamItem


class PartitionError(RuntimeError):
    """Raised when the validated bank cannot fill both partitions."""

    def __init__(self, reason: str, available: int, required: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.available = available
        self.required = required


def partition_bank(
    items: list[ExamItem],
    selection_count: int,
    holdout_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Deterministically split validated item ids into (selection, holdout)."""
    ids = sorted(item.question_id for item in items)
    required = selection_count + holdout_count
    if len(ids) < required:
        raise PartitionError("insufficient_validated_items", len(ids), required)
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    selection = sorted(shuffled[:selection_count])
    holdout = sorted(shuffled[selection_count : selection_count + holdout_count])
    if set(selection) & set(holdout):
        raise PartitionError("partitions_overlap", len(ids), required)
    return selection, holdout
