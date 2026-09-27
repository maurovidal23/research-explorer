"""Seed-discovery diagnostics and empty-frontier classification.

The colony records one :class:`SeedDiscovery` per agent during initialization.
A valid seed whose initial frontier has no traversable candidates is a completed
but degraded run; exactly one primary reason code explains why.
"""

from __future__ import annotations

from dataclasses import dataclass

from research_explorer.events.models import (
    REASON_NO_NEIGHBORS_DISCOVERED,
    REASON_NO_TRAVERSABLE_IDENTIFIERS,
    REASON_REFERENCE_EXTRACTION_FAILED,
    REASON_SEED_DISCOVERY_FAILED,
)


@dataclass
class SeedDiscovery:
    """Structured per-agent result of discovering the seed's neighbors.

    ``extraction_attempted``/``extraction_failed`` are observed seed-discovery
    signals: the agent sets them only when it actually tried to extract the
    seed's bibliography and produced no usable entries. They are never inferred
    from provider capability.
    """

    agent_id: str
    refs: int = 0
    cits: int = 0
    traversable: int = 0
    failed: bool = False
    extraction_attempted: bool = False
    extraction_failed: bool = False


def classify_empty_frontier(records: list[SeedDiscovery]) -> str | None:
    """Return the single primary reason an initial frontier is empty.

    ``None`` means at least one traversable candidate survived, so the run
    may proceed. Otherwise exactly one of the four reason codes is returned.
    """
    if not records:
        return REASON_SEED_DISCOVERY_FAILED
    if all(record.failed for record in records):
        return REASON_SEED_DISCOVERY_FAILED
    neighbors = sum(record.refs + record.cits for record in records)
    traversable = sum(record.traversable for record in records)
    if traversable > 0:
        return None
    if neighbors == 0:
        if any(record.failed for record in records):
            return REASON_SEED_DISCOVERY_FAILED
        if any(record.extraction_failed for record in records):
            return REASON_REFERENCE_EXTRACTION_FAILED
        return REASON_NO_NEIGHBORS_DISCOVERED
    return REASON_NO_TRAVERSABLE_IDENTIFIERS
