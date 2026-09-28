"""Replay helpers: load durable events and hydrate missing detail.

The live TUI and replay share one :class:`RunProjection`. Replay additionally
recovers full evaluation rationales and narrative content from the existing
``evaluation_results`` and ``artifacts`` tables, without duplicating records
already carried inline by the trace events.
"""

from __future__ import annotations

import re

from research_explorer.events.models import STATUS_INTERRUPTED, RunEvent
from research_explorer.events.projection import RunProjection
from research_explorer.replay.trace import RunTraceStore

_NARRATIVE_RE = re.compile(r"^narrative_(?P<agent>.+?)(?:_t\d+)?\.md$")


def apply_reconciled_status(projection: RunProjection, run: dict) -> RunProjection:
    """Project a reconciled ``interrupted`` row as a non-durable status event.

    A killed process leaves no terminal event, so replay must surface the
    heartbeat reconciliation without inventing a winner or touching the trace.
    """
    status = str(run.get("status") or "")
    if status == STATUS_INTERRUPTED and projection.state.status != STATUS_INTERRUPTED:
        reason = str(run.get("interrupt_reason") or "run was interrupted")
        projection.apply(
            RunEvent(
                seq=0,
                type="run_interrupted",
                payload={"reason": reason, "outcome": STATUS_INTERRUPTED},
            )
        )
    return projection


def hydrate_from_store(
    projection: RunProjection, store: RunTraceStore, run_id: str
) -> RunProjection:
    """Merge persisted evaluation/artifact detail into a projected run."""
    for detail in store.list_evaluations(run_id):
        projection.hydrate_evaluation(detail)
    for artifact in store.list_artifacts(run_id):
        if artifact.get("kind") != "narrative":
            continue
        full = store.get_artifact(artifact["artifact_id"])
        if full is None:
            continue
        name = str(full.get("name", ""))
        match = _NARRATIVE_RE.match(name)
        agent_id = match.group("agent") if match else ""
        projection.hydrate_narrative(agent_id, str(full.get("content", "")))
    return projection
