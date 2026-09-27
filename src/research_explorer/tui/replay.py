"""Replay helpers: load durable events and hydrate missing detail.

The live TUI and replay share one :class:`RunProjection`. Replay additionally
recovers full evaluation rationales and narrative content from the existing
``evaluation_results`` and ``artifacts`` tables, without duplicating records
already carried inline by the trace events.
"""

from __future__ import annotations

import re

from research_explorer.events.projection import RunProjection
from research_explorer.replay.trace import RunTraceStore

_NARRATIVE_RE = re.compile(r"^narrative_(?P<agent>.+?)(?:_t\d+)?\.md$")


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
