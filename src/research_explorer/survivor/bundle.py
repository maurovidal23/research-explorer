"""Build a frozen :class:`SurvivorBundle` from evaluated structured memory."""

from __future__ import annotations

from typing import Any

from research_explorer.examination.models import EvidencePack
from research_explorer.memory.models import ResearchMemory
from research_explorer.survivor.models import SelectionMetadata, SurvivorBundle


def build_bundle(
    agent_id: str,
    memory: ResearchMemory,
    synthesis: str,
    pack: EvidencePack,
    selection: SelectionMetadata,
    config_fingerprint: str = "",
    model_ids: dict[str, str] | None = None,
    prompt_versions: dict[str, str] | None = None,
    schema_versions: dict[str, str] | None = None,
) -> SurvivorBundle:
    """Freeze exactly one survivor's structured memory and evidence index."""
    if agent_id != memory.agent_id:
        raise ValueError("bundle agent id must match the memory agent id")
    bundle = SurvivorBundle(
        survivor_id=agent_id,
        scope=memory.scope,
        scope_origin=memory.scope_origin,
        dossiers=dict(memory.dossiers),
        claims=dict(memory.claims),
        concepts=dict(memory.concepts),
        relations=list(memory.relations),
        gaps=list(memory.gaps),
        evidence_index=memory.evidence_index(),
        synthesis=synthesis,
        source_catalog=list(pack.sources),
        config_fingerprint=config_fingerprint,
        model_ids=dict(model_ids or {}),
        prompt_versions=dict(prompt_versions or {}),
        schema_versions=dict(schema_versions or {}),
        selection=selection,
    )
    return bundle.freeze()


def bundle_from_state(
    state: Any,
    pack: EvidencePack,
    selection: SelectionMetadata,
    config_fingerprint: str = "",
    model_ids: dict[str, str] | None = None,
    prompt_versions: dict[str, str] | None = None,
    schema_versions: dict[str, str] | None = None,
) -> SurvivorBundle:
    from research_explorer.memory.extract import memory_from_state

    return build_bundle(
        agent_id=state.id,
        memory=memory_from_state(state),
        synthesis=state.synthesis or state.regenerate_synthesis(),
        pack=pack,
        selection=selection,
        config_fingerprint=config_fingerprint,
        model_ids=model_ids,
        prompt_versions=prompt_versions,
        schema_versions=schema_versions,
    )
