"""Deterministic synthetic benchmark corpus (network-free test profile).

Provides two candidate agents with different evidence reach so the matched
naive baseline can reproduce a descriptive uplift without any network access.
"""

from __future__ import annotations

from research_explorer.examination.models import EvidencePack
from research_explorer.examination.pack import build_evidence_pack
from research_explorer.graph.models import Paper
from research_explorer.memory.extract import (
    dossier_from_paper,
    research_memory_from_dossiers,
)
from research_explorer.memory.models import ContentKind, PaperDossier, ResearchMemory

DEFAULT_SCOPE = "synthetic benchmark scope"


def build_synthetic_dossiers(count: int = 6, scope: str = DEFAULT_SCOPE) -> dict[str, PaperDossier]:
    dossiers: dict[str, PaperDossier] = {}
    for index in range(count):
        paper = Paper(
            id=f"syn-{index:02d}",
            title=f"Synthetic study {index}",
            abstract=f"Synthetic abstract {index} on the benchmark scope.",
            provider="arxiv",
            fulltext=f"Synthetic full text {index}. " * 120,
            authors=["A. Author", "B. Author"],
            year=2000 + index,
        )
        dossier = dossier_from_paper(
            paper,
            {
                "research_problem": f"Problem addressed by study {index}",
                "contribution": f"Contribution of study {index}",
                "key_concepts": [f"concept-{index}"],
                "findings": [
                    f"Finding {index}-a supported by study {index}",
                    f"Finding {index}-b supported by study {index}",
                ],
                "limitations": [f"Limitation {index}"],
            },
            acquisition_event=index,
            scope=scope,
        )
        dossiers[dossier.paper_id] = dossier
    return dossiers


def build_synthetic_corpus(
    count: int = 6,
    scope: str = DEFAULT_SCOPE,
) -> tuple[
    EvidencePack,
    list[tuple[str, float, ResearchMemory]],
    dict[str, dict[str, ContentKind]],
]:
    """Return ``(pack, candidates, acquired_index)`` for a network-free run."""
    dossiers = build_synthetic_dossiers(count, scope)
    seed_id = sorted(dossiers)[0]
    distances = {
        paper_id: ("seed" if paper_id == seed_id else "direct_reference")
        for paper_id in dossiers
    }
    pack = build_evidence_pack(seed_id, scope, dossiers, distances=distances).freeze()
    rich = research_memory_from_dossiers(
        "survivor-candidate", scope, dossiers
    )
    seed_only = research_memory_from_dossiers(
        "seed-only-candidate", scope, {seed_id: dossiers[seed_id]}
    )
    acquired = {
        rich.agent_id: {
            paper_id: dossier.content_kind
            for paper_id, dossier in dossiers.items()
        },
        seed_only.agent_id: {seed_id: dossiers[seed_id].content_kind},
    }
    return pack, [(rich.agent_id, 0.6, rich), (seed_only.agent_id, 0.4, seed_only)], acquired
