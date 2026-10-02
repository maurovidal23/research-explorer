"""Frozen survivor bundle round-trip and later-question answering (SURV-5).

PRD §4 case 6: "A survivor bundle round-trips and can answer a later question
without exploration." These tests build a bundle from structured memory, persist
and reload it, and answer a previously unseen question using only the frozen
bundle contents.
"""

from __future__ import annotations

import asyncio

from research_explorer.examination.models import EvidencePack
from research_explorer.examination.pack import build_evidence_pack
from research_explorer.examination.synthetic import build_synthetic_dossiers
from research_explorer.memory.extract import research_memory_from_dossiers
from research_explorer.memory.models import (
    ClaimStatus,
    ContentKind,
    DossierEvidence,
    LedgerClaim,
    PaperDossier,
    ResearchMemory,
    content_hash,
)
from research_explorer.survivor.answer import (
    answer_later_question,
    answer_later_question_with_model,
    assemble_context,
)
from research_explorer.survivor.bundle import build_bundle
from research_explorer.survivor.models import SelectionMetadata, SurvivorBundle


def _memory_and_pack() -> tuple[ResearchMemory, EvidencePack]:
    dossiers = build_synthetic_dossiers(count=6, scope="survivor scope")
    seed_id = sorted(dossiers)[0]
    distances = {
        paper_id: ("seed" if paper_id == seed_id else "direct_reference")
        for paper_id in dossiers
    }
    pack = build_evidence_pack(seed_id, "survivor scope", dossiers, distances=distances).freeze()
    memory = research_memory_from_dossiers("agent-a", "survivor scope", dossiers)

    first = dossiers[seed_id]
    evidence = next(ref for ref in first.evidence if ref.resolves)
    other = DossierEvidence(
        evidence_id="ev-other",
        paper_id=first.paper_id,
        content_kind=ContentKind.FULL_TEXT,
        content_hash=content_hash("other"),
        excerpt="other",
    )
    memory.claims["claim-disputed"] = LedgerClaim(
        id="claim-disputed",
        text="A claim that later work contradicts",
        status=ClaimStatus.DISPUTED,
        contradicting=[evidence.model_copy()],
        provenance=first.paper_id,
    )
    memory.claims["claim-unknown"] = LedgerClaim(
        id="claim-unknown",
        text="An unresolved open question",
        status=ClaimStatus.UNKNOWN,
    )
    memory.claims["claim-plausible"] = LedgerClaim(
        id="claim-plausible",
        text="A provisional interpretation of the evidence",
        status=ClaimStatus.PROPOSED,
        supporting=[other],
        provenance=first.paper_id,
    )
    return memory, pack


def _bundle() -> SurvivorBundle:
    memory, pack = _memory_and_pack()
    selection = SelectionMetadata(
        terminal_score=0.8,
        selection_accuracy=0.75,
        process_score=0.6,
        grounding_score=0.9,
        candidate_ranking=["agent-a", "agent-b"],
        process_peak_agent="agent-b",
    )
    return build_bundle(
        agent_id="agent-a",
        memory=memory,
        synthesis="Established findings\n- synthesized view",
        pack=pack,
        selection=selection,
        config_fingerprint="cfg-1",
        model_ids={"answer_model": "fake-answer-v1"},
        prompt_versions={"answer": "v1"},
        schema_versions={"bundle": "survivor-bundle/1"},
    )


def test_bundle_round_trips_through_persisted_json(tmp_path) -> None:
    bundle = _bundle()
    assert bundle.state_hash
    assert bundle.state_hash == bundle.compute_state_hash()
    assert bundle.bundle_version == "survivor-bundle/1"

    path = tmp_path / "survivor.json"
    path.write_text(bundle.model_dump_json(), encoding="utf-8")
    reloaded = SurvivorBundle.model_validate_json(path.read_text(encoding="utf-8"))

    assert reloaded.state_hash == bundle.state_hash
    assert reloaded.compute_state_hash() == bundle.state_hash
    assert set(reloaded.dossiers) == set(bundle.dossiers)
    assert set(reloaded.claims) == set(bundle.claims)
    assert reloaded.source_catalog == bundle.source_catalog
    assert reloaded.model_ids == bundle.model_ids
    assert reloaded.prompt_versions == bundle.prompt_versions
    assert reloaded.schema_versions == bundle.schema_versions
    assert reloaded.selection.candidate_ranking == ["agent-a", "agent-b"]
    assert reloaded.synthesis


def test_bundle_answers_later_question_without_exploration() -> None:
    bundle = _bundle()
    answer = answer_later_question(bundle, "Which finding is supported by study?")

    assert answer.question
    assert answer.supported, "supported claims must be usable for later answers"
    assert answer.assembled_context_chars > 0
    valid_sources = {entry.source_id for entry in bundle.source_catalog}
    valid_sources.update(bundle.evidence_index)
    assert set(answer.citations) <= valid_sources
    assert all(_cited(text) <= valid_sources for text in answer.supported)
    assert all(_cited(text) for text in answer.supported)


def test_answer_statuses_distinguish_supported_plausible_disputed_unknown() -> None:
    bundle = _bundle()
    answer = answer_later_question(bundle, "contradicts provisional unresolved findings")
    joined = " ".join(
        answer.supported + answer.plausible + answer.disputed + answer.unknown
    )
    assert "contradicts" in joined
    assert answer.unknown, "unknown/superseded claims are reported as unknowns"
    assert answer.disputed, "disputed claims are reported separately"
    assert answer.plausible, "supported-proposed claims are plausible"
    assert not set(answer.supported) & set(answer.disputed)


def test_retrieval_never_invents_sources_absent_from_bundle() -> None:
    bundle = _bundle()
    bundle.source_catalog = []
    bundle.evidence_index = {}
    answer = answer_later_question(bundle, "any question at all")
    assert answer.citations == []
    assert all("[" not in text for text in answer.supported)


def test_assemble_context_excludes_metadata_only_and_stays_bounded() -> None:
    memory, pack = _memory_and_pack()
    metadata = PaperDossier(
        paper_id="arxiv:metadata-only",
        title="Navigational node",
        content_kind=ContentKind.METADATA,
    )
    memory.dossiers[metadata.paper_id] = metadata
    bundle = build_bundle(
        agent_id="agent-a",
        memory=memory,
        synthesis="view",
        pack=pack,
        selection=SelectionMetadata(),
    )
    context = assemble_context(bundle, "question", max_chars=200)
    assert "arxiv:metadata-only" not in context
    assert len(context) <= 200


def test_later_answer_with_model_uses_bounded_context() -> None:
    bundle = _bundle()

    class _Client:
        model_id = "fake-later-answer"

        def __init__(self) -> None:
            self.got_context = ""

        async def answer(self, question: str, context: str) -> str:
            self.got_context = context
            return "Model synthesis grounded in the bundle."

    client = _Client()
    answer = asyncio.run(
        answer_later_question_with_model(bundle, "What follows?", client, max_chars=500)
    )
    assert answer.answered_with_model == "fake-later-answer"
    assert answer.supported
    assert len(client.got_context) <= 500


def _cited(text: str) -> set[str]:
    if "[" not in text:
        return set()
    inside = text.split("[", 1)[1].split("]", 1)[0]
    return {part.strip() for part in inside.split(",") if part.strip()}
