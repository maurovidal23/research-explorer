"""Revert-failing tests for structured research memory (SURV-2/3/4, SURV-5)."""

from __future__ import annotations

from research_explorer.agents.state import AgentState
from research_explorer.graph.models import Paper
from research_explorer.memory.extract import (
    claims_from_dossier,
    derive_content_kind,
    dossier_from_paper,
    memory_from_state,
)
from research_explorer.memory.ledger import (
    dedupe_claims,
    enforce_claim_ledger,
    evidence_resolves,
    validate_claim,
)
from research_explorer.memory.models import (
    ClaimStatus,
    ContentKind,
    DossierEvidence,
    LedgerClaim,
    PaperDossier,
    ResearchMemory,
    content_hash,
)
from research_explorer.memory.synthesis import synthesize_memory


def _full_text_paper(pid: str = "1", fulltext: str | None = "FULL TEXT BODY") -> Paper:
    return Paper(
        id=pid,
        title=f"Paper {pid}",
        abstract="SHORT ABSTRACT",
        provider="arxiv",
        fulltext=fulltext,
    )


def test_full_text_reaches_dossier_extraction_not_abstract() -> None:
    paper = _full_text_paper(fulltext="UNIQUE FULL TEXT SENTINEL")
    dossier = dossier_from_paper(paper, {"findings": ["finding"]})

    assert dossier.content_kind is ContentKind.FULL_TEXT
    assert dossier.evidence, "acquired full text must produce evidence"
    assert dossier.evidence[0].content_hash == content_hash("UNIQUE FULL TEXT SENTINEL")
    assert dossier.evidence[0].content_hash != content_hash("SHORT ABSTRACT")
    assert dossier.has_evidence_credit


def test_abstract_and_metadata_eligibility_are_distinct() -> None:
    abstract = _full_text_paper("1", fulltext=None)
    metadata = Paper(id="2", title="Metadata only", provider="arxiv")

    assert derive_content_kind(abstract) is ContentKind.ABSTRACT
    assert derive_content_kind(metadata) is ContentKind.METADATA

    abstract_dossier = dossier_from_paper(abstract)
    metadata_dossier = dossier_from_paper(metadata)
    assert abstract_dossier.has_evidence_credit
    assert not metadata_dossier.has_evidence_credit
    assert metadata_dossier.evidence == []


def test_supported_claim_cannot_reference_unacquired_graph_only_paper() -> None:
    acquired = {"arxiv:1": ContentKind.FULL_TEXT}
    graph_only_ref = DossierEvidence(
        evidence_id="ev-graph",
        paper_id="arxiv:999",
        content_kind=ContentKind.METADATA,
        content_hash="deadbeef",
    )
    claim = LedgerClaim(
        id="c1",
        text="A graph-only claim",
        status=ClaimStatus.SUPPORTED,
        supporting=[graph_only_ref],
    )
    assert not evidence_resolves(graph_only_ref, acquired)
    assert validate_claim(claim, acquired) == ["supported_without_acquired_evidence"]

    corrected, demoted = enforce_claim_ledger({"c1": claim}, acquired)
    assert demoted == ["c1"]
    assert corrected["c1"].status is ClaimStatus.PROPOSED


def test_failed_reextraction_preserves_prior_valid_evidence() -> None:
    state = AgentState(id="a", pos="arxiv:1")
    valid = dossier_from_paper(_full_text_paper())
    state.record_dossier(valid)

    failed = PaperDossier(
        paper_id=valid.paper_id,
        content_kind=ContentKind.METADATA,
        extraction_status="failed",
        extraction_error="model unavailable",
    )
    state.record_dossier(failed)

    assert state.dossiers[valid.paper_id].has_evidence_credit
    assert state.dossiers[valid.paper_id].evidence == valid.evidence
    assert any("model unavailable" in failure for failure in state.extraction_failures)


def test_equivalent_claims_merge_without_losing_provenance() -> None:
    paper = _full_text_paper()
    dossier = dossier_from_paper(paper, {"findings": ["Shared finding"]})
    claims = claims_from_dossier(dossier, {"findings": ["Shared finding"]})
    base = claims[0]

    other_evidence = DossierEvidence(
        evidence_id="ev-other",
        paper_id="arxiv:2",
        content_kind=ContentKind.ABSTRACT,
        content_hash=content_hash("other"),
        excerpt="other",
    )
    duplicate = base.model_copy(
        update={"id": "claim-other", "supporting": [other_evidence]}
    )
    merged = dedupe_claims([base, duplicate])
    assert len(merged) == 1
    merged_claim = next(iter(merged.values()))
    refs = {ref.evidence_id for ref in merged_claim.supporting}
    assert refs == {"ev-other", base.supporting[0].evidence_id}


def test_synthesis_regeneration_does_not_mutate_memory_and_is_configurable() -> None:
    paper = _full_text_paper()
    dossier = dossier_from_paper(
        paper, {"findings": ["finding one", "finding two"], "limitations": ["limit"]}
    )
    memory = ResearchMemory(
        agent_id="a",
        scope="scope",
        dossiers={dossier.paper_id: dossier},
        claims={c.id: c for c in claims_from_dossier(dossier, {"findings": dossier.results})},
    )
    before = memory.model_dump()
    short = synthesize_memory(memory, max_words=50)
    long = synthesize_memory(memory, max_words=2000)
    assert memory.model_dump() == before
    assert len(short.split()) <= 50
    assert len(long.split()) >= len(short.split())
    assert "Established findings" in long


def test_state_memory_view_round_trips() -> None:
    state = AgentState(id="a", pos="arxiv:1", research_scope="scope", scope_origin="user")
    dossier = dossier_from_paper(_full_text_paper())
    state.record_dossier(dossier)
    memory = memory_from_state(state)
    assert memory.agent_id == "a"
    assert memory.scope_origin == "user"
    assert memory.dossiers[dossier.paper_id].has_evidence_credit
