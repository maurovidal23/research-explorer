"""Final answer as an evidence-integrity view over the structured state."""

from __future__ import annotations

from research_explorer.research.answer import build_final_answer
from research_explorer.research.evaluator import DeterministicIntegrity
from research_explorer.research.models import (
    AgentNotebook,
    Claim,
    ClaimStatus,
    EvidenceRef,
    OpenQuestion,
    OpenQuestionStatus,
    ResearchEvaluation,
    ResearchObjective,
    ResearchState,
)


def _objective() -> ResearchObjective:
    return ResearchObjective(
        run_id="r", seed_paper_id="openalex:seed", question="How do fields arise?"
    )


def _state(claims=None, questions=None, evidence=None, evaluation=None) -> ResearchState:
    return ResearchState(
        objective=_objective(),
        notebook=AgentNotebook(agent_id="a"),
        claims=claims or {},
        questions=questions or {},
        evidence=evidence or [],
        latest_evaluation=evaluation,
    )


def test_unsupported_proposed_claim_is_never_presented_as_fact() -> None:
    state = _state(
        claims={"c1": Claim(id="c1", text="Unbacked guess", status=ClaimStatus.PROPOSED)}
    )
    answer = build_final_answer(_objective(), state)
    assert answer.supported_conclusions == []
    assert answer.plausible_interpretations == []
    assert "Unbacked guess" not in answer.render_markdown()


def test_supported_claim_cites_resolvable_evidence() -> None:
    ref = EvidenceRef(paper_id="openalex:W1", content_hash="h")
    state = _state(
        claims={
            "c1": Claim(
                id="c1",
                text="Fields arise from currents",
                status=ClaimStatus.SUPPORTED,
                confidence=0.8,
                supporting=[ref],
            )
        },
        evidence=[ref],
    )
    answer = build_final_answer(_objective(), state)
    assert answer.supported_conclusions == ["Fields arise from currents [openalex:W1]"]
    assert answer.citations == ["openalex:W1"]
    integrity = DeterministicIntegrity(
        paper_exists=lambda pid: pid == "openalex:W1"
    ).evaluate(state, answer)
    assert integrity.checks["final_answer_citations_resolve"]
    assert integrity.passed


def test_disputed_claim_is_disclosed_as_contradiction() -> None:
    state = _state(
        claims={
            "c1": Claim(
                id="c1",
                text="Fields scale linearly",
                status=ClaimStatus.DISPUTED,
                contradicting=[EvidenceRef(paper_id="openalex:W2")],
            )
        }
    )
    answer = build_final_answer(_objective(), state)
    assert answer.contradictions == [
        "Fields scale linearly (contradicted by openalex:W2)"
    ]
    assert answer.citations == ["openalex:W2"]


def test_open_questions_become_unknowns() -> None:
    state = _state(
        questions={
            "q1": OpenQuestion(
                id="q1", text="Which currents?", status=OpenQuestionStatus.OPEN
            )
        }
    )
    answer = build_final_answer(_objective(), state)
    assert answer.unknowns == ["Which currents?"]


def test_no_supported_claim_adds_limitation() -> None:
    answer = build_final_answer(_objective(), _state())
    assert any(
        "No claim reached supported status" in limitation
        for limitation in answer.limitations
    )
    assert answer.evidence_sufficient is False
    assert "insufficient" in answer.render_markdown().casefold()


def test_evaluator_missing_knowledge_becomes_limitation() -> None:
    state = _state(
        claims={
            "c1": Claim(
                id="c1",
                text="Supported",
                status=ClaimStatus.SUPPORTED,
                supporting=[EvidenceRef(paper_id="openalex:W1")],
            )
        },
        evaluation=ResearchEvaluation(missing_knowledge=["no dose-response data"]),
    )
    answer = build_final_answer(_objective(), state)
    assert "no dose-response data" in answer.limitations


def test_supported_claim_renders_optional_locator() -> None:
    ref = EvidenceRef(
        paper_id="openalex:W1", locator="Results 3.2", content_hash="h"
    )
    state = _state(
        claims={
            "c1": Claim(
                id="c1",
                text="Fields arise",
                status=ClaimStatus.SUPPORTED,
                supporting=[ref],
            )
        },
        evidence=[ref],
    )
    answer = build_final_answer(_objective(), state)
    assert answer.supported_conclusions == [
        "Fields arise [openalex:W1 (Results 3.2)]"
    ]
    assert "Results 3.2" in answer.render_markdown()


def test_provisional_tracks_evidence_volume_threshold() -> None:
    ref = EvidenceRef(paper_id="openalex:W1", content_hash="h", setting="oncology")
    claim = Claim(
        id="c1", text="c", status=ClaimStatus.SUPPORTED, supporting=[ref]
    )
    sparse = _state(claims={"c1": claim}, evidence=[ref])
    assert build_final_answer(_objective(), sparse).provisional is True


def test_provisional_requires_cross_setting_coverage() -> None:
    ref = EvidenceRef(paper_id="openalex:W1", content_hash="h", setting="oncology")
    claim = Claim(
        id="c1", text="c", status=ClaimStatus.SUPPORTED, supporting=[ref]
    )
    same_setting = [
        ref,
        EvidenceRef(paper_id="openalex:W2", content_hash="h", setting="oncology"),
        EvidenceRef(paper_id="openalex:W3", content_hash="h", setting="oncology"),
    ]
    same_answer = build_final_answer(
        _objective(), _state(claims={"c1": claim}, evidence=same_setting)
    )
    assert same_answer.provisional is True
    assert "conclusions are provisional" in " ".join(
        same_answer.limitations
    ).casefold()

    cross_setting = [
        ref,
        EvidenceRef(paper_id="openalex:W2", content_hash="h", setting="cardiology"),
        EvidenceRef(paper_id="openalex:W3", content_hash="h", setting="neurology"),
    ]
    cross_answer = build_final_answer(
        _objective(), _state(claims={"c1": claim}, evidence=cross_setting)
    )
    assert cross_answer.provisional is False
    assert "conclusions are provisional" not in " ".join(
        cross_answer.limitations
    ).casefold()


def test_limitations_are_deduplicated_and_provenance_disproved_dropped() -> None:
    ref = EvidenceRef(
        paper_id="openalex:W1",
        locator="reported sample size 12",
        content_hash="h",
    )
    evaluation = ResearchEvaluation(
        missing_knowledge=["alpha gap", "Alpha   gap", "sample size 12"]
    )
    state = _state(evidence=[ref], evaluation=evaluation)
    answer = build_final_answer(_objective(), state)
    lowered = [limitation.casefold() for limitation in answer.limitations]
    assert sum("alpha" in text and "gap" in text for text in lowered) == 1
    assert not any("sample size 12" in text for text in lowered)
