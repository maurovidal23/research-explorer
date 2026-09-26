"""Malformed agent/evaluator containment and evaluator composition."""

from __future__ import annotations

import pytest

from research_explorer.research.agent import (
    AgentOutputError,
    LLMResearchAgent,
    parse_agent_brief,
)
from research_explorer.research.context import PromptBundle
from research_explorer.research.evaluator import (
    CompositeEvaluator,
    DeterministicIntegrity,
    LLMRubricEvaluator,
    normalize_claim_text,
)
from research_explorer.research.models import (
    AgentNotebook,
    Claim,
    ClaimStatus,
    EvidenceRef,
    FinalAnswer,
    ResearchObjective,
    ResearchState,
)


class _FakeLLM:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    async def chat_json(self, messages, **kwargs):
        if self.error is not None:
            raise self.error
        return self.response


def _objective() -> ResearchObjective:
    return ResearchObjective(run_id="r", seed_paper_id="seed", question="Q")


def _state() -> ResearchState:
    return ResearchState(objective=_objective(), notebook=AgentNotebook(agent_id="a"))


def _prompt() -> PromptBundle:
    return PromptBundle(messages=[{"role": "user", "content": "hi"}], template_version="t")


def test_parse_agent_brief_ignores_malformed_entries() -> None:
    brief = parse_agent_brief(
        {
            "action_summary": "did work",
            "claim_mutations": [
                {"op": "propose", "text": "A claim", "evidence": [{"paper_id": "p1"}]},
                "not-a-dict",
                {"op": "propose", "confidence": 9.0, "text": "bad range"},
                {"op": "unknown_op", "text": "ignored op still valid"},
            ],
            "evidence": [{"paper_id": "p1"}, {"nope": 1}],
            "contradictions": ["c1", 5, "c2"],
            "proposed_next_actions": "not-a-list",
        }
    )
    assert brief.action_summary == "did work"
    assert [m.text for m in brief.claim_mutations] == ["A claim", "ignored op still valid"]
    assert [e.paper_id for e in brief.evidence] == ["p1"]
    assert brief.contradictions == ["c1", "c2"]
    assert brief.proposed_next_actions == []


def test_parse_agent_brief_rejects_non_object() -> None:
    with pytest.raises(AgentOutputError):
        parse_agent_brief(["not", "an", "object"])


async def test_llm_agent_wraps_transport_error() -> None:
    agent = LLMResearchAgent(_FakeLLM(error=RuntimeError("boom")))
    with pytest.raises(AgentOutputError):
        await agent.propose_brief(_objective(), _state(), _prompt())


async def test_rubric_evaluator_accepts_valid_scores() -> None:
    llm = _FakeLLM(
        response={
            "dimension_scores": {"relevance": 0.8, "coverage": 0.5, "redundancy": "bad"},
            "missing_knowledge": ["gaps"],
        }
    )
    result = await LLMRubricEvaluator(llm).evaluate(_objective(), _state())
    assert result.ok
    assert result.dimension_scores == {"relevance": 0.8, "coverage": 0.5}
    assert result.missing_knowledge == ["gaps"]


async def test_rubric_evaluator_records_invalid_output() -> None:
    result = await LLMRubricEvaluator(_FakeLLM(response={"dimension_scores": "nope"})).evaluate(
        _objective(), _state()
    )
    assert not result.ok
    assert result.error


def test_integrity_flags_unsupported_supported_claim() -> None:
    state = _state()
    state.claims["c1"] = Claim(
        id="c1", text="unsupported", status=ClaimStatus.SUPPORTED
    )
    result = DeterministicIntegrity().evaluate(state)
    assert not result.passed
    assert "c1" in result.unsupported_claims


def test_integrity_flags_duplicate_claims() -> None:
    state = _state()
    state.claims["c1"] = Claim(id="c1", text="Same Claim!", status=ClaimStatus.PROPOSED)
    state.claims["c2"] = Claim(id="c2", text="same   claim", status=ClaimStatus.PROPOSED)
    assert normalize_claim_text("Same Claim!") == normalize_claim_text("same   claim")
    result = DeterministicIntegrity().evaluate(state)
    assert set(result.duplicate_claims) == {"c1", "c2"}


def test_integrity_flags_unresolved_final_citations() -> None:
    state = _state()
    state.claims["c1"] = Claim(
        id="c1",
        text="supported",
        status=ClaimStatus.SUPPORTED,
        supporting=[EvidenceRef(paper_id="openalex:W1")],
    )
    answer = FinalAnswer(question="Q", supported_conclusions=["supported"], citations=["missing"])
    result = DeterministicIntegrity(paper_exists=lambda pid: pid == "openalex:W1").evaluate(
        state, answer
    )
    assert result.unresolved_citations == ["missing"]


async def test_composite_keeps_deterministic_results_when_rubric_fails() -> None:
    state = _state()
    state.claims["c1"] = Claim(
        id="c1",
        text="ok",
        status=ClaimStatus.SUPPORTED,
        supporting=[EvidenceRef(paper_id="openalex:W1")],
    )
    evaluator = CompositeEvaluator(
        integrity=DeterministicIntegrity(paper_exists=lambda pid: True),
        rubric=LLMRubricEvaluator(_FakeLLM(response={"dimension_scores": 3})),
        weights={"integrity": 1.0},
    )
    evaluation = await evaluator.evaluate(_objective(), state)
    assert evaluation.rubric is not None and not evaluation.rubric.ok
    assert evaluation.dimension_scores["integrity"] == 1.0
    assert evaluation.overall == 1.0
