"""Malformed agent/evaluator containment and evaluator composition."""

from __future__ import annotations

import pytest

from research_explorer.graph.models import Paper
from research_explorer.research.agent import (
    AgentOutputError,
    LLMReferenceMapper,
    LLMResearchAgent,
    ReferenceMappingError,
    parse_agent_brief,
    reference_schema,
)
from research_explorer.research.context import (
    RUBRIC_DIMENSIONS,
    PromptBundle,
    rubric_schema,
)
from research_explorer.research.evaluator import (
    CompositeEvaluator,
    DeterministicIntegrity,
    LLMRubricEvaluator,
)
from research_explorer.research.models import (
    AgentNotebook,
    Claim,
    ClaimStatus,
    EvidenceRef,
    FinalAnswer,
    ResearchObjective,
    ResearchState,
    VerdictAssessment,
    normalize_claim_text,
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


def _dimension_scores(**overrides) -> dict[str, float]:
    scores = {dimension: 0.5 for dimension in RUBRIC_DIMENSIONS}
    scores.update(overrides)
    return scores


def _rubric_payload(claims=(), **overrides) -> dict:
    payload: dict = {
        "dimension_scores": _dimension_scores(),
        "missing_knowledge": [],
        "unsupported_claims": [],
        "contradictions": [],
        "recommended_questions": [],
        "claim_verdicts": [
            {"claim_id": cid, "assessment": "supported", "reason": "ok"}
            for cid in claims
        ],
    }
    payload.update(overrides)
    return payload


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
        response=_rubric_payload(
            missing_knowledge=["gaps"],
            dimension_scores=_dimension_scores(relevance=0.8, coverage=0.5),
        )
    )
    result = await LLMRubricEvaluator(llm).evaluate(_objective(), _state())
    assert result.ok
    assert result.dimension_scores == _dimension_scores(relevance=0.8, coverage=0.5)
    assert result.missing_knowledge == ["gaps"]


async def test_rubric_evaluator_rejects_omitted_required_dimension() -> None:
    llm = _FakeLLM(response=_rubric_payload(dimension_scores={"relevance": 0.8}))
    result = await LLMRubricEvaluator(llm).evaluate(_objective(), _state())
    assert not result.ok
    assert result.error and "coverage" in result.error
    assert result.dimension_scores == {}


async def test_rubric_evaluator_rejects_invalid_dimension_value() -> None:
    scores = _dimension_scores()
    scores["coverage"] = "bad"
    llm = _FakeLLM(response=_rubric_payload(dimension_scores=scores))
    result = await LLMRubricEvaluator(llm).evaluate(_objective(), _state())
    assert not result.ok
    assert result.error and "coverage" in result.error


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


def test_integrity_flags_claim_refs_not_resolved_to_acquired() -> None:
    state = _state()
    state.claims["c1"] = Claim(
        id="c1",
        text="graph resident but never acquired",
        status=ClaimStatus.SUPPORTED,
        supporting=[EvidenceRef(paper_id="openalex:W9")],
    )
    result = DeterministicIntegrity(paper_exists=lambda pid: True).evaluate(state)
    assert result.checks["claim_refs_resolve_to_acquired"] is False
    assert not result.passed
    assert any("never acquired" in issue for issue in result.issues)


def test_integrity_passes_when_claim_refs_resolve_to_acquired() -> None:
    ref = EvidenceRef(paper_id="openalex:W1", content_hash="h")
    state = _state()
    state.evidence.append(ref)
    state.claims["c1"] = Claim(
        id="c1",
        text="grounded",
        status=ClaimStatus.SUPPORTED,
        supporting=[ref],
    )
    result = DeterministicIntegrity(paper_exists=lambda pid: True).evaluate(state)
    assert result.checks["claim_refs_resolve_to_acquired"] is True
    assert result.passed


async def test_composite_keeps_deterministic_results_when_rubric_fails() -> None:
    state = _state()
    state.claims["c1"] = Claim(
        id="c1",
        text="ok",
        status=ClaimStatus.SUPPORTED,
        supporting=[EvidenceRef(paper_id="openalex:W1")],
    )
    state.evidence.append(EvidenceRef(paper_id="openalex:W1", content_hash="h"))
    evaluator = CompositeEvaluator(
        integrity=DeterministicIntegrity(paper_exists=lambda pid: True),
        rubric=LLMRubricEvaluator(_FakeLLM(response={"dimension_scores": 3})),
        weights={"integrity": 1.0},
    )
    evaluation = await evaluator.evaluate(_objective(), state)
    assert evaluation.rubric is not None and not evaluation.rubric.ok
    assert evaluation.dimension_scores["integrity"] == 1.0
    assert evaluation.overall == 1.0


def test_rubric_evaluator_default_judge_model() -> None:
    assert LLMRubricEvaluator(_FakeLLM()).model == "glm-5.3-flash"


async def test_rubric_evaluator_parses_structured_verdicts() -> None:
    state = _state()
    state.claims["c1"] = Claim(id="c1", text="claim", status=ClaimStatus.PROPOSED)
    state.claims["c2"] = Claim(id="c2", text="other", status=ClaimStatus.PROPOSED)
    llm = _FakeLLM(
        response=_rubric_payload(
            claim_verdicts=[
                {"claim_id": "c1", "assessment": "unsupported", "reason": "nope"},
                {"claim_id": "c2", "assessment": "supported", "reason": "yes"},
                {
                    "claim_id": "unknown",
                    "assessment": "unsupported",
                    "reason": "unknown id",
                },
            ]
        )
    )
    result = await LLMRubricEvaluator(llm).evaluate(_objective(), state)
    assert result.ok
    assert [v.claim_id for v in result.claim_verdicts] == ["c1", "c2"]
    assert result.claim_verdicts[0].assessment is VerdictAssessment.UNSUPPORTED
    assert result.claim_verdicts[0].reason == "nope"


async def test_rubric_evaluator_rejects_omitted_known_claim_verdict() -> None:
    state = _state()
    state.claims["c1"] = Claim(id="c1", text="claim", status=ClaimStatus.SUPPORTED)
    state.claims["c2"] = Claim(id="c2", text="other", status=ClaimStatus.PROPOSED)
    llm = _FakeLLM(
        response=_rubric_payload(
            claim_verdicts=[
                {"claim_id": "c1", "assessment": "unsupported", "reason": "nope"},
            ]
        )
    )
    result = await LLMRubricEvaluator(llm).evaluate(_objective(), state)
    assert not result.ok
    assert result.error and "c2" in result.error
    assert result.claim_verdicts == []


async def test_rubric_evaluator_rejects_malformed_structured_verdicts() -> None:
    state = _state()
    state.claims["c1"] = Claim(id="c1", text="claim", status=ClaimStatus.PROPOSED)
    llm = _FakeLLM(
        response=_rubric_payload(
            claim_verdicts=[
                {"claim_id": "c1", "assessment": "unsupported", "reason": "ok"},
                {"claim_id": "c1", "assessment": "bogus", "reason": "bad"},
            ]
        )
    )
    result = await LLMRubricEvaluator(llm).evaluate(_objective(), state)
    assert not result.ok
    assert result.error and "claim_verdicts" in result.error
    assert result.claim_verdicts == []


async def test_rubric_evaluator_rejects_malformed_aggregate_fields() -> None:
    llm = _FakeLLM(response=_rubric_payload(missing_knowledge="not-a-list"))
    result = await LLMRubricEvaluator(llm).evaluate(_objective(), _state())
    assert not result.ok
    assert result.error and "missing_knowledge" in result.error


async def test_rubric_evaluator_rejects_omitted_aggregate_field() -> None:
    payload = _rubric_payload()
    payload.pop("recommended_questions")
    result = await LLMRubricEvaluator(_FakeLLM(response=payload)).evaluate(
        _objective(), _state()
    )
    assert not result.ok
    assert result.error and "recommended_questions" in result.error


async def test_rubric_evaluator_rejects_omitted_claim_verdicts() -> None:
    payload = _rubric_payload()
    payload.pop("claim_verdicts")
    result = await LLMRubricEvaluator(_FakeLLM(response=payload)).evaluate(
        _objective(), _state()
    )
    assert not result.ok
    assert result.error and "claim_verdicts" in result.error
    assert result.claim_verdicts == []


def _verdict_payload(entry: dict) -> dict:
    return _rubric_payload(claim_verdicts=[entry])


async def test_rubric_evaluator_rejects_verdict_without_reason() -> None:
    state = _state()
    state.claims["c1"] = Claim(id="c1", text="claim", status=ClaimStatus.PROPOSED)
    llm = _FakeLLM(
        response=_verdict_payload({"claim_id": "c1", "assessment": "unsupported"})
    )
    result = await LLMRubricEvaluator(llm).evaluate(_objective(), state)
    assert not result.ok
    assert result.error and "reason" in result.error
    assert result.claim_verdicts == []


async def test_rubric_evaluator_validates_reason_before_ignoring_unknown_id() -> None:
    llm = _FakeLLM(
        response=_verdict_payload(
            {"claim_id": "does-not-exist", "assessment": "unsupported"}
        )
    )
    result = await LLMRubricEvaluator(llm).evaluate(_objective(), _state())
    assert not result.ok
    assert result.error and "reason" in result.error


class _SequencedLLM:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def chat_json(self, messages, **kwargs):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _reference_paper() -> Paper:
    return Paper(
        id="2301.00001",
        provider="arxiv",
        title="Seed",
        ref_entries=["Vaswani. Attention is all you need. arXiv:1706.03762"],
    )


async def test_reference_mapper_retries_malformed_within_bound() -> None:
    llm = _SequencedLLM(
        [
            ValueError("bad json"),
            {"references": [{"title": "Attention", "arxiv_id": "1706.03762"}]},
        ]
    )
    mapper = LLMReferenceMapper(llm, max_attempts=2)
    refs = await mapper.map_references(_reference_paper(), "q", 5)
    assert llm.calls == 2
    assert [ref.id for ref in refs] == ["1706.03762"]


async def test_reference_mapper_bounded_failure_raises() -> None:
    llm = _SequencedLLM([ValueError("bad"), ValueError("bad"), ValueError("bad")])
    mapper = LLMReferenceMapper(llm, max_attempts=2)
    with pytest.raises(ReferenceMappingError):
        await mapper.map_references(_reference_paper(), "q", 5)
    assert llm.calls == 2


class _RecordingLLM:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def chat_json(self, messages, **kwargs):
        self.calls.append(kwargs)
        return self.response


async def test_rubric_evaluator_requests_schema_constrained_output() -> None:
    llm = _RecordingLLM({"dimension_scores": {"relevance": 0.5}})
    await LLMRubricEvaluator(llm).evaluate(_objective(), _state())
    assert llm.calls[0]["schema"] == rubric_schema()
    assert "claim_verdicts" in llm.calls[0]["schema"]["properties"]


async def test_reference_mapper_requests_schema_constrained_output() -> None:
    llm = _RecordingLLM({"references": []})
    assert await LLMReferenceMapper(llm).map_references(_reference_paper(), "q", 5) == []
    assert llm.calls[0]["schema"] == reference_schema()
    assert "references" in llm.calls[0]["schema"]["required"]
