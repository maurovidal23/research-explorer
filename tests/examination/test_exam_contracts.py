"""Leakage, matching, and unavailable-arm contracts (EXAM-2/5/6/7/8/9).

PRD §4 cases 11, 14, 15, 17: test takers receive no private fields, the two
holdout arms are matched, the frozen survivor is the examined state, and an
unavailable arm is reported as unavailable rather than as zero.
"""

from __future__ import annotations

import asyncio

from research_explorer.examination.benchmark import (
    OUTCOME_BENCHMARKED,
    OUTCOME_SURVIVOR_UNBENCHMARKED,
    BenchmarkConfig,
    BenchmarkRunner,
)
from research_explorer.examination.clients import FakeAnswerClient
from research_explorer.examination.events import survivor_payloads
from research_explorer.examination.generator import FakeExaminer
from research_explorer.examination.models import (
    CATEGORY_CONCEPTS,
    DIFFICULTY_EASY,
    CandidateMemory,
    ExamItem,
    ExamOption,
    SelectionWeights,
)
from research_explorer.examination.selection import (
    candidate_eligibility,
    compute_selection_accuracy,
    grounding_score,
)
from research_explorer.examination.synthetic import build_synthetic_corpus
from research_explorer.memory.models import ContentKind


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        selection_count=6,
        holdout_count=4,
        partition_seed=7,
        weights=SelectionWeights(),
        model_ids={"answer_model": "fake-answer-v1"},
        prompt_versions={"examiner": "v1", "answer": "v1"},
    )


def _run(client, config: BenchmarkConfig | None = None):
    pack, candidates, acquired = build_synthetic_corpus()
    runner = BenchmarkRunner(pack, FakeExaminer(), client, config or _config())
    result = asyncio.run(runner.run(candidates, acquired))
    return runner, result


class _RecordingClient(FakeAnswerClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, list[dict]]] = []

    async def answer(self, responder, questions, context):
        self.calls.append((responder, [q.model_dump() for q in questions]))
        return await super().answer(responder, questions, context)


class _RecordingTracer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type: str, **payload) -> None:
        self.events.append((event_type, payload))


def test_benchmark_runner_emits_candidate_test_events() -> None:
    pack, candidates, acquired = build_synthetic_corpus()
    tracer = _RecordingTracer()
    runner = BenchmarkRunner(
        pack, FakeExaminer(), FakeAnswerClient(), _config(), tracer=tracer
    )
    result = asyncio.run(runner.run(candidates, acquired))
    assert result.outcome == OUTCOME_BENCHMARKED
    types = [event_type for event_type, _ in tracer.events]
    assert "candidate_test_started" in types
    assert "candidate_test_completed" in types
    started = [payload for event_type, payload in tracer.events if event_type == "candidate_test_started"]
    partitions = {payload["partition"] for payload in started}
    assert {"selection", "holdout"} <= partitions


def test_test_takers_never_receive_private_fields() -> None:
    client = _RecordingClient()
    _runner, result = _run(client)
    assert result.outcome == OUTCOME_BENCHMARKED
    assert client.calls
    public_fields = {"question_id", "category", "difficulty", "question", "options"}
    for _responder, questions in client.calls:
        assert questions
        for question in questions:
            assert set(question) == public_fields
            serialized = str(question)
            for forbidden in ("correct_option_id", "rationale", "evidence_refs", "answer_key"):
                assert forbidden not in serialized


def test_both_arms_use_identical_public_questions_and_model() -> None:
    client = _RecordingClient()
    runner, result = _run(client)
    holdout = {responder: questions for responder, questions in client.calls if responder in ("survivor", "naive")}
    assert set(holdout) == {"survivor", "naive"}
    assert holdout["survivor"] == holdout["naive"]
    assert result.model_ids.get("answer_model") == client.model_id
    assert runner.answer_client is client


def test_frozen_state_hash_is_reported_and_matches_examined_memory() -> None:
    runner, result = _run(FakeAnswerClient())
    bundle = runner.survivor_bundle
    assert bundle is not None
    assert result.state_hash == bundle.state_hash
    frozen = dict(survivor_payloads(result))["survivor_frozen"]
    assert frozen["state_hash"] == bundle.state_hash
    assert frozen["survivor_id"] == result.survivor_id


class _NaiveOfflineClient:
    model_id = "fake-answer-v1"

    async def answer(self, responder, questions, context):
        if responder == "naive":
            raise RuntimeError("naive arm offline")
        return {q.question_id: "A" for q in questions}


def test_baseline_failure_is_unavailable_not_zero() -> None:
    _runner, result = _run(_NaiveOfflineClient())
    assert result.outcome == OUTCOME_SURVIVOR_UNBENCHMARKED
    assert result.reason_code == "baseline_failed"
    assert result.survivor_accuracy is None
    assert result.naive_accuracy is None
    assert result.uplift is None
    assert result.survivor_score is None
    assert result.naive_score is None


class _TinyExaminer:
    model_id = "tiny-examiner"

    async def generate(self, pack, spec):
        items = []
        for index in range(3):
            source_id = pack.sources[index % len(pack.sources)].source_id
            filler = " ".join(f"token{index}x{n}" for n in range(20))
            items.append(
                ExamItem(
                    question_id=f"tiny-{index}",
                    category=CATEGORY_CONCEPTS,
                    difficulty=DIFFICULTY_EASY,
                    question=f"Unique grounded question {index} {filler}?",
                    options=[
                        ExamOption(id="A", text=f"Entailed {index} about {source_id}."),
                        ExamOption(id="B", text=f"False {index} about {source_id}."),
                        ExamOption(id="C", text=f"Incomplete {index} about {source_id}."),
                        ExamOption(id="D", text=f"Unrelated {index} about {source_id}."),
                    ],
                    correct_option_id="A",
                    evidence_refs=[source_id],
                )
            )
        return items


def test_insufficient_validated_items_has_stable_reason_and_no_holdout_shrink() -> None:
    pack, candidates, acquired = build_synthetic_corpus()
    runner = BenchmarkRunner(pack, _TinyExaminer(), FakeAnswerClient(), _config())
    result = asyncio.run(runner.run(candidates, acquired))
    assert result.outcome == "completed_exam_insufficient"
    assert result.reason_code == "insufficient_validated_questions"
    assert runner.bank is None
    assert result.survivor_accuracy is None
    assert result.naive_accuracy is None
    assert result.uplift is None


def test_selection_accuracy_helper_handles_empty() -> None:
    accuracy, coverage = compute_selection_accuracy(3, 4)
    assert accuracy == 0.75
    assert coverage == 0.75
    empty, empty_coverage = compute_selection_accuracy(0, 0)
    assert empty is None
    assert empty_coverage == 0.0


def test_eligibility_gates_cover_provenance_and_coverage() -> None:
    base = dict(
        process_score=0.5,
        grounding_score=0.5,
        has_evidence_bearing_dossier=True,
        selection_accuracy=0.5,
        selection_answered=5,
        selection_coverage=1.0,
    )
    assert candidate_eligibility(CandidateMemory(agent_id="a", **base), 0.5) == ""
    assert (
        candidate_eligibility(
            CandidateMemory(agent_id="a", **{**base, "provenance_integrity": False}), 0.5
        )
        == "provenance_integrity_failed"
    )
    assert (
        candidate_eligibility(
            CandidateMemory(agent_id="a", **{**base, "selection_answered": 0}), 0.5
        )
        == "no_selection_response"
    )
    assert (
        candidate_eligibility(
            CandidateMemory(agent_id="a", **{**base, "selection_coverage": 0.1}), 0.5
        )
        == "below_min_examination_coverage"
    )


def test_grounding_score_rewards_evidence_and_preserves_integrity() -> None:
    from research_explorer.examination.synthetic import build_synthetic_dossiers
    from research_explorer.memory.extract import research_memory_from_dossiers

    dossiers = build_synthetic_dossiers(count=3)
    memory = research_memory_from_dossiers("agent-a", "scope", dossiers)
    acquired = {pid: d.content_kind for pid, d in dossiers.items()}
    score = grounding_score(memory, acquired)
    assert 0.0 <= score <= 1.0
    empty = grounding_score(
        research_memory_from_dossiers("agent-b", "scope", {}), {}
    )
    assert empty == 0.0
    assert ContentKind.FULL_TEXT.has_evidence
