"""Revert-failing tests for the hidden examination (EXAM-1..9, OBS-1)."""

from __future__ import annotations

from research_explorer.examination.benchmark import (
    OUTCOME_BENCHMARKED,
    BenchmarkConfig,
    BenchmarkRunner,
    _naive_context,
)
from research_explorer.examination.clients import FakeAnswerClient, build_answer_set
from research_explorer.examination.events import (
    assert_no_private_key,
    evidence_pack_frozen_payload,
)
from research_explorer.examination.generator import (
    FakeExaminer,
    GenerationSpec,
    build_answer_key,
)
from research_explorer.examination.models import (
    AnswerKey,
    CandidateMemory,
    EvidencePack,
    ExamBank,
    ExamItem,
    ExamOption,
    SelectionWeights,
    StudentQuestion,
)
from research_explorer.examination.pack import build_evidence_pack
from research_explorer.examination.partition import PartitionError, partition_bank
from research_explorer.examination.report import (
    benchmark_report_markdown,
    benchmark_result_json,
    private_key_artifact,
    public_exam_artifact,
)
from research_explorer.examination.scoring import score_answers
from research_explorer.examination.selection import (
    candidate_eligibility,
    select_survivor,
)
from research_explorer.examination.synthetic import build_synthetic_corpus
from research_explorer.examination.validation import validate_bank, validate_item
from research_explorer.memory.models import ContentKind, PaperDossier


def _student_questions(bank: ExamBank, ids: list[str]) -> list[StudentQuestion]:
    wanted = set(ids)
    return [
        StudentQuestion(
            question_id=i.question_id,
            category=i.category,
            difficulty=i.difficulty,
            question=i.question,
            options=list(i.options),
        )
        for i in bank.items
        if i.question_id in wanted
    ]


def _valid_options(questions: list[StudentQuestion]) -> dict[str, set[str]]:
    return {q.question_id: {o.id for o in q.options} for q in questions}


def _synthetic_bank() -> tuple[ExamBank, EvidencePack]:
    from research_explorer.graph.models import Paper
    from research_explorer.memory.extract import dossier_from_paper

    dossiers = {}
    for i in range(6):
        paper = Paper(
            id=f"syn-{i}",
            title=f"Synthetic {i}",
            provider="arxiv",
            fulltext=f"Unique body {i}. " * 90,
        )
        dossier = dossier_from_paper(paper, {"findings": [f"f{i}"]}, acquisition_event=i)
        dossiers[dossier.paper_id] = dossier
    pack = build_evidence_pack(
        "arxiv:syn-0",
        "scope",
        dossiers,
        distances={pid: ("seed" if pid == "arxiv:syn-0" else "direct_reference") for pid in dossiers},
    ).freeze()
    items = _run_generate_sync(pack)
    accepted, _rejected, _reasons = validate_bank(items, pack)
    bank = ExamBank(items=accepted, accepted_count=len(accepted), rejected_count=1)
    bank.selection_ids, bank.holdout_ids = partition_bank(accepted, 6, 4, 7)
    bank.partition_frozen = True
    return bank, pack


def _run_generate_sync(pack):
    import asyncio

    return asyncio.run(FakeExaminer().generate(pack, GenerationSpec(6, 4, seed=7)))


def test_evidence_pack_contains_no_agent_artifacts() -> None:
    dossiers = {}
    from research_explorer.graph.models import Paper
    from research_explorer.memory.extract import dossier_from_paper

    for i in range(3):
        paper = Paper(
            id=f"p{i}", title=f"P{i}", provider="arxiv", fulltext=f"body {i} " * 50
        )
        dossier = dossier_from_paper(paper)
        dossiers[dossier.paper_id] = dossier
    metadata = PaperDossier(paper_id="arxiv:meta", content_kind=ContentKind.METADATA)
    dossiers[metadata.paper_id] = metadata
    pack = build_evidence_pack("arxiv:p0", "scope", dossiers).freeze()
    serialized = pack.model_dump_json()
    assert "narrative" not in serialized.lower()
    assert "agent" not in serialized.lower()
    assert all(e.reason == "metadata_only" for e in pack.excluded)
    assert_no_private_key(evidence_pack_frozen_payload(pack))


def test_public_payload_excludes_private_fields() -> None:
    bank, _pack = _synthetic_bank()
    payload = bank.public_payload(bank.selection_ids)
    serialized = public_exam_artifact(bank)
    assert payload
    for item in payload:
        assert set(item) == {"question_id", "category", "difficulty", "question", "options"}
        assert "correct_option_id" not in item
    for forbidden in ("correct_option_id", "rationale", "evidence_refs"):
        assert forbidden not in serialized
    key_artifact = private_key_artifact(build_answer_key(bank, 7))
    assert "correct_option_id" in key_artifact


def test_ambiguous_item_with_two_defensible_answers_is_rejected() -> None:
    _bank, pack = _synthetic_bank()
    items = _run_generate_sync(pack)
    _accepted, rejected, reasons = validate_bank(items, pack)
    assert rejected, "the ambiguous item must be rejected"
    assert rejected[0].rejection_reasons == ["multiple_defensible_options"]
    assert "multiple_defensible_options" in reasons


def test_keyed_answer_without_resolving_evidence_is_rejected() -> None:
    _bank, pack = _synthetic_bank()
    item = ExamItem(
        question_id="q-x",
        category="concepts_definitions",
        difficulty="easy",
        question="Which claim holds?",
        options=[ExamOption(id=c, text=f"option {c}") for c in ("A", "B", "C", "D")],
        correct_option_id="A",
        evidence_refs=["ghost-source"],
    )
    reasons = validate_item(item, pack)
    assert "unresolved_evidence" in reasons


def test_partitions_are_deterministic_and_disjoint() -> None:
    _bank, pack = _synthetic_bank()
    items = _run_generate_sync(pack)
    accepted, _rejected, _reasons = validate_bank(items, pack)
    first = partition_bank(accepted, 6, 4, 7)
    second = partition_bank(accepted, 6, 4, 7)
    assert first == second
    assert not (set(first[0]) & set(first[1]))
    other = partition_bank(accepted, 6, 4, 99)
    assert other[0] != first[0]
    try:
        partition_bank(accepted, 100, 100, 7)
    except PartitionError as exc:
        assert exc.reason == "insufficient_validated_items"
    else:  # pragma: no cover - defensive
        raise AssertionError("expected PartitionError")


def test_invalid_and_missing_answers_score_incorrect_without_crashing() -> None:
    bank, _pack = _synthetic_bank()
    ids = bank.selection_ids[:2]
    questions = _student_questions(bank, ids)
    key: AnswerKey = build_answer_key(bank, 7)

    raw = {ids[0]: "Z"}  # invalid option; ids[1] missing
    answer_set = build_answer_set("r", "m", raw, questions, _valid_options(questions))
    score = score_answers(answer_set, bank, ids, key)
    assert score.correct == 0
    assert score.total == 2
    assert all(outcome is False for outcome in score.item_outcomes.values())

    extra = {**raw, "unknown-q": "A"}
    answer_set = build_answer_set("r", "m", extra, questions, _valid_options(questions))
    assert answer_set.answers["unknown-q"].invalid_reason == "unknown_question"


def test_terminal_selection_follows_weights_eligibility_and_ties() -> None:
    weights = SelectionWeights()
    assert abs(weights.total - 1.0) < 1e-9

    base = dict(
        process_score=0.9,
        grounding_score=0.5,
        has_evidence_bearing_dossier=True,
        provenance_integrity=True,
        selection_answered=5,
        selection_coverage=1.0,
    )
    tie_a = CandidateMemory(agent_id="a", selection_accuracy=0.5, **base)
    tie_b = CandidateMemory(agent_id="b", selection_accuracy=0.5, **base)
    winner = select_survivor([tie_b, tie_a], weights)
    assert winner.survivor_id == "a"
    assert abs(winner.terminal_score - (0.7 * 0.5 + 0.2 * 0.9 + 0.1 * 0.5)) < 1e-9

    ineligible = CandidateMemory(
        agent_id="c",
        selection_accuracy=1.0,
        process_score=1.0,
        grounding_score=1.0,
        has_evidence_bearing_dossier=False,
    )
    assert candidate_eligibility(ineligible, 0.5) == "no_evidence_bearing_dossier"
    none = select_survivor([ineligible], weights)
    assert not none.eligible


def test_naive_context_contains_only_seed_evidence() -> None:
    pack, _candidates, _acquired = build_synthetic_corpus()
    context = _naive_context(pack, 40_000)
    seed_sources = [s for s in pack.sources if s.source_distance == "seed"]
    non_seed = [s for s in pack.sources if s.source_distance != "seed"]
    assert seed_sources
    for source in seed_sources:
        assert f"[{source.source_id}]" in context
    for source in non_seed:
        assert f"[{source.source_id}]" not in context


class _RecordingAnswerClient(FakeAnswerClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, list[StudentQuestion], str]] = []

    async def answer(self, responder, questions, context):
        self.calls.append((responder, list(questions), context))
        return await super().answer(responder, questions, context)


def _run_sync(pack, candidates, acquired, client, config=None):
    import asyncio

    default = BenchmarkConfig(
        6,
        4,
        7,
        model_ids={"answer_model": client.model_id},
        prompt_versions={"examiner": "v1", "answer": "v1"},
    )
    runner = BenchmarkRunner(pack, FakeExaminer(), client, config or default)
    result = asyncio.run(runner.run(candidates, acquired))
    return runner, result


def test_arms_share_model_and_public_payload() -> None:
    pack, candidates, acquired = build_synthetic_corpus()
    client = _RecordingAnswerClient()
    runner, result = _run_sync(pack, candidates, acquired, client)
    assert result.outcome == OUTCOME_BENCHMARKED
    selection_calls = [c for c in client.calls if c[0] in {cand[0] for cand in candidates}]
    holdout_calls = {c[0]: c for c in client.calls if c[0] in ("survivor", "naive")}
    assert set(holdout_calls) == {"survivor", "naive"}
    survivor_q = [q.model_dump() for q in holdout_calls["survivor"][1]]
    naive_q = [q.model_dump() for q in holdout_calls["naive"][1]]
    assert survivor_q == naive_q
    assert runner.answer_client is client
    assert selection_calls  # selection exam also ran


def test_survivor_snapshot_matches_examined_structured_memory() -> None:
    pack, candidates, acquired = build_synthetic_corpus()
    runner, result = _run_sync(pack, candidates, acquired, FakeAnswerClient())
    bundle = runner.survivor_bundle
    assert bundle is not None
    examined = next(memory for agent_id, _p, memory in candidates if agent_id == result.survivor_id)
    assert bundle.state_hash == bundle.compute_state_hash()
    assert set(bundle.dossiers) == set(examined.dossiers)
    assert set(bundle.claims) == set(examined.claims)


def test_unavailable_arm_values_remain_unavailable() -> None:
    pack, candidates, acquired = build_synthetic_corpus()
    runner = BenchmarkRunner(
        pack,
        FakeExaminer(unavailable_reason="examiner offline"),
        FakeAnswerClient(),
        BenchmarkConfig(6, 4, 7),
    )
    import asyncio

    result = asyncio.run(runner.run(candidates, acquired))
    assert result.outcome == "failed"
    assert result.reason_code == "examiner_unavailable"
    assert result.survivor_accuracy is None
    assert result.naive_accuracy is None
    assert result.uplift is None


class _NaiveFavouringClient:
    model_id = "naive-favouring"

    async def answer(self, responder, questions, context):
        choice = "A" if responder == "naive" else "B"
        return {q.question_id: choice for q in questions}


def test_negative_uplift_is_a_valid_benchmark_result() -> None:
    pack, candidates, acquired = build_synthetic_corpus()
    _runner, result = _run_sync(pack, candidates, acquired, _NaiveFavouringClient())
    assert result.outcome == OUTCOME_BENCHMARKED
    assert result.uplift is not None and result.uplift < 0
    assert result.survivor_accuracy == 0.0
    assert result.naive_accuracy == 1.0


def test_private_keys_are_absent_from_public_artifacts_and_events() -> None:
    pack, candidates, acquired = build_synthetic_corpus()
    runner, result = _run_sync(pack, candidates, acquired, FakeAnswerClient())
    assert runner.bank is not None
    assert runner.answer_key is not None

    from research_explorer.examination.events import survivor_payloads

    public = public_exam_artifact(runner.bank)
    for forbidden in ("correct_option_id", "rationale", "evidence_refs"):
        assert forbidden not in public
    for _event_type, payload in survivor_payloads(result):
        assert_no_private_key(payload)
    assert "correct_option_id" in private_key_artifact(runner.answer_key)


def test_report_reconstructs_scores_partitions_models_and_reasons() -> None:
    pack, candidates, acquired = build_synthetic_corpus()
    runner, result = _run_sync(pack, candidates, acquired, FakeAnswerClient())
    report = benchmark_report_markdown(
        result, pack, scope=pack.scope, bank=runner.bank
    )
    assert result.survivor_id in report
    assert "Survivor accuracy:" in report
    assert "Naive accuracy:" in report
    assert "Partition seed: 7" in report
    assert "Selection items: 6" in report
    assert "Holdout items: 4" in report
    assert "fake-answer-v1" in report
    assert result.outcome in report
    json_blob = benchmark_result_json(result)
    assert result.survivor_id in json_blob
