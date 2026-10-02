from research_explorer.fixed_benchmark.models import (
    BenchmarkManifest,
    BenchmarkPaper,
    BenchmarkQuestion,
    EvidenceLocator,
    QuestionOption,
)
from research_explorer.fixed_benchmark.scoring import score_responses
from research_explorer.fixed_benchmark.validation import validate_suite


def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        benchmark_id="fixture",
        version="1.0.0",
        description="fixture",
        question_count_per_paper=1,
        options_per_question=3,
        papers=[
            BenchmarkPaper(
                paper_id="paper-1",
                arxiv_id="1234.56789",
                version=1,
                title="Paper",
                cluster="test",
                abstract_url="https://arxiv.org/abs/1234.56789v1",
                pdf_url="https://arxiv.org/pdf/1234.56789v1",
            )
        ],
        categories={"method": 1},
        difficulties={"hard": 1},
        correct_option_counts={"2": 1},
        generator_model="generator",
        critic_model="critic",
        prompt_version="v1",
        created_at="2026-10-01T00:00:00Z",
    )


def _question() -> BenchmarkQuestion:
    return BenchmarkQuestion(
        question_id="paper-1-q001",
        paper_id="paper-1",
        category="method",
        difficulty="hard",
        question="Select every supported statement.",
        options=[
            QuestionOption(option_id="A", text="Supported A"),
            QuestionOption(option_id="B", text="Distractor"),
            QuestionOption(option_id="C", text="Supported C"),
        ],
        correct_option_ids=["C", "A"],
        evidence=[EvidenceLocator(paper_id="paper-1", page=2, quote="support")],
        rationale="A and C are supported.",
        critic_model="critic",
    )


def test_valid_multiselect_suite() -> None:
    question = _question()
    assert question.correct_option_ids == ["A", "C"]
    assert validate_suite(_manifest(), [question]) == []
    assert "correct_option_ids" not in question.public_payload()


def test_exact_set_and_partial_scoring() -> None:
    question = _question()
    exact = score_responses([question], {question.question_id: ["C", "A"]})
    partial = score_responses([question], {question.question_id: ["A"]})
    overselected = score_responses([question], {question.question_id: ["A", "B", "C"]})
    assert exact.exact_accuracy == 1.0
    assert partial.exact_accuracy == 0.0
    assert partial.partial_credit == 0.5
    assert overselected.partial_credit == 2 / 3


def test_validator_rejects_distribution_and_option_errors() -> None:
    question = _question().model_copy(
        update={"options": _question().options[:2], "difficulty": "easy"}
    )
    codes = {issue.code for issue in validate_suite(_manifest(), [question])}
    assert "invalid_options" in codes
    assert "invalid_answer_key" in codes
    assert "invalid_difficulty_distribution" in codes
