from __future__ import annotations

from pydantic import BaseModel, Field

from research_explorer.fixed_benchmark.models import BenchmarkQuestion


class BenchmarkScore(BaseModel):
    exact_correct: int = 0
    total: int = 0
    exact_accuracy: float = 0.0
    partial_credit: float = 0.0
    by_paper: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    by_category: dict[str, dict[str, float | int]] = Field(default_factory=dict)
    item_scores: dict[str, float] = Field(default_factory=dict)


def _jaccard(expected: set[str], actual: set[str]) -> float:
    union = expected | actual
    return len(expected & actual) / len(union) if union else 1.0


def _bump(
    buckets: dict[str, dict[str, float | int]], label: str, exact: bool, partial: float
) -> None:
    bucket = buckets.setdefault(label, {"exact_correct": 0, "total": 0, "partial": 0.0})
    bucket["total"] = int(bucket["total"]) + 1
    bucket["partial"] = float(bucket["partial"]) + partial
    if exact:
        bucket["exact_correct"] = int(bucket["exact_correct"]) + 1


def score_responses(
    questions: list[BenchmarkQuestion], responses: dict[str, list[str]]
) -> BenchmarkScore:
    score = BenchmarkScore(total=len(questions))
    partial_sum = 0.0
    for question in questions:
        option_ids = {option.option_id for option in question.options}
        expected = set(question.correct_option_ids)
        supplied = responses.get(question.question_id, [])
        actual = {value for value in supplied if value in option_ids}
        exact = actual == expected and len(supplied) == len(actual)
        partial = _jaccard(expected, actual)
        score.item_scores[question.question_id] = partial
        partial_sum += partial
        if exact:
            score.exact_correct += 1
        _bump(score.by_paper, question.paper_id, exact, partial)
        _bump(score.by_category, question.category, exact, partial)
    if score.total:
        score.exact_accuracy = score.exact_correct / score.total
        score.partial_credit = partial_sum / score.total
    return score
