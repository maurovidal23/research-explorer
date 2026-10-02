from __future__ import annotations

import hashlib
import json
from collections import Counter

from pydantic import BaseModel

from research_explorer.fixed_benchmark.models import BenchmarkManifest, BenchmarkQuestion


class ValidationIssue(BaseModel):
    code: str
    detail: str
    question_id: str = ""


def canonical_bank_sha256(questions: list[BenchmarkQuestion]) -> str:
    payload = "\n".join(
        question.model_dump_json(exclude_none=True)
        for question in sorted(questions, key=lambda item: item.question_id)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _distribution_issues(
    paper_id: str,
    label: str,
    actual: Counter[str],
    expected: dict[str, int],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if dict(actual) != expected:
        issues.append(
            ValidationIssue(
                code=f"invalid_{label}_distribution",
                detail=f"{paper_id}: expected={json.dumps(expected, sort_keys=True)} "
                f"actual={json.dumps(dict(actual), sort_keys=True)}",
            )
        )
    return issues


def validate_suite(
    manifest: BenchmarkManifest, questions: list[BenchmarkQuestion]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    paper_ids = {paper.paper_id for paper in manifest.papers}
    seen: set[str] = set()
    grouped: dict[str, list[BenchmarkQuestion]] = {paper_id: [] for paper_id in paper_ids}
    for question in questions:
        if question.question_id in seen:
            issues.append(
                ValidationIssue(
                    code="duplicate_question_id",
                    detail=question.question_id,
                    question_id=question.question_id,
                )
            )
        seen.add(question.question_id)
        if question.paper_id not in paper_ids:
            issues.append(
                ValidationIssue(
                    code="unknown_paper",
                    detail=question.paper_id,
                    question_id=question.question_id,
                )
            )
            continue
        grouped[question.paper_id].append(question)
        option_ids = [option.option_id for option in question.options]
        expected_ids = [chr(ord("A") + index) for index in range(manifest.options_per_question)]
        if option_ids != expected_ids:
            issues.append(
                ValidationIssue(
                    code="invalid_options",
                    detail=f"expected={expected_ids} actual={option_ids}",
                    question_id=question.question_id,
                )
            )
        keys = set(question.correct_option_ids)
        if not keys or not keys.issubset(set(option_ids)):
            issues.append(
                ValidationIssue(
                    code="invalid_answer_key",
                    detail=str(question.correct_option_ids),
                    question_id=question.question_id,
                )
            )
        if any(locator.paper_id not in paper_ids for locator in question.evidence):
            issues.append(
                ValidationIssue(
                    code="unknown_evidence_paper",
                    detail="evidence locator references an unknown paper",
                    question_id=question.question_id,
                )
            )
        if not question.evidence:
            issues.append(
                ValidationIssue(
                    code="missing_evidence",
                    detail="at least one evidence locator is required",
                    question_id=question.question_id,
                )
            )
        if not question.critic_model:
            issues.append(
                ValidationIssue(
                    code="missing_critic_validation",
                    detail="question has not passed the independent critic",
                    question_id=question.question_id,
                )
            )
    for paper_id, paper_questions in sorted(grouped.items()):
        if len(paper_questions) != manifest.question_count_per_paper:
            issues.append(
                ValidationIssue(
                    code="invalid_paper_question_count",
                    detail=f"{paper_id}: expected={manifest.question_count_per_paper} "
                    f"actual={len(paper_questions)}",
                )
            )
        issues.extend(
            _distribution_issues(
                paper_id,
                "category",
                Counter(item.category for item in paper_questions),
                manifest.categories,
            )
        )
        issues.extend(
            _distribution_issues(
                paper_id,
                "difficulty",
                Counter(item.difficulty for item in paper_questions),
                manifest.difficulties,
            )
        )
        issues.extend(
            _distribution_issues(
                paper_id,
                "correct_option_count",
                Counter(str(len(item.correct_option_ids)) for item in paper_questions),
                manifest.correct_option_counts,
            )
        )
    if manifest.bank_sha256 and canonical_bank_sha256(questions) != manifest.bank_sha256:
        issues.append(
            ValidationIssue(code="bank_hash_mismatch", detail="question bank hash differs")
        )
    return issues
