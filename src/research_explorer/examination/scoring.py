"""Exam scoring with category/difficulty/kind/distance breakdowns (EXAM-8)."""

from __future__ import annotations

from research_explorer.examination.models import (
    AnswerKey,
    AnswerSet,
    CategoryScore,
    EvidencePack,
    ExamBank,
    ExamScore,
)


def _bump(mapping: dict[str, CategoryScore], label: str, correct: bool) -> None:
    bucket = mapping.setdefault(label, CategoryScore())
    bucket.total += 1
    if correct:
        bucket.correct += 1


def score_answers(
    answer_set: AnswerSet,
    bank: ExamBank,
    question_ids: list[str],
    key: AnswerKey,
    pack: EvidencePack | None = None,
) -> ExamScore:
    score = ExamScore()
    distance_by_source = (
        {entry.source_id: entry.source_distance for entry in pack.sources}
        if pack is not None
        else {}
    )
    for question_id in sorted(question_ids):
        item = bank.by_id(question_id)
        if item is None:
            continue
        key_entry = key.entries.get(question_id, {})
        correct_option = key_entry.get("correct_option_id")
        record = answer_set.answers.get(question_id)
        is_correct = bool(
            record is not None
            and record.valid
            and record.option_id is not None
            and record.option_id == correct_option
        )
        score.item_outcomes[question_id] = is_correct
        score.total += 1
        if is_correct:
            score.correct += 1
        _bump(score.by_category, item.category, is_correct)
        _bump(score.by_difficulty, item.difficulty, is_correct)
        distance = item.source_distance
        if item.evidence_refs and item.evidence_refs[0] in distance_by_source:
            distance = distance_by_source[item.evidence_refs[0]]
        _bump(score.by_source_distance, distance, is_correct)
        if pack is not None:
            for ref in item.evidence_refs:
                entry = pack.source(ref)
                if entry is not None:
                    _bump(score.by_content_kind, entry.content_kind.value, is_correct)
                    break
    return score


def paired_outcomes(
    survivor: ExamScore, naive: ExamScore
) -> dict[str, dict[str, bool]]:
    paired: dict[str, dict[str, bool]] = {}
    for question_id in sorted(set(survivor.item_outcomes) | set(naive.item_outcomes)):
        paired[question_id] = {
            "survivor": bool(survivor.item_outcomes.get(question_id, False)),
            "naive": bool(naive.item_outcomes.get(question_id, False)),
        }
    return paired
