"""Test-taker clients and the exact-option answer protocol (EXAM-6).

Test takers receive only public question fields and options. They never receive
answer keys, evidence annotations, examiner rationales, validation results, or
the other partition. Chain-of-thought, confidence prose, and free-form answers
are neither requested nor scored.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from research_explorer.examination.models import AnswerRecord, AnswerSet, StudentQuestion
from research_explorer.logging_setup import get_logger

log = get_logger("answer")

_SOURCE_RE = re.compile(r"\[([^\[\]\s]+)\]")


@runtime_checkable
class AnswerClient(Protocol):
    async def answer(
        self,
        responder: str,
        questions: list[StudentQuestion],
        context: str,
    ) -> dict[str, str | None]:
        ...


class FakeAnswerClient:
    """Deterministic, network-free test taker.

    It answers correctly only for questions whose source appears in its own
    assembled context, so a richer survivor context produces higher accuracy
    than a seed-only naive context without ever seeing the answer key.
    """

    def __init__(self, correct_option: str = "A", wrong_option: str = "B") -> None:
        self.correct_option = correct_option
        self.wrong_option = wrong_option
        self.model_id = "fake-answer-v1"

    async def answer(
        self,
        responder: str,
        questions: list[StudentQuestion],
        context: str,
    ) -> dict[str, str | None]:
        available = set(_SOURCE_RE.findall(context))
        answers: dict[str, str | None] = {}
        for question in questions:
            answers[question.question_id] = (
                self.correct_option
                if self._supports(question, available)
                else self.wrong_option
            )
        return answers

    def _supports(self, question: StudentQuestion, available: set[str]) -> bool:
        haystack = " ".join(
            [question.question] + [option.text for option in question.options]
        )
        return any(source_id in haystack for source_id in available)


class LLMAnswerClient:
    """Schema-constrained LLM test taker (temperature zero, no tools)."""

    def __init__(
        self,
        llm,
        model_id: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        reasoning_effort: str | None = None,
    ) -> None:
        self.llm = llm
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.last_usage: dict | None = None

    async def answer(
        self,
        responder: str,
        questions: list[StudentQuestion],
        context: str,
    ) -> dict[str, str | None]:
        payload = {
            "questions": [q.model_dump() for q in questions],
        }
        schema = {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                }
            },
            "required": ["answers"],
        }
        extra_body = (
            {"reasoning_effort": self.reasoning_effort}
            if self.reasoning_effort
            else None
        )
        try:
            result = await self.llm.chat_json(
                [
                    {"role": "system", "content": _ANSWER_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Evidence context:\n{context}\n\nQuestions:\n{payload}",
                    },
                ],
                model=self.model_id,
                schema=schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=extra_body,
                purpose=f"exam_answer:{responder}",
            )
        except Exception as exc:
            log.warning("answer_client_failed", responder=responder, error=str(exc))
            raise
        self.last_usage = getattr(self.llm, "last_usage", None)
        answers = result.get("answers", {})
        return {str(k): (str(v) if v is not None else None) for k, v in answers.items()}


_ANSWER_SYSTEM = (
    "You are taking a multiple-choice examination. Return ONLY a JSON object "
    '{"answers": {"<question_id>": "<option_id>"}} with exactly one option id '
    "per question. Do not explain, cite, or add any other field."
)


def build_answer_set(
    responder: str,
    model_id: str,
    raw: dict[str, str | None],
    questions: list[StudentQuestion],
    valid_options: dict[str, set[str]],
) -> AnswerSet:
    """Validate a raw mapping into an :class:`AnswerSet` deterministically.

    Unknown question ids, unknown option ids, and missing answers are rejected;
    missing or invalid answers simply score as incorrect (they never crash).
    """
    answer_set = AnswerSet(responder=responder, model_id=model_id)
    known = {q.question_id for q in questions}
    for question_id in sorted(known):
        if question_id not in raw:
            answer_set.answers[question_id] = AnswerRecord(
                question_id=question_id,
                option_id=None,
                valid=False,
                invalid_reason="missing_answer",
            )
            continue
        option_id = raw.get(question_id)
        if option_id is None:
            answer_set.answers[question_id] = AnswerRecord(
                question_id=question_id,
                option_id=None,
                valid=False,
                invalid_reason="missing_answer",
            )
            continue
        if question_id not in valid_options or option_id not in valid_options[question_id]:
            answer_set.answers[question_id] = AnswerRecord(
                question_id=question_id,
                option_id=option_id,
                valid=False,
                invalid_reason="invalid_option",
            )
            continue
        answer_set.answers[question_id] = AnswerRecord(
            question_id=question_id, option_id=option_id, valid=True
        )
    for extra in sorted(set(raw) - known):
        answer_set.answers[extra] = AnswerRecord(
            question_id=extra,
            option_id=raw[extra],
            valid=False,
            invalid_reason="unknown_question",
        )
    return answer_set
