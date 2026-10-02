"""Stub-LLM coverage for the live examiner/answer clients (EXAM-3/4/6)."""

from __future__ import annotations

import asyncio

from research_explorer.examination.clients import LLMAnswerClient
from research_explorer.examination.generator import (
    GenerationSpec,
    LLMExaminer,
    _exam_json_schema,
    _parse_items,
)
from research_explorer.examination.models import ExamOption, StudentQuestion
from research_explorer.examination.synthetic import build_synthetic_corpus


def _pack():
    pack, _candidates, _acquired = build_synthetic_corpus(count=3)
    return pack


def test_exam_schema_and_parse_include_defensible_option_ids() -> None:
    schema = _exam_json_schema()
    item_props = schema["properties"]["items"]["items"]["properties"]
    assert "defensible_option_ids" in item_props

    items = _parse_items(
        {
            "items": [
                {
                    "question": "Which claim holds?",
                    "options": [{"id": c, "text": f"opt {c}"} for c in "ABCD"],
                    "correct_option_id": "A",
                    "defensible_option_ids": ["A", "B"],
                    "evidence_refs": ["arxiv:syn-0"],
                }
            ]
        },
        "m",
    )
    assert items[0].defensible_option_ids == ["A", "B"]


class _StubExaminerLLM:
    def __init__(self, generation: dict, critic: dict) -> None:
        self.generation = generation
        self.critic = critic
        self.purposes: list[str] = []

    async def chat_json(self, messages, *, purpose, **kwargs) -> dict:
        self.purposes.append(purpose)
        return self.generation if purpose == "exam_generation" else self.critic


def test_llm_examiner_critic_annotates_defensible_options() -> None:
    pack = _pack()
    source = pack.sources[0].source_id
    generation = {
        "items": [
            {
                "question_id": "q-1",
                "question": "Which claim holds?",
                "category": "concepts_definitions",
                "difficulty": "easy",
                "options": [{"id": c, "text": f"opt {c}"} for c in "ABCD"],
                "correct_option_id": "A",
                "evidence_refs": [source],
            }
        ]
    }
    critic = {
        "items": [
            {
                "question_id": "a0-q-1",
                "defensible_option_ids": ["A", "B"],
                "accepted": False,
                "rejection_reason": "ambiguous",
            }
        ]
    }
    llm = _StubExaminerLLM(generation, critic)
    examiner = LLMExaminer(llm, "m", 1000)
    items = asyncio.run(
        examiner.generate(pack, GenerationSpec(1, 0, seed=1, max_validation_attempts=1))
    )
    assert items[0].defensible_option_ids == ["A", "B"]
    assert items[0].critic_status == "rejected"
    assert llm.purposes == ["exam_generation", "exam_validation"]


class _RetryExaminerLLM:
    def __init__(self, source: str) -> None:
        self.source = source
        self.generation_calls = 0

    async def chat_json(self, messages, *, purpose, **kwargs) -> dict:
        if purpose == "exam_validation":
            return {
                "items": [
                    {
                        "question_id": f"a{attempt}-q-1",
                        "defensible_option_ids": ["A"],
                        "accepted": True,
                    }
                    for attempt in range(2)
                ]
            }
        self.generation_calls += 1
        leaked = self.generation_calls == 1
        return {
            "items": [
                {
                    "question_id": "q-1",
                    "question": f"Question {self.generation_calls}?",
                    "category": "concepts_definitions",
                    "difficulty": "easy",
                    "options": [
                        {"id": "A", "text": "a much longer keyed answer" if leaked else "alpha"},
                        {"id": "B", "text": "beta"},
                        {"id": "C", "text": "gamma"},
                        {"id": "D", "text": "delta"},
                    ],
                    "correct_option_id": "A",
                    "evidence_refs": [self.source],
                }
            ]
        }


def test_llm_examiner_retries_until_enough_items_validate() -> None:
    pack = _pack()
    llm = _RetryExaminerLLM(pack.sources[0].source_id)
    examiner = LLMExaminer(llm, "m", 1000)

    items = asyncio.run(
        examiner.generate(pack, GenerationSpec(1, 0, seed=1, max_validation_attempts=2))
    )

    assert llm.generation_calls == 2
    assert [item.question_id for item in items] == ["a0-q-1", "a1-q-1"]


class _StubAnswerLLM:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    async def chat_json(self, messages, **kwargs) -> dict:
        self.kwargs = kwargs
        return {"answers": {"q-1": "A"}}


def test_llm_answer_client_returns_option_mapping() -> None:
    llm = _StubAnswerLLM()
    client = LLMAnswerClient(llm, "answer-model", temperature=0.0, max_tokens=100)
    questions = [
        StudentQuestion(
            question_id="q-1",
            category="concepts_definitions",
            difficulty="easy",
            question="Which claim holds?",
            options=[ExamOption(id=c, text=f"opt {c}") for c in "ABCD"],
        )
    ]
    answers = asyncio.run(client.answer("survivor", questions, "[arxiv:syn-0] body"))
    assert answers == {"q-1": "A"}
    assert llm.kwargs is not None
    assert llm.kwargs["model"] == "answer-model"
    assert llm.kwargs["temperature"] == 0.0
