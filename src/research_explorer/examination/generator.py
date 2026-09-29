"""Exam generation: a pluggable examiner boundary plus a deterministic fake.

The examiner never sees candidate narratives, memories, paths, identities, or
process scores. It receives only the frozen evidence pack and a generation
spec. A deterministic fake examiner makes the whole benchmark runnable without
network access (CFG-1 test profile).
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from research_explorer.examination.models import (
    CATEGORIES,
    CATEGORY_CONCEPTS,
    CATEGORY_DISTRIBUTION,
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_MEDIUM,
    EXAM_SCHEMA_VERSION,
    AnswerKey,
    EvidencePack,
    ExamBank,
    ExamItem,
    ExamOption,
)
from research_explorer.logging_setup import get_logger

log = get_logger("examiner")

_QUESTION_WORDS = (
    "alignment",
    "baseline",
    "causality",
    "dataset",
    "entropy",
    "framework",
    "gradient",
    "hypothesis",
    "inference",
    "junction",
    "kernel",
    "latency",
    "manifold",
    "neuron",
    "objective",
    "protocol",
    "quantile",
    "robustness",
    "sampling",
    "topology",
    "uncertainty",
    "variance",
    "wavelength",
    "yield",
)


class EvidenceInsufficientError(RuntimeError):
    """Raised when the evidence pack cannot support the requested bank."""


class GenerationSpec:
    def __init__(
        self,
        selection_count: int = 30,
        holdout_count: int = 20,
        exam_version: str = EXAM_SCHEMA_VERSION,
        seed: int = 0,
        options_per_item: int = 4,
    ) -> None:
        self.selection_count = selection_count
        self.holdout_count = holdout_count
        self.exam_version = exam_version
        self.seed = seed
        self.options_per_item = options_per_item

    @property
    def total(self) -> int:
        return self.selection_count + self.holdout_count


@runtime_checkable
class ExamGenerator(Protocol):
    async def generate(self, pack: EvidencePack, spec: GenerationSpec) -> list[ExamItem]:
        ...


def _category_counts(total: int) -> list[str]:
    counts: dict[str, int] = {}
    assigned = 0
    for category, share in CATEGORY_DISTRIBUTION.items():
        n = int(total * share)
        counts[category] = n
        assigned += n
    remainder = total - assigned
    for category in CATEGORIES:
        if remainder <= 0:
            break
        counts[category] += 1
        remainder -= 1
    ordered: list[str] = []
    for category in CATEGORIES:
        ordered.extend([category] * counts[category])
    return ordered


class FakeExaminer:
    """Deterministic, network-free examiner used by tests and the test profile."""

    def __init__(
        self,
        *,
        ambiguous_at: int | None = 0,
        model_id: str = "fake-examiner-v1",
        unavailable_reason: str | None = None,
    ) -> None:
        self.ambiguous_at = ambiguous_at
        self.model_id = model_id
        self.unavailable_reason = unavailable_reason

    async def generate(self, pack: EvidencePack, spec: GenerationSpec) -> list[ExamItem]:
        if self.unavailable_reason is not None:
            raise EvidenceInsufficientError(self.unavailable_reason)
        if not pack.sources:
            raise EvidenceInsufficientError("evidence pack has no eligible sources")
        rng = random.Random(spec.seed)
        total = spec.total + 1
        categories = _category_counts(total)
        items: list[ExamItem] = []
        for index in range(total):
            source = pack.sources[index % len(pack.sources)]
            category = categories[index % len(categories)]
            item = self._make_item(index, source, category, rng)
            if self.ambiguous_at is not None and index == self.ambiguous_at:
                item.defensible_option_ids = ["A", "B"]
            items.append(item)
        return items

    def _make_item(self, index, source, category, rng: random.Random) -> ExamItem:
        difficulty = (DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD)[index % 3]
        correct_id = "A"
        focus = _QUESTION_WORDS[index % len(_QUESTION_WORDS)]
        option_texts = {
            "A": f"Supported claim about {source.source_id} regarding {focus}.",
            "B": f"Contradicted variant about {source.source_id} regarding {focus}.",
            "C": f"Overgeneralized variant about {source.source_id} regarding {focus}.",
            "D": f"Unrelated distractor about {source.source_id} regarding {focus}.",
        }
        options = [ExamOption(id=label, text=option_texts[label]) for label in ("A", "B", "C", "D")]
        return ExamItem(
            question_id=f"q-{index + 1:04d}",
            exam_version=EXAM_SCHEMA_VERSION,
            category=category,
            difficulty=difficulty,
            question=(
                f"Which {focus} statement about {source.source_id} is entailed by "
                f"the evidence?"
            ),
            options=options,
            correct_option_id=correct_id,
            evidence_refs=[source.source_id],
            rationale=f"Key grounded in {source.source_id}.",
            source_distance=source.source_distance,
            has_insufficient_evidence_option=False,
            generator_model=self.model_id,
        )


class LLMExaminer:
    """LLM-backed examiner (non-deterministic path; not used by automated tests)."""

    def __init__(self, llm, model_id: str, max_tokens: int = 4000) -> None:
        self.llm = llm
        self.model_id = model_id
        self.max_tokens = max_tokens

    async def generate(self, pack: EvidencePack, spec: GenerationSpec) -> list[ExamItem]:
        schema = _exam_json_schema()
        prompt = self._prompt(pack, spec)
        payload = await self.llm.chat_json(
            [
                {"role": "system", "content": _EXAMINER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=self.model_id,
            schema=schema,
            temperature=0.2,
            max_tokens=self.max_tokens,
            purpose="exam_generation",
        )
        items = _parse_items(payload, self.model_id)
        if not items:
            raise EvidenceInsufficientError("examiner returned no items")
        return items

    def _prompt(self, pack: EvidencePack, spec: GenerationSpec) -> str:
        return (
            f"Generate {spec.total} grounded multiple-choice questions from the "
            f"evidence pack below. Category targets: {CATEGORY_DISTRIBUTION}. "
            f"Exactly one defensible correct option per item; distractors must be "
            f"false or incomplete. Return JSON {{\"items\":[...]}}.\n\n"
            f"Evidence pack hash: {pack.pack_hash}\n\n{pack.bounded_context()}"
        )


_EXAMINER_SYSTEM = (
    "You are an independent scientific examiner. Use only the supplied evidence "
    "pack. Never reference candidate agents, narratives, or quality scores. Every "
    "keyed answer must be entailed by the pack. Return strict JSON only."
)


def _exam_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "category": {"type": "string"},
                        "difficulty": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "text": {"type": "string"},
                                },
                                "required": ["id", "text"],
                            },
                        },
                        "correct_option_id": {"type": "string"},
                        "evidence_refs": {"type": "array", "items": {"type": "string"}},
                        "rationale": {"type": "string"},
                        "source_distance": {"type": "string"},
                    },
                    "required": ["question", "options", "correct_option_id"],
                },
            }
        },
        "required": ["items"],
    }


def _parse_items(payload: dict, model_id: str) -> list[ExamItem]:
    items: list[ExamItem] = []
    for index, raw in enumerate(payload.get("items", [])):
        try:
            options = [
                ExamOption(id=str(o["id"]), text=str(o["text"]))
                for o in raw.get("options", [])
            ]
            items.append(
                ExamItem(
                    question_id=raw.get("question_id") or f"q-{index + 1:04d}",
                    category=str(raw.get("category", CATEGORY_CONCEPTS)),
                    difficulty=str(raw.get("difficulty", DIFFICULTY_MEDIUM)),
                    question=str(raw["question"]),
                    options=options,
                    correct_option_id=raw.get("correct_option_id"),
                    evidence_refs=[str(r) for r in raw.get("evidence_refs", [])],
                    rationale=str(raw.get("rationale", "")),
                    source_distance=str(raw.get("source_distance", "seed")),
                    generator_model=model_id,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("examiner_item_parse_failed", error=str(exc))
    return items


def build_answer_key(bank: ExamBank, partition_seed: int) -> AnswerKey:
    entries = {item.question_id: item.answer_key() for item in bank.items}
    return AnswerKey(partition_seed=partition_seed, entries=entries).freeze()
