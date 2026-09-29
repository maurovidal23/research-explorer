"""Exam generation: a pluggable examiner boundary plus a deterministic fake.

The examiner never sees candidate narratives, memories, paths, identities, or
process scores. It receives only the frozen evidence pack and a generation
spec. A deterministic fake examiner makes the whole benchmark runnable without
network access (CFG-1 test profile).
"""

from __future__ import annotations

import json
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
    STATUS_REJECTED,
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
        max_validation_attempts: int = 2,
    ) -> None:
        self.selection_count = selection_count
        self.holdout_count = holdout_count
        self.exam_version = exam_version
        self.seed = seed
        self.options_per_item = options_per_item
        self.max_validation_attempts = max(1, max_validation_attempts)

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

    def __init__(
        self,
        llm,
        model_id: str,
        max_tokens: int = 4000,
        *,
        temperature: float = 0.2,
        reasoning_effort: str | None = None,
        evidence_max_chars: int = 40_000,
    ) -> None:
        self.llm = llm
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.evidence_max_chars = evidence_max_chars

    async def generate(self, pack: EvidencePack, spec: GenerationSpec) -> list[ExamItem]:
        collected: list[ExamItem] = []
        seen: set[str] = set()
        for attempt in range(spec.max_validation_attempts):
            payload = await self._generate_payload(pack, spec)
            items = _parse_items(payload, self.model_id, id_prefix=f"a{attempt}")
            items = await self._critique(items, pack)
            for item in items:
                if item.question_id in seen:
                    continue
                seen.add(item.question_id)
                collected.append(item)
            if len(collected) >= spec.total:
                break
        if not collected:
            raise EvidenceInsufficientError("examiner returned no items")
        return collected

    async def _generate_payload(self, pack: EvidencePack, spec: GenerationSpec) -> dict:
        extra_body = (
            {"reasoning_effort": self.reasoning_effort}
            if self.reasoning_effort
            else None
        )
        return await self.llm.chat_json(
            [
                {"role": "system", "content": _EXAMINER_SYSTEM},
                {"role": "user", "content": self._prompt(pack, spec)},
            ],
            model=self.model_id,
            schema=_exam_json_schema(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            purpose="exam_generation",
            extra_body=extra_body,
        )

    async def _critique(self, items: list[ExamItem], pack: EvidencePack) -> list[ExamItem]:
        """Independent critic pass, isolated from the generator's rationale.

        The critic sees only public question fields plus the keyed option id and
        evidence references (never the private rationale). It contributes
        ``defensible_option_ids`` so the deterministic validator can reject
        ambiguous items, and can reject items it cannot ground.
        """
        if not items:
            return items
        public = []
        for item in items:
            payload = item.public_payload()
            payload["correct_option_id"] = item.correct_option_id
            payload["evidence_refs"] = list(item.evidence_refs)
            public.append(payload)
        extra_body = (
            {"reasoning_effort": self.reasoning_effort}
            if self.reasoning_effort
            else None
        )
        try:
            verdicts = await self.llm.chat_json(
                [
                    {"role": "system", "content": _CRITIC_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Evidence pack:\n{pack.bounded_context(self.evidence_max_chars)}\n\n"
                            f"Items to validate:\n{json.dumps(public)}"
                        ),
                    },
                ],
                model=self.model_id,
                schema=_critic_json_schema(),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                purpose="exam_validation",
                extra_body=extra_body,
            )
        except Exception as exc:  # pragma: no cover - live path defensive
            log.warning("examiner_critique_failed", error=str(exc))
            return items
        verdict_by_id = {
            str(v.get("question_id")): v
            for v in verdicts.get("items", [])
            if isinstance(v, dict)
        }
        for item in items:
            verdict = verdict_by_id.get(item.question_id)
            if verdict is None:
                continue
            defensible = [
                str(option_id) for option_id in verdict.get("defensible_option_ids", [])
            ]
            if defensible:
                item.defensible_option_ids = sorted(
                    set(item.defensible_option_ids) | set(defensible)
                )
            if verdict.get("accepted") is False:
                item.critic_status = STATUS_REJECTED
                reason = str(verdict.get("rejection_reason") or "critic_rejected")
                if reason not in item.critic_reasons:
                    item.critic_reasons.append(reason)
        return items

    def _prompt(self, pack: EvidencePack, spec: GenerationSpec) -> str:
        return (
            f"Generate {spec.total} grounded multiple-choice questions from the "
            f"evidence pack below. Category targets: {CATEGORY_DISTRIBUTION}. "
            f"Exactly one defensible correct option per item; distractors must be "
            f"false or incomplete. Return JSON {{\"items\":[...]}}.\n\n"
            f"Evidence pack hash: {pack.pack_hash}\n\n"
            f"{pack.bounded_context(self.evidence_max_chars)}"
        )


_EXAMINER_SYSTEM = (
    "You are an independent scientific examiner. Use only the supplied evidence "
    "pack. Never reference candidate agents, narratives, or quality scores. Every "
    "keyed answer must be entailed by the pack. Return strict JSON only."
)

_CRITIC_SYSTEM = (
    "You are an independent examination critic. You receive candidate "
    "multiple-choice items with their keyed option and evidence references, plus "
    "the frozen evidence pack. For each item, decide whether exactly one option "
    "is defensible from the evidence alone. Return strict JSON "
    '{"items": [{"question_id": str, "defensible_option_ids": [str], '
    '"accepted": bool, "rejection_reason": str}]}. Never read or use any '
    "candidate narrative, memory, identity, path, or process score."
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
                        "question_id": {"type": "string"},
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
                        "defensible_option_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
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


def _critic_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_id": {"type": "string"},
                        "defensible_option_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "accepted": {"type": "boolean"},
                        "rejection_reason": {"type": "string"},
                    },
                    "required": ["question_id", "defensible_option_ids", "accepted"],
                },
            }
        },
        "required": ["items"],
    }


def _parse_items(payload: dict, model_id: str, id_prefix: str = "") -> list[ExamItem]:
    items: list[ExamItem] = []
    for index, raw in enumerate(payload.get("items", [])):
        try:
            options = [
                ExamOption(id=str(o["id"]), text=str(o["text"]))
                for o in raw.get("options", [])
            ]
            prefix = f"{id_prefix}-" if id_prefix else ""
            items.append(
                ExamItem(
                    question_id=str(
                        raw.get("question_id") or f"{prefix}q-{index + 1:04d}"
                    ),
                    category=str(raw.get("category", CATEGORY_CONCEPTS)),
                    difficulty=str(raw.get("difficulty", DIFFICULTY_MEDIUM)),
                    question=str(raw["question"]),
                    options=options,
                    correct_option_id=raw.get("correct_option_id"),
                    defensible_option_ids=[
                        str(option_id)
                        for option_id in raw.get("defensible_option_ids", [])
                    ],
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
