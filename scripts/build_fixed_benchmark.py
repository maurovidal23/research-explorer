#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import httpx
from pypdf import PdfReader

from research_explorer.agents.opencode_client import OpenCodeLLMClient
from research_explorer.fixed_benchmark.io import load_bank, load_manifest
from research_explorer.fixed_benchmark.models import (
    BenchmarkManifest,
    BenchmarkQuestion,
    CriticVerdict,
)
from research_explorer.fixed_benchmark.validation import canonical_bank_sha256, validate_suite

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks" / "research_explorer_v1"
MANIFEST = SUITE / "manifest.json"
CACHE = ROOT / "data" / "fixed_benchmark" / "research_explorer_v1"
BANK = SUITE / "bank.private.jsonl"
PUBLIC_BANK = SUITE / "bank.public.jsonl"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(manifest: BenchmarkManifest) -> None:
    MANIFEST.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


async def acquire(manifest: BenchmarkManifest) -> None:
    source_dir = CACHE / "sources"
    text_dir = CACHE / "text"
    source_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        for paper in manifest.papers:
            pdf_path = source_dir / f"{paper.paper_id}.pdf"
            if not pdf_path.exists():
                response = await client.get(paper.pdf_url)
                response.raise_for_status()
                pdf_path.write_bytes(response.content)
            pdf_bytes = pdf_path.read_bytes()
            reader = PdfReader(pdf_path)
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(
                f"<<<PAGE {index}>>>\n{page}" for index, page in enumerate(pages, start=1)
            )
            text_path = text_dir / f"{paper.paper_id}.txt"
            text_path.write_text(text, encoding="utf-8")
            paper.source_sha256 = _sha256(pdf_bytes)
            paper.text_sha256 = _sha256(text.encode())
            paper.page_count = len(pages)
            print(f"acquired {paper.paper_id}: {len(pages)} pages")
    _write_manifest(manifest)


def _blueprint(manifest: BenchmarkManifest, paper_id: str) -> list[dict[str, Any]]:
    categories = [
        label for label, count in manifest.categories.items() for _ in range(count)
    ]
    difficulties = [
        label for label, count in manifest.difficulties.items() for _ in range(count)
    ]
    key_counts = [
        int(label)
        for label, count in manifest.correct_option_counts.items()
        for _ in range(count)
    ]
    rng = random.Random(f"{manifest.version}:{paper_id}")
    rng.shuffle(categories)
    rng.shuffle(difficulties)
    rng.shuffle(key_counts)
    return [
        {
            "question_id": f"{paper_id}-q{index + 1:03d}",
            "category": categories[index],
            "difficulty": difficulties[index],
            "correct_option_count": key_counts[index],
        }
        for index in range(manifest.question_count_per_paper)
    ]


def _question_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_id": {"type": "string"},
                        "paper_id": {"type": "string"},
                        "category": {"type": "string"},
                        "difficulty": {"type": "string"},
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "option_id": {"type": "string"},
                                    "text": {"type": "string"},
                                },
                                "required": ["option_id", "text"],
                            },
                        },
                        "correct_option_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "paper_id": {"type": "string"},
                                    "page": {"type": "integer"},
                                    "section": {"type": "string"},
                                    "quote": {"type": "string"},
                                },
                                "required": ["paper_id", "page", "section", "quote"],
                            },
                        },
                        "rationale": {"type": "string"},
                        "related_paper_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "trap_type": {"type": "string"},
                    },
                    "required": [
                        "question_id",
                        "paper_id",
                        "category",
                        "difficulty",
                        "question",
                        "options",
                        "correct_option_ids",
                        "evidence",
                        "rationale",
                        "related_paper_ids",
                        "trap_type",
                    ],
                },
            }
        },
        "required": ["questions"],
    }


def _critic_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_id": {"type": "string"},
                        "accepted": {"type": "boolean"},
                        "defensible_option_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "evidence_supported": {"type": "boolean"},
                        "distractors_valid": {"type": "boolean"},
                        "issue": {"type": "string"},
                    },
                    "required": [
                        "question_id",
                        "accepted",
                        "defensible_option_ids",
                        "evidence_supported",
                        "distractors_valid",
                        "issue",
                    ],
                },
            }
        },
        "required": ["verdicts"],
    }


SYSTEM = """You are constructing a permanent expert-level scientific benchmark. Read the entire supplied paper, including appendices, tables, captions, and limitations. Return strict JSON only. Questions must be answerable from the paper but difficult even for a strong model. Use five plausible options A-E. The test taker must select every correct option, but is not told how many are correct. Never use all/none of the above. Avoid trivia, wording clues, and answer-length clues. Prefer exact experimental conditions, causal distinctions, ablations, quantitative comparisons, limitations, cross-section synthesis, and traps based on nearby methods or reversed relationships. Each correct answer needs page-level evidence using a short verbatim quote from the supplied page. Incorrect options must be genuinely false, not merely absent. Do not rely on knowledge outside the supplied text."""

CRITIC_SYSTEM = """You are the independent final critic for a permanent scientific benchmark. Read the supplied full paper and evaluate each item from scratch. Accept an item only when the proposed correct option set is exactly the set defensible from the paper, every cited quote occurs on the stated page and supports the key, every distractor is false rather than merely unmentioned, and wording gives no clue to the answer. Reject ambiguity, outside-knowledge dependence, trivial questions, misleading evidence, or multiple reasonable interpretations. Return strict JSON only. Do not repair or rewrite items."""


def _generation_prompt(
    manifest: BenchmarkManifest,
    paper_id: str,
    title: str,
    specs: list[dict[str, Any]],
    paper_text: str,
) -> str:
    ids = {paper.paper_id for paper in manifest.papers}
    return f"""Benchmark: {manifest.benchmark_id} {manifest.version}
Paper ID: {paper_id}
Title: {title}
Known benchmark paper IDs for optional related_paper_ids: {sorted(ids)}

Produce exactly one question for every blueprint row and preserve its question_id, category, and difficulty. Each row's correct_option_count is exact. Set paper_id to {paper_id}. Options must appear in A, B, C, D, E order. Evidence may cite this paper only. A cross_paper_distinctions question must test a distinction the paper itself makes in its related-work or comparison discussion; do not assume the other paper's claims independently.

Blueprint:
{json.dumps(specs)}

Full paper text with page markers:
{paper_text}"""


def _load_existing(path: Path) -> dict[str, BenchmarkQuestion]:
    if not path.exists():
        return {}
    questions = load_bank(path)
    return {question.question_id: question for question in questions}


def _save_questions(path: Path, questions: dict[str, BenchmarkQuestion]) -> None:
    ordered = [questions[key] for key in sorted(questions)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(question.model_dump_json() + "\n" for question in ordered),
        encoding="utf-8",
    )


async def generate(
    manifest: BenchmarkManifest, batch_size: int, paper_filter: str | None
) -> None:
    client = OpenCodeLLMClient()
    work_dir = CACHE / "questions"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        for paper in manifest.papers:
            if paper_filter and paper.paper_id != paper_filter:
                continue
            text_path = CACHE / "text" / f"{paper.paper_id}.txt"
            if not text_path.exists():
                raise FileNotFoundError(f"missing extracted paper: {text_path}")
            paper_text = text_path.read_text(encoding="utf-8")
            output_path = work_dir / f"{paper.paper_id}.jsonl"
            current = _load_existing(output_path)
            pending = [
                spec
                for spec in _blueprint(manifest, paper.paper_id)
                if spec["question_id"] not in current
            ]
            for start in range(0, len(pending), batch_size):
                specs = pending[start : start + batch_size]
                payload = await client.chat_json(
                    [
                        {"role": "system", "content": SYSTEM},
                        {
                            "role": "user",
                            "content": _generation_prompt(
                                manifest, paper.paper_id, paper.title, specs, paper_text
                            ),
                        },
                    ],
                    model=manifest.generator_model,
                    schema=_question_schema(),
                    max_tokens=40_000,
                    purpose=f"fixed_benchmark:{paper.paper_id}",
                    attempts=3,
                    extra_body={"reasoning_effort": "medium"},
                )
                parsed = [
                    BenchmarkQuestion.model_validate(item).model_copy(
                        update={"generator_model": manifest.generator_model}
                    )
                    for item in payload.get("questions", [])
                ]
                expected = {spec["question_id"] for spec in specs}
                actual = {question.question_id for question in parsed}
                if actual != expected:
                    raise ValueError(
                        f"{paper.paper_id}: expected IDs {sorted(expected)}, got {sorted(actual)}"
                    )
                current.update({question.question_id: question for question in parsed})
                _save_questions(output_path, current)
                print(f"generated {paper.paper_id}: {len(current)}/100")
    finally:
        await client.aclose()


def _critic_payload(question: BenchmarkQuestion) -> dict[str, Any]:
    return {
        **question.public_payload(),
        "proposed_correct_option_ids": question.correct_option_ids,
        "evidence": [locator.model_dump() for locator in question.evidence],
    }


def _load_verdicts(path: Path) -> dict[str, CriticVerdict]:
    if not path.exists():
        return {}
    verdicts: dict[str, CriticVerdict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                verdict = CriticVerdict.model_validate_json(line)
                verdicts[verdict.question_id] = verdict
    return verdicts


def _save_verdicts(path: Path, verdicts: dict[str, CriticVerdict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(verdicts[key].model_dump_json() + "\n" for key in sorted(verdicts)),
        encoding="utf-8",
    )


async def critique(
    manifest: BenchmarkManifest, batch_size: int, paper_filter: str | None
) -> None:
    client = OpenCodeLLMClient()
    verdict_dir = CACHE / "critic"
    try:
        for paper in manifest.papers:
            if paper_filter and paper.paper_id != paper_filter:
                continue
            question_path = CACHE / "questions" / f"{paper.paper_id}.jsonl"
            if not question_path.exists():
                continue
            questions = load_bank(question_path)
            text = (CACHE / "text" / f"{paper.paper_id}.txt").read_text(encoding="utf-8")
            verdict_path = verdict_dir / f"{paper.paper_id}.jsonl"
            current = _load_verdicts(verdict_path)
            pending = [q for q in questions if q.question_id not in current]
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                payload = await client.chat_json(
                    [
                        {"role": "system", "content": CRITIC_SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                f"Paper ID: {paper.paper_id}\nTitle: {paper.title}\n\n"
                                f"Items:\n{json.dumps([_critic_payload(q) for q in batch])}"
                                f"\n\nFull paper text with page markers:\n{text}"
                            ),
                        },
                    ],
                    model=manifest.critic_model,
                    schema=_critic_schema(),
                    max_tokens=12_000,
                    purpose=f"fixed_benchmark_critic:{paper.paper_id}",
                    attempts=3,
                    extra_body={"reasoning_effort": "high"},
                )
                parsed = [
                    CriticVerdict.model_validate(item)
                    for item in payload.get("verdicts", [])
                ]
                expected = {question.question_id for question in batch}
                actual = {verdict.question_id for verdict in parsed}
                if actual != expected:
                    raise ValueError(
                        f"{paper.paper_id}: expected verdicts {sorted(expected)}, "
                        f"got {sorted(actual)}"
                    )
                current.update({verdict.question_id: verdict for verdict in parsed})
                _save_verdicts(verdict_path, current)
                print(f"criticized {paper.paper_id}: {len(current)}/{len(questions)}")
    finally:
        await client.aclose()


def prune_rejected(manifest: BenchmarkManifest, paper_filter: str | None) -> None:
    for paper in manifest.papers:
        if paper_filter and paper.paper_id != paper_filter:
            continue
        question_path = CACHE / "questions" / f"{paper.paper_id}.jsonl"
        verdict_path = CACHE / "critic" / f"{paper.paper_id}.jsonl"
        if not question_path.exists() or not verdict_path.exists():
            continue
        questions = _load_existing(question_path)
        verdicts = _load_verdicts(verdict_path)
        rejected = {
            question_id
            for question_id, verdict in verdicts.items()
            if question_id not in questions
            or not verdict.accepted
            or not verdict.evidence_supported
            or not verdict.distractors_valid
            or set(verdict.defensible_option_ids)
            != set(questions[question_id].correct_option_ids)
        }
        for question_id in rejected:
            questions.pop(question_id, None)
            verdicts.pop(question_id, None)
        _save_questions(question_path, questions)
        _save_verdicts(verdict_path, verdicts)
        print(f"pruned {paper.paper_id}: {len(rejected)} rejected slots")


def assemble(manifest: BenchmarkManifest) -> None:
    questions: dict[str, BenchmarkQuestion] = {}
    for paper in manifest.papers:
        path = CACHE / "questions" / f"{paper.paper_id}.jsonl"
        verdict_path = CACHE / "critic" / f"{paper.paper_id}.jsonl"
        verdicts = _load_verdicts(verdict_path)
        for question_id, question in _load_existing(path).items():
            verdict = verdicts.get(question_id)
            if verdict is None or not verdict.accepted:
                continue
            if set(verdict.defensible_option_ids) != set(question.correct_option_ids):
                continue
            if not verdict.evidence_supported or not verdict.distractors_valid:
                continue
            questions[question_id] = question.model_copy(
                update={"critic_model": manifest.critic_model}
            )
    _save_questions(BANK, questions)
    bank = [questions[key] for key in sorted(questions)]
    PUBLIC_BANK.write_text(
        "".join(json.dumps(question.public_payload()) + "\n" for question in bank),
        encoding="utf-8",
    )
    manifest.bank_sha256 = canonical_bank_sha256(bank)
    _write_manifest(manifest)
    print(f"assembled {len(bank)} questions")


def validate(manifest: BenchmarkManifest) -> int:
    questions = load_bank(BANK)
    issues = validate_suite(manifest, questions)
    for issue in issues:
        print(f"{issue.code}: {issue.question_id} {issue.detail}")
    print(f"validated {len(questions)} questions with {len(issues)} issues")
    return 1 if issues else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("acquire", "generate", "critique", "prune", "assemble", "validate"),
    )
    parser.add_argument("--paper")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    manifest = load_manifest(MANIFEST)
    if args.command == "acquire":
        asyncio.run(acquire(manifest))
    elif args.command == "generate":
        asyncio.run(generate(manifest, max(1, args.batch_size), args.paper))
    elif args.command == "critique":
        asyncio.run(critique(manifest, max(1, args.batch_size), args.paper))
    elif args.command == "prune":
        prune_rejected(manifest, args.paper)
    elif args.command == "assemble":
        assemble(manifest)
    else:
        raise SystemExit(validate(manifest))


if __name__ == "__main__":
    main()
