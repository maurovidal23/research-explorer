"""Network-free benchmark orchestrator: selection exam, survivor, naive baseline.

The same answer model, prompt contract, question order, and tool restrictions
are used for both arms. The naive arm receives only seed-paper evidence. An
unavailable arm is represented as unavailable with a reason, never as zero.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from research_explorer.events.models import (
    OUTCOME_BENCHMARKED,
    OUTCOME_DEGRADED_NO_SURVIVOR,
    OUTCOME_EXAM_INSUFFICIENT,
    OUTCOME_FAILED,
    OUTCOME_SURVIVOR_UNBENCHMARKED,
    REASON_BASELINE_FAILED,
    REASON_EXAMINER_UNAVAILABLE,
    REASON_INSUFFICIENT_QUESTIONS,
    REASON_INVALID_CANDIDATE_RESPONSE,
    REASON_NO_ELIGIBLE_SURVIVOR,
    EventType,
)
from research_explorer.examination.clients import AnswerClient, build_answer_set
from research_explorer.examination.generator import (
    EvidenceInsufficientError,
    ExamGenerator,
    GenerationSpec,
    build_answer_key,
)
from research_explorer.examination.models import (
    AnswerKey,
    AnswerSet,
    BenchmarkResult,
    CandidateMemory,
    EvidencePack,
    ExamBank,
    ExamItem,
    SelectionWeights,
    StudentQuestion,
    SurvivorSelection,
)
from research_explorer.examination.partition import PartitionError, partition_bank
from research_explorer.examination.scoring import (
    paired_confidence_interval,
    paired_outcomes,
    score_answers,
)
from research_explorer.examination.selection import (
    compute_selection_accuracy,
    grounding_score,
    select_survivor,
    terminal_formula,
)
from research_explorer.examination.validation import validate_bank
from research_explorer.logging_setup import get_logger
from research_explorer.memory.ledger import enforce_claim_ledger, validate_claim
from research_explorer.memory.models import ContentKind, ResearchMemory, canonical_json
from research_explorer.memory.synthesis import synthesize_memory
from research_explorer.survivor.answer import assemble_context
from research_explorer.survivor.bundle import build_bundle
from research_explorer.survivor.models import SelectionMetadata, SurvivorBundle

log = get_logger("benchmark")

OUTCOME_DEGRADED = OUTCOME_DEGRADED_NO_SURVIVOR
NAIVE_DISTANCE = "seed"


class BenchmarkConfig:
    def __init__(
        self,
        selection_count: int = 30,
        holdout_count: int = 20,
        partition_seed: int = 0,
        weights: SelectionWeights | None = None,
        min_coverage: float = 0.5,
        options_per_item: int = 4,
        synthesis_words: int = 2000,
        config_fingerprint: str = "",
        model_ids: dict[str, str] | None = None,
        prompt_versions: dict[str, str] | None = None,
        context_max_chars: int = 40_000,
        answer_batch_size: int = 5,
        max_validation_attempts: int = 2,
    ) -> None:
        self.selection_count = selection_count
        self.holdout_count = holdout_count
        self.partition_seed = partition_seed
        self.weights = weights or SelectionWeights()
        self.min_coverage = min_coverage
        self.options_per_item = options_per_item
        self.synthesis_words = synthesis_words
        self.config_fingerprint = config_fingerprint
        self.model_ids = model_ids or {}
        self.prompt_versions = prompt_versions or {}
        self.context_max_chars = context_max_chars
        self.answer_batch_size = max(1, answer_batch_size)
        self.max_validation_attempts = max(1, max_validation_attempts)

    @property
    def total(self) -> int:
        return self.selection_count + self.holdout_count


def _student_questions(bank: ExamBank, ids: list[str]) -> list[StudentQuestion]:
    wanted = set(ids)
    return [
        StudentQuestion(
            question_id=item.question_id,
            category=item.category,
            difficulty=item.difficulty,
            question=item.question,
            options=list(item.options),
        )
        for item in bank.items
        if item.question_id in wanted
    ]


def _valid_options(questions: list[StudentQuestion]) -> dict[str, set[str]]:
    return {
        q.question_id: {option.id for option in q.options} for q in questions
    }


def _dossier_context(
    memory: ResearchMemory, source_ids: set[str], max_chars: int
) -> str:
    blocks: list[str] = []
    for paper_id in sorted(memory.dossiers):
        if paper_id not in source_ids:
            continue
        dossier = memory.dossiers[paper_id]
        if not dossier.has_evidence_credit:
            continue
        excerpt = ""
        for ref in dossier.evidence:
            if ref.resolves:
                excerpt = ref.excerpt
                break
        blocks.append(f"[{paper_id}] {dossier.title}\n{excerpt}")
    return "\n\n".join(blocks)[:max_chars]


def _naive_context(pack: EvidencePack, max_chars: int) -> str:
    seed_sources = [s for s in pack.sources if s.source_distance == NAIVE_DISTANCE]
    blocks = [f"[{s.source_id}] {s.title}\n{s.excerpt}" for s in seed_sources]
    return "\n\n".join(blocks)[:max_chars]


class BenchmarkRunner:
    """Runs the full hidden examination and matched naive baseline."""

    def __init__(
        self,
        pack: EvidencePack,
        generator: ExamGenerator,
        answer_client: AnswerClient,
        config: BenchmarkConfig | None = None,
        tracer: Any | None = None,
        fallback_generator: ExamGenerator | None = None,
    ) -> None:
        self.pack = pack
        self.generator = generator
        self.answer_client = answer_client
        self.cfg = config or BenchmarkConfig()
        self.tracer = tracer
        self.fallback_generator = fallback_generator
        self.fallback_model = getattr(fallback_generator, "model_id", "")
        self.bank: ExamBank | None = None
        self.answer_key: AnswerKey | None = None
        self.survivor_bundle: SurvivorBundle | None = None
        self.token_usage: dict[str, int] = {}
        self.latency_seconds: float = 0.0

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self.tracer is not None:
            self.tracer.emit(event_type, **payload)

    def _accumulate_usage(self) -> None:
        usage = getattr(self.answer_client, "last_usage", None)
        if not isinstance(usage, dict):
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                self.token_usage[key] = self.token_usage.get(key, 0) + value

    async def _generate_with_fallback(
        self,
        spec: GenerationSpec,
        primary_error: EvidenceInsufficientError,
        result: BenchmarkResult,
    ) -> list[ExamItem] | None:
        """Use the explicitly configured fallback examiner, never a silent one.

        Returns the generated items, or ``None`` after recording an explicit
        examiner-unavailable outcome. A fallback is only consulted when one was
        explicitly configured (EXAM-2).
        """
        fallback = self.fallback_generator
        if fallback is None:
            result.outcome = OUTCOME_FAILED
            result.reason_code = REASON_EXAMINER_UNAVAILABLE
            result.reason = str(primary_error)
            return None
        try:
            items = await fallback.generate(self.pack, spec)
        except EvidenceInsufficientError as fallback_error:
            result.outcome = OUTCOME_FAILED
            result.reason_code = REASON_EXAMINER_UNAVAILABLE
            result.reason = (
                f"{primary_error}; configured fallback "
                f"{self.fallback_model or 'unknown'} failed: {fallback_error}"
            )
            return None
        result.examiner_fallback_used = True
        return items

    async def _run_partition(
        self,
        bank: ExamBank,
        question_ids: list[str],
        contexts: dict[str, str],
        responder_model: str,
        partition: str = "selection",
    ) -> dict[str, AnswerSet]:
        questions = _student_questions(bank, question_ids)
        opts = _valid_options(questions)
        batch_size = self.cfg.answer_batch_size
        results: dict[str, AnswerSet] = {}
        for agent_id in sorted(contexts):
            raw: dict[str, str | None] = {}
            latencies: dict[str, float] = {}
            for start in range(0, len(questions), batch_size):
                batch = questions[start : start + batch_size]
                self._emit(
                    EventType.CANDIDATE_TEST_STARTED,
                    agent_id=agent_id,
                    partition=partition,
                    question_count=len(batch),
                )
                started = time.monotonic()
                try:
                    partial = await self.answer_client.answer(
                        agent_id, batch, contexts[agent_id]
                    )
                finally:
                    elapsed = time.monotonic() - started
                    for question in batch:
                        latencies[question.question_id] = elapsed
                self.latency_seconds += elapsed
                raw.update(partial)
                self._accumulate_usage()
                self._emit(
                    EventType.CANDIDATE_TEST_COMPLETED,
                    agent_id=agent_id,
                    partition=partition,
                    question_count=len(batch),
                    elapsed=round(elapsed, 6),
                )
            answer_set = build_answer_set(
                agent_id, responder_model, raw, questions, opts
            )
            for question_id, record in answer_set.answers.items():
                record.latency_seconds = latencies.get(question_id)
            results[agent_id] = answer_set
        return results

    async def run(
        self,
        candidates: list[tuple[str, float, ResearchMemory]],
        memory_acquired: dict[str, dict[str, ContentKind]] | None = None,
    ) -> BenchmarkResult:
        acquired = memory_acquired or {}
        spec = GenerationSpec(
            selection_count=self.cfg.selection_count,
            holdout_count=self.cfg.holdout_count,
            seed=self.cfg.partition_seed,
            options_per_item=self.cfg.options_per_item,
            max_validation_attempts=self.cfg.max_validation_attempts,
        )
        result = BenchmarkResult(config_fingerprint=self.cfg.config_fingerprint)
        result.model_ids = dict(self.cfg.model_ids)
        result.outcome = OUTCOME_DEGRADED
        result.examiner_fallback_model = self.fallback_model
        if self.fallback_model:
            result.model_ids.setdefault("examiner_fallback_model", self.fallback_model)

        # 1. Generate and independently validate the bank.
        raw_items: list[ExamItem] | None
        try:
            raw_items = await self.generator.generate(self.pack, spec)
        except EvidenceInsufficientError as exc:
            raw_items = await self._generate_with_fallback(spec, exc, result)
            if raw_items is None:
                return result
        accepted, _rejected, _reasons = validate_bank(raw_items, self.pack)
        if len(accepted) < self.cfg.total:
            result.outcome = OUTCOME_EXAM_INSUFFICIENT
            result.reason_code = REASON_INSUFFICIENT_QUESTIONS
            result.reason = (
                f"accepted={len(accepted)} required={self.cfg.total}"
            )
            return result
        bank = ExamBank(
            items=accepted,
            accepted_count=len(accepted),
            rejected_count=len(raw_items) - len(accepted),
            rejection_reasons=list(_reasons),
            partition_seed=self.cfg.partition_seed,
        )
        try:
            selection_ids, holdout_ids = partition_bank(
                accepted, self.cfg.selection_count, self.cfg.holdout_count, self.cfg.partition_seed
            )
        except PartitionError as exc:
            result.outcome = OUTCOME_EXAM_INSUFFICIENT
            result.reason_code = REASON_INSUFFICIENT_QUESTIONS
            result.reason = exc.reason
            return result
        bank.selection_ids = selection_ids
        bank.holdout_ids = holdout_ids
        bank.partition_frozen = True
        result.selection_count = len(selection_ids)
        result.holdout_count = len(holdout_ids)
        key: AnswerKey = build_answer_key(bank, self.cfg.partition_seed)
        self.bank = bank
        self.answer_key = key

        # 2. Selection exam (public payload only).
        candidates_by_id: dict[str, CandidateMemory] = {}
        selection_contexts: dict[str, str] = {}
        for agent_id, process_score, memory in candidates:
            memory_ctx = _dossier_context(
                memory, {s.source_id for s in self.pack.sources}, self.cfg.context_max_chars
            )
            selection_contexts[agent_id] = memory_ctx
            acquired_index = acquired.get(agent_id, {})
            grounded = grounding_score(memory, acquired_index)
            corrected_claims, _ = enforce_claim_ledger(memory.claims, acquired_index)
            provenance_ok = all(
                not validate_claim(claim, acquired_index)
                for claim in corrected_claims.values()
            )
            candidates_by_id[agent_id] = CandidateMemory(
                agent_id=agent_id,
                process_score=process_score,
                grounding_score=grounded,
                evidence_ids=sorted(memory.evidence_index()),
                has_evidence_bearing_dossier=bool(memory.evidence_bearing_papers()),
                provenance_integrity=provenance_ok,
            )
        responder_model = self.cfg.model_ids.get("answer_model", "fake-answer-v1")
        try:
            selection_sets = await self._run_partition(
                bank, selection_ids, selection_contexts, responder_model, "selection"
            )
        except Exception as exc:
            result.outcome = OUTCOME_DEGRADED
            result.reason_code = REASON_INVALID_CANDIDATE_RESPONSE
            result.reason = str(exc)
            return result

        selection_questions = _student_questions(bank, selection_ids)
        for agent_id, candidate in candidates_by_id.items():
            answer_set = selection_sets.get(agent_id)
            if answer_set is None:
                continue
            score = score_answers(answer_set, bank, selection_ids, key, self.pack)
            candidate.selection_accuracy, _ = compute_selection_accuracy(
                score.correct, len(selection_questions)
            )
            valid_answers = sum(
                1
                for rec in answer_set.answers.values()
                if rec.valid and bool(rec.option_id)
            )
            candidate.selection_answered = valid_answers
            total = len(selection_questions)
            candidate.selection_coverage = (valid_answers / total) if total else 0.0

        selection: SurvivorSelection = select_survivor(
            list(candidates_by_id.values()), self.cfg.weights, self.cfg.min_coverage
        )
        result.selection = selection
        if not selection.eligible:
            result.outcome = OUTCOME_DEGRADED
            result.reason_code = REASON_NO_ELIGIBLE_SURVIVOR
            result.reason = selection.ineligible_reason
            return result
        result.survivor_id = selection.survivor_id

        survivor_memory = next(
            memory for agent_id, _process, memory in candidates if agent_id == selection.survivor_id
        )
        metadata = SelectionMetadata(
            terminal_score=selection.terminal_score,
            selection_accuracy=selection.selection_accuracy,
            process_score=selection.process_score,
            grounding_score=selection.grounding_score,
            weights=self.cfg.weights,
            formula=terminal_formula(self.cfg.weights),
            candidate_ranking=selection.ranking,
            process_peak_agent=selection.process_peak_agent,
        )
        bundle = build_bundle(
            agent_id=selection.survivor_id,
            memory=survivor_memory,
            synthesis=survivor_memory_synthesis(survivor_memory, self.cfg.synthesis_words),
            pack=self.pack,
            selection=metadata,
            config_fingerprint=self.cfg.config_fingerprint,
            model_ids=self.cfg.model_ids,
            prompt_versions=self.cfg.prompt_versions,
        )
        self.survivor_bundle = bundle
        result.state_hash = bundle.state_hash

        # 3. Hidden holdout benchmark: survivor vs matched naive baseline.
        survivor_ctx = assemble_context(
            bundle, "holdout examination", max_chars=self.cfg.context_max_chars
        )
        naive_ctx = _naive_context(self.pack, self.cfg.context_max_chars)
        try:
            holdout_sets = await self._run_partition(
                bank,
                holdout_ids,
                {"survivor": survivor_ctx, "naive": naive_ctx},
                responder_model,
                "holdout",
            )
        except Exception as exc:
            result.outcome = OUTCOME_SURVIVOR_UNBENCHMARKED
            result.reason_code = REASON_BASELINE_FAILED
            result.reason = str(exc)
            return result

        survivor_score = score_answers(
            holdout_sets["survivor"], bank, holdout_ids, key, self.pack
        )
        naive_score = score_answers(
            holdout_sets["naive"], bank, holdout_ids, key, self.pack
        )
        result.survivor_score = survivor_score
        result.naive_score = naive_score
        result.survivor_accuracy = survivor_score.accuracy
        result.naive_accuracy = naive_score.accuracy
        if survivor_score.accuracy is not None and naive_score.accuracy is not None:
            result.uplift = round(survivor_score.accuracy - naive_score.accuracy, 6)
        result.paired_outcomes = paired_outcomes(survivor_score, naive_score)
        interval = paired_confidence_interval(result.paired_outcomes)
        if interval is not None:
            result.uplift_ci_low, result.uplift_ci_high = interval
        result.token_usage = dict(self.token_usage)
        result.latency_seconds = round(self.latency_seconds, 6)
        result.outcome = OUTCOME_BENCHMARKED
        result.reason_code = ""
        result.reason = ""
        return result


def survivor_memory_synthesis(memory: ResearchMemory, max_words: int) -> str:
    return synthesize_memory(memory, max_words)


def config_fingerprint(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]
