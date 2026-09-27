"""Composite evaluator: deterministic integrity + optional LLM rubric.

The two outputs are separable and both stored. The LLM rubric receives no
controller score and no policy name. Invalid judge output produces a recorded
evaluator failure while deterministic results remain available, and it never
aborts the research run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from research_explorer.agents.llm_client import LLMClient
from research_explorer.redaction import redact_secrets
from research_explorer.research.context import RUBRIC_DIMENSIONS, build_evaluation_prompt
from research_explorer.research.models import (
    ClaimStatus,
    FinalAnswer,
    IntegrityResult,
    ResearchEvaluation,
    ResearchObjective,
    ResearchState,
    RubricResult,
)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]+")


def normalize_claim_text(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", text.casefold())).strip()


class DeterministicIntegrity:
    """Deterministic, provider-free integrity checks over the research state."""

    def __init__(self, paper_exists: Callable[[str], bool] | None = None) -> None:
        self.paper_exists = paper_exists

    def evaluate(
        self, state: ResearchState, final_answer: FinalAnswer | None = None
    ) -> IntegrityResult:
        checks: dict[str, bool] = {}
        issues: list[str] = []

        unsupported: list[str] = []
        invalid_evidence: list[str] = []
        for claim in state.claims.values():
            if claim.status in (ClaimStatus.SUPPORTED, ClaimStatus.DISPUTED) and not claim.has_evidence:
                unsupported.append(claim.id)
                issues.append(f"claim {claim.id} is {claim.status.value} without evidence")
            if not (0.0 <= claim.confidence <= 1.0):
                issues.append(f"claim {claim.id} confidence out of range")
            for ref in claim.supporting + claim.contradicting:
                if not ref.paper_id:
                    invalid_evidence.append(claim.id)
        checks["supported_claims_have_evidence"] = not unsupported
        checks["evidence_references_valid"] = not invalid_evidence

        referenced = {ref.paper_id for ref in state.evidence}
        for claim in state.claims.values():
            referenced.update(ref.paper_id for ref in claim.supporting + claim.contradicting)
        missing: list[str] = []
        if self.paper_exists is not None:
            missing = sorted(pid for pid in referenced if pid and not self.paper_exists(pid))
        checks["referenced_papers_exist"] = not missing
        if missing:
            issues.append(f"missing referenced papers: {missing}")

        checks["confidence_ranges_valid"] = all(
            0.0 <= c.confidence <= 1.0 for c in state.claims.values()
        )

        duplicates: dict[str, list[str]] = {}
        for claim in state.claims.values():
            duplicates.setdefault(normalize_claim_text(claim.text), []).append(claim.id)
        duplicate_ids = sorted(
            cid for ids in duplicates.values() if len(ids) > 1 for cid in ids
        )
        checks["no_duplicate_claims"] = not duplicate_ids
        if duplicate_ids:
            issues.append(f"duplicate claims: {duplicate_ids}")

        unresolved: list[str] = []
        if final_answer is not None:
            unresolved = sorted(
                cid for cid in final_answer.citations if cid and cid not in referenced
            )
        checks["final_answer_citations_resolve"] = not unresolved
        if unresolved:
            issues.append(f"unresolved citations: {unresolved}")

        budget = state.objective.budget
        checks["budget_accounting_consistent"] = (
            budget.fetches_used >= 0
            and budget.tokens_used >= 0
            and budget.fetches_used <= budget.max_fetches
            and budget.tokens_used <= budget.max_tokens
        )

        return IntegrityResult(
            passed=all(checks.values()),
            checks=checks,
            issues=issues,
            duplicate_claims=duplicate_ids,
            unsupported_claims=unsupported,
            unresolved_citations=unresolved,
        )


class LLMRubricEvaluator:
    """LLM rubric over relevance, coverage, depth, evidence, and calibration."""

    def __init__(
        self,
        llm: LLMClient,
        model: str = "deepseek-v4-flash",
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> None:
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def evaluate(self, objective: ResearchObjective, state: ResearchState) -> RubricResult:
        prompt = build_evaluation_prompt(objective, state)
        try:
            raw = await self.llm.chat_json(
                prompt.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, prompt.output_reserve or self.max_tokens),
                purpose="rubric_evaluation",
            )
        except Exception as exc:
            error = redact_secrets(str(exc))
            raw_text = json.dumps({"error": error}, sort_keys=True)
            return RubricResult(ok=False, error=error, raw=raw_text)
        raw_text = redact_secrets(json.dumps(raw, sort_keys=True, default=str))
        if not isinstance(raw, dict):
            return RubricResult(
                ok=False, error="evaluation output was not an object", raw=raw_text
            )
        raw_scores = raw.get("dimension_scores")
        if not isinstance(raw_scores, dict):
            return RubricResult(
                ok=False, error="missing dimension_scores object", raw=raw_text
            )
        scores: dict[str, float] = {}
        for dimension in RUBRIC_DIMENSIONS:
            value = raw_scores.get(dimension)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            scores[dimension] = max(0.0, min(1.0, float(value)))
        if not scores:
            return RubricResult(ok=False, error="no valid dimension scores", raw=raw_text)

        def _str_list(value: object) -> list[str]:
            if not isinstance(value, list):
                return []
            return [item for item in value if isinstance(item, str)]

        return RubricResult(
            ok=True,
            dimension_scores=scores,
            missing_knowledge=_str_list(raw.get("missing_knowledge")),
            unsupported_claims=_str_list(raw.get("unsupported_claims")),
            contradictions=_str_list(raw.get("contradictions")),
            recommended_questions=_str_list(raw.get("recommended_questions")),
            raw=raw_text,
        )


class CompositeEvaluator:
    """Combine integrity and rubric outputs into a stored evaluation."""

    def __init__(
        self,
        integrity: DeterministicIntegrity,
        rubric: LLMRubricEvaluator | None,
        weights: dict[str, float],
        rubric_version: str = "v1",
    ) -> None:
        self.integrity = integrity
        self.rubric = rubric
        self.weights = weights
        self.rubric_version = rubric_version

    def _integrity_score(self, result: IntegrityResult) -> float:
        if not result.checks:
            return 1.0
        return sum(1 for value in result.checks.values() if value) / len(result.checks)

    async def evaluate(
        self,
        objective: ResearchObjective,
        state: ResearchState,
        final_answer: FinalAnswer | None = None,
    ) -> ResearchEvaluation:
        integrity = self.integrity.evaluate(state, final_answer)
        rubric: RubricResult | None = None
        if self.rubric is not None:
            rubric = await self.rubric.evaluate(objective, state)

        dimension_scores: dict[str, float] = {
            "integrity": self._integrity_score(integrity)
        }
        if rubric is not None and rubric.ok:
            dimension_scores.update(rubric.dimension_scores)

        total_w = 0.0
        accum = 0.0
        for dimension, score in dimension_scores.items():
            weight = float(self.weights.get(dimension, 1.0))
            total_w += weight
            accum += weight * score
        overall = accum / total_w if total_w > 0 else 0.0

        previous_overall = 0.0
        previous = state.latest_evaluation
        if previous is not None and previous.rubric_version == self.rubric_version:
            previous_overall = previous.overall

        return ResearchEvaluation(
            rubric_version=self.rubric_version,
            dimension_scores=dimension_scores,
            overall=max(0.0, min(1.0, overall)),
            previous_overall=previous_overall,
            delta_quality=overall - previous_overall,
            missing_knowledge=(
                list(rubric.missing_knowledge) if rubric is not None and rubric.ok else []
            ),
            unsupported_claims=(
                list(rubric.unsupported_claims)
                if rubric is not None and rubric.ok and rubric.unsupported_claims
                else integrity.unsupported_claims
            ),
            contradictions=(
                list(rubric.contradictions)
                if rubric is not None and rubric.ok
                else []
            ),
            recommended_questions=(
                list(rubric.recommended_questions)
                if rubric is not None and rubric.ok
                else []
            ),
            evaluator_id="composite",
            evaluator_version="v1",
            integrity=integrity,
            rubric=rubric,
        )
