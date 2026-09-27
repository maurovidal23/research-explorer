"""Composite evaluator: deterministic integrity + optional LLM rubric.

The two outputs are separable and both stored. The LLM rubric receives no
controller score and no policy name. Invalid judge output produces a recorded
evaluator failure while deterministic results remain available, and it never
aborts the research run.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from research_explorer.agents.llm_client import LLMClient
from research_explorer.redaction import redact_secrets
from research_explorer.research.context import (
    RUBRIC_DIMENSIONS,
    approx_tokens,
    build_evaluation_prompt,
    rubric_schema,
)
from research_explorer.research.models import (
    ClaimStatus,
    ClaimVerdict,
    FinalAnswer,
    IntegrityResult,
    ResearchEvaluation,
    ResearchObjective,
    ResearchState,
    RubricResult,
    VerdictAssessment,
    normalize_claim_text,
)


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

        acquired = {ref.paper_id for ref in state.evidence if ref.paper_id}
        unresolved_claims: list[str] = []
        for claim in state.claims.values():
            if claim.status not in (ClaimStatus.SUPPORTED, ClaimStatus.DISPUTED):
                continue
            refs = claim.supporting + claim.contradicting
            if any(ref.paper_id not in acquired for ref in refs):
                unresolved_claims.append(claim.id)
        checks["claim_refs_resolve_to_acquired"] = not unresolved_claims
        if unresolved_claims:
            issues.append(
                "claims reference evidence that was never acquired: "
                f"{sorted(set(unresolved_claims))}"
            )

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
        model: str = "glm-5.3-flash",
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
                schema=rubric_schema(),
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, prompt.output_reserve or self.max_tokens),
            )
        except Exception as exc:
            error = redact_secrets(str(exc))
            raw_text = json.dumps({"error": error}, sort_keys=True)
            return RubricResult(
                ok=False,
                error=error,
                raw=raw_text,
                input_tokens=prompt.approx_input_tokens,
                output_tokens=approx_tokens(raw_text),
            )
        raw_text = redact_secrets(json.dumps(raw, sort_keys=True, default=str))
        input_tokens = prompt.approx_input_tokens
        output_tokens = approx_tokens(raw_text)
        if not isinstance(raw, dict):
            return RubricResult(
                ok=False,
                error="evaluation output was not an object",
                raw=raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        raw_scores = raw.get("dimension_scores")
        if not isinstance(raw_scores, dict):
            return RubricResult(
                ok=False,
                error="missing dimension_scores object",
                raw=raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        scores: dict[str, float] = {}
        invalid_dimensions: list[str] = []
        for dimension in RUBRIC_DIMENSIONS:
            value = raw_scores.get(dimension)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                invalid_dimensions.append(dimension)
                continue
            scores[dimension] = max(0.0, min(1.0, float(value)))
        if invalid_dimensions:
            return RubricResult(
                ok=False,
                error=f"missing or invalid dimension scores: {invalid_dimensions}",
                raw=raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        aggregate_fields = {
            "missing_knowledge": raw.get("missing_knowledge"),
            "unsupported_claims": raw.get("unsupported_claims"),
            "contradictions": raw.get("contradictions"),
            "recommended_questions": raw.get("recommended_questions"),
        }
        aggregates: dict[str, list[str]] = {}
        for field_name, value in aggregate_fields.items():
            parsed_list = _parse_str_list(value)
            if parsed_list is None:
                return RubricResult(
                    ok=False,
                    error=f"malformed {field_name} list",
                    raw=raw_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            aggregates[field_name] = parsed_list
        try:
            verdicts = parse_claim_verdicts(raw.get("claim_verdicts"), state)
        except ValueError as exc:
            return RubricResult(
                ok=False,
                error=f"malformed claim_verdicts: {exc}",
                raw=raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return RubricResult(
            ok=True,
            dimension_scores=scores,
            missing_knowledge=aggregates["missing_knowledge"],
            unsupported_claims=aggregates["unsupported_claims"],
            contradictions=aggregates["contradictions"],
            recommended_questions=aggregates["recommended_questions"],
            claim_verdicts=verdicts,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw=raw_text,
        )


def _parse_str_list(value: object) -> list[str] | None:
    """Validate an aggregate string list; return ``None`` when malformed.

    Every aggregate field is required by the rubric schema, so an absent or
    non-list-of-strings field is malformed and must become an explicit
    evaluator failure rather than a silent empty list.
    """
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return list(value)


def parse_claim_verdicts(value: object, state: ResearchState) -> list[ClaimVerdict]:
    """Validate structured verdicts for every known claim id.

    The ``claim_verdicts`` collection is required by the rubric schema, so an
    absent or malformed collection raises ``ValueError`` rather than being
    treated as an empty verdict set. Each entry must carry a claim id, a known
    assessment, and a non-empty reason; structure is validated before an
    unknown claim id is ignored, so a malformed verdict cannot escape
    governance by naming an unknown claim. Every known claim must be covered by
    a structurally valid verdict; extra structurally valid unknown ids are
    ignored.
    """
    if not isinstance(value, list):
        raise ValueError("claim_verdicts must be a list")
    verdicts: list[ClaimVerdict] = []
    covered: set[str] = set()
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("claim_verdicts entries must be objects")
        claim_id = entry.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            raise ValueError("claim verdict missing a claim_id")
        assessment = entry.get("assessment")
        if isinstance(assessment, VerdictAssessment):
            parsed = assessment
        elif isinstance(assessment, str):
            try:
                parsed = VerdictAssessment(assessment.strip().casefold())
            except ValueError as exc:
                raise ValueError(f"unknown claim assessment {assessment!r}") from exc
        else:
            raise ValueError("claim verdict missing an assessment")
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("claim verdict missing a reason")
        if claim_id in covered:
            continue
        covered.add(claim_id)
        if claim_id not in state.claims:
            continue
        verdicts.append(
            ClaimVerdict(claim_id=claim_id, assessment=parsed, reason=reason)
        )
    missing = sorted(set(state.claims) - covered)
    if missing:
        raise ValueError(f"missing claim verdicts for known claims: {missing}")
    return verdicts


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
            claim_verdicts=(
                list(rubric.claim_verdicts)
                if rubric is not None and rubric.ok
                else []
            ),
            evaluator_id="composite",
            evaluator_version="v1",
            integrity=integrity,
            rubric=rubric,
        )
