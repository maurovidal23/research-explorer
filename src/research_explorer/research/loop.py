"""Single-agent closed-loop research kernel.

The controller coordinates only: it asks the policy for actions, delegates
evidence acquisition to an injected gateway, asks the agent for a typed brief,
applies validated mutations, evaluates, and snapshots. It never calls providers
or writes evidence itself, which keeps the loop deterministic and testable with
frozen fakes.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from research_explorer.graph.models import Paper, PaperSummary, normalize_id, parse_normalized_id
from research_explorer.graph.store import GraphStore
from research_explorer.logging_setup import get_logger
from research_explorer.providers.base import ResilientProvider, TransientProviderError
from research_explorer.providers.routing import provider_by_name
from research_explorer.redaction import redact_secrets
from research_explorer.research.agent import AgentOutputError, ResearchAgent
from research_explorer.research.answer import build_final_answer
from research_explorer.research.context import approx_tokens, build_agent_prompt
from research_explorer.research.evaluator import CompositeEvaluator
from research_explorer.research.models import (
    ActionKind,
    AgentBrief,
    CandidateAction,
    Claim,
    ClaimStatus,
    EvidenceRef,
    FinalAnswer,
    OpenQuestion,
    OpenQuestionStatus,
    PlannedAction,
    ProviderOutcome,
    ResearchEvent,
    ResearchObjective,
    ResearchState,
    SlotState,
)
from research_explorer.research.policy import ExplorationPolicy
from research_explorer.research.store import ResearchStore

log = get_logger("research.kernel")


@dataclass
class KernelOptions:
    max_actions_per_turn: int = 1
    total_slots: int = 1
    max_transient_attempts: int = 2
    plateau_turns: int = 3
    convergence_epsilon: float = 0.01
    snapshot_interval: int = 1
    context_input_target: int = 6000
    output_reserve: int = 1500
    eval_interval: int = 1
    evaluator_enabled: bool = True


@dataclass
class Acquisition:
    paper_id: str
    outcome: ProviderOutcome
    provider: str = ""
    title: str = ""
    abstract: str = ""
    evidence: EvidenceRef | None = None
    candidates: list[CandidateAction] = field(default_factory=list)


class EvidenceGateway(Protocol):
    """Fetches and integrates evidence, classifying provider outcomes."""

    async def acquire(self, paper_id: str, turn: int) -> Acquisition: ...

    def paper_exists(self, paper_id: str) -> bool: ...

    def close(self) -> None: ...


def _content_hash(paper: Paper) -> str:
    payload = f"{paper.title}\n{paper.abstract or ''}".encode()
    return hashlib.sha256(payload).hexdigest()


class GraphEvidenceGateway:
    """Provider + graph-store gateway with explicit outcome classification."""

    def __init__(
        self,
        graph: GraphStore,
        providers: dict[str, ResilientProvider],
        default_provider_name: str,
        *,
        ref_limit: int = 20,
    ) -> None:
        self.graph = graph
        self.providers = providers
        self.default_provider_name = default_provider_name
        self.ref_limit = ref_limit

    def _provider_for(self, paper_id: str) -> ResilientProvider:
        provider_name, _ = parse_normalized_id(paper_id)
        return provider_by_name(
            self.providers, provider_name, self.default_provider_name
        )

    def paper_exists(self, paper_id: str) -> bool:
        return self.graph.get_paper_summary(paper_id) is not None

    async def acquire(self, paper_id: str, turn: int) -> Acquisition:
        provider = self._provider_for(paper_id)
        _, native_id = parse_normalized_id(paper_id)
        native = native_id if native_id else paper_id
        try:
            async with provider.strict_outcomes():
                paper = await provider.get_paper(native)
        except TransientProviderError as exc:
            log.warning(
                "kernel_provider_transient",
                paper_id=paper_id,
                provider=exc.provider,
                reason=exc.reason,
            )
            return Acquisition(paper_id, ProviderOutcome.TRANSIENT, provider=exc.provider)
        except Exception as exc:
            log.warning("kernel_provider_error", paper_id=paper_id, error=str(exc))
            return Acquisition(paper_id, ProviderOutcome.TRANSIENT, provider=provider.name)

        if paper is None:
            return Acquisition(paper_id, ProviderOutcome.ABSENT, provider=provider.name)

        self.graph.cache_paper(paper)
        nid = normalize_id(paper.provider, paper.id)
        evidence = EvidenceRef(
            paper_id=nid,
            content_hash=_content_hash(paper),
            acquisition_event=turn,
        )
        return Acquisition(
            paper_id=nid,
            outcome=ProviderOutcome.SUCCESS,
            provider=provider.name,
            title=paper.title,
            abstract=paper.abstract or "",
            evidence=evidence,
            candidates=self._candidates(paper),
        )

    def _candidates(self, paper: Paper) -> list[CandidateAction]:
        nid = normalize_id(paper.provider, paper.id)
        candidates: list[CandidateAction] = []
        for summary in list(paper.references)[: self.ref_limit]:
            candidates.append(self._candidate(summary, nid, "ref"))
        for summary in list(paper.citations)[: self.ref_limit]:
            candidates.append(self._candidate(summary, nid, "cites"))
        return candidates

    @staticmethod
    def _candidate(summary: PaperSummary, source: str, mode: str) -> CandidateAction:
        pid = normalize_id(summary.provider, summary.id)
        score = 0.5
        if summary.abstract:
            score += 0.3
        if summary.citation_count:
            score += min(0.2, summary.citation_count / 5000)
        return CandidateAction(
            paper_id=pid,
            source=source,
            mode=mode,
            score=round(min(1.0, score), 6),
            reason=f"{mode}_candidate",
        )

    def close(self) -> None:
        return None


class ResearchKernel:
    """Coordinates one reproducible single-agent research loop."""

    def __init__(
        self,
        *,
        store: ResearchStore,
        gateway: EvidenceGateway,
        policy: ExplorationPolicy,
        agent: ResearchAgent,
        evaluator: CompositeEvaluator,
        options: KernelOptions | None = None,
    ) -> None:
        self.store = store
        self.gateway = gateway
        self.policy = policy
        self.agent = agent
        self.evaluator = evaluator
        self.options = options or KernelOptions()

    async def run(
        self,
        objective: ResearchObjective,
        cancel: asyncio.Event | None = None,
    ) -> FinalAnswer:
        opts = self.options
        run_id = objective.run_id
        state = self.store.create_run(objective)
        self._emit(state, "kernel_start", payload={"question": objective.question})

        pool: dict[str, CandidateAction] = {
            objective.seed_paper_id: CandidateAction(
                paper_id=objective.seed_paper_id, score=1.0, reason="seed"
            )
        }
        attempts: dict[str, int] = {}
        no_progress = 0
        terminal = "no_eligible_actions"
        slots = SlotState(total_slots=opts.total_slots)
        start = time.monotonic()
        budget = state.objective.budget

        while True:
            if cancel is not None and cancel.is_set():
                terminal = "cancelled"
                break
            budget.time_used = time.monotonic() - start
            if budget.exhausted:
                terminal = "budget_exhausted"
                break

            candidates = [c for c in pool.values() if c.paper_id not in state.visited]
            actions = await self.policy.select_actions(state, candidates, budget, slots)
            considered = getattr(self.policy, "last_considered", candidates)
            reasons = getattr(self.policy, "last_reasons", {})
            self._emit(
                state,
                "controller_decision",
                payload={
                    "considered": [c.model_dump(mode="json") for c in considered],
                    "reason_codes": dict(reasons),
                    "chosen": [a.model_dump(mode="json") for a in actions],
                },
            )
            if not actions:
                terminal = "no_eligible_actions"
                break

            action = actions[0]
            if action.kind is ActionKind.STOP or action.paper_id is None:
                terminal = "policy_stop"
                break

            self._emit(
                state,
                "turn_start",
                turn=budget.turns_used + 1,
                payload={"action": action.model_dump(mode="json")},
            )
            fetch_start = time.monotonic()
            acquisition = await self.gateway.acquire(action.paper_id, budget.turns_used + 1)
            fetch_seconds = time.monotonic() - fetch_start
            budget.fetches_used += 1

            if acquisition.outcome is ProviderOutcome.TRANSIENT:
                attempts[action.paper_id] = attempts.get(action.paper_id, 0) + 1
                self._emit(
                    state,
                    "provider_failure",
                    turn=budget.turns_used + 1,
                    fetches=1,
                    seconds=fetch_seconds,
                    payload={
                        "paper_id": action.paper_id,
                        "provider": acquisition.provider,
                        "classification": ProviderOutcome.TRANSIENT.value,
                        "attempt": attempts[action.paper_id],
                    },
                )
                budget.turns_used += 1
                no_progress += 1
                if attempts[action.paper_id] >= opts.max_transient_attempts:
                    pool.pop(action.paper_id, None)
                if no_progress >= opts.plateau_turns:
                    terminal = "converged"
                    break
                continue

            if acquisition.outcome is ProviderOutcome.ABSENT:
                self._emit(
                    state,
                    "evidence_absent",
                    turn=budget.turns_used + 1,
                    fetches=1,
                    seconds=fetch_seconds,
                    payload={
                        "paper_id": action.paper_id,
                        "provider": acquisition.provider,
                        "classification": ProviderOutcome.ABSENT.value,
                    },
                )
                pool.pop(action.paper_id, None)
                budget.turns_used += 1
                continue

            # Successful evidence acquisition.
            if acquisition.evidence is not None:
                state.evidence.append(acquisition.evidence)
            if acquisition.paper_id not in state.visited:
                state.visited.append(acquisition.paper_id)
            pool.pop(action.paper_id, None)
            for candidate in acquisition.candidates:
                pool.setdefault(candidate.paper_id, candidate)
            self._emit(
                state,
                "evidence_acquired",
                turn=budget.turns_used + 1,
                fetches=1,
                seconds=fetch_seconds,
                output_refs=[acquisition.paper_id],
                payload={
                    "paper_id": acquisition.paper_id,
                    "provider": acquisition.provider,
                    "classification": ProviderOutcome.SUCCESS.value,
                },
            )

            prompt = build_agent_prompt(
                objective,
                state,
                acquisition.paper_id,
                acquisition.title,
                input_target=opts.context_input_target,
                output_reserve=opts.output_reserve,
            )
            self._emit(
                state,
                "context_assembled",
                turn=budget.turns_used + 1,
                payload=prompt.as_event_payload(),
            )
            try:
                brief = await self.agent.propose_brief(objective, state, prompt)
            except AgentOutputError as exc:
                error = redact_secrets(str(exc))
                log.warning("kernel_agent_failed", error=error)
                self._emit(
                    state,
                    "agent_turn_failed",
                    turn=budget.turns_used + 1,
                    payload={"error": error},
                )
                budget.turns_used += 1
                no_progress += 1
                if no_progress >= opts.plateau_turns:
                    terminal = "converged"
                    break
                continue

            budget.tokens_used += prompt.approx_input_tokens + approx_tokens(
                brief.model_dump_json()
            )
            self._apply_brief(state, brief, budget.turns_used + 1)
            budget.turns_used += 1

            should_evaluate = (
                opts.evaluator_enabled
                and opts.eval_interval > 0
                and budget.turns_used % opts.eval_interval == 0
            )
            if should_evaluate:
                eval_start = time.monotonic()
                evaluation = await self.evaluator.evaluate(objective, state)
                eval_seconds = time.monotonic() - eval_start
                if evaluation.rubric is not None and evaluation.rubric.raw is not None:
                    evaluation.raw_artifact_ref = self.store.save_artifact(
                        run_id,
                        "evaluation_raw.json",
                        "evaluation_raw",
                        evaluation.rubric.raw,
                    )
                state.latest_evaluation = evaluation
                await self.policy.observe([action], [evaluation])
                eval_event = self._emit(
                    state,
                    "evaluation_complete",
                    turn=budget.turns_used,
                    tokens=budget.tokens_used,
                    seconds=eval_seconds,
                    payload={
                        "overall": round(evaluation.overall, 6),
                        "delta_quality": round(evaluation.delta_quality, 6),
                        "dimensions": evaluation.dimension_scores,
                        "integrity_passed": evaluation.integrity.passed,
                        "rubric_ok": evaluation.rubric.ok if evaluation.rubric else None,
                        "budget_before": {
                            "fetches": budget.fetches_used - 1,
                            "turns": budget.turns_used - 1,
                        },
                        "budget_after": {
                            "fetches": budget.fetches_used,
                            "turns": budget.turns_used,
                        },
                    },
                )
                if (
                    opts.snapshot_interval > 0
                    and budget.turns_used % opts.snapshot_interval == 0
                ):
                    self.store.snapshot(run_id, eval_event.seq, state)
                if evaluation.delta_quality <= opts.convergence_epsilon:
                    no_progress += 1
                else:
                    no_progress = 0
                if no_progress >= opts.plateau_turns:
                    terminal = "converged"
                    break

            if budget.exhausted:
                terminal = "budget_exhausted"
                break

        answer = self._finish(state, terminal)
        return answer

    # ---- Mutation application -------------------------------------------

    def _apply_brief(self, state: ResearchState, brief: AgentBrief, turn: int) -> int:
        valid = self._valid_evidence(brief.evidence, state)
        for ref in valid:
            if all(existing.paper_id != ref.paper_id for existing in state.evidence):
                state.evidence.append(ref)

        for mutation in brief.claim_mutations:
            if mutation.op == "propose":
                self._propose_claim(state, mutation, turn)
            elif mutation.op in ("support", "dispute", "reject", "supersede"):
                self._mutate_claim(state, mutation, turn)
            elif mutation.op == "open_question":
                self._open_question(state, mutation)
            elif mutation.op == "answer_question":
                self._answer_question(state, mutation)

        notebook = state.notebook
        if brief.changed_understanding:
            notebook.thesis = brief.changed_understanding
        notebook.accepted_claims = sorted(
            c.id for c in state.claims.values() if c.status is ClaimStatus.SUPPORTED
        )
        notebook.disputed_claims = sorted(
            c.id for c in state.claims.values() if c.status is ClaimStatus.DISPUTED
        )
        notebook.open_questions = sorted(
            q.id
            for q in state.questions.values()
            if q.status in (OpenQuestionStatus.OPEN, OpenQuestionStatus.INVESTIGATING)
        )
        notebook.planned_actions = [
            PlannedAction(action=item, reason="agent_proposed")
            for item in brief.proposed_next_actions
        ]
        if brief.action_summary:
            notebook.history.append(brief.action_summary)
        notebook.bump()

        event = self._emit(
            state,
            "knowledge_mutation",
            turn=turn,
            payload={
                "mutations": len(brief.claim_mutations),
                "notebook_revision": notebook.revision,
                "claims": len(state.claims),
                "questions": len(state.questions),
                "contradictions": brief.contradictions,
            },
        )
        return event.seq

    def _valid_evidence(
        self, refs: list[EvidenceRef], state: ResearchState
    ) -> list[EvidenceRef]:
        known = {ref.paper_id for ref in state.evidence} | set(state.visited)
        out: list[EvidenceRef] = []
        for ref in refs:
            if not ref.paper_id:
                continue
            if ref.paper_id in known or self.gateway.paper_exists(ref.paper_id):
                out.append(ref)
        return out

    def _next_id(self, prefix: str, existing: Mapping[str, object]) -> str:
        highest = 0
        for key in existing:
            head, _, tail = key.partition("-")
            if head == prefix and tail.isdigit():
                highest = max(highest, int(tail))
        return f"{prefix}-{highest + 1}"

    def _propose_claim(self, state: ResearchState, mutation, turn: int) -> None:
        text = mutation.text.strip() or mutation.reason.strip()
        if not text:
            return
        cid = self._next_id("claim", state.claims)
        evidence = self._valid_evidence(mutation.evidence, state)
        status = mutation.status or ClaimStatus.PROPOSED
        if status in (ClaimStatus.SUPPORTED, ClaimStatus.DISPUTED) and not evidence:
            status = ClaimStatus.PROPOSED
        state.claims[cid] = Claim(
            id=cid,
            text=text,
            status=status,
            confidence=mutation.confidence if mutation.confidence is not None else 0.5,
            supporting=evidence if status is ClaimStatus.SUPPORTED else [],
            contradicting=evidence if status is ClaimStatus.DISPUTED else [],
            creator=state.notebook.agent_id,
            created_seq=turn,
            updated_seq=turn,
        )

    def _mutate_claim(self, state: ResearchState, mutation, turn: int) -> None:
        if not mutation.claim_id:
            return
        claim = state.claims.get(mutation.claim_id)
        if claim is None:
            return
        evidence = self._valid_evidence(mutation.evidence, state)
        if mutation.op == "support":
            claim.supporting.extend(evidence)
        elif mutation.op == "dispute":
            claim.contradicting.extend(evidence)
        if mutation.text:
            claim.text = mutation.text
        if mutation.confidence is not None:
            claim.confidence = mutation.confidence
        if mutation.status is not None:
            claim.status = mutation.status
        elif mutation.op == "support":
            claim.status = ClaimStatus.SUPPORTED
        elif mutation.op == "dispute":
            claim.status = ClaimStatus.DISPUTED
        elif mutation.op == "reject":
            claim.status = ClaimStatus.REJECTED
        # A claim presented as supported/disputed without evidence is downgraded.
        if claim.status in (ClaimStatus.SUPPORTED, ClaimStatus.DISPUTED) and not claim.has_evidence:
            claim.status = ClaimStatus.PROPOSED
        claim.updated_seq = turn

    def _open_question(self, state: ResearchState, mutation) -> None:
        text = mutation.text.strip() or mutation.reason.strip()
        if not text:
            return
        qid = mutation.question_id or self._next_id("q", state.questions)
        if qid in state.questions:
            state.questions[qid].status = OpenQuestionStatus.INVESTIGATING
            return
        state.questions[qid] = OpenQuestion(
            id=qid,
            text=text,
            priority=mutation.confidence if mutation.confidence is not None else 0.5,
            related_claims=[mutation.claim_id] if mutation.claim_id else [],
        )

    def _answer_question(self, state: ResearchState, mutation) -> None:
        if not mutation.question_id:
            return
        question = state.questions.get(mutation.question_id)
        if question is None:
            return
        evidence = self._valid_evidence(mutation.evidence, state)
        if not evidence:
            question.status = OpenQuestionStatus.INVESTIGATING
            return
        question.status = OpenQuestionStatus.RESOLVED
        question.resolution_evidence.extend(evidence)

    # ---- Finalization ----------------------------------------------------

    def _finish(self, state: ResearchState, terminal: str) -> FinalAnswer:
        state.terminal_reason = terminal
        answer = build_final_answer(state.objective, state)
        state.final_answer = answer.render_markdown()
        run_id = state.objective.run_id
        artifact_id = self.store.save_artifact(
            run_id, "final_answer.md", "final_answer", state.final_answer
        )
        self._emit(
            state,
            "final_answer",
            output_refs=[artifact_id],
            payload={
                "terminal_reason": terminal,
                "citations": answer.citations,
                "supported": len(answer.supported_conclusions),
                "plausible": len(answer.plausible_interpretations),
                "unknowns": len(answer.unknowns),
            },
        )
        self._emit(
            state,
            "run_complete",
            payload={"terminal_reason": terminal},
        )
        self.store.finish_run(run_id, "completed", terminal)
        return answer

    def _emit(
        self,
        state: ResearchState,
        type: str,
        *,
        payload: dict | None = None,
        turn: int | None = None,
        wave: int | None = None,
        output_refs: list[str] | None = None,
        tokens: int | None = None,
        fetches: int | None = None,
        seconds: float | None = None,
    ) -> ResearchEvent:
        event = self.store.append_event(
            state.objective.run_id,
            type,
            state=state,
            payload=payload,
            turn=turn,
            wave=wave,
            output_refs=output_refs,
            tokens=tokens,
            fetches=fetches,
            seconds=seconds,
        )
        outcome = None
        if payload:
            outcome = payload.get("classification") or payload.get("terminal_reason")
        log.info(
            "research_event",
            run_id=state.objective.run_id,
            seq=event.seq,
            turn=turn,
            actor=event.actor,
            event_type=type,
            outcome=outcome,
            tokens=tokens,
            fetches=fetches,
            seconds=seconds,
        )
        return event
