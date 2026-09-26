"""Kernel loop stop conditions, transient retention, and failure containment."""

from __future__ import annotations

import asyncio

from research_explorer.research.agent import AgentOutputError
from research_explorer.research.evaluator import (
    CompositeEvaluator,
    DeterministicIntegrity,
    LLMRubricEvaluator,
)
from research_explorer.research.loop import (
    Acquisition,
    KernelOptions,
    ResearchKernel,
)
from research_explorer.research.models import (
    AgentBrief,
    BudgetState,
    ClaimMutation,
    ClaimStatus,
    EvidenceRef,
    IntegrityResult,
    ProviderOutcome,
    ResearchEvaluation,
    ResearchObjective,
)
from research_explorer.research.policy import GreedyPolicy
from research_explorer.research.store import ResearchStore

SEED = "openalex:seed"
SENTINEL = "SENTINEL-SECRET-LOOP"


class ScriptedGateway:
    """Outcome per paper id; everything else is definitively absent."""

    def __init__(self, outcomes: dict[str, ProviderOutcome]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    async def acquire(self, paper_id: str, turn: int) -> Acquisition:
        self.calls.append(paper_id)
        outcome = self.outcomes.get(paper_id, ProviderOutcome.ABSENT)
        if outcome is ProviderOutcome.SUCCESS:
            return Acquisition(
                paper_id=paper_id,
                outcome=outcome,
                provider="fake",
                title="title",
                evidence=EvidenceRef(paper_id=paper_id, content_hash="h"),
            )
        return Acquisition(paper_id=paper_id, outcome=outcome, provider="fake")

    def paper_exists(self, paper_id: str) -> bool:
        return True

    def close(self) -> None:
        return None


class RaisingAgent:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def propose_brief(self, objective, state, prompt) -> AgentBrief:
        raise self.error


class BriefAgent:
    def __init__(self, brief: AgentBrief) -> None:
        self.brief = brief

    async def propose_brief(self, objective, state, prompt) -> AgentBrief:
        return self.brief


class _FailingLLM:
    async def chat_json(self, messages, **kwargs):
        raise RuntimeError("judge backend down")


class ZeroEvaluator:
    async def evaluate(self, objective, state) -> ResearchEvaluation:
        return ResearchEvaluation(
            overall=0.0, delta_quality=0.0, integrity=IntegrityResult(passed=True)
        )


def _objective(run_id: str, **budget: int) -> ResearchObjective:
    values = {"max_fetches": 10, "max_turns": 10, "max_tokens": 100000}
    values.update(budget)
    return ResearchObjective(
        run_id=run_id,
        seed_paper_id=SEED,
        question="How do extracellular fields originate?",
        budget=BudgetState(**values),
    )


def _options(**overrides) -> KernelOptions:
    base = {
        "max_transient_attempts": 2,
        "plateau_turns": 50,
        "convergence_epsilon": 0.0,
        "snapshot_interval": 1,
        "evaluator_enabled": False,
    }
    base.update(overrides)
    return KernelOptions(**base)


def _integrity_evaluator() -> CompositeEvaluator:
    return CompositeEvaluator(
        integrity=DeterministicIntegrity(paper_exists=lambda pid: True),
        rubric=None,
        weights={"integrity": 1.0},
    )


def _kernel(store, gateway, agent, evaluator=None, **option_overrides) -> ResearchKernel:
    return ResearchKernel(
        store=store,
        gateway=gateway,
        policy=GreedyPolicy(seed=0),
        agent=agent,
        evaluator=evaluator or _integrity_evaluator(),
        options=_options(**option_overrides),
    )


async def test_cancel_terminates_before_any_fetch(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    gateway = ScriptedGateway({SEED: ProviderOutcome.SUCCESS})
    cancel = asyncio.Event()
    cancel.set()
    try:
        answer = await _kernel(store, gateway, BriefAgent(AgentBrief())).run(
            _objective("run-cancel"), cancel=cancel
        )
        run = store.get_run("run-cancel")
        assert run["terminal_reason"] == "cancelled"
        assert gateway.calls == []
        assert answer.question
    finally:
        store.close()


async def test_budget_exhaustion_terminates(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    gateway = ScriptedGateway({SEED: ProviderOutcome.SUCCESS})
    try:
        await _kernel(store, gateway, BriefAgent(AgentBrief())).run(
            _objective("run-budget", max_turns=1)
        )
        run = store.get_run("run-budget")
        assert run["status"] == "completed"
        assert run["terminal_reason"] == "budget_exhausted"
        state = store.reconstruct("run-budget")
        assert state.objective.budget.turns_used == 1
    finally:
        store.close()


async def test_convergence_on_quality_plateau(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    gateway = ScriptedGateway({SEED: ProviderOutcome.SUCCESS})
    try:
        answer = await _kernel(
            store,
            gateway,
            BriefAgent(AgentBrief()),
            evaluator=ZeroEvaluator(),
            plateau_turns=1,
            evaluator_enabled=True,
        ).run(_objective("run-converge"))
        assert store.get_run("run-converge")["terminal_reason"] == "converged"
        assert answer.question == "How do extracellular fields originate?"
    finally:
        store.close()


async def test_transient_failure_retains_then_releases_after_bounded_attempts(
    tmp_path,
) -> None:
    store = ResearchStore(tmp_path / "research.db")
    gateway = ScriptedGateway({SEED: ProviderOutcome.TRANSIENT})
    try:
        await _kernel(store, gateway, BriefAgent(AgentBrief())).run(
            _objective("run-transient", max_turns=10)
        )
        events = store.list_events("run-transient")
        failures = [e for e in events if e.type == "provider_failure"]
        assert [e.payload["attempt"] for e in failures] == [1, 2]
        assert all(e.payload["classification"] == "transient" for e in failures)
        assert all(e.payload["paper_id"] == SEED for e in failures)
        # A transient failure never masquerades as acquired evidence.
        assert not [e for e in events if e.type == "evidence_acquired"]
        assert store.get_run("run-transient")["terminal_reason"] == "no_eligible_actions"
        state = store.reconstruct("run-transient")
        assert state.visited == []
        assert state.objective.budget.fetches_used == 2
    finally:
        store.close()


async def test_agent_failure_is_recorded_and_does_not_abort_run(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    gateway = ScriptedGateway({SEED: ProviderOutcome.SUCCESS})
    agent = RaisingAgent(AgentOutputError(f"malformed brief token={SENTINEL}"))
    try:
        await _kernel(store, gateway, agent).run(_objective("run-agent-fail"))
        events = store.list_events("run-agent-fail")
        failed = [e for e in events if e.type == "agent_turn_failed"]
        assert failed
        assert SENTINEL not in failed[0].payload["error"]
        assert store.get_run("run-agent-fail")["status"] == "completed"
    finally:
        store.close()


async def test_invalid_evaluator_output_does_not_abort_run(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    gateway = ScriptedGateway({SEED: ProviderOutcome.SUCCESS})
    evaluator = CompositeEvaluator(
        integrity=DeterministicIntegrity(paper_exists=lambda pid: True),
        rubric=LLMRubricEvaluator(_FailingLLM()),
        weights={"integrity": 1.0},
    )
    try:
        await _kernel(
            store, gateway, BriefAgent(AgentBrief()), evaluator=evaluator,
            evaluator_enabled=True,
        ).run(_objective("run-eval-fail"))
        completed = [
            e for e in store.list_events("run-eval-fail") if e.type == "evaluation_complete"
        ]
        assert completed
        assert completed[0].payload["rubric_ok"] is False
        assert completed[0].payload["integrity_passed"] is True
        assert store.get_run("run-eval-fail")["status"] == "completed"
    finally:
        store.close()


async def test_supported_claim_without_evidence_is_downgraded(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    gateway = ScriptedGateway({SEED: ProviderOutcome.SUCCESS})
    agent = BriefAgent(
        AgentBrief(
            action_summary="overclaim",
            claim_mutations=[
                ClaimMutation(
                    op="propose",
                    text="Unsupported assertion",
                    status=ClaimStatus.SUPPORTED,
                    evidence=[],
                )
            ],
        )
    )
    try:
        answer = await _kernel(store, gateway, agent).run(_objective("run-overclaim"))
        claim = next(iter(store.reconstruct("run-overclaim").claims.values()))
        assert claim.status is ClaimStatus.PROPOSED
        assert claim.supporting == []
        assert not answer.supported_conclusions
        assert "Unsupported assertion" not in answer.render_markdown()
    finally:
        store.close()
