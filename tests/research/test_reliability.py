"""Evidence-reliability regressions for the research kernel (REL-1..REL-9).

These tests use frozen fake gateways, agents, and evaluators: no network and no
live LLM. They reproduce the observed failure where an agent promoted a broad
claim and the evaluator's finding did not govern the published answer.
"""

from __future__ import annotations

from research_explorer.logging_setup import ACORenderer
from research_explorer.research.context import build_agent_prompt, build_evaluation_prompt
from research_explorer.research.evaluator import DeterministicIntegrity
from research_explorer.research.loop import (
    Acquisition,
    KernelOptions,
    ResearchKernel,
    SearchOutcome,
)
from research_explorer.research.models import (
    AgentBrief,
    AgentNotebook,
    BudgetState,
    CandidateAction,
    Claim,
    ClaimMutation,
    ClaimStatus,
    ClaimVerdict,
    EvidenceRef,
    FinalAnswer,
    IntegrityResult,
    ProviderOutcome,
    ResearchEvaluation,
    ResearchObjective,
    ResearchState,
    RubricResult,
    VerdictAssessment,
    normalize_query,
)
from research_explorer.research.policy import GreedyPolicy
from research_explorer.research.store import ResearchStore

SEED = "openalex:seed"
R1 = "openalex:r1"
R2 = "openalex:r2"
QUESTION = "How do fields arise?"


class FakeGateway:
    """Deterministic gateway with acquired evidence and bounded search."""

    def __init__(
        self,
        corpus: dict[str, dict],
        *,
        absent: set[str] | None = None,
        transient_once: set[str] | None = None,
        search_results: dict[str, list[CandidateAction]] | None = None,
        transient_search: set[str] | None = None,
    ) -> None:
        self.corpus = corpus
        self.absent = set(absent or ())
        self.transient_once = set(transient_once or ())
        self.search_results = search_results or {}
        self.transient_search = {normalize_query(q) for q in (transient_search or ())}
        self.acquired: list[str] = []
        self.searches: list[str] = []
        self.search_attempts: dict[str, int] = {}

    async def acquire(self, paper_id: str, turn: int) -> Acquisition:
        if paper_id in self.transient_once:
            self.transient_once.discard(paper_id)
            return Acquisition(paper_id, ProviderOutcome.TRANSIENT, provider="fake")
        if paper_id in self.absent or paper_id not in self.corpus:
            return Acquisition(paper_id, ProviderOutcome.ABSENT, provider="fake")
        self.acquired.append(paper_id)
        entry = self.corpus[paper_id]
        candidates = [
            CandidateAction(
                paper_id=ref, source=paper_id, mode="ref", score=1.0
            )
            for ref in entry.get("refs", [])
        ]
        return Acquisition(
            paper_id=paper_id,
            outcome=ProviderOutcome.SUCCESS,
            provider="fake",
            title=entry.get("title", "title"),
            abstract=entry.get("abstract", ""),
            content=entry.get("abstract", ""),
            evidence=EvidenceRef(paper_id=paper_id, content_hash="h"),
            candidates=candidates,
        )

    async def search(self, query: str, limit: int) -> SearchOutcome:
        normalized = normalize_query(query)
        self.searches.append(normalized)
        self.search_attempts[normalized] = self.search_attempts.get(normalized, 0) + 1
        if normalized in self.transient_search:
            return SearchOutcome(
                query=query, outcome=ProviderOutcome.TRANSIENT, provider="fake"
            )
        candidates = self.search_results.get(normalized, [])
        outcome = (
            ProviderOutcome.SUCCESS if candidates else ProviderOutcome.ABSENT
        )
        return SearchOutcome(
            query=query, outcome=outcome, provider="fake", candidates=candidates
        )

    def paper_exists(self, paper_id: str) -> bool:
        return True

    def close(self) -> None:
        return None


class CallableAgent:
    def __init__(self, build) -> None:
        self._build = build

    async def propose_brief(self, objective, state, prompt) -> AgentBrief:
        return self._build(state, prompt)


class StubEvaluator:
    """Evaluator whose integrity is real and whose verdicts are scripted."""

    def __init__(self, integrity: DeterministicIntegrity, build) -> None:
        self.integrity = integrity
        self._build = build

    async def evaluate(self, objective, state) -> ResearchEvaluation:
        return self._build(state)


def _evaluation(
    *,
    verdicts: list[ClaimVerdict] | None = None,
    missing: list[str] | None = None,
    recommended: list[str] | None = None,
) -> ResearchEvaluation:
    verdicts = verdicts or []
    missing = missing or []
    recommended = recommended or []
    return ResearchEvaluation(
        overall=0.1,
        delta_quality=0.1,
        missing_knowledge=missing,
        recommended_questions=recommended,
        claim_verdicts=verdicts,
        integrity=IntegrityResult(passed=True),
        rubric=RubricResult(
            ok=True,
            dimension_scores={"relevance": 0.5},
            claim_verdicts=verdicts,
            missing_knowledge=missing,
            recommended_questions=recommended,
        ),
    )


def _integrity() -> DeterministicIntegrity:
    return DeterministicIntegrity(paper_exists=lambda pid: True)


def _objective(run_id: str, **budget) -> ResearchObjective:
    values = {"max_fetches": 10, "max_turns": 12, "max_tokens": 100000}
    values.update(budget)
    return ResearchObjective(
        run_id=run_id,
        seed_paper_id=SEED,
        question=QUESTION,
        budget=BudgetState(**values),
    )


def _options(**overrides) -> KernelOptions:
    base = {
        "max_transient_attempts": 2,
        "plateau_turns": 100,
        "convergence_epsilon": 0.0,
        "snapshot_interval": 1,
        "evaluator_enabled": True,
        "search_enabled": True,
        "max_search_queries": 5,
    }
    base.update(overrides)
    return KernelOptions(**base)


def _kernel(store, gateway, agent, evaluator, **option_overrides) -> ResearchKernel:
    return ResearchKernel(
        store=store,
        gateway=gateway,
        policy=GreedyPolicy(seed=0, max_search_queries=5),
        agent=agent,
        evaluator=evaluator,
        options=_options(**option_overrides),
    )


def _support_seed_claim(state, prompt) -> AgentBrief:
    return AgentBrief(
        action_summary="broad claim",
        claim_mutations=[
            ClaimMutation(
                op="propose",
                text="Extracellular fields drive computation broadly",
                status=ClaimStatus.SUPPORTED,
                evidence=[EvidenceRef(paper_id=SEED)],
            )
        ],
    )


# ---- REL-1 / REL-2 ----------------------------------------------------------


async def test_unsupported_verdict_removes_claim_from_supported(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})

    def build(state):
        verdicts = [
            ClaimVerdict(
                claim_id=c.id,
                assessment=VerdictAssessment.UNSUPPORTED,
                reason="source does not entail the claim",
            )
            for c in state.claims.values()
        ]
        return _evaluation(verdicts=verdicts)

    evaluator = StubEvaluator(_integrity(), build)
    try:
        answer = await _kernel(
            store, gateway, CallableAgent(_support_seed_claim), evaluator
        ).run(_objective("run-unsupported"))
        assert answer.supported_conclusions == []
        assert answer.evidence_sufficient is False
        assert "insufficient" in answer.render_markdown().casefold()
        state = store.reconstruct("run-unsupported")
        assert all(c.status is not ClaimStatus.SUPPORTED for c in state.claims.values())
        transitions = [
            e for e in store.list_events("run-unsupported") if e.type == "verdict_transition"
        ]
        assert transitions
        assert transitions[0].payload["to_status"] == "proposed"
    finally:
        store.close()


async def test_unknown_verdict_claim_id_cannot_mutate_state(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})
    evaluator = StubEvaluator(
        _integrity(),
        lambda state: _evaluation(
            verdicts=[
                ClaimVerdict(
                    claim_id="claim-999",
                    assessment=VerdictAssessment.UNSUPPORTED,
                    reason="unknown",
                )
            ]
        ),
    )
    try:
        await _kernel(
            store, gateway, CallableAgent(_support_seed_claim), evaluator
        ).run(_objective("run-unknown-verdict"))
        run = store.get_run("run-unknown-verdict")
        assert run["status"] == "completed"
        state = store.reconstruct("run-unknown-verdict")
        assert any(c.status is ClaimStatus.SUPPORTED for c in state.claims.values())
    finally:
        store.close()


async def test_malformed_evaluator_does_not_mutate_claim_state(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})

    def build(state):
        return ResearchEvaluation(
            overall=0.0,
            integrity=IntegrityResult(passed=True),
            rubric=RubricResult(ok=False, error="bad json"),
        )

    try:
        await _kernel(
            store, gateway, CallableAgent(_support_seed_claim), StubEvaluator(_integrity(), build)
        ).run(_objective("run-bad-eval"))
        state = store.reconstruct("run-bad-eval")
        assert any(c.status is ClaimStatus.SUPPORTED for c in state.claims.values())
    finally:
        store.close()


# ---- REL-3 --------------------------------------------------------------


async def test_graph_resident_paper_never_acquired_cannot_support_claim(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})

    def build(state, prompt) -> AgentBrief:
        return AgentBrief(
            claim_mutations=[
                ClaimMutation(
                    op="propose",
                    text="Ungrounded claim from the graph store",
                    status=ClaimStatus.SUPPORTED,
                    evidence=[EvidenceRef(paper_id=R2)],
                )
            ]
        )

    evaluator = StubEvaluator(_integrity(), lambda state: _evaluation())
    try:
        answer = await _kernel(
            store,
            gateway,
            CallableAgent(build),
            evaluator,
            evaluator_enabled=False,
        ).run(_objective("run-ungrounded"))
        state = store.reconstruct("run-ungrounded")
        claim = next(iter(state.claims.values()))
        assert claim.status is ClaimStatus.PROPOSED
        assert claim.supporting == []
        assert not any(R2 in citation for citation in answer.citations)
    finally:
        store.close()


# ---- REL-4 / REL-8 ------------------------------------------------------


async def test_sparse_run_renders_insufficient_answer_without_contradiction(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}}, absent={SEED})
    evaluator = StubEvaluator(_integrity(), lambda state: _evaluation())
    try:
        answer = await _kernel(
            store, gateway, CallableAgent(lambda s, p: AgentBrief()), evaluator
        ).run(_objective("run-sparse"))
        run = store.get_run("run-sparse")
        assert run["terminal_reason"] == "frontier_exhausted"
        assert answer.supported_conclusions == []
        assert answer.contradictions == []
        rendered = answer.render_markdown()
        assert "insufficient" in rendered.casefold()
        assert "frontier_exhausted" in rendered
        final_event = next(
            e for e in store.list_events("run-sparse") if e.type == "final_answer"
        )
        assert final_event.payload["evidence_sufficient"] is False
    finally:
        store.close()


# ---- REL-5 --------------------------------------------------------------


async def test_empty_frontier_schedules_one_search_per_normalized_query(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})
    evaluator = StubEvaluator(
        _integrity(),
        lambda state: _evaluation(
            missing=["Alpha finding", "Beta finding"],
            recommended=["  alpha   finding "],
        ),
    )
    try:
        await _kernel(
            store, gateway, CallableAgent(lambda s, p: AgentBrief()), evaluator
        ).run(_objective("run-search"))
        assert gateway.searches == [
            "alpha finding",
            "beta finding",
            normalize_query(QUESTION),
        ]
        assert len(set(gateway.searches)) == len(gateway.searches)
        state = store.reconstruct("run-search")
        assert state.search_queries == [
            "alpha finding",
            "beta finding",
            normalize_query(QUESTION),
        ]
    finally:
        store.close()


async def test_search_results_are_deduplicated_and_become_candidates(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway(
        {SEED: {"title": "Seed"}, R1: {"title": "R1"}, R2: {"title": "R2"}},
        search_results={
            "gap": [
                CandidateAction(paper_id=R1, source="gap", mode="search", score=0.9),
                CandidateAction(paper_id=R1, source="gap", mode="search", score=0.9),
                CandidateAction(paper_id=R2, source="gap", mode="search", score=0.8),
            ]
        },
    )
    evaluator = StubEvaluator(
        _integrity(), lambda state: _evaluation(missing=["gap"])
    )
    try:
        await _kernel(
            store, gateway, CallableAgent(lambda s, p: AgentBrief()), evaluator
        ).run(_objective("run-search-dedup"))
        acquired = [
            e.payload["paper_id"]
            for e in store.list_events("run-search-dedup")
            if e.type == "evidence_acquired"
        ]
        assert acquired.count(R1) == 1
        assert acquired.count(R2) == 1
        state = store.reconstruct("run-search-dedup")
        assert R1 in state.visited and R2 in state.visited
    finally:
        store.close()


async def test_transient_search_retries_then_blocks_equivalent_query(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}}, transient_search={"gap"})
    evaluator = StubEvaluator(
        _integrity(), lambda state: _evaluation(missing=["gap"])
    )
    try:
        await _kernel(
            store,
            gateway,
            CallableAgent(lambda s, p: AgentBrief()),
            evaluator,
            max_transient_attempts=2,
        ).run(_objective("run-search-transient"))
        question = normalize_query(QUESTION)
        assert gateway.searches == ["gap", "gap", question]
        assert gateway.search_attempts["gap"] == 2
        failures = [
            e
            for e in store.list_events("run-search-transient")
            if e.type == "provider_failure" and "query" in e.payload
        ]
        assert [e.payload["attempt"] for e in failures] == [1, 2]
        assert all(e.payload["classification"] == "transient" for e in failures)
        assert all(e.payload["query"] == "gap" for e in failures)
        state = store.reconstruct("run-search-transient")
        assert "gap" in state.search_queries
        assert "gap" not in state.blocked_search_queries
    finally:
        store.close()


# ---- REL-9 --------------------------------------------------------------


async def test_reconstruction_after_absent_turn_returns_final_totals(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}}, absent={SEED})
    evaluator = StubEvaluator(_integrity(), lambda state: _evaluation())
    try:
        await _kernel(
            store,
            gateway,
            CallableAgent(lambda s, p: AgentBrief()),
            evaluator,
            evaluator_enabled=False,
        ).run(_objective("run-absent-recon"))
        budget = store.reconstruct("run-absent-recon").objective.budget
        assert budget.fetches_used == 1
        assert budget.turns_used == 1
        assert budget.time_used > 0
    finally:
        store.close()


async def test_reconstruction_after_transient_turn_returns_final_totals(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway(
        {SEED: {"title": "Seed"}}, transient_once={SEED}
    )
    evaluator = StubEvaluator(_integrity(), lambda state: _evaluation())
    try:
        await _kernel(
            store,
            gateway,
            CallableAgent(lambda s, p: AgentBrief()),
            evaluator,
            evaluator_enabled=False,
            max_transient_attempts=1,
        ).run(_objective("run-transient-recon"))
        state = store.reconstruct("run-transient-recon")
        assert state.objective.budget.fetches_used == 1
        assert state.objective.budget.turns_used == 1
        assert state.objective.budget.time_used > 0
        assert store.get_run("run-transient-recon")["terminal_reason"] == "provider_failure"
    finally:
        store.close()


def test_context_selected_ids_are_unique_preserving_order() -> None:
    objective = _objective("run-context")
    state = ResearchState(
        objective=objective,
        notebook=AgentNotebook(agent_id="a"),
        evidence=[EvidenceRef(paper_id=SEED), EvidenceRef(paper_id=SEED)],
    )
    prompt = build_agent_prompt(
        objective, state, SEED, "Seed", selected_content="body"
    )
    assert len(prompt.selected_ids) == len(set(prompt.selected_ids))
    assert prompt.selected_ids.count(SEED) == 1


def test_research_event_console_fields_stay_separated() -> None:
    renderer = ACORenderer()
    line = renderer(
        None,
        "info",
        {
            "event": "research_event",
            "level": "info",
            "run_id": "run-1",
            "seq": 5,
            "turn": 2,
            "actor": "controller",
            "event_type": "evidence_acquired",
            "outcome": "success",
            "tokens": None,
            "fetches": 1,
            "seconds": 0.25,
        },
    )
    assert " | " in line
    assert "fetches=1 | seconds=0.25" in line
    assert "tokens=None" not in line


async def test_final_answer_integrity_receives_rendered_answer(tmp_path) -> None:
    seen: list[FinalAnswer] = []

    class SpyIntegrity(DeterministicIntegrity):
        def evaluate(self, state, final_answer=None):  # type: ignore[override]
            if final_answer is not None:
                seen.append(final_answer)
            return super().evaluate(state, final_answer)

    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})
    evaluator = StubEvaluator(SpyIntegrity(paper_exists=lambda pid: True), lambda state: _evaluation())
    try:
        await _kernel(
            store,
            gateway,
            CallableAgent(_support_seed_claim),
            evaluator,
            evaluator_enabled=False,
        ).run(_objective("run-final-gate"))
        assert seen
        state = store.reconstruct("run-final-gate")
        assert seen[-1].render_markdown() == state.final_answer
    finally:
        store.close()


# ---- Frozen end-to-end failure reproduction ------------------------------


def test_final_integrity_gate_downgrades_unresolved_claim(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})
    evaluator = StubEvaluator(_integrity(), lambda state: _evaluation())
    kernel = _kernel(
        store,
        gateway,
        CallableAgent(lambda s, p: AgentBrief()),
        evaluator,
        evaluator_enabled=False,
    )
    try:
        state = store.create_run(_objective("run-final-integrity"))
        state.evidence.append(EvidenceRef(paper_id=SEED, content_hash="h"))
        state.claims["claim-1"] = Claim(
            id="claim-1",
            text="Unresolved claim",
            status=ClaimStatus.SUPPORTED,
            supporting=[EvidenceRef(paper_id=R2)],
        )
        answer = kernel._finish(state, "frontier_exhausted")
        assert answer.supported_conclusions == []
        assert answer.evidence_sufficient is False
        assert state.claims["claim-1"].status is ClaimStatus.PROPOSED
        transitions = [
            e
            for e in store.list_events("run-final-integrity")
            if e.type == "verdict_transition"
            and e.payload.get("assessment") == "integrity_gate"
        ]
        assert transitions
        assert transitions[0].payload["from_status"] == "supported"
        assert transitions[0].payload["to_status"] == "proposed"
    finally:
        store.close()


async def test_frozen_run_never_publishes_rejected_claim(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway(
        {SEED: {"title": "Seed", "refs": [R1]}, R1: {"title": "Reference"}}
    )

    def build(state):
        verdicts = [
            ClaimVerdict(
                claim_id=c.id,
                assessment=VerdictAssessment.UNSUPPORTED,
                reason="single source does not entail the broad claim",
            )
            for c in state.claims.values()
        ]
        return _evaluation(verdicts=verdicts, missing=["unsupported extrapolation"])

    evaluator = StubEvaluator(_integrity(), build)
    try:
        answer = await _kernel(
            store, gateway, CallableAgent(_support_seed_claim), evaluator
        ).run(_objective("run-frozen-failure"))

        assert answer.supported_conclusions == []
        assert answer.evidence_sufficient is False
        rendered = answer.render_markdown()
        assert "insufficient" in rendered.casefold()
        assert "Extracellular fields drive computation broadly" not in rendered
        state = store.reconstruct("run-frozen-failure")
        assert all(c.status is not ClaimStatus.SUPPORTED for c in state.claims.values())
        run = store.get_run("run-frozen-failure")
        assert run["terminal_reason"] == "frontier_exhausted"
    finally:
        store.close()


# ---- REL-2 contradicted resolution --------------------------------------


async def test_contradicted_verdict_only_disputes_when_evidence_resolves(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway(
        {SEED: {"title": "Seed", "refs": [R1]}, R1: {"title": "Contradiction"}}
    )

    def build(state, prompt) -> AgentBrief:
        if prompt.selected_ids[0] == SEED:
            return _support_seed_claim(state, prompt)
        proposed = [c for c in state.claims.values() if c.status is ClaimStatus.PROPOSED]
        if prompt.selected_ids[0] == R1 and proposed:
            return AgentBrief(
                claim_mutations=[
                    ClaimMutation(
                        op="dispute",
                        claim_id=proposed[0].id,
                        evidence=[EvidenceRef(paper_id=R1)],
                    )
                ]
            )
        return AgentBrief()

    def evaluate(state):
        verdicts = [
            ClaimVerdict(
                claim_id=c.id,
                assessment=VerdictAssessment.CONTRADICTED,
                reason="source contradicts the claim",
            )
            for c in state.claims.values()
        ]
        return _evaluation(verdicts=verdicts)

    evaluator = StubEvaluator(_integrity(), evaluate)
    try:
        answer = await _kernel(
            store, gateway, CallableAgent(build), evaluator
        ).run(_objective("run-contradicted"))
        state = store.reconstruct("run-contradicted")
        claim = next(iter(state.claims.values()))
        assert claim.status is ClaimStatus.DISPUTED
        assert answer.supported_conclusions == []
        assert answer.contradictions
    finally:
        store.close()


async def test_contradicted_verdict_without_acquired_evidence_downgrades_to_proposed(
    tmp_path,
) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})

    def build(state):
        verdicts = [
            ClaimVerdict(
                claim_id=c.id,
                assessment=VerdictAssessment.CONTRADICTED,
                reason="source contradicts the claim but no contradiction was acquired",
            )
            for c in state.claims.values()
        ]
        return _evaluation(verdicts=verdicts)

    evaluator = StubEvaluator(_integrity(), build)
    try:
        answer = await _kernel(
            store, gateway, CallableAgent(_support_seed_claim), evaluator
        ).run(_objective("run-contradicted-unacquired"))
        state = store.reconstruct("run-contradicted-unacquired")
        claim = next(iter(state.claims.values()))
        assert claim.status is ClaimStatus.PROPOSED
        assert claim.contradicting == []
        assert answer.supported_conclusions == []
        transitions = [
            e
            for e in store.list_events("run-contradicted-unacquired")
            if e.type == "verdict_transition"
        ]
        assert transitions
        assert transitions[0].payload["to_status"] == "proposed"
        assert transitions[0].payload["reason"]
    finally:
        store.close()


async def test_evaluator_tokens_are_accounted_separately(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})

    def build(state):
        evaluation = _evaluation()
        assert evaluation.rubric is not None
        evaluation.rubric.input_tokens = 100
        evaluation.rubric.output_tokens = 50
        return evaluation

    evaluator = StubEvaluator(_integrity(), build)
    try:
        await _kernel(
            store, gateway, CallableAgent(lambda s, p: AgentBrief()), evaluator
        ).run(_objective("run-tokens"))
        budget = store.reconstruct("run-tokens").objective.budget
        assert budget.evaluator_tokens_used == 150
        assert budget.agent_tokens_used > 0
        assert budget.tokens_used == (
            budget.agent_tokens_used + budget.evaluator_tokens_used
        )
        event = next(
            e
            for e in store.list_events("run-tokens")
            if e.type == "evaluation_complete"
        )
        assert event.payload["evaluator_tokens"] == 150
    finally:
        store.close()


# ---- REL-2 supported verdict is not an evidence factory -------------------


async def test_supported_verdict_cannot_upgrade_evidence_free_claim(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})

    def agent(state, prompt) -> AgentBrief:
        return AgentBrief(
            claim_mutations=[
                ClaimMutation(
                    op="propose",
                    text="Evidence-free proposal",
                    status=ClaimStatus.PROPOSED,
                    evidence=[],
                )
            ]
        )

    def build(state):
        return _evaluation(
            verdicts=[
                ClaimVerdict(
                    claim_id=c.id,
                    assessment=VerdictAssessment.SUPPORTED,
                    reason="evaluator approves",
                )
                for c in state.claims.values()
            ]
        )

    evaluator = StubEvaluator(_integrity(), build)
    try:
        answer = await _kernel(
            store, gateway, CallableAgent(agent), evaluator
        ).run(_objective("run-supported-verdict"))
        state = store.reconstruct("run-supported-verdict")
        claim = next(iter(state.claims.values()))
        assert claim.status is ClaimStatus.PROPOSED
        assert claim.supporting == []
        assert answer.supported_conclusions == []
        assert answer.evidence_sufficient is False
        supported_transitions = [
            e
            for e in store.list_events("run-supported-verdict")
            if e.type == "verdict_transition" and e.payload["assessment"] == "supported"
        ]
        assert supported_transitions == []
        assert "Evidence-free proposal" not in answer.render_markdown()
    finally:
        store.close()


# ---- REL-3 provenance / locator boundary ----------------------------------


async def test_locator_provenance_survives_into_state_and_answer(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})

    def agent(state, prompt) -> AgentBrief:
        return AgentBrief(
            claim_mutations=[
                ClaimMutation(
                    op="propose",
                    text="Grounded conclusion",
                    status=ClaimStatus.SUPPORTED,
                    evidence=[EvidenceRef(paper_id=SEED, locator="Results 2.1")],
                )
            ]
        )

    evaluator = StubEvaluator(_integrity(), lambda state: _evaluation())
    try:
        answer = await _kernel(
            store,
            gateway,
            CallableAgent(agent),
            evaluator,
            evaluator_enabled=False,
        ).run(_objective("run-locator"))
        state = store.reconstruct("run-locator")
        claim = next(iter(state.claims.values()))
        assert claim.supporting[0].paper_id == SEED
        assert claim.supporting[0].locator == "Results 2.1"
        assert claim.supporting[0].content_hash == "h"
        assert "Results 2.1" in answer.supported_conclusions[0]
        assert "Results 2.1" in answer.render_markdown()
    finally:
        store.close()


def test_evaluation_prompt_exposes_acquired_provenance() -> None:
    objective = _objective("run-eval-provenance")
    state = ResearchState(
        objective=objective,
        notebook=AgentNotebook(agent_id="a"),
        evidence=[
            EvidenceRef(
                paper_id=SEED,
                locator="Methods 3",
                content_hash="abc123",
                acquisition_event=4,
            )
        ],
    )
    prompt = build_evaluation_prompt(objective, state)
    blob = " ".join(message["content"] for message in prompt.messages)
    assert "Methods 3" in blob
    assert "abc123" in blob
    assert "acquisition_event" in blob


# ---- REL-5 bounded expansion cap ------------------------------------------


async def test_search_expansion_respects_query_cap(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})
    evaluator = StubEvaluator(
        _integrity(),
        lambda state: _evaluation(
            missing=["gap one", "gap two", "gap three"]
        ),
    )
    kernel = ResearchKernel(
        store=store,
        gateway=gateway,
        policy=GreedyPolicy(seed=0, max_search_queries=1),
        agent=CallableAgent(lambda s, p: AgentBrief()),
        evaluator=evaluator,
        options=_options(max_search_queries=1),
    )
    try:
        await kernel.run(_objective("run-search-cap"))
        assert gateway.searches == ["gap one"]
        state = store.reconstruct("run-search-cap")
        assert state.search_queries == ["gap one"]
    finally:
        store.close()


async def test_search_cap_blocks_without_recording_issued_query(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = FakeGateway({SEED: {"title": "Seed"}})
    evaluator = StubEvaluator(
        _integrity(),
        lambda state: _evaluation(missing=["gap one", "gap two"]),
    )
    kernel = ResearchKernel(
        store=store,
        gateway=gateway,
        policy=GreedyPolicy(seed=0, max_search_queries=5),
        agent=CallableAgent(lambda s, p: AgentBrief()),
        evaluator=evaluator,
        options=_options(max_search_queries=1),
    )
    try:
        await kernel.run(_objective("run-search-cap-blocked"))
        assert gateway.searches == ["gap one"]
        state = store.reconstruct("run-search-cap-blocked")
        assert state.search_queries == ["gap one"]
        assert "gap two" in state.blocked_search_queries
        assert "gap two" not in state.search_queries
    finally:
        store.close()


# ---- REL-6 mapping failure is recorded in events --------------------------


class MappingFailureGateway(FakeGateway):
    async def acquire(self, paper_id: str, turn: int) -> Acquisition:
        acquisition = await super().acquire(paper_id, turn)
        acquisition.mapping_attempted = True
        acquisition.mapping_failed = True
        acquisition.mapping_fallback = True
        acquisition.mapping_recovered = 0
        acquisition.mapped_count = 0
        return acquisition


async def test_kernel_records_reference_mapping_fallback_in_event(tmp_path) -> None:
    store = ResearchStore(tmp_path / "r.db")
    gateway = MappingFailureGateway({SEED: {"title": "Seed"}})
    evaluator = StubEvaluator(_integrity(), lambda state: _evaluation())
    try:
        await _kernel(
            store,
            gateway,
            CallableAgent(lambda s, p: AgentBrief()),
            evaluator,
            evaluator_enabled=False,
        ).run(_objective("run-mapping-fallback"))
        event = next(
            e
            for e in store.list_events("run-mapping-fallback")
            if e.type == "evidence_acquired"
        )
        assert event.payload["reference_mapping"] == {
            "attempted": True,
            "failed": True,
            "fallback": True,
            "recovered": 0,
            "mapped": 0,
        }
    finally:
        store.close()
