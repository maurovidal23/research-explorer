"""Frozen-corpus end-to-end single-agent loop.

The corpus contains a seed, several references, one transient fetch failure,
and one contradictory source, plus enough evidence to resolve one question.
No live provider or LLM is used.
"""

from __future__ import annotations

from research_explorer.research.evaluator import CompositeEvaluator, DeterministicIntegrity
from research_explorer.research.loop import (
    Acquisition,
    KernelOptions,
    ResearchKernel,
)
from research_explorer.research.models import (
    AgentBrief,
    BudgetState,
    CandidateAction,
    ClaimMutation,
    ClaimStatus,
    EvidenceRef,
    OpenQuestionStatus,
    ProviderOutcome,
    ResearchObjective,
)
from research_explorer.research.policy import GreedyPolicy
from research_explorer.research.store import ResearchStore

SEED = "openalex:seed"
R1 = "openalex:r1"
R2 = "openalex:r2"
R3 = "openalex:r3"
R4 = "openalex:r4"
C1 = "openalex:c1"

CORPUS: dict[str, dict] = {
    SEED: {"title": "Seed", "abstract": "seed abstract", "refs": [R3], "cits": []},
    R3: {
        "title": "Bridge",
        "abstract": "bridge abstract",
        "refs": [R1, R2, R4],
        "cits": [C1],
    },
    R1: {"title": "Supporting A", "abstract": "a", "refs": [R4], "cits": []},
    R2: {"title": "Contradictory source", "abstract": "contradiction", "refs": [], "cits": []},
    R4: {"title": "Supporting B", "abstract": "b", "refs": [], "cits": []},
    C1: {"title": "Answer paper", "abstract": "c", "refs": [], "cits": []},
}


class FakeGateway:
    def __init__(self, transient_once: set[str]) -> None:
        self.transient_once = set(transient_once)
        self.failed: set[str] = set()
        self.acquired: set[str] = set()

    async def acquire(self, paper_id: str, turn: int) -> Acquisition:
        if paper_id in self.transient_once and paper_id not in self.failed:
            self.failed.add(paper_id)
            return Acquisition(paper_id, ProviderOutcome.TRANSIENT, provider="fake")
        if paper_id not in CORPUS:
            return Acquisition(paper_id, ProviderOutcome.ABSENT, provider="fake")
        self.acquired.add(paper_id)
        entry = CORPUS[paper_id]
        candidates = [
            CandidateAction(paper_id=rid, source=paper_id, mode="ref", score=1.0)
            for rid in entry.get("refs", [])
        ] + [
            CandidateAction(paper_id=cid, source=paper_id, mode="cites", score=0.8)
            for cid in entry.get("cits", [])
        ]
        return Acquisition(
            paper_id=paper_id,
            outcome=ProviderOutcome.SUCCESS,
            provider="fake",
            title=entry["title"],
            abstract=entry.get("abstract", ""),
            evidence=EvidenceRef(paper_id=paper_id, content_hash="h"),
            candidates=candidates,
        )

    def paper_exists(self, paper_id: str) -> bool:
        return paper_id in CORPUS or paper_id in self.acquired

    def close(self) -> None:
        return None


class FakeAgent:
    """Deterministic briefs keyed by the acquired source paper."""

    async def propose_brief(self, objective, state, prompt) -> AgentBrief:
        source = prompt.selected_ids[0]
        if source == SEED:
            return AgentBrief(
                action_summary="seed read",
                claim_mutations=[
                    ClaimMutation(
                        op="propose",
                        text="Extracellular fields originate from neuronal currents",
                        evidence=[EvidenceRef(paper_id=SEED)],
                    ),
                    ClaimMutation(
                        op="open_question",
                        text="Which neuronal currents generate the field?",
                    ),
                ],
                evidence=[EvidenceRef(paper_id=SEED)],
            )
        if source == R3:
            return AgentBrief(
                action_summary="bridge read",
                claim_mutations=[
                    ClaimMutation(
                        op="propose",
                        text="Field magnitude scales with synchronous firing",
                        evidence=[EvidenceRef(paper_id=R3)],
                    )
                ],
                evidence=[EvidenceRef(paper_id=R3)],
            )
        if source == R4:
            proposed = [c.id for c in state.claims.values() if c.status is ClaimStatus.PROPOSED]
            if proposed:
                return AgentBrief(
                    action_summary="support B",
                    claim_mutations=[
                        ClaimMutation(
                            op="support",
                            claim_id=proposed[0],
                            evidence=[EvidenceRef(paper_id=R4)],
                        )
                    ],
                )
            return AgentBrief(action_summary="nothing to support")
        if source == R2:
            disputed = [c.id for c in state.claims.values() if c.status is ClaimStatus.PROPOSED]
            if disputed:
                return AgentBrief(
                    action_summary="contradiction found",
                    claim_mutations=[
                        ClaimMutation(
                            op="dispute",
                            claim_id=disputed[0],
                            evidence=[EvidenceRef(paper_id=R2)],
                        )
                    ],
                    contradictions=["source R2 disagrees"],
                )
            return AgentBrief(action_summary="no claim to dispute")
        if source == C1:
            open_q = [
                q.id
                for q in state.questions.values()
                if q.status in (OpenQuestionStatus.OPEN, OpenQuestionStatus.INVESTIGATING)
            ]
            if open_q:
                return AgentBrief(
                    action_summary="answer question",
                    claim_mutations=[
                        ClaimMutation(
                            op="answer_question",
                            question_id=open_q[0],
                            evidence=[EvidenceRef(paper_id=C1)],
                        )
                    ],
                )
        return AgentBrief(action_summary=f"read {source}")


def _options() -> KernelOptions:
    return KernelOptions(
        max_transient_attempts=2,
        plateau_turns=100,
        convergence_epsilon=0.0,
        snapshot_interval=1,
    )


async def test_frozen_corpus_run_completes_with_evidence(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    gateway = FakeGateway(transient_once={R3})
    objective = ResearchObjective(
        run_id="run-frozen",
        seed_paper_id=SEED,
        question="How do extracellular fields originate?",
        budget=BudgetState(max_fetches=12, max_turns=20, max_tokens=100000),
        random_seed=0,
    )
    evaluator = CompositeEvaluator(
        integrity=DeterministicIntegrity(paper_exists=gateway.paper_exists),
        rubric=None,
        weights={"integrity": 1.0},
    )
    kernel = ResearchKernel(
        store=store,
        gateway=gateway,
        policy=GreedyPolicy(seed=0),
        agent=FakeAgent(),
        evaluator=evaluator,
        options=_options(),
    )
    try:
        answer = await kernel.run(objective)

        # A supported claim resolves to stored evidence.
        assert answer.supported_conclusions
        assert any(cid in answer.citations for cid in (R4, R3, R1))

        # A question was resolved with evidence.
        state = store.reconstruct("run-frozen")
        resolved = [
            q for q in state.questions.values() if q.status is OpenQuestionStatus.RESOLVED
        ]
        assert resolved
        assert resolved[0].resolution_evidence

        # The transient failure is visibly classified and retained, not treated as absence.
        failures = [e for e in store.list_events("run-frozen") if e.type == "provider_failure"]
        assert failures
        assert failures[0].payload["classification"] == "transient"
        assert failures[0].payload["paper_id"] == R3
        assert R3 in state.visited  # later succeeded

        # Final-answer citations all resolve through the gateway.
        integrity = DeterministicIntegrity(paper_exists=gateway.paper_exists).evaluate(
            state, answer
        )
        assert integrity.checks["final_answer_citations_resolve"]
        assert "final_answer" in [e.type for e in store.list_events("run-frozen")]
    finally:
        store.close()


async def test_reconstruction_equals_live_snapshots(tmp_path) -> None:
    from research_explorer.research.models import canonical_json

    store = ResearchStore(tmp_path / "research.db")
    gateway = FakeGateway(transient_once={R3})
    objective = ResearchObjective(
        run_id="run-recon",
        seed_paper_id=SEED,
        question="How do extracellular fields originate?",
        budget=BudgetState(max_fetches=12, max_turns=20),
    )
    kernel = ResearchKernel(
        store=store,
        gateway=gateway,
        policy=GreedyPolicy(seed=0),
        agent=FakeAgent(),
        evaluator=CompositeEvaluator(
            integrity=DeterministicIntegrity(paper_exists=gateway.paper_exists),
            rubric=None,
            weights={"integrity": 1.0},
        ),
        options=_options(),
    )
    try:
        await kernel.run(objective)

        first = canonical_json(store.reconstruct("run-recon"))
        second = canonical_json(store.reconstruct("run-recon"))
        assert first == second

        for snapshot in store.list_snapshots("run-recon"):
            rebuilt = store.reconstruct("run-recon", snapshot["seq"])
            assert canonical_json(rebuilt) == snapshot["state_json"]

        run = store.get_run("run-recon")
        assert run is not None
        assert run["status"] == "completed"
        assert run["terminal_reason"] == "evidence_sufficient"
        seqs = [e.seq for e in store.list_events("run-recon")]
        assert seqs == sorted(set(seqs))
    finally:
        store.close()
