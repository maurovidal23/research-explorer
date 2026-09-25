"""Integration tests for ACO selection, claims, private pheromone, metadata
frontier accounting, and candidate trace payloads."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_explorer.aco.frontier import SharedFrontier
from research_explorer.aco.scheduler import Scheduler
from research_explorer.agents.explorer import ExplorerAgent
from research_explorer.agents.state import AgentState
from research_explorer.config import Config
from research_explorer.replay.models import CandidateScore, CandidateSelection
from research_explorer.replay.trace import RunTracer, RunTraceStore


class _FixedRng:
    """Deterministic RNG: random() returns rand; choice/choices return pick."""

    def __init__(self, rand: float = 0.0, pick: int = 0):
        self._rand = rand
        self._pick = pick

    def random(self) -> float:
        return self._rand

    def choice(self, seq):
        return seq[self._pick]

    def choices(self, seq, weights=None, k: int = 1):
        return [seq[self._pick]]


class _Graph:
    def __init__(self, papers: dict):
        self._papers = papers

    def get_paper_summary(self, nid):
        return self._papers.get(nid)

    def get_paper_embedding(self, nid):
        return None

    def set_paper_embedding(self, nid, vec):
        return None


def _cfg(alpha=1.0, beta=2.0, epsilon=0.0) -> Config:
    return Config()


class _DummyProvider:
    name = "semantic_scholar"
    supports_fulltext = False

    async def get_paper(self, native):
        return None

    async def get_fulltext_and_refs(self, native, max_chars=0, ref_limit=0):
        return None


def _make_agent(agent_id="agent-000", caste="mixto", rng=None, papers=None,
                cfg=None, shared_visited=None, frontier=None) -> ExplorerAgent:
    cfg = cfg or Config()
    a = ExplorerAgent.__new__(ExplorerAgent)
    a.state = AgentState(id=agent_id, pos="s2:seed", caste=caste,
                         visited=["s2:seed"], budget=5)
    a.cfg = cfg
    a.seed_embedding = None  # avoid embedding calls -> sim stays neutral 0.5
    a.graph = _Graph(papers or {})
    a._shared_visited = shared_visited if shared_visited is not None else set()
    a._frontier = frontier if frontier is not None else SharedFrontier()
    a._eta_cache = {}
    a._llm_priority = {}
    a.tracer = None
    a.rng = rng if rng is not None else _FixedRng(rand=0.5, pick=0)
    a.provider = _DummyProvider()
    a.providers = {"semantic_scholar": a.provider}
    a.embedding = None
    return a


def _summary(pid: str, title: str, year: int = 2020, cites: int = 10, provider="semantic_scholar"):
    from research_explorer.graph.models import PaperSummary

    return PaperSummary(id=pid, title=title, year=year, citation_count=cites,
                        provider=provider)


async def test_private_pheromone_changes_selection() -> None:
    """Depositing private pheromone on an edge boosts that candidate for this agent."""
    papers = {"s2:a": _summary("s2:a", "AAA"), "s2:b": _summary("s2:b", "BBB")}

    # agent1 has private pheromone on the seed->a edge (its own frontier)
    f1 = SharedFrontier()
    f1.add(["s2:a", "s2:b"], source="s2:seed", mode="ref")
    a1 = _make_agent("agent-1", caste="mixto", papers=papers, frontier=f1)
    a1.state.set_pheromone("s2:seed", "s2:a", "ref", 5.0)
    sel1 = await a1._select_candidate()
    assert sel1 is not None
    assert sel1.probabilities["s2:a"] > sel1.probabilities["s2:b"]

    # agent2 has no private pheromone -> equal tau -> equal probability
    f2 = SharedFrontier()
    f2.add(["s2:a", "s2:b"], source="s2:seed", mode="ref")
    a2 = _make_agent("agent-2", caste="mixto", papers=papers, frontier=f2)
    sel2 = await a2._select_candidate()
    assert sel2 is not None
    assert sel2.probabilities["s2:a"] == pytest.approx(0.5)
    assert sel2.probabilities["s2:b"] == pytest.approx(0.5)


async def test_shared_claim_prevents_concurrent_selection() -> None:
    """A node claimed by another agent is excluded from a peer's selection."""
    frontier = SharedFrontier()
    frontier.add(["s2:a", "s2:b"], source="s2:seed", mode="ref")
    papers = {"s2:a": _summary("s2:a", "AAA"), "s2:b": _summary("s2:b", "BBB")}

    _make_agent("agent-1", caste="mixto", papers=papers, frontier=frontier)
    a2 = _make_agent("agent-2", caste="mixto", papers=papers, frontier=frontier)

    # agent1 claims 'a' (in-progress fetch)
    assert frontier.claim_for("s2:a", "agent-1") is True

    sel2 = await a2._select_candidate()
    assert sel2 is not None
    assert sel2.chosen != "s2:a"
    assert "s2:a" not in sel2.probabilities

    # releasing the claim makes 'a' selectable again (reusable path)
    frontier.release("s2:a", "agent-1")
    sel2b = await a2._select_candidate()
    assert "s2:a" in sel2b.probabilities or sel2b.chosen == "s2:a"


async def test_caste_shifts_direction_preference_in_selection() -> None:
    """impacto caste should weight cites-mode candidates above ref-mode peers."""
    papers = {
        "s2:refc": _summary("s2:refc", "RRR"),
        "s2:citc": _summary("s2:citc", "CCC"),
    }

    f_imp = SharedFrontier()
    f_imp.add(["s2:refc"], source="s2:seed", mode="ref")
    f_imp.add(["s2:citc"], source="s2:seed", mode="cites")

    f_fund = SharedFrontier()
    f_fund.add(["s2:refc"], source="s2:seed", mode="ref")
    f_fund.add(["s2:citc"], source="s2:seed", mode="cites")

    imp = _make_agent("impacto", caste="impacto", papers=papers, frontier=f_imp)
    fund = _make_agent("fundaciones", caste="fundaciones", papers=papers, frontier=f_fund)

    # impactor favors the cites-mode candidate
    sel_imp = await imp._select_candidate()
    assert sel_imp.probabilities["s2:citc"] > sel_imp.probabilities["s2:refc"]

    # fundaciones favors the ref-mode candidate
    sel_fund = await fund._select_candidate()
    assert sel_fund.probabilities["s2:refc"] > sel_fund.probabilities["s2:citc"]


async def test_metadata_only_frontier_nodes_are_rewarded_as_work() -> None:
    """A metadata-only transit consumes budget and is counted as fetch work."""
    class Expander:
        async def expand(self, node_id, paper=None, extracted=None, tracer=None):
            from types import SimpleNamespace as Ns
            return Ns(node_id=node_id,
                      incoming=Ns(direction="incoming", provider=None, fallback_used=False,
                                  discovered=0, rejected=0, node_ids=[]),
                      outgoing=Ns(direction="outgoing", provider=None, fallback_used=False,
                                  discovered=0, rejected=0, node_ids=[]))

    a = _make_agent("metadata", caste="mixto",
                    papers={"s2:meta": _summary("s2:meta", "Meta only")})
    a.expander = Expander()
    a.state.budget = 3
    a.state.mark_discovered("s2:seed")
    # a metadata-only node in the frontier (metadata cached, no readable content)
    a._frontier.add(["s2:meta"], source="s2:seed", mode="ref")

    edges = await a.take_turn(2)

    # metadata transit: no traversal edge, but work is counted
    assert edges == []
    assert a.state.delta_work == 1
    assert a.state.fetch_work == 1
    assert a.state.budget == 2
    assert "s2:meta" in a.state.metadata_transits


# ---- Scheduler accounting ---------------------------------------------------


def _async_empty_turn(delta_work: int):
    async def _turn(k):
        return []
    return _turn


class _FakeColony:
    def __init__(self, agents):
        self.llm = object()
        self.agents = agents
        self.seed_query = "q"
        self.graph = SimpleNamespace(get_paper=lambda nid: None)
        self.shared_frontier = SharedFrontier()
        self.agent_states = [a.state for a in agents]
        self.best_quality = 0.0
        self.best_snapshot_agent = ""

    def update_best(self, oleada):
        peak = max(a.state.quality for a in self.agents)
        if peak > self.best_quality:
            self.best_quality = peak
            self.best_snapshot_agent = max(self.agents, key=lambda a: a.state.quality).state.id

    def pheromone_concentration(self):
        return 0.0


class _FakePheromone:
    def update(self, *args, **kwargs):
        pass


async def test_scheduler_counts_metadata_work_consistently() -> None:
    agent = _make_agent("agent-work")
    agent.state.budget = 4
    agent.state.delta_q = 0.0

    async def fake_turn(k):
        agent.state.delta_work = 2  # e.g. 2 metadata transits this turn
        return []

    agent.take_turn = fake_turn  # type: ignore[method-assign]

    colony = _FakeColony([agent])
    scheduler = Scheduler(colony, Config(), _FakePheromone(), SimpleNamespace())
    scheduler._pick_top_k = lambda: [agent]
    await scheduler.run_oleada()
    assert scheduler.total_fetches == 2


# ---- Trace payloads ----------------------------------------------------------


async def test_candidate_score_and_selected_trace_payloads(tmp_path) -> None:
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "q")
    tracer = RunTracer(store, run_id)

    score = CandidateScore(
        paper_id="openalex:W1", agent_id="agent-1", provider="openalex",
        components={"sim": 0.8, "citations": 0.3, "recency": 0.5,
                    "confidence": 0.9, "llm": 0.4},
        weights={"w_sim": 0.6, "w_citas": 0.2, "w_recencia": 0.2,
                 "w_confidence": 0.1, "w_llm": 0.2},
        eta=0.62, llm_used=True, source="openalex:seed", mode="ref",
    )
    tracer.record_candidate_score(score)

    sel = CandidateSelection(
        agent_id="agent-1", paper_id="openalex:W1", src="openalex:seed",
        mode="ref", caste="mixto", dir_modifier=0.7, tau=3.0, alpha=1.0, beta=2.0,
        eta=0.62, final_weight=3.0 * 0.62 ** 2 * 0.7, probability=0.9,
        epsilon_branch=False, chosen=True, rationale="caste=mixto mode=ref branch=weighted",
    )
    tracer.record_candidate_selected(sel)

    events = store.list_events(run_id)
    types = [e["type"] for e in events]
    assert "candidate_score" in types
    assert "candidate_selected" in types

    score_evt = next(e for e in events if e["type"] == "candidate_score")
    assert score_evt["payload"]["paper_id"] == "openalex:W1"
    assert score_evt["payload"]["components"]["sim"] == 0.8
    assert score_evt["payload"]["weights"]["w_llm"] == 0.2
    assert score_evt["payload"]["eta"] == 0.62
    assert score_evt["payload"]["llm_used"] is True

    sel_evt = next(e for e in events if e["type"] == "candidate_selected")
    p = sel_evt["payload"]
    assert p["paper_id"] == "openalex:W1"
    assert p["mode"] == "ref"
    assert p["caste"] == "mixto"
    assert p["dir_modifier"] == 0.7
    assert p["tau"] == 3.0
    assert p["alpha"] == 1.0
    assert p["beta"] == 2.0
    assert p["epsilon_branch"] is False
    assert p["chosen"] is True
    assert "branch=weighted" in p["rationale"]
    store.close()


async def test_candidate_events_appear_on_backward_compatible_api(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from research_explorer.replay.server import build_app

    db = tmp_path / "replay.db"
    store = RunTraceStore(db)
    run_id = store.create_run("seed", "q")
    tracer = RunTracer(store, run_id)
    tracer.record_candidate_selected(CandidateSelection(
        agent_id="agent-1", paper_id="openalex:W1", src="s", mode="ref",
        caste="mixto", dir_modifier=0.7, tau=1.0, alpha=1.0, beta=1.0,
        eta=0.5, final_weight=0.35, probability=1.0, chosen=True,
    ))
    store.close()

    app = build_app(db, default_run_id=run_id)
    with TestClient(app) as client:
        events = client.get(f"/api/runs/{run_id}/events").json()
        runs = client.get("/api/runs").json()

    assert any(e["type"] == "candidate_selected" for e in events)
    assert runs[0]["seed_paper_id"] == "seed"


async def test_eta_components_expose_normalized_provider_confidence() -> None:
    """Eta exposes provider confidence per provider, adjusted by metadata evidence."""
    # openalex with DOI + abstract -> high confidence
    oa = _summary("openalex:W1", "Title", year=2021, cites=5, provider="openalex")
    oa.doi = "10.123/x"
    oa.abstract = "an abstract"

    # arxiv with title only -> lower base
    ax = _summary("arxiv:2001.00001", "Title", year=2020, cites=0, provider="arxiv")

    a = _make_agent("conf", papers={
        "openalex:W1": oa, "arxiv:2001.00001": ax,
    })
    c1 = await a._eta_components("openalex:W1")
    c2 = await a._eta_components("arxiv:2001.00001")
    assert c1.components()["confidence"] > c2.components()["confidence"]
    assert c1.components()["confidence"] == pytest.approx(1.0)  # 0.95 + 0.05 + 0.05
    assert c2.components()["confidence"] == pytest.approx(0.8)
    assert c1.components()["recency"] >= c2.components()["recency"]


async def test_selection_emits_candidate_events_via_tracer(tmp_path) -> None:
    """Selection emits candidate_score + candidate_selected trace events."""
    store = RunTraceStore(tmp_path / "replay.db")
    run_id = store.create_run("seed", "q")
    tracer = RunTracer(store, run_id)

    frontier = SharedFrontier()
    frontier.add(["s2:a"], source="s2:seed", mode="ref")
    papers = {"s2:a": _summary("s2:a", "AAA")}
    a = _make_agent("agent-1", papers=papers, frontier=frontier)
    a.tracer = tracer

    sel = await a._select_candidate()
    assert sel is not None
    assert sel.chosen == "s2:a"

    events = store.list_events(run_id)
    types = [e["type"] for e in events]
    assert "candidate_score" in types
    assert "candidate_selected" in types
    score_evt = next(e for e in events if e["type"] == "candidate_score")
    assert score_evt["payload"]["paper_id"] == "s2:a"
    assert set(score_evt["payload"]["components"]) == {
        "sim", "citations", "recency", "confidence", "llm",
    }
    sel_evt = next(e for e in events if e["type"] == "candidate_selected")
    p = sel_evt["payload"]
    assert p["paper_id"] == "s2:a"
    assert p["chosen"] is True
    assert p["alpha"] == a.cfg.aco.alpha
    assert p["mode"] == "ref"
    store.close()
