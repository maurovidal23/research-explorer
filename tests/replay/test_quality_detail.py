"""Quality assess_detail retention tests (self reasoning, peer votes, coverage)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_explorer.agents.state import AgentState
from research_explorer.config import Config
from research_explorer.evaluation.peer_vote import VOTE_SCHEMA
from research_explorer.evaluation.quality import QualityAssessor
from research_explorer.evaluation.self_assess import SCORE_SCHEMA
from research_explorer.evaluation.virgin_judge import JUDGE_SCHEMA
from research_explorer.replay.models import StructuralComponentsDetail


class FakeLLM:
    """Scripted chat_json that returns schema-appropriate structured output."""

    async def chat_json(self, messages, *, model="", schema=None, temperature=0.0, max_tokens=0):
        if schema is None:
            return {"score": 0.5}
        if schema is SCORE_SCHEMA:
            return {"score": 0.8, "reasoning": "I judge my narrative directly relevant"}
        if schema is VOTE_SCHEMA:
            content = " ".join(m.get("content") or "" for m in messages)
            if "beta" in content:
                return {"score": 0.6, "reasoning": "agent peer2 finds it partially relevant"}
            return {"score": 0.8, "reasoning": "agent peer1 sees strong advancement"}
        if schema is JUDGE_SCHEMA:
            return {
                "score": 0.6,
                "coverage": "Covers foundational mechanisms",
                "gaps": "Does not address applications",
            }
        return {"score": 0.5}


class FakeStructural:
    def compute_detail(self, state) -> StructuralComponentsDetail:
        return StructuralComponentsDetail(
            coverage=0.4, diversity=0.6, depth=0.5, coherence=1.0, r=0.625
        )


def _agent(agent_id: str, narrative: str = "narrative") -> SimpleNamespace:
    return SimpleNamespace(
        state=AgentState(id=agent_id, pos="seed", narrative=narrative, quality=0.0, turn_count=1)
    )


@pytest.fixture
def assessor():
    cfg = Config()
    return QualityAssessor(FakeLLM(), cfg, FakeStructural())


async def test_assess_detail_retains_self_reasoning(assessor):
    target = _agent("agent-000-target")
    peers = [_agent("agent-000-peer1", "alpha narrative"), _agent("agent-000-peer2", "beta narrative")]
    rec = await assessor.assess_detail(target, [target, *peers], "research line", oleada=2)

    assert rec.agent_id == "agent-000-target"
    assert rec.oleada == 2
    assert rec.turn == 1
    assert rec.self_assessment.score == 0.8
    assert "directly relevant" in rec.self_assessment.reasoning

    assert rec.peers.num_votes == 2
    votes_by_voter = {v.voter_id: v for v in rec.peers.votes}
    assert votes_by_voter["agent-000-peer1"].reasoning == "agent peer1 sees strong advancement"
    assert votes_by_voter["agent-000-peer2"].score == 0.6
    assert all(v.reasoning for v in rec.peers.votes)

    assert rec.virgin_judge.score == 0.6
    assert rec.virgin_judge.coverage == "Covers foundational mechanisms"
    assert rec.virgin_judge.gaps == "Does not address applications"

    assert rec.structural.coverage == 0.4
    assert rec.structural.r == 0.625

    assert rec.weights == {"S": 0.25, "P": 0.25, "J": 0.25, "R": 0.25}
    expected_q = 0.25 * 0.8 + 0.25 * 0.7 + 0.25 * 0.6 + 0.25 * 0.625
    assert rec.q == pytest.approx(expected_q)


async def test_assess_detail_tracks_delta_q(assessor):
    target = _agent("agent-000-target")
    target.state.quality = 0.5
    rec = await assessor.assess_detail(target, [target], "research line")
    assert rec.old_quality == 0.5
    assert rec.delta_q == pytest.approx(rec.q - 0.5)


async def test_assess_surfaces_breakdown(assessor):
    target = _agent("agent-000-target")
    peer1 = _agent("agent-000-peer1", "alpha narrative")
    q, breakdown = await assessor.assess(target, [target, peer1], "research line")
    assert q == pytest.approx(
        0.25 * 0.8 + 0.25 * 0.8 + 0.25 * 0.6 + 0.25 * 0.625
    )
    assert set(breakdown) == {"S", "P", "J", "R"}
    assert breakdown["S"] == 0.8


class FailingLLM:
    """LLM that always fails, simulating JSON parse errors."""

    async def chat_json(self, messages, *, model="", schema=None, temperature=0.0, max_tokens=0):
        raise ValueError("simulated LLM failure")


async def test_failed_llm_components_score_zero():
    cfg = Config()
    assessor = QualityAssessor(FailingLLM(), cfg, FakeStructural())
    target = _agent("agent-fail", narrative="some narrative")
    rec = await assessor.assess_detail(target, [target], "research line", oleada=1)

    assert rec.self_assessment.score == 0.0
    assert "failed" in rec.self_assessment.reasoning
    assert rec.peers.aggregated_score == 0.0
    assert rec.peers.num_votes == 0
    assert rec.virgin_judge.score == 0.0
    assert "failed" in rec.virgin_judge.gaps

    assert rec.q == pytest.approx(0.25 * 0.0 + 0.25 * 0.0 + 0.25 * 0.0 + 0.25 * 0.625)
    assert rec.q < 0.2, "failed evaluator must not produce competitive Q"
