"""Typed evaluation-replay model tests."""

from __future__ import annotations

from research_explorer.replay.models import (
    DetailedEvaluation,
    PeerVoteDetail,
    PeerVotesDetail,
    SelfAssessmentDetail,
    StructuralComponentsDetail,
    VirginJudgeDetail,
)


def test_detailed_evaluation_defaults():
    rec = DetailedEvaluation(agent_id="agent-000-abc")
    assert rec.agent_id == "agent-000-abc"
    assert rec.oleada == 0
    assert rec.turn == 0
    assert rec.q == 0.0
    assert rec.peers.votes == []
    assert rec.peers.num_votes == 0
    assert rec.virgin_judge.coverage == ""
    assert rec.structural.coverage == 0.0
    assert rec.new_papers == []


def test_detailed_evaluation_breakdown_property():
    rec = DetailedEvaluation(
        agent_id="a",
        self_assessment=SelfAssessmentDetail(score=0.2, reasoning="r"),
        peers=PeerVotesDetail(aggregated_score=0.4),
        virgin_judge=VirginJudgeDetail(score=0.6),
        structural=StructuralComponentsDetail(r=0.8),
    )
    assert rec.breakdown == {"S": 0.2, "P": 0.4, "J": 0.6, "R": 0.8}


def test_peer_votes_retain_individual_reasoning():
    rec = DetailedEvaluation(
        agent_id="a",
        peers=PeerVotesDetail(
            votes=[
                PeerVoteDetail(voter_id="p1", score=0.9, reasoning="vote one"),
                PeerVoteDetail(voter_id="p2", score=0.5, reasoning="vote two"),
            ],
            aggregated_score=0.7,
            aggregation_method="median",
            num_votes=2,
        ),
    )
    assert len(rec.peers.votes) == 2
    assert rec.peers.votes[0].voter_id == "p1"
    assert rec.peers.votes[0].reasoning == "vote one"


def test_detailed_evaluation_json_roundtrip():
    rec = DetailedEvaluation(
        agent_id="a",
        oleada=3,
        turn=2,
        q=0.42,
        self_assessment=SelfAssessmentDetail(score=0.5, reasoning="self"),
        peers=PeerVotesDetail(
            votes=[PeerVoteDetail(voter_id="p", score=0.6, reasoning="peer")],
            aggregated_score=0.6,
            num_votes=1,
        ),
        virgin_judge=VirginJudgeDetail(score=0.4, coverage="cov", gaps="gap"),
        structural=StructuralComponentsDetail(
            coverage=0.25, diversity=0.5, depth=0.75, coherence=1.0, r=0.6
        ),
        new_papers=["s2:abc"],
    )
    restored = DetailedEvaluation.model_validate_json(rec.model_dump_json())
    assert restored.agent_id == "a"
    assert restored.oleada == 3
    assert restored.peers.votes[0].reasoning == "peer"
    assert restored.virgin_judge.gaps == "gap"
    assert restored.structural.coherence == 1.0


def test_app_js_handles_evaluation_skipped():
    from pathlib import Path

    js_path = Path(__file__).resolve().parents[2] / "src" / "research_explorer" / "replay" / "static" / "app.js"
    content = js_path.read_text()
    assert "evaluation_skipped" in content
    assert "Skipped" in content
    assert "Quality unchanged" in content
