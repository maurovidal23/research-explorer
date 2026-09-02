"""Typed evaluation-replay data models.

These records retain the full, explicit rationale behind every quality score:
self-assessment reasoning, individual peer reasoning, virgin-judge coverage and
gaps, and the deterministic structural breakdown. They are the persisted,
queryable payloads stored by the RunTraceStore.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_weights() -> dict[str, float]:
    return {"S": 0.25, "P": 0.25, "J": 0.25, "R": 0.25}


class SelfAssessmentDetail(BaseModel):
    """S component: the agent's own judgement of its narrative."""

    score: float = 0.0
    reasoning: str = Field(default="", description="Explicit rationale behind the score")


class PeerVoteDetail(BaseModel):
    """One peer's individual vote on the target agent's recent contribution."""

    voter_id: str
    score: float = 0.0
    reasoning: str = Field(default="", description="The voting peer's explicit rationale")


class PeerVotesDetail(BaseModel):
    """P component: aggregate peer vote plus every individual vote."""

    votes: list[PeerVoteDetail] = Field(default_factory=list)
    aggregated_score: float = 0.5
    aggregation_method: str = "median"
    num_votes: int = 0


class VirginJudgeDetail(BaseModel):
    """J component: impartial judge with coverage assessment and gaps."""

    score: float = 0.0
    coverage: str = Field(default="", description="What the narrative covers")
    gaps: str = Field(default="", description="Explicit gaps the judge identified")


class StructuralComponentsDetail(BaseModel):
    """R component: deterministic structural breakdown."""

    coverage: float = 0.0
    diversity: float = 0.0
    depth: float = 0.0
    coherence: float = 0.0
    r: float = Field(default=0.0, description="Aggregate structural score")


class DetailedEvaluation(BaseModel):
    """Full quality evaluation record for one agent turn."""

    agent_id: str
    oleada: int = 0
    turn: int = 0
    q: float = Field(default=0.0, description="Final weighted quality in [0, 1]")
    old_quality: float = 0.0
    delta_q: float = 0.0
    weights: dict[str, float] = Field(default_factory=_default_weights)
    self_assessment: SelfAssessmentDetail = Field(default_factory=SelfAssessmentDetail)
    peers: PeerVotesDetail = Field(default_factory=PeerVotesDetail)
    virgin_judge: VirginJudgeDetail = Field(default_factory=VirginJudgeDetail)
    structural: StructuralComponentsDetail = Field(default_factory=StructuralComponentsDetail)
    new_papers: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=utc_now)

    @property
    def breakdown(self) -> dict[str, float]:
        return {
            "S": self.self_assessment.score,
            "P": self.peers.aggregated_score,
            "J": self.virgin_judge.score,
            "R": self.structural.r,
        }
