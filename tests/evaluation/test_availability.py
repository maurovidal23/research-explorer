"""Component availability: unavailable reasons stay distinct from a real zero."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from research_explorer.agents.state import AgentState
from research_explorer.config import Config
from research_explorer.evaluation import availability
from research_explorer.evaluation.peer_vote import PeerVoting
from research_explorer.evaluation.self_assess import SelfAssessment
from research_explorer.evaluation.virgin_judge import VirginJudge
from research_explorer.events.models import EvaluationState, RunViewState
from research_explorer.tui import text as render


class _StubLLM:
    def __init__(self, result=None, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error

    async def chat_json(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return self.result


def test_classify_failure_maps_stable_reasons() -> None:
    assert availability.classify_failure(asyncio.TimeoutError()) == availability.REASON_TIMEOUT
    assert availability.classify_failure(TimeoutError()) == availability.REASON_TIMEOUT
    assert (
        availability.classify_failure(json.JSONDecodeError("bad", "x", 0))
        == availability.REASON_PARSE_FAILED
    )
    assert availability.classify_failure(ValueError("bad")) == availability.REASON_PARSE_FAILED
    assert availability.classify_failure(RuntimeError("boom")) == availability.REASON_MODEL_FAILED


async def test_model_zero_is_available_not_unavailable() -> None:
    llm = _StubLLM(result={"score": 0.0, "reasoning": "nothing useful"})
    detail = await SelfAssessment(llm, Config()).score_detail("a narrative", "query")
    assert detail.score == 0.0
    assert detail.available is True
    assert detail.unavailable_reason == ""


async def test_empty_narrative_is_unavailable_zero() -> None:
    llm = _StubLLM(result={"score": 0.0, "reasoning": "unused"})
    detail = await SelfAssessment(llm, Config()).score_detail("   ", "query")
    assert detail.score == 0.0
    assert detail.available is False
    assert detail.unavailable_reason == availability.REASON_EMPTY_NARRATIVE


async def test_parse_failure_and_timeout_are_not_ordinary_zeros() -> None:
    parse = await SelfAssessment(
        _StubLLM(error=json.JSONDecodeError("bad", "x", 0)), Config()
    ).score_detail("a narrative", "query")
    assert parse.score == 0.0
    assert parse.available is False
    assert parse.unavailable_reason == availability.REASON_PARSE_FAILED

    timeout = await SelfAssessment(
        _StubLLM(error=asyncio.TimeoutError()), Config()
    ).score_detail("a narrative", "query")
    assert timeout.score == 0.0
    assert timeout.available is False
    assert timeout.unavailable_reason == availability.REASON_TIMEOUT

    model = await VirginJudge(
        _StubLLM(error=RuntimeError("provider down")), Config()
    ).judge_detail("a narrative", "query")
    assert model.score == 0.0
    assert model.available is False
    assert model.unavailable_reason == availability.REASON_MODEL_FAILED


async def test_peer_parse_failure_is_not_hidden_as_model_failure() -> None:
    target = SimpleNamespace(
        state=AgentState(id="target", pos="seed", narrative="target narrative")
    )
    voter = SimpleNamespace(
        state=AgentState(id="voter", pos="seed", narrative="voter narrative")
    )
    detail = await PeerVoting(
        _StubLLM(error=json.JSONDecodeError("bad", "", 0)), Config()
    ).vote_detail(target, [target, voter], "query")
    assert detail.available is False
    assert detail.unavailable_reason == availability.REASON_PARSE_FAILED


def test_evaluation_render_separates_zero_from_unavailable() -> None:
    state = RunViewState()
    state.current_wave = 1
    state.evaluation_states["1:a0"] = EvaluationState(
        agent_id="a0",
        wave=1,
        status="complete",
        q=0.0,
        q_delta=0.0,
        components={"S": 0.0, "P": 0.0, "J": 0.0, "R": 0.0},
        unavailable={},
        evidence_papers=1,
    )
    zero_body = "\n".join(render.render_evaluation_state(state, "a0", 1))
    assert "S: 0.0000" in zero_body
    assert "unavailable" not in zero_body

    state.evaluation_states["1:a1"] = EvaluationState(
        agent_id="a1",
        wave=1,
        status="complete",
        q=0.0809,
        q_delta=0.0809,
        components={"S": None, "P": None, "J": None, "R": 0.32},
        unavailable={"S": "timeout", "P": "no_peers", "J": "parse_failed"},
        evidence_papers=1,
    )
    partial_body = "\n".join(render.render_evaluation_state(state, "a1", 1))
    assert "S: unavailable (timeout)" in partial_body
    assert "J: unavailable (parse_failed)" in partial_body
    assert "R: 0.3200" in partial_body
