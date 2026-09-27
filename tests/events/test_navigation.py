"""Derived agent-first navigation tests."""

from __future__ import annotations

from research_explorer.events.models import RunEvent
from research_explorer.events.navigation import (
    active_agent_id,
    active_leaf_entry,
    build_agent_navigation,
    find_node,
    flatten_navigation,
)
from research_explorer.events.projection import RunProjection


def _event(seq: int, type: str, **payload) -> RunEvent:
    return RunEvent(seq=seq, type=type, payload=payload)


def _state(*events: RunEvent):
    start = _event(
        1,
        "orchestrator_start",
        run_id="r",
        colony_size=3,
        K=2,
        k_per_turn=2,
    )
    colony = _event(2, "colony_initialized", size=3, agents=["a0", "a1", "a2"])
    return RunProjection.from_events([start, colony, *events]).state


def test_agent_roots_follow_colony_order() -> None:
    roots = build_agent_navigation(_state())
    assert [root.agent_id for root in roots] == ["a0", "a1", "a2"]
    assert all(root.kind == "agent" for root in roots)


def test_waves_turns_and_leaves_nest_under_agent() -> None:
    state = _state(
        _event(3, "oleada_start", oleada=1, active=["a0"]),
        _event(4, "agent_turn_start", agent_id="a0", oleada=1, turn=0, caste="mixto"),
        _event(
            5,
            "paper_integration_completed",
            agent_id="a0",
            paper_id="p1",
            title="P1",
            oleada=1,
            turn=0,
        ),
        _event(
            6,
            "evaluation_skipped",
            agent_id="a0",
            oleada=1,
            turn=0,
            reason="no_new_evidence",
        ),
    )
    roots = build_agent_navigation(state)
    wave = find_node(roots, "agent:a0:wave:1")
    assert wave is not None
    turn = find_node(roots, "turn:1:a0:0")
    assert turn is not None
    kinds = [child.kind for child in turn.children]
    assert "leaf" in kinds
    leaves = [child.label for child in turn.children]
    assert any("read" in label for label in leaves)
    assert any("skipped" in label for label in leaves)
    skipped = find_node(roots, "eval:1:a0:0")
    assert skipped is not None and skipped.status == "skipped"


def test_seed_discovery_is_grouped_as_wave_zero() -> None:
    state = _state(
        _event(
            3,
            "neighbor_discovery_completed",
            agent_id="a0",
            paper_id="seed",
            oleada=0,
            turn=0,
            refs=2,
            cits=1,
        )
    )
    roots = build_agent_navigation(state)
    seed = find_node(roots, "agent:a0:wave:0")
    assert seed is not None and seed.label == "Seed"


def test_missing_agent_context_is_derived_from_surrounding_turn() -> None:
    state = _state(
        _event(3, "oleada_start", oleada=1, active=["a0"]),
        _event(4, "agent_turn_start", agent_id="a0", oleada=1, turn=0),
        RunEvent(
            seq=5,
            type="paper_integration_completed",
            payload={"paper_id": "p1", "title": "P1", "turn": 0},
        ),
    )
    roots = build_agent_navigation(state)
    turn = find_node(roots, "turn:1:a0:0")
    assert turn is not None
    assert any("P1" in child.label for child in turn.children)


def test_duplicate_delivery_does_not_duplicate_nodes() -> None:
    events = [
        _event(3, "oleada_start", oleada=1, active=["a0"]),
        _event(4, "agent_turn_start", agent_id="a0", oleada=1, turn=0),
    ]
    projection = RunProjection.from_events(
        [
            _event(1, "orchestrator_start", run_id="r", colony_size=1, K=1),
            _event(2, "colony_initialized", size=1, agents=["a0"]),
            *events,
            *events,
        ]
    )
    roots = build_agent_navigation(projection.state)
    assert len(flatten_navigation(roots, {"agent:a0", "agent:a0:wave:1"})) == 3


def test_flatten_honors_expansion() -> None:
    state = _state(
        _event(3, "oleada_start", oleada=1, active=["a0"]),
        _event(4, "agent_turn_start", agent_id="a0", oleada=1, turn=0),
    )
    roots = build_agent_navigation(state)
    collapsed = flatten_navigation(roots, set())
    assert [node.kind for _, node in collapsed] == ["agent", "agent", "agent"]
    expanded = flatten_navigation(roots, {"agent:a0", "agent:a0:wave:1"})
    assert any(node.kind == "turn" for _, node in expanded)


def test_active_agent_and_leaf_tracking() -> None:
    state = _state(
        _event(3, "oleada_start", oleada=1, active=["a0", "a1"]),
        _event(4, "agent_turn_start", agent_id="a0", oleada=1, turn=0),
        _event(
            5,
            "paper_integration_completed",
            agent_id="a0",
            paper_id="p1",
            title="P1",
            oleada=1,
            turn=0,
        ),
    )
    assert active_agent_id(state) == "a0"
    assert active_leaf_entry(state, "a0") == "paper:read:a0:1:0:p1"


def test_unknown_events_are_ignored_by_navigation() -> None:
    state = _state(_event(3, "future_event", payload={"x": 1}))
    roots = build_agent_navigation(state)
    assert len(flatten_navigation(roots, {"agent:a0"})) == 3
