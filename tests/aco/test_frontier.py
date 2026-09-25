"""Tests for the shared frontier: claims, eligibility, and non-destructive reuse."""

from __future__ import annotations

from research_explorer.aco.frontier import SharedFrontier


def _frontier(*pids: str) -> SharedFrontier:
    f = SharedFrontier()
    f.add(list(pids), source="seed", mode="ref")
    return f


def test_add_dedup_and_sources() -> None:
    f = _frontier("a", "b")
    f.add(["a", "c"], source="x", mode="cites")
    assert f.papers == ["a", "b", "c"]
    assert f.sources["c"] == ("x", "cites")
    # source not overwritten for existing paper
    assert f.sources["a"] == ("seed", "ref")


def test_add_excludes_visited() -> None:
    f = SharedFrontier()
    f.add(["a", "b"], source="s", mode="ref", exclude={"a"})
    assert f.papers == ["b"]


def test_remove_clears_claim() -> None:
    f = _frontier("a")
    f.claim_for("a", "agent1")
    f.remove("a")
    assert "a" not in f.papers
    assert not f.is_claimed("a")


def test_claim_for_acquires_and_blocks_other() -> None:
    f = _frontier("a")
    assert f.claim_for("a", "agent1") is True
    assert f.is_claimed("a")
    assert f.claimed_by("a") == "agent1"
    # a different agent cannot claim while agent1 holds it
    assert f.claim_for("a", "agent2") is False
    # same agent re-claim is allowed (idempotent)
    assert f.claim_for("a", "agent1") is True


def test_empty_claim_on_missing_paper() -> None:
    f = _frontier("a")
    assert f.claim_for("zz", "agent1") is False


def test_release_returns_paper_to_pool() -> None:
    f = _frontier("a")
    f.claim_for("a", "agent1")
    assert not f.eligible()
    # releasing by a different agent is a no-op
    f.release("a", "agent2")
    assert f.is_claimed("a")
    f.release("a", "agent1")
    assert not f.is_claimed("a")
    assert f.eligible() == ["a"]


def test_eligible_excludes_visited_and_claimed() -> None:
    f = _frontier("a", "b", "c")
    f.claim_for("b", "agent1")
    # 'a' excluded as visited
    assert set(f.eligible(exclude={"a"})) == {"c"}
    assert set(f.eligible()) == {"a", "c"}


def test_best_ignores_claimed_and_visited() -> None:
    f = _frontier("a", "b")
    f.set_score("a", 0.9)
    f.set_score("b", 0.8)
    assert f.best() == "a"
    f.claim_for("a", "agent1")
    assert f.best() == "b"
    assert f.best(exclude={"b"}) is None


def test_claims_do_not_erase_discovered_paths() -> None:
    # A claim blocks the node only during the concurrent window; the already
    # revealed neighbors stay in the frontier (reusable paths preserved).
    f = SharedFrontier()
    f.add(["meta"], source="seed", mode="ref")
    f.add(["meta-ref-1", "meta-cit-1"], source="meta", mode="ref")
    f.claim_for("meta", "agent1")
    assert "meta-ref-1" in f.papers
    assert "meta-cit-1" in f.papers
    assert set(f.eligible()) == {"meta-ref-1", "meta-cit-1"}
