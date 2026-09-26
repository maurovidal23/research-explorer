"""STAB-3: frontier retention, bounded retry, and exception-safe release."""

from __future__ import annotations

from research_explorer.aco.frontier import SharedFrontier


def _frontier(*pids: str) -> SharedFrontier:
    f = SharedFrontier()
    f.add(list(pids), source="seed", mode="ref")
    return f


def test_transient_failure_retains_candidate_and_releases_claim() -> None:
    f = _frontier("a")
    assert f.claim_for("a", "agent1") is True
    f.record_transient_failure("a", "agent1", turn=3)
    assert "a" in f.papers
    assert not f.is_claimed("a")
    assert f.attempt_count("a") == 1


def test_transient_failure_delays_eligibility() -> None:
    f = _frontier("a")
    f.record_transient_failure("a", "agent1", turn=3)
    assert f.eligible(turn=3) == []
    assert f.eligible(turn=4) == ["a"]


def test_attempts_are_bounded() -> None:
    f = _frontier("a")
    f.max_attempts = 2
    f.record_transient_failure("a", "agent1", turn=0)
    f.record_transient_failure("a", "agent1", turn=2)
    assert f.is_exhausted("a")
    assert "a" in f.papers  # retained, not evicted
    assert f.eligible(turn=99) == []


def test_definitive_absence_removes_candidate() -> None:
    f = _frontier("a")
    f.record_absence("a")
    assert "a" not in f.papers


def test_release_all_releases_only_that_agent() -> None:
    f = _frontier("a", "b", "c")
    f.claim_for("a", "agent1")
    f.claim_for("b", "agent1")
    f.claim_for("c", "agent2")
    f.release_all("agent1")
    assert not f.is_claimed("a")
    assert not f.is_claimed("b")
    assert f.is_claimed("c")


def test_remove_clears_retry_bookkeeping() -> None:
    f = _frontier("a")
    f.record_transient_failure("a", "agent1", turn=0)
    f.remove("a")
    assert f.attempt_count("a") == 0
    assert f.eligible(turn=10) == []
