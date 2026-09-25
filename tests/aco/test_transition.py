"""Unit tests for the ACO transition-weighting and eta formulas.

Covers: transition weight formula, eta component combination, caste direction
weights, weighted vs epsilon selection, tau/beta sensitivity, deterministic
RNG injection, and probability normalization.
"""

from __future__ import annotations

import random

import pytest

from research_explorer.aco.transition import (
    IMPACTO,
    MIXTO,
    Candidate,
    EtaComponents,
    caste_direction_weights,
    combine_eta,
    mode_direction_modifier,
    select_candidate,
    transition_weight,
)


class _FixedRng:
    """Deterministic RNG stub: random() returns rand, choice/choices return pick."""

    def __init__(self, rand: float = 0.0, pick: int = 0):
        self._rand = rand
        self._pick = pick

    def random(self) -> float:
        return self._rand

    def choice(self, seq):
        return seq[self._pick]

    def choices(self, seq, weights=None, k: int = 1):
        return [seq[self._pick]]


def _cand(pid: str, tau: float = 1.0, eta: float = 0.5, mode: str = "ref", dm: float = 1.0) -> Candidate:
    return Candidate(pid=pid, src="s", mode=mode, tau=tau, eta=eta, dir_modifier=dm)


# ---- transition_weight -----------------------------------------------------


def test_transition_weight_formula() -> None:
    # tau^alpha * eta^beta * dir
    w = transition_weight(tau=2.0, alpha=2.0, eta=0.5, beta=1.0, dir_modifier=0.8)
    assert w == pytest.approx((2.0 ** 2) * (0.5 ** 1) * 0.8)


def test_transition_weight_zero_values() -> None:
    assert transition_weight(tau=0.0, alpha=1.0, eta=0.5, beta=1.0, dir_modifier=1.0) > 0.0
    assert transition_weight(tau=2.0, alpha=1.0, eta=0.0, beta=1.0, dir_modifier=1.0) == 0.0
    assert transition_weight(tau=2.0, alpha=1.0, eta=0.5, beta=1.0, dir_modifier=0.0) == 0.0


def test_transition_weight_eta_power_zero_is_neutral() -> None:
    # beta=0 -> eta^0 = 1, so weight independent of eta
    w1 = transition_weight(1.0, 1.0, 0.1, 0.0, 1.0)
    w2 = transition_weight(1.0, 1.0, 0.9, 0.0, 1.0)
    assert w1 == w2


# ---- combine_eta -----------------------------------------------------------


def test_combine_eta_sigmoid_range() -> None:
    v = combine_eta(1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.2, 0.2, 0.1, 0.2, True)
    assert 0.0 < v < 1.0


def test_combine_eta_monotonic_in_components() -> None:
    low = combine_eta(0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.2, 0.2, 0.1, 0.2, True)
    high = combine_eta(1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.2, 0.2, 0.1, 0.2, True)
    assert high > low


def test_combine_eta_llm_only_used_when_enabled() -> None:
    off = combine_eta(0.5, 0.5, 0.5, 0.5, 1.0, 0.2, 0.2, 0.2, 0.2, 0.5, False)
    on = combine_eta(0.5, 0.5, 0.5, 0.5, 1.0, 0.2, 0.2, 0.2, 0.2, 0.5, True)
    assert on > off


def test_combine_eta_respects_component_weights() -> None:
    # w_citas dominates -> citation component drives the value most
    high_cit = combine_eta(0.5, 1.0, 0.5, 0.5, 0.0, 0.2, 0.6, 0.1, 0.1, 0.0, False)
    low_cit = combine_eta(0.5, 0.0, 0.5, 0.5, 0.0, 0.2, 0.6, 0.1, 0.1, 0.0, False)
    assert high_cit > low_cit


# ---- caste direction weights ----------------------------------------------


def test_caste_direction_is_normalized() -> None:
    for caste in ("fundaciones", IMPACTO, MIXTO):
        ref, cites = caste_direction_weights(caste, 0.7, 0.3)
        assert ref + cites == pytest.approx(1.0)


def test_fundaciones_prefers_references() -> None:
    ref, cites = caste_direction_weights("fundaciones", 0.7, 0.3)
    assert ref > cites


def test_impacto_prefers_cited_by() -> None:
    ref, cites = caste_direction_weights(IMPACTO, 0.7, 0.3)
    assert cites > ref


def test_mixto_uses_config_baseline() -> None:
    ref, cites = caste_direction_weights(MIXTO, 0.7, 0.3)
    assert ref == pytest.approx(0.7)
    assert cites == pytest.approx(0.3)


def test_caste_ordering() -> None:
    ref_f, _ = caste_direction_weights("fundaciones", 0.7, 0.3)
    ref_m, _ = caste_direction_weights(MIXTO, 0.7, 0.3)
    ref_i, _ = caste_direction_weights(IMPACTO, 0.7, 0.3)
    assert ref_f > ref_m > ref_i


def test_mode_direction_modifier() -> None:
    assert mode_direction_modifier("ref", 0.8, 0.2) == 0.8
    assert mode_direction_modifier("cites", 0.8, 0.2) == 0.2


# ---- select_candidate: weighted draw ---------------------------------------


def test_select_candidate_weighted_probabilities_normalized() -> None:
    cands = [_cand("a", tau=1.0, eta=0.9), _cand("b", tau=1.0, eta=0.1)]
    # epsilon=0 -> weighted; rng.rand=0.5 not < 0
    sel = select_candidate(cands, alpha=1.0, beta=2.0, epsilon=0.0, rng=_FixedRng(rand=0.5, pick=0))
    assert sel is not None
    assert not sel.epsilon_branch
    assert sum(sel.probabilities.values()) == pytest.approx(1.0)
    # high-eta candidate gets probability > 0.5
    assert sel.probabilities["a"] > sel.probabilities["b"]


def test_select_candidate_empty_returns_none() -> None:
    assert select_candidate([], alpha=1.0, beta=1.0, epsilon=0.1, rng=random.Random(0)) is None


def test_select_candidate_falls_back_to_uniform_when_all_zero() -> None:
    cands = [_cand("a", eta=0.0), _cand("b", eta=0.0)]
    sel = select_candidate(cands, alpha=1.0, beta=1.0, epsilon=0.0, rng=_FixedRng(0.5, 0))
    assert sel is not None
    assert sel.probabilities["a"] == pytest.approx(sel.probabilities["b"])


# ---- select_candidate: tau / beta sensitivity -------------------------------


def test_select_candidate_tau_sensitivity() -> None:
    # same eta, different tau -> higher tau (on a) wins higher probability
    cands = [_cand("a", tau=5.0, eta=0.5), _cand("b", tau=1.0, eta=0.5)]
    sel = select_candidate(cands, alpha=1.0, beta=1.0, epsilon=0.0, rng=_FixedRng(0.5, 0))
    assert sel.probabilities["a"] > sel.probabilities["b"]
    # with alpha=0 tau is ignored -> equal probabilities
    sel0 = select_candidate(cands, alpha=0.0, beta=1.0, epsilon=0.0, rng=_FixedRng(0.5, 0))
    assert sel0.probabilities["a"] == pytest.approx(0.5)
    assert sel0.probabilities["b"] == pytest.approx(0.5)


def test_select_candidate_beta_sensitivity() -> None:
    cands = [_cand("a", eta=0.9), _cand("b", eta=0.1)]
    sel_low = select_candidate(cands, alpha=1.0, beta=0.5, epsilon=0.0, rng=_FixedRng(0.5, 0))
    sel_high = select_candidate(cands, alpha=1.0, beta=5.0, epsilon=0.0, rng=_FixedRng(0.5, 0))
    # higher beta amplifies the eta gap between a and b
    gap_low = sel_low.probabilities["a"] - sel_low.probabilities["b"]
    gap_high = sel_high.probabilities["a"] - sel_high.probabilities["b"]
    assert gap_high > gap_low


def test_select_candidate_direction_modifier_affects_weight() -> None:
    cands = [_cand("ref1", eta=0.5, mode="ref", dm=0.9), _cand("cites1", eta=0.5, mode="cites", dm=0.1)]
    sel = select_candidate(cands, alpha=1.0, beta=1.0, epsilon=0.0, rng=_FixedRng(0.5, 0))
    # ref direction (higher dir modifier) dominates for equal tau/eta
    assert sel.probabilities["ref1"] > sel.probabilities["cites1"]


# ---- select_candidate: epsilon / RNG ---------------------------------------


def test_epsilon_branch_ignores_tau_and_eta() -> None:
    cands = [_cand("a", tau=100.0, eta=0.99), _cand("b", tau=1.0, eta=0.01)]
    sel = select_candidate(cands, alpha=2.0, beta=5.0, epsilon=0.5, rng=_FixedRng(rand=0.1, pick=1))
    assert sel is not None
    assert sel.epsilon_branch
    # uniform: both probabilities equal despite tau/eta differences
    assert sel.probabilities["a"] == pytest.approx(sel.probabilities["b"])
    # the chosen candidate is the uniform pick (b), not the weighted-favored one
    assert sel.chosen == "b"


def test_epsilon_branch_deterministic_with_injected_rng() -> None:
    cands = [_cand("a"), _cand("b"), _cand("c")]
    s1 = select_candidate(cands, 1.0, 1.0, 0.5, _FixedRng(rand=0.0, pick=2))
    s2 = select_candidate(cands, 1.0, 1.0, 0.5, _FixedRng(rand=0.0, pick=2))
    assert s1.chosen == "c"
    assert s1.chosen == s2.chosen


def test_real_rng_is_reproducible() -> None:
    cands = [_cand("a"), _cand("b")]
    r1 = random.Random(42)
    r2 = random.Random(42)
    seq1 = [select_candidate(cands, 1.0, 1.0, 0.0, r1).chosen for _ in range(20)]
    seq2 = [select_candidate(cands, 1.0, 1.0, 0.0, r2).chosen for _ in range(20)]
    assert seq1 == seq2


# ---- EtaComponents ---------------------------------------------------------


def test_eta_components_as_dict_and_components() -> None:
    eta = EtaComponents(sim=0.7, citations=0.2, recency=0.3, confidence=0.5,
                        llm=0.1, weights={"w_sim": 0.6}, value=0.8)
    comps = eta.components()
    assert comps["sim"] == 0.7
    assert comps["confidence"] == 0.5
    assert "llm" in comps
    d = eta.as_dict()
    assert d["value"] == 0.8
    assert d["weights"] == {"w_sim": 0.6}
