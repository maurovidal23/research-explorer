"""Candidate transition weighting for ACO traversal.

Implements the two-level move model from docs/model.md:

  1. Eta(v) is a weighted combination of normalized components:
        eta(v) = sigmoid(w_sim*sim + w_cites*cit + w_rec*rec + w_conf*conf
                          + w_llm*llm)
     Components (each in [0, 1], normalized where needed):
        sim       normalized semantic similarity to the seed
        cit       normalized citation impact (seminality)
        rec       normalized recency
        conf      provider/verification confidence
        llm       optional LLM priority (only used when the LLM-eta mode is on)
  2. Direction weights are adjusted per caste:
        fundaciones favors the references (outgoing) direction
        impacto favors the cited-by (incoming) direction
        mixto uses the config baseline (balanced)
  3. A candidate's transition weight is:
        weight(v) = tau_d(src, v)^alpha * eta(v)^beta * dir_modifier(mode)
     and the selection probability is weight(v) / sum(weights).
  4. With probability epsilon the move is purely exploratory (uniform over the
     eligible candidates), ignoring tau; this uses an injectable RNG.

All transitions are pure functions of their inputs so the formulas are
deterministic and unit-testable.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

# How strongly a caste skews the config baseline toward its preferred direction.
CASTE_SHIFT = 0.5

FUNDACIONES = "fundaciones"
IMPACTO = "impacto"
MIXTO = "mixto"

_EPS = 1e-12


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _normalize_dir(ref: float, cites: float) -> tuple[float, float]:
    total = ref + cites
    if total <= _EPS:
        return 0.5, 0.5
    return ref / total, cites / total


def caste_direction_weights(
    caste: str, ref_weight: float, cites_weight: float
) -> tuple[float, float]:
    """Return the normalized (ref, cites) direction weights for a caste.

    Args:
        caste: one of "fundaciones", "impacto", "mixto".
        ref_weight: baseline references weight (config).
        cites_weight: baseline cited-by weight (config); typically 1 - ref_weight.

    Returns:
        A (ref, cites) tuple that sums to 1-epsilon normalization.
    """
    if caste == FUNDACIONES:
        ref = ref_weight + (1.0 - ref_weight) * CASTE_SHIFT
        cites = cites_weight * (1.0 - CASTE_SHIFT)
    elif caste == IMPACTO:
        ref = ref_weight * (1.0 - CASTE_SHIFT)
        cites = cites_weight + (1.0 - cites_weight) * CASTE_SHIFT
    else:  # mixto / unknown -> config baseline
        ref, cites = ref_weight, cites_weight
    return _normalize_dir(ref, cites)


@dataclass
class EtaComponents:
    """The per-candidate eta decomposition (all components normalized to [0,1])."""

    sim: float = 0.0
    citations: float = 0.0
    recency: float = 0.0
    confidence: float = 0.0
    llm: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    value: float = 0.0

    def components(self) -> dict[str, float]:
        return {
            "sim": round(self.sim, 6),
            "citations": round(self.citations, 6),
            "recency": round(self.recency, 6),
            "confidence": round(self.confidence, 6),
            "llm": round(self.llm, 6),
        }

    def as_dict(self) -> dict:
        return {
            "components": self.components(),
            "weights": dict(self.weights),
            "value": round(self.value, 6),
        }


def combine_eta(
    sim: float,
    citations: float,
    recency: float,
    confidence: float,
    llm: float,
    w_sim: float,
    w_cites: float,
    w_recency: float,
    w_confidence: float,
    w_llm: float,
    use_llm: bool,
) -> float:
    """Combine normalized eta components into the sigmoid-weighted value."""
    value = (
        w_sim * _clamp01(sim)
        + w_cites * _clamp01(citations)
        + w_recency * _clamp01(recency)
        + w_confidence * _clamp01(confidence)
    )
    if use_llm:
        value += w_llm * _clamp01(llm)
    return _sigmoid(value)


def transition_weight(
    tau: float, alpha: float, eta: float, beta: float, dir_modifier: float
) -> float:
    """weight(v) = tau^alpha * eta^beta * dir_modifier."""
    if tau <= 0.0:
        tau = _EPS
    if eta <= 0.0:
        eta = 0.0
    return max(0.0, tau ** alpha) * max(0.0, eta ** beta) * max(0.0, dir_modifier)


@dataclass
class Candidate:
    """One eligible frontier candidate with its transition-relevant factors."""

    pid: str
    src: str
    mode: str
    tau: float = 1.0
    eta: float = 0.5
    dir_modifier: float = 1.0


@dataclass
class Selection:
    """The outcome of a single candidate draw (greedy or epsilon)."""

    chosen: str
    src: str
    mode: str
    weights: dict[str, float] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)
    epsilon_branch: bool = False
    alpha: float = 0.0
    beta: float = 0.0
    tau: float = 0.0
    eta: float = 0.0
    dir_modifier: float = 0.0
    final_weight: float = 0.0
    chosen_probability: float = 0.0

    @property
    def chosen_tau(self) -> float:
        return self.tau

    def as_dict(self) -> dict:
        return {
            "chosen": self.chosen,
            "src": self.src,
            "mode": self.mode,
            "weights": {k: round(v, 8) for k, v in self.weights.items()},
            "probabilities": {k: round(v, 8) for k, v in self.probabilities.items()},
            "epsilon_branch": self.epsilon_branch,
            "alpha": self.alpha,
            "beta": self.beta,
            "tau": round(self.tau, 8),
            "eta": round(self.eta, 8),
            "dir_modifier": round(self.dir_modifier, 8),
            "final_weight": round(self.final_weight, 8),
            "chosen_probability": round(self.chosen_probability, 8),
        }


def select_candidate(
    candidates: Sequence[Candidate],
    alpha: float,
    beta: float,
    epsilon: float,
    rng: random.Random,
) -> Selection | None:
    """Draw a candidate with probability proportional to weights.

    With probability ``epsilon`` a candidate is chosen uniformly (pure
    exploration, ignoring tau and eta); otherwise the weighted draw is used.
    The RNG is injectable so the outcome is deterministic for tests.

    Returns None when there are no eligible candidates.
    """
    if not candidates:
        return None

    epsilon = max(0.0, epsilon)
    eps_branch = rng.random() < epsilon

    if eps_branch:
        chosen = rng.choice(list(candidates))
        weights = {c.pid: 1.0 for c in candidates}
        probs = {c.pid: 1.0 / len(candidates) for c in candidates}
        e = weights.get(chosen.pid, 1.0)
        p = probs.get(chosen.pid, 1.0 / len(candidates))
        return Selection(
            chosen=chosen.pid,
            src=chosen.src,
            mode=chosen.mode,
            weights=weights,
            probabilities=probs,
            epsilon_branch=True,
            alpha=alpha,
            beta=beta,
            tau=chosen.tau,
            eta=chosen.eta,
            dir_modifier=chosen.dir_modifier,
            final_weight=e,
            chosen_probability=p,
        )

    weights = {
        c.pid: transition_weight(c.tau, alpha, c.eta, beta, c.dir_modifier)
        for c in candidates
    }
    total = sum(weights.values())
    if total <= _EPS:
        weights = {c.pid: 1.0 for c in candidates}
        total = float(len(candidates))
    probs = {pid: w / total for pid, w in weights.items()}

    pids = list(candidates)
    chosen = rng.choices(pids, weights=[weights[c.pid] for c in pids], k=1)[0]
    chosen_w = weights.get(chosen.pid, 0.0)
    chosen_p = probs.get(chosen.pid, 0.0)
    return Selection(
        chosen=chosen.pid,
        src=chosen.src,
        mode=chosen.mode,
        weights=weights,
        probabilities=probs,
        epsilon_branch=False,
        alpha=alpha,
        beta=beta,
        tau=chosen.tau,
        eta=chosen.eta,
        dir_modifier=chosen.dir_modifier,
        final_weight=max(0.0, chosen_w),
        chosen_probability=chosen_p,
    )


def mode_direction_modifier(mode: str, ref_weight: float, cites_weight: float) -> float:
    """Return the direction modifier for a candidate's mode."""
    return ref_weight if mode == "ref" else cites_weight
