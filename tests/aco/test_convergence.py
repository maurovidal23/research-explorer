"""Tests for the convergence checker."""

from research_explorer.aco.convergence import ConvergenceChecker
from research_explorer.config import BudgetConfig, Config, ConvergenceConfig


def test_budget_fetches_stop() -> None:
    cfg = Config(budget=BudgetConfig(type="fetches", max_fetches=100))
    checker = ConvergenceChecker(cfg)
    assert not checker.should_stop(0.5, 50, 1.0)
    assert checker.should_stop(0.5, 100, 1.0)
    assert checker.should_stop(0.5, 150, 1.0)


def test_quality_plateau() -> None:
    cfg = Config(
        convergence=ConvergenceConfig(plateau_T=3, epsilon=0.01),
        budget=BudgetConfig(type="convergence", max_fetches=10000),
    )
    checker = ConvergenceChecker(cfg)
    # Add quality values that plateau
    for _ in range(5):
        checker.should_stop(0.5, 0, 1.0)
    # 5th call: history[-1] - history[-3] = 0 < 0.01 -> stop
    assert checker.should_stop(0.5, 0, 1.0)


def test_pheromone_concentration() -> None:
    cfg = Config(
        convergence=ConvergenceConfig(theta=5.0),
        budget=BudgetConfig(type="convergence", max_fetches=10000),
    )
    checker = ConvergenceChecker(cfg)
    assert not checker.should_stop(0.5, 0, 3.0)
    assert checker.should_stop(0.5, 0, 6.0)


def test_no_stop_early() -> None:
    cfg = Config(
        convergence=ConvergenceConfig(plateau_T=10, epsilon=0.01),
        budget=BudgetConfig(type="convergence", max_fetches=10000),
    )
    checker = ConvergenceChecker(cfg)
    for i in range(5):
        assert not checker.should_stop(0.1 * i, 0, 1.0)  # improving quality
