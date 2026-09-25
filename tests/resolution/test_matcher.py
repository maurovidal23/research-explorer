"""Tests for deterministic title/author/year scoring."""

from __future__ import annotations

import pytest

from research_explorer.resolution.matcher import (
    MatchThresholds,
    author_overlap,
    hard_gate_failures,
    normalize_title,
    score_candidate,
    surname,
    title_similarity,
    title_tokens,
    year_matches,
)
from research_explorer.resolution.models import BibliographicEntry, RejectReason


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Attention Is All You Need!", "attention is all you need"),
        ("  Spaced   out\t title  ", "spaced out title"),
        ("Hyphen-ated: punctuation?", "hyphen ated punctuation"),
    ],
)
def test_normalize_title(raw: str, expected: str) -> None:
    assert normalize_title(raw) == expected


def test_title_tokens_drop_stop_words() -> None:
    assert title_tokens("The Origin of Extracellular Fields") == {
        "origin",
        "extracellular",
        "fields",
    }


def test_title_similarity_exact_after_normalization() -> None:
    assert title_similarity("Attention Is All You Need!", "attention is all you need") == 1.0


def test_title_similarity_empty_inputs() -> None:
    assert title_similarity("", "x") == 0.0
    assert title_similarity("x", "") == 0.0


def test_title_similarity_partial() -> None:
    score = title_similarity("Deep residuals for image recognition", "Deep residual learning")
    assert 0.0 < score < 1.0


def test_surname_formats() -> None:
    assert surname("Ashish Vaswani") == "vaswani"
    assert surname("Vaswani, Ashish") == "vaswani"
    assert surname("  O'Connor, John  ") == "oconnor"
    assert surname("") == ""


def test_author_overlap_fraction_of_expected() -> None:
    expected = ["Vaswani", "Shazeer", "Uszkoreit"]
    actual = ["A. Vaswani", "N. Shazeer", "Someone Else"]
    assert author_overlap(expected, actual) == pytest.approx(2 / 3)


def test_author_overlap_empty_sides() -> None:
    assert author_overlap([], ["X"]) == 0.0
    assert author_overlap(["X"], []) == 0.0


def test_year_matches_tolerance() -> None:
    assert year_matches(2017, 2019, tolerance=2)
    assert not year_matches(2017, 2020, tolerance=2)
    assert year_matches(None, 2020, tolerance=2)
    assert year_matches(2017, None, tolerance=2)


def test_score_candidate_renormalizes_over_available_evidence() -> None:
    entry = BibliographicEntry(title="Attention is all you need")
    thresholds = MatchThresholds()
    score = score_candidate(entry, "Attention is all you need", [], None, thresholds)
    assert score.title_checked
    assert not score.authors_checked
    assert not score.year_checked
    assert score.confidence == pytest.approx(score.title_similarity)
    assert score.confidence == pytest.approx(1.0)


def test_score_candidate_full_evidence() -> None:
    entry = BibliographicEntry(
        title="Attention is all you need",
        authors=["Vaswani"],
        year=2017,
    )
    thresholds = MatchThresholds()
    score = score_candidate(entry, "Attention is all you need", ["A. Vaswani"], 2017, thresholds)
    expected = 0.6 * 1.0 + 0.25 * 1.0 + 0.15 * 1.0
    assert score.confidence == pytest.approx(expected)
    assert score.year_ok


def test_score_candidate_year_failure_lowers_confidence() -> None:
    entry = BibliographicEntry(title="Attention is all you need", year=2017)
    thresholds = MatchThresholds()
    score = score_candidate(entry, "Attention is all you need", [], 2020, thresholds)
    expected = (0.6 * 1.0 + 0.15 * 0.0) / 0.75
    assert score.confidence == pytest.approx(expected)
    assert not score.year_ok


def test_score_candidate_no_overlap_returns_zero() -> None:
    entry = BibliographicEntry()
    score = score_candidate(entry, "", [], None, MatchThresholds())
    assert score.confidence == 0.0


def test_hard_gate_failures_ordered() -> None:
    entry = BibliographicEntry(title="Attention is all you need", authors=["Vaswani"], year=2017)
    thresholds = MatchThresholds()
    score = score_candidate(entry, "Completely different words here", ["Shazeer"], 2020, thresholds)
    failures = hard_gate_failures(score, thresholds)
    assert failures == [
        RejectReason.TITLE_MISMATCH,
        RejectReason.AUTHOR_MISMATCH,
        RejectReason.YEAR_MISMATCH,
    ]


def test_hard_gate_failures_empty_when_passing() -> None:
    entry = BibliographicEntry(title="Attention is all you need", authors=["Vaswani"], year=2017)
    score = score_candidate(entry, "Attention is all you need", ["A. Vaswani"], 2018, MatchThresholds())
    assert hard_gate_failures(score, MatchThresholds()) == []


def test_from_config_uses_partial_attributes() -> None:
    class Cfg:
        min_title_similarity = 0.9

    thresholds = MatchThresholds.from_config(Cfg())
    assert thresholds.min_title_similarity == 0.9
    assert thresholds.min_author_overlap == MatchThresholds.min_author_overlap
    assert thresholds.min_confidence == MatchThresholds.min_confidence
    assert thresholds.year_tolerance == MatchThresholds.year_tolerance
