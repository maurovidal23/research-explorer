"""Tests for DOI and arXiv identifier normalization."""

from __future__ import annotations

import pytest

from research_explorer.resolution.resolver import (
    alias_key_for_arxiv,
    alias_key_for_doi,
    is_valid_arxiv,
    is_valid_doi,
    normalize_arxiv,
    normalize_doi,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.1038/nrn3241", "10.1038/nrn3241"),
        (" 10.1038/nrn3241 ", "10.1038/nrn3241"),
        ("10.1038/NRN3241", "10.1038/nrn3241"),
        ("https://doi.org/10.1038/nrn3241", "10.1038/nrn3241"),
        ("http://doi.org/10.1038/nrn3241", "10.1038/nrn3241"),
        ("https://dx.doi.org/10.1038/nrn3241", "10.1038/nrn3241"),
        ("doi.org/10.1038/nrn3241", "10.1038/nrn3241"),
        ("doi:10.1038/nrn3241", "10.1038/nrn3241"),
        ("DOI: 10.1038/nrn3241", "10.1038/nrn3241"),
        ("info:doi/10.1038/nrn3241", "10.1038/nrn3241"),
        ("https://doi.org/DOI:10.1038/nrn3241", "10.1038/nrn3241"),
    ],
)
def test_normalize_doi(raw: str, expected: str) -> None:
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2301.00001", "2301.00001"),
        ("arXiv:2301.00001", "2301.00001"),
        ("arxiv:2301.00001v2", "2301.00001"),
        ("https://arxiv.org/abs/2301.00001v3", "2301.00001"),
        ("http://arxiv.org/abs/2301.00001", "2301.00001"),
        ("arxiv.org/abs/2301.00001v10", "2301.00001"),
        ("arXiv: 2301.00001 ", "2301.00001"),
        ("cs/0112017", "cs/0112017"),
        ("arXiv:cs/0112017v1", "cs/0112017"),
        ("https://arxiv.org/abs/cs/0112017v2", "cs/0112017"),
        ("math.GT/0309136", "math.gt/0309136"),
        ("hep-th/9901001", "hep-th/9901001"),
    ],
)
def test_normalize_arxiv(raw: str, expected: str) -> None:
    assert normalize_arxiv(raw) == expected


@pytest.mark.parametrize("value", ["10.1038/nrn3241", "10.5555/3295222.3295349"])
def test_valid_doi(value: str) -> None:
    assert is_valid_doi(value)


@pytest.mark.parametrize(
    "value", ["", "not-a-doi", "10.1038", "11.1038/nrn3241", "doi:10.1038/x"]
)
def test_invalid_doi(value: str) -> None:
    assert not is_valid_doi(value)


@pytest.mark.parametrize(
    "value",
    ["2301.00001", "2301.00001v2", "cs/0112017", "math.gt/0309136", "hep-th/9901001"],
)
def test_valid_arxiv(value: str) -> None:
    assert is_valid_arxiv(value)


@pytest.mark.parametrize(
    "value",
    ["", "2301.000", "2301.000011", "not-an-id", "2301_00001", "cs/01120"],
)
def test_invalid_arxiv(value: str) -> None:
    assert not is_valid_arxiv(value)


def test_alias_keys() -> None:
    assert alias_key_for_doi("https://doi.org/10.1038/NRN3241") == "doi:10.1038/nrn3241"
    assert alias_key_for_arxiv("arXiv:2301.00001v2") == "arxiv:2301.00001"
    assert alias_key_for_doi("junk") is None
    assert alias_key_for_arxiv("junk") is None


def test_normalization_is_idempotent() -> None:
    assert normalize_doi(normalize_doi("doi:10.1/x")) == "10.1/x"
    assert normalize_arxiv(normalize_arxiv("arXiv:2301.00001v2")) == "2301.00001"
