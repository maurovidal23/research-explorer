"""Shared test fixtures."""

from __future__ import annotations

import pytest

from research_explorer.graph.models import Paper, PaperSummary


@pytest.fixture
def sample_summary_s2() -> PaperSummary:
    return PaperSummary(
        id="da82f8e6abc",
        doi="10.1038/nrn3241",
        title="The origin of extracellular fields",
        year=2012,
        authors=["Buzsáki", "Anastassiou", "Koch"],
        citation_count=4061,
        abstract="Neuronal activity in the brain...",
        provider="semantic_scholar",
    )


@pytest.fixture
def sample_paper_s2(sample_summary_s2: PaperSummary) -> Paper:
    return Paper(
        **sample_summary_s2.model_dump(),
        references=[
            PaperSummary(
                id="ref1",
                title="A reference paper",
                year=2005,
                provider="semantic_scholar",
            )
        ],
        citations=[
            PaperSummary(
                id="cit1",
                title="A citing paper",
                year=2015,
                provider="semantic_scholar",
            )
        ],
        external_ids={"DOI": "10.1038/nrn3241", "PubMed": "22595786"},
        fields_of_study=["Neuroscience"],
        tldr="Extracellular fields originate from neuronal currents.",
    )
