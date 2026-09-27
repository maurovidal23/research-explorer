"""Unit tests for the deterministic bibliography helpers (FRG-1).

Network-free: hashing, raw-entry trimming, identifier recovery, and the bounded
unstructured-bibliography segmentation used when a provider cannot supply entry
boundaries.
"""

from __future__ import annotations

from research_explorer.references.bibliography import (
    MAX_RAW_TEXT_CHARS,
    extract_arxiv_candidates,
    extract_doi_candidates,
    extract_pmid_candidates,
    provisional_reference_id,
    raw_entry_hash,
    segment_bibliography,
    sha256_text,
    source_content_hash,
    trim_raw_entry,
)


def test_source_content_hash_is_stable_and_sensitive_to_content() -> None:
    entries = ["Author A. Work one. 2020.", "Author B. Work two. 2021."]
    assert source_content_hash(entries) == source_content_hash(list(entries))
    assert source_content_hash(entries) != source_content_hash(entries[:1])
    assert len(sha256_text("x")) == 64


def test_raw_entry_hash_ignores_whitespace_but_not_content() -> None:
    assert raw_entry_hash("Author  A.\n Work one.") == raw_entry_hash(
        "Author A. Work one."
    )
    assert raw_entry_hash("Work one.") != raw_entry_hash("Work two.")


def test_trim_raw_entry_collapses_and_bounds_length() -> None:
    long = "word " * (MAX_RAW_TEXT_CHARS * 2)
    trimmed = trim_raw_entry(long)
    assert len(trimmed) == MAX_RAW_TEXT_CHARS
    assert "\n" not in trim_raw_entry("a\nb\t c", limit=100)
    assert trim_raw_entry("  padded  ") == "padded"


def test_provisional_reference_id_is_stable_and_provider_non_authoritative() -> None:
    source = source_content_hash(["e"])
    entry = raw_entry_hash("raw")
    prov = provisional_reference_id(source, entry)
    assert prov == provisional_reference_id(source, entry)
    assert prov.count(":") == 2
    assert prov.split(":")[0] == "bib"
    assert prov != provisional_reference_id(source, raw_entry_hash("other"))


def test_identifier_candidate_extraction() -> None:
    text = (
        "Author. Title. doi:10.1234/abc.def. arXiv:2106.09685v2. "
        "arXiv:cs.LG/0701001. PMID: 12345678."
    )
    assert extract_doi_candidates(text) == ["10.1234/abc.def"]
    assert set(extract_arxiv_candidates(text)) == {"2106.09685", "cs.LG/0701001"}
    assert extract_pmid_candidates(text) == ["12345678"]


def test_segment_bibliography_bracketed_entries() -> None:
    text = (
        "Body text with no entries.\nReferences\n"
        "[1] Author A. A sufficiently long first reference. 2020.\n"
        "[2] Author B. A sufficiently long second reference. 2021.\n"
    )
    entries = segment_bibliography(text)
    assert len(entries) == 2
    assert "first reference" in entries[0]
    assert "second reference" in entries[1]
    assert segment_bibliography(text, max_entries=1) == entries[:1]


def test_segment_bibliography_numbered_paragraph_fallback() -> None:
    text = (
        "Body.\nReferences\n"
        "1. Author A. A sufficiently long first reference. 2020.\n"
        "2. Author B. A sufficiently long second reference. 2021.\n"
    )
    entries = segment_bibliography(text)
    assert len(entries) == 2
    assert "first reference" in entries[0]


def test_segment_bibliography_without_heading_is_empty() -> None:
    assert segment_bibliography("Just body text, no bibliography at all.") == []
