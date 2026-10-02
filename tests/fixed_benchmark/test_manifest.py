from pathlib import Path

from research_explorer.fixed_benchmark.io import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks" / "research_explorer_v1" / "manifest.json"


def test_fixed_manifest_pins_25_complete_sources() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest.version.endswith("-draft")
    assert len(manifest.papers) == 25
    assert len({paper.paper_id for paper in manifest.papers}) == 25
    assert sum(manifest.categories.values()) == 100
    assert sum(manifest.difficulties.values()) == 100
    assert sum(manifest.correct_option_counts.values()) == 100
    assert all(paper.versioned_arxiv_id in paper.pdf_url for paper in manifest.papers)
    assert all(len(paper.source_sha256) == 64 for paper in manifest.papers)
    assert all(len(paper.text_sha256) == 64 for paper in manifest.papers)
    assert all(paper.page_count > 0 for paper in manifest.papers)
    assert manifest.expected_question_count == 2500
    assert manifest.bank_sha256 == ""
