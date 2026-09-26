"""Tests for the arXiv provider (parsing logic, no network)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from research_explorer.providers.arxiv import ArxivProvider

_ENTRY_XML = """<entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <id>http://arxiv.org/abs/2301.00001v1</id>
  <updated>2023-01-02T00:00:00Z</updated>
  <published>2023-01-01T00:00:00Z</published>
  <title>Attention Is All\n  You Need</title>
  <summary>The dominant sequence transduction models are based on complex recurrent networks.</summary>
  <author><name>Author A</name></author>
  <author><name>Author B</name></author>
  <arxiv:comment>15 pages</arxiv:comment>
  <arxiv:journal_ref>Nature 2023</arxiv:journal_ref>
  <arxiv:doi>10.1/xxx</arxiv:doi>
  <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  <category term="cs.CL"/>
  <category term="cs.AI"/>
</entry>"""

_ENTRY_NO_META_XML = """<entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <id>http://arxiv.org/abs/cs.AI/0703001</id>
  <published>2007-03-01T00:00:00Z</published>
  <title>Old style paper</title>
  <summary>An abstract.</summary>
  <author><name>Solo Author</name></author>
</entry>"""


def _entry(xml: str) -> ET.Element:
    return ET.fromstring(xml)


def test_format_id_auto_arxiv_prefix() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    assert p._format_id("arXiv:2301.00001", "auto") == "2301.00001"


def test_format_id_auto_plain() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    assert p._format_id("2301.00001", "auto") == "2301.00001"


def test_format_id_explicit() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    assert p._format_id("2301.00001", "arxiv") == "2301.00001"


def test_build_search_query_plain_text() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    assert p._build_search_query("machine learning") == "all:machine learning"


def test_build_search_query_with_prefix() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    assert p._build_search_query("ti:neural") == "ti:neural"
    assert p._build_search_query("cat:cs.AI") == "cat:cs.AI"
    assert p._build_search_query("au:smith") == "au:smith"


def test_extract_id_strips_version() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    assert p._extract_id(_entry(_ENTRY_XML)) == "2301.00001"


def test_extract_id_old_style() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    assert p._extract_id(_entry(_ENTRY_NO_META_XML)) == "cs.AI/0703001"


def test_parse_summary() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    s = p._parse_summary(_entry(_ENTRY_XML))
    assert s.id == "2301.00001"
    assert s.doi == "10.1/xxx"
    assert s.title == "Attention Is All You Need"
    assert s.year == 2023
    assert s.authors == ["Author A", "Author B"]
    assert s.citation_count is None
    assert s.abstract == (
        "The dominant sequence transduction models are based on complex recurrent networks."
    )
    assert s.provider == "arxiv"


def test_parse_summary_no_meta() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    s = p._parse_summary(_entry(_ENTRY_NO_META_XML))
    assert s.id == "cs.AI/0703001"
    assert s.doi is None
    assert s.year == 2007
    assert s.authors == ["Solo Author"]
    assert s.abstract == "An abstract."


def test_parse_paper() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    paper = p._parse_paper(_entry(_ENTRY_XML))
    assert paper.id == "2301.00001"
    assert paper.doi == "10.1/xxx"
    assert paper.title == "Attention Is All You Need"
    assert paper.year == 2023
    assert paper.authors == ["Author A", "Author B"]
    assert paper.citation_count is None
    assert paper.provider == "arxiv"
    assert paper.external_ids == {"DOI": "10.1/xxx"}
    assert paper.references == []
    assert paper.citations == []
    assert paper.fields_of_study[0] == "cs.CL"
    assert "cs.AI" in paper.fields_of_study


def test_parse_paper_no_meta() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    paper = p._parse_paper(_entry(_ENTRY_NO_META_XML))
    assert paper.id == "cs.AI/0703001"
    assert paper.doi is None
    assert paper.external_ids == {}
    assert paper.fields_of_study == []
    assert paper.references == []
    assert paper.citations == []


async def test_get_references_and_citations_are_empty() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    assert await p.get_references("2301.00001") == []
    assert await p.get_citations("2301.00001") == []


async def test_atom_client_requests_atom_xml(tmp_path) -> None:
    p = ArxivProvider(cache_dir=str(tmp_path))
    try:
        assert p.client.headers["accept"] == "application/atom+xml"
    finally:
        await p.aclose()


# ---- Full-text + bibliography parsing (no network) --------------------------

_HTML = """<html><head><title>X</title><style>a{}</style></head><body>
<section><h1>1 Introduction</h1><p>This paper studies attention.</p></section>
<section class="ltx_bibliography" id="bib">
<h2 class="ltx_title ltx_title_bibliography">References</h2>
<ul class="ltx_biblist">
<li class="ltx_bibitem" id="bib.bib1">
<span class="ltx_tag">Vaswani et al. (2017)</span>
<span class="ltx_bibblock">Vaswani, A. Attention is all you need. arXiv:1706.03762, 2017.</span>
</li>
<li class="ltx_bibitem" id="bib.bib2">
<span class="ltx_tag">Angrist (2009)</span>
<span class="ltx_bibblock">Angrist, J. Mostly harmless econometrics. 2009.</span>
</li>
</ul>
</section>
<script>var x=1;</script>
</body></html>"""


def test_html_to_text_and_refs() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    text, refs = p._html_to_text_and_refs(_HTML, max_chars=10000, ref_limit=100)
    assert "Attention is all you need" in text
    assert "1 Introduction" in text
    assert "<" not in text  # tags stripped
    assert len(refs) == 2
    assert "Vaswani" in refs[0]
    assert "arXiv:1706.03762" in refs[0]
    assert "Angrist" in refs[1]


def test_html_to_text_truncates() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    text, _ = p._html_to_text_and_refs(_HTML, max_chars=50, ref_limit=100)
    assert len(text) <= 50


def test_html_to_text_no_bibliography() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    text, refs = p._html_to_text_and_refs(
        "<html><body><p>No refs here.</p></body></html>",
        max_chars=10000, ref_limit=100,
    )
    assert "No refs here." in text
    assert refs == []


def test_extract_bib_entries_strips_tags() -> None:
    p = ArxivProvider.__new__(ArxivProvider)
    entries = p._extract_bib_entries(_HTML, ref_limit=100)
    assert len(entries) == 2
    assert "<span" not in entries[0]
    assert "1706.03762" in entries[0]


def test_supports_fulltext_flag() -> None:
    assert ArxivProvider.supports_fulltext is True
