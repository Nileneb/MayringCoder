"""Tests for paper ingestion pipeline: fetcher, chunker, wiki rules."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from unittest.mock import patch, MagicMock

import pytest

from src.memory.paper_fetcher import ArxivPaper, normalize_arxiv_id, fetch_arxiv, fetch_multiple
from src.memory.chunker import chunk_paper
from src.memory.wiki import (
    find_citation_pairs, find_keyword_overlap,
    find_shared_concepts, find_method_chains, find_dataset_pairs,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

_SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Attention Is All You Need</title>
    <summary>We propose a new simple network architecture, the Transformer.</summary>
    <published>2017-06-12T00:00:00Z</published>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <category term="cs.CL"/>
    <category term="cs.AI"/>
  </entry>
</feed>"""


@pytest.fixture
def sample_paper() -> ArxivPaper:
    return ArxivPaper(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        abstract="We propose a new simple network architecture, the Transformer.",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        categories=["cs.CL", "cs.AI"],
        published="2017-06-12",
        full_text=None,
    )


@pytest.fixture
def paper_with_sections() -> ArxivPaper:
    return ArxivPaper(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        abstract="Abstract text here.",
        authors=["Vaswani"],
        categories=["cs.CL"],
        published="2017-06-12",
        full_text="Introduction\nThis paper introduces the Transformer.\n\nMethods\nWe use self-attention.\n\nConclusion\nTransformers work well.",
    )


# ─── Fetcher Tests ────────────────────────────────────────────────────────────

def test_fetch_arxiv_returns_dataclass():
    """fetch_arxiv with mocked HTTP returns populated ArxivPaper."""
    import io
    mock_response = MagicMock()
    mock_response.read.return_value = _SAMPLE_XML
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        paper = fetch_arxiv("1706.03762")

    assert isinstance(paper, ArxivPaper)
    assert paper.arxiv_id == "1706.03762"
    assert paper.title == "Attention Is All You Need"
    assert "Vaswani" in paper.authors[0]
    assert "cs.CL" in paper.categories
    assert paper.full_text is None


def test_fetch_arxiv_normalizes_url():
    """normalize_arxiv_id strips URL and version suffix."""
    assert normalize_arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"
    assert normalize_arxiv_id("https://arxiv.org/abs/1706.03762v3") == "1706.03762"
    assert normalize_arxiv_id("arxiv:1706.03762") == "1706.03762"
    assert normalize_arxiv_id("1706.03762") == "1706.03762"


def test_normalize_arxiv_id_raises_on_garbage():
    """normalize_arxiv_id raises ValueError for unparseable input."""
    with pytest.raises(ValueError):
        normalize_arxiv_id("not-an-id")


def test_fetch_multiple_skips_failed_ids():
    """fetch_multiple returns successful papers, warns on failures."""
    import io
    mock_response = MagicMock()
    mock_response.read.return_value = _SAMPLE_XML
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    def _mock_urlopen(url, timeout=30):
        if "bad_id" in url:
            raise ValueError("bad id")
        return mock_response

    with patch("urllib.request.urlopen", side_effect=_mock_urlopen):
        with pytest.warns(UserWarning):
            results = fetch_multiple(["1706.03762", "bad_id"])

    assert len(results) == 1
    assert results[0].arxiv_id == "1706.03762"


# ─── Chunker Tests ────────────────────────────────────────────────────────────

def test_chunk_paper_abstract_only(sample_paper):
    """Paper without full_text produces exactly 1 abstract chunk."""
    chunks = chunk_paper(sample_paper, "paper:arxiv:1706.03762")
    assert len(chunks) == 1
    assert chunks[0].chunk_level == "abstract"
    assert "Attention Is All You Need" in chunks[0].text
    assert "Vaswani" in chunks[0].text


def test_chunk_paper_with_sections(paper_with_sections):
    """Paper with full_text produces abstract chunk + section chunks."""
    chunks = chunk_paper(paper_with_sections, "paper:arxiv:1706.03762")
    assert len(chunks) >= 2
    assert chunks[0].chunk_level == "abstract"
    levels = [c.chunk_level for c in chunks[1:]]
    # At least one section detected
    assert any(lvl in ("introduction", "methods", "conclusion") for lvl in levels)


# ─── Wiki Rules Tests ─────────────────────────────────────────────────────────

def _make_chunk(source_id: str, text: str):
    """Simple namespace-style chunk for wiki rule tests."""
    class _C:
        pass
    c = _C()
    c.source_id = source_id
    c.text = text
    return c


def test_find_keyword_overlap_basic():
    """Two papers sharing >20% keywords produce an edge."""
    text_a = "deep learning neural network transformer attention mechanism self-attention"
    text_b = "neural network transformer attention mechanism multi-head attention layers"
    chunks = [
        _make_chunk("paper:arxiv:0001.00001", text_a),
        _make_chunk("paper:arxiv:0001.00002", text_b),
    ]
    edges = find_keyword_overlap({}, chunks)
    assert len(edges) >= 1
    assert edges[0].rule == "keyword_overlap"
    assert edges[0].weight >= 0.2


def test_find_method_chains_bert():
    """Two papers mentioning BERT produce a method edge."""
    chunks = [
        _make_chunk("paper:arxiv:0002.00001", "We fine-tune bert on downstream tasks"),
        _make_chunk("paper:arxiv:0002.00002", "Our model outperforms bert baseline"),
    ]
    edges = find_method_chains(chunks, None, None, "", "")
    assert any(e.rule.startswith("method:bert") for e in edges)


def test_find_shared_concepts_no_chroma_returns_empty():
    """find_shared_concepts returns [] when chroma is None."""
    edges = find_shared_concepts([], None, None, "", "")
    assert edges == []


def test_find_dataset_pairs_squad():
    """Two papers mentioning squad produce a dataset edge."""
    chunks = [
        _make_chunk("paper:arxiv:0003.00001", "Evaluated on squad question answering benchmark"),
        _make_chunk("paper:arxiv:0003.00002", "Our model achieves 91 F1 on squad"),
    ]
    edges = find_dataset_pairs(chunks, None, None, "", "")
    assert any(e.rule.startswith("dataset:squad") for e in edges)
