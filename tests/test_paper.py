"""Tests for paper ingestion pipeline: chunker, wiki rules, pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.memory.chunker import chunk_paper, extract_pdf_text
from src.memory.wiki import (
    find_citation_pairs, find_keyword_overlap,
    find_shared_concepts, find_method_chains, find_dataset_pairs,
)


# ─── Chunker Tests ────────────────────────────────────────────────────────────

_TEXT_WITH_SECTIONS = """\
Introduction
This paper introduces the Transformer architecture.

Methods
We use self-attention mechanisms across all layers.

Results
Our model achieves state-of-the-art on WMT 2014.

Conclusion
Transformers outperform recurrent models.
"""

_TEXT_NO_SECTIONS = "Some plain text without any recognizable section headers."


def test_chunk_paper_with_sections():
    chunks = chunk_paper(_TEXT_WITH_SECTIONS, "paper:1706.03762")
    assert len(chunks) >= 2
    levels = {c.chunk_level for c in chunks}
    assert any(lvl in ("introduction", "methods", "results", "conclusion") for lvl in levels)


def test_chunk_paper_no_sections_fallback():
    chunks = chunk_paper(_TEXT_NO_SECTIONS, "paper:plain")
    assert len(chunks) == 1
    assert chunks[0].chunk_level == "file"


def test_chunk_paper_empty_text():
    chunks = chunk_paper("", "paper:empty")
    assert len(chunks) == 1


def test_chunk_paper_source_ids_all_match():
    chunks = chunk_paper(_TEXT_WITH_SECTIONS, "paper:test123")
    assert all(c.source_id == "paper:test123" for c in chunks)


def test_extract_pdf_text_returns_none_on_missing_file():
    result = extract_pdf_text("/tmp/does_not_exist_xyzabc.pdf")
    assert result is None


# ─── Pipeline Test ────────────────────────────────────────────────────────────

def test_run_ingest_paper_scans_directory():
    from src.pipeline import run_ingest_paper

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "2305.10601.txt").write_text(
            "Introduction\nThis is paper one.\n\nConclusion\nDone.", encoding="utf-8"
        )
        (Path(tmpdir) / "2406.12345.txt").write_text(
            "Introduction\nThis is paper two.\n\nMethods\nMethod section.", encoding="utf-8"
        )

        mock_conn = MagicMock()
        mock_chroma = MagicMock()

        with (
            patch("src.memory.store.init_memory_db", return_value=mock_conn),
            patch("src.memory.ingest.get_or_create_chroma_collection", return_value=mock_chroma),
            patch("src.memory.ingest.ingest", return_value={"chunks": 2, "skipped": False}) as mock_ingest,
        ):
            result = run_ingest_paper(
                papers_dir=tmpdir,
                ollama_url="http://localhost:11434",
                model="test-model",
            )

        assert result["total"] == 2
        assert mock_ingest.call_count == 2


def test_run_ingest_paper_missing_dir():
    from src.pipeline import run_ingest_paper

    result = run_ingest_paper(
        papers_dir="/tmp/does_not_exist_mayring_xyz",
        ollama_url="",
        model="",
    )
    assert result == {"ingested": 0, "skipped": 0, "failed": 0, "total": 0}


def test_run_ingest_paper_skips_unchanged(tmp_path):
    """ingest() returning skipped=True increments skipped counter."""
    from src.pipeline import run_ingest_paper

    (tmp_path / "paper.txt").write_text("Introduction\nContent.", encoding="utf-8")

    mock_conn = MagicMock()
    mock_chroma = MagicMock()

    with (
        patch("src.memory.store.init_memory_db", return_value=mock_conn),
        patch("src.memory.ingest.get_or_create_chroma_collection", return_value=mock_chroma),
        patch("src.memory.ingest.ingest", return_value={"chunks": 0, "skipped": True}),
    ):
        result = run_ingest_paper(papers_dir=str(tmp_path), ollama_url="", model="")

    assert result["skipped"] == 1
    assert result["ingested"] == 0


# ─── Wiki Rules Tests ─────────────────────────────────────────────────────────

def _make_chunk(source_id: str, text: str):
    class _C:
        pass
    c = _C()
    c.source_id = source_id
    c.text = text
    return c


def test_find_keyword_overlap_basic():
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
    chunks = [
        _make_chunk("paper:arxiv:0002.00001", "We fine-tune bert on downstream tasks"),
        _make_chunk("paper:arxiv:0002.00002", "Our model outperforms bert baseline"),
    ]
    edges = find_method_chains(chunks, None, None, "", "")
    assert any(e.rule.startswith("method:bert") for e in edges)


def test_find_shared_concepts_no_chroma_returns_empty():
    assert find_shared_concepts([], None, None, "", "") == []


def test_find_dataset_pairs_squad():
    chunks = [
        _make_chunk("paper:arxiv:0003.00001", "Evaluated on squad question answering benchmark"),
        _make_chunk("paper:arxiv:0003.00002", "Our model achieves 91 F1 on squad"),
    ]
    edges = find_dataset_pairs(chunks, None, None, "", "")
    assert any(e.rule.startswith("dataset:squad") for e in edges)


def test_paper_rules_return_empty_for_no_papers():
    non_paper = [_make_chunk("repo_file:src/foo.py", "def foo(): pass")]
    assert find_citation_pairs({}, non_paper) == []
    assert find_keyword_overlap({}, non_paper) == []
    assert find_method_chains(non_paper, None, None, "", "") == []
    assert find_dataset_pairs(non_paper, None, None, "", "") == []
