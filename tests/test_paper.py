"""Tests for paper ingestion pipeline: chunker, wiki rules, pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.memory.chunker import chunk_paper, extract_pdf_text


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


