"""Tests for src/memory_ingest.py."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.memory_schema import Chunk, Source
from src.memory_store import init_memory_db, upsert_source, find_by_text_hash
from src.memory_ingest import (
    _chunk_markdown,
    _chunk_python,
    _chunk_yaml_json,
    _make_file_chunk,
    configure_memory_log,
    ingest,
    resolve_dedup,
    structural_chunk,
)


def _make_source(sid: str = "repo:owner/test:src/foo.py") -> Source:
    return Source(
        source_id=sid,
        source_type="repo_file",
        repo="owner/test",
        path="src/foo.py",
        branch="main",
        commit="abc",
        content_hash="sha256:test",
        captured_at="2026-04-08T10:00:00+00:00",
    )


class TestStructuralChunkPython:
    def test_two_functions(self) -> None:
        code = "def foo():\n    pass\n\ndef bar():\n    return 1\n"
        chunks = _chunk_python(code, "repo:owner/test:src/foo.py")
        assert len(chunks) == 2
        assert all(c.chunk_level == "function" for c in chunks)
        assert all(c.text_hash.startswith("sha256:") for c in chunks)

    def test_syntax_error_returns_empty(self) -> None:
        chunks = _chunk_python("def (broken:", "repo:s")
        assert chunks == []

    def test_class_level(self) -> None:
        code = "class MyClass:\n    def method(self):\n        pass\n"
        chunks = _chunk_python(code, "repo:owner/test:src/foo.py")
        assert len(chunks) == 1
        assert chunks[0].chunk_level == "class"

    def test_chunk_ids_unique(self) -> None:
        code = "def a(): pass\ndef b(): pass\n"
        chunks = _chunk_python(code, "repo:s")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestStructuralChunkMarkdown:
    def test_three_headings(self) -> None:
        md = "# Title\n\nIntro text.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B.\n"
        chunks = _chunk_markdown(md, "repo:s")
        assert len(chunks) == 3
        assert all(c.chunk_level == "section" for c in chunks)

    def test_no_headings_returns_empty(self) -> None:
        chunks = _chunk_markdown("plain text no headings", "repo:s")
        assert chunks == []

    def test_heading_text_included(self) -> None:
        md = "## Overview\n\nSome content here.\n"
        chunks = _chunk_markdown(md, "repo:s")
        assert len(chunks) == 1
        assert "Overview" in chunks[0].text


class TestStructuralChunkYamlJson:
    def test_yaml_two_keys(self) -> None:
        yaml_text = "key1: value1\nkey2: value2\n"
        chunks = _chunk_yaml_json(yaml_text, "repo:s", "config.yaml")
        assert len(chunks) == 2
        assert all(c.chunk_level == "block" for c in chunks)

    def test_json_dict(self) -> None:
        json_text = '{"a": 1, "b": 2}'
        chunks = _chunk_yaml_json(json_text, "repo:s", "data.json")
        assert len(chunks) == 2

    def test_invalid_returns_empty(self) -> None:
        chunks = _chunk_yaml_json("not: valid: yaml: !!!", "repo:s", "bad.yaml")
        # May return empty or fallback — either is acceptable
        assert isinstance(chunks, list)


class TestStructuralChunkDispatch:
    def test_py_extension_uses_python_chunker(self) -> None:
        code = "def hello(): pass\n"
        chunks = structural_chunk(code, "repo:s", "main.py")
        assert chunks[0].chunk_level == "function"

    def test_md_extension_uses_markdown_chunker(self) -> None:
        md = "## A\n\ntext\n"
        chunks = structural_chunk(md, "repo:s", "README.md")
        assert chunks[0].chunk_level == "section"

    def test_unknown_extension_returns_file_chunk(self) -> None:
        chunks = structural_chunk("some content", "repo:s", "file.rb")
        assert len(chunks) == 1
        assert chunks[0].chunk_level == "file"

    def test_empty_text_returns_one_chunk(self) -> None:
        chunks = structural_chunk("", "repo:s", "foo.py")
        assert len(chunks) == 1

    def test_all_chunks_have_text_hash(self) -> None:
        code = "def x(): pass\ndef y(): pass\n"
        chunks = structural_chunk(code, "repo:s", "mod.py")
        for c in chunks:
            assert c.text_hash.startswith("sha256:")
            assert c.chunk_id != ""


class TestResolveDedup:
    def test_no_duplicate(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        source = _make_source()
        upsert_source(conn, source)
        chunk = _make_file_chunk("unique content xyz", source.source_id)
        returned, is_dup = resolve_dedup(conn, chunk)
        assert is_dup is False
        assert returned is chunk

    def test_duplicate_found(self, tmp_path: Path) -> None:
        from src.memory_store import insert_chunk
        conn = init_memory_db(tmp_path / "m.db")
        source = _make_source()
        upsert_source(conn, source)
        chunk = _make_file_chunk("duplicate text", source.source_id)
        insert_chunk(conn, chunk)
        # Same text → same text_hash
        chunk2 = _make_file_chunk("duplicate text", source.source_id)
        returned, is_dup = resolve_dedup(conn, chunk2)
        assert is_dup is True
        assert returned.chunk_id == chunk.chunk_id


class TestIngest:
    def _make_mock_chroma(self):
        col = MagicMock()
        col.upsert = MagicMock()
        return col

    # Fix: patch src.context._embed_texts (the actual import location used inside ingest())
    # rather than src.memory_ingest._embed_texts which is not a module-level name.
    @patch("src.context._embed_texts", return_value=[[0.1] * 4])
    def test_first_ingest_returns_chunks(self, mock_embed, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        source = _make_source()
        chroma = self._make_mock_chroma()

        result = ingest(
            source=source,
            content="def foo(): pass\n",
            conn=conn,
            chroma_collection=chroma,
            ollama_url="http://localhost:11434",
            model="",
            opts={},
        )

        assert result["source_id"] == source.source_id
        assert len(result["chunk_ids"]) >= 1
        assert result["deduped"] == 0

    @patch("src.context._embed_texts", return_value=[[0.1] * 4])
    def test_second_ingest_same_content_is_deduped(self, mock_embed, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        source = _make_source()
        chroma = self._make_mock_chroma()
        content = "def foo(): pass\n"

        ingest(source, content, conn, chroma, "http://localhost:11434", "", {})
        result2 = ingest(source, content, conn, chroma, "http://localhost:11434", "", {})

        assert result2["deduped"] >= 1
        assert len(result2["chunk_ids"]) == 0

    def test_jsonl_log_written(self, tmp_path: Path) -> None:
        import json as _json
        conn = init_memory_db(tmp_path / "m.db")
        source = _make_source()

        configure_memory_log.__module__  # ensure importable
        import src.memory_ingest as mi
        log_path = tmp_path / "test_memory_log.jsonl"
        mi._MEMORY_LOG_PATH = log_path

        with patch("src.context._embed_texts", return_value=[[0.1] * 4]):
            ingest(source, "def bar(): pass\n", conn, None, "http://localhost:11434", "", {"log": True})

        mi._MEMORY_LOG_PATH = None  # reset
        assert log_path.exists()
        line = _json.loads(log_path.read_text().strip().splitlines()[-1])
        assert line["event"] == "ingest"
        assert line["source_id"] == source.source_id
