"""Tests for src/memory_store.py."""
import json
from pathlib import Path

import pytest

from src.memory.schema import Chunk, Source
from src.memory.store import (
    add_feedback,
    deactivate_chunks_by_source,
    find_by_text_hash,
    get_chunk,
    get_chunks_by_source,
    get_source,
    init_memory_db,
    insert_chunk,
    kv_get,
    kv_invalidate_by_ids,
    kv_put,
    log_ingestion_event,
    supersede_chunk,
    upsert_source,
)


def _make_source(source_id: str = "repo:owner/test:src/foo.py") -> Source:
    return Source(
        source_id=source_id,
        source_type="repo_file",
        repo="owner/test",
        path="src/foo.py",
        branch="main",
        commit="abc123",
        content_hash="sha256:aaa",
        captured_at="2026-04-08T10:00:00+00:00",
    )


def _make_chunk(source_id: str = "repo:owner/test:src/foo.py", ordinal: int = 0) -> Chunk:
    text = f"def foo_{ordinal}(): pass"
    text_hash = Chunk.compute_text_hash(text)
    chunk_id = Chunk.make_id(source_id, ordinal, "function")
    return Chunk(
        chunk_id=chunk_id,
        source_id=source_id,
        chunk_level="function",
        ordinal=ordinal,
        text=text,
        text_hash=text_hash,
        category_labels=["utility", "function"],
        created_at="2026-04-08T10:00:00+00:00",
    )


class TestInitMemoryDb:
    def test_creates_tables(self, tmp_path: Path) -> None:
        db = tmp_path / "memory.db"
        conn = init_memory_db(db)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"sources", "chunks", "chunk_feedback", "ingestion_log"}.issubset(tables)
        conn.close()

    def test_wal_mode(self, tmp_path: Path) -> None:
        db = tmp_path / "memory.db"
        conn = init_memory_db(db)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "memory.db"
        conn1 = init_memory_db(db)
        conn1.close()
        # Second call must not raise
        conn2 = init_memory_db(db)
        conn2.close()


class TestUpsertSource:
    def test_insert_and_get(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source()
        upsert_source(conn, src)
        result = get_source(conn, src.source_id)
        assert result is not None
        assert result.source_id == src.source_id
        assert result.repo == "owner/test"

    def test_replace_on_conflict(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source()
        upsert_source(conn, src)
        # Update content_hash
        src2 = Source(
            source_id=src.source_id,
            source_type=src.source_type,
            repo=src.repo,
            path=src.path,
            branch=src.branch,
            commit="def456",
            content_hash="sha256:bbb",
            captured_at=src.captured_at,
        )
        upsert_source(conn, src2)
        result = get_source(conn, src.source_id)
        assert result is not None
        assert result.commit == "def456"
        assert result.content_hash == "sha256:bbb"

    def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        assert get_source(conn, "nonexistent") is None


class TestInsertChunk:
    def test_insert_and_get(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source()
        upsert_source(conn, src)
        chunk = _make_chunk()
        insert_chunk(conn, chunk)
        result = get_chunk(conn, chunk.chunk_id)
        assert result is not None
        assert result.chunk_id == chunk.chunk_id
        assert result.chunk_level == "function"

    def test_category_labels_roundtrip(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source()
        upsert_source(conn, src)
        chunk = _make_chunk()
        chunk.category_labels = ["auth", "state-transition", "api"]
        insert_chunk(conn, chunk)
        result = get_chunk(conn, chunk.chunk_id)
        assert result is not None
        assert result.category_labels == ["auth", "state-transition", "api"]

    def test_get_chunks_by_source(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source()
        upsert_source(conn, src)
        for i in range(3):
            insert_chunk(conn, _make_chunk(ordinal=i))
        results = get_chunks_by_source(conn, src.source_id)
        assert len(results) == 3

    def test_get_chunk_nonexistent(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        assert get_chunk(conn, "nonexistent") is None


class TestFindByTextHash:
    def test_finds_existing(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        upsert_source(conn, _make_source())
        chunk = _make_chunk()
        insert_chunk(conn, chunk)
        result = find_by_text_hash(conn, chunk.text_hash)
        assert result is not None
        assert result.chunk_id == chunk.chunk_id

    def test_returns_none_for_unknown(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        assert find_by_text_hash(conn, "sha256:unknown") is None

    def test_ignores_inactive(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        upsert_source(conn, _make_source())
        chunk = _make_chunk()
        insert_chunk(conn, chunk)
        conn.execute("UPDATE chunks SET is_active = 0 WHERE chunk_id = ?", (chunk.chunk_id,))
        conn.commit()
        assert find_by_text_hash(conn, chunk.text_hash) is None


class TestSupersede:
    def test_supersede_sets_inactive(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        upsert_source(conn, _make_source())
        old = _make_chunk(ordinal=0)
        new = _make_chunk(ordinal=1)
        insert_chunk(conn, old)
        insert_chunk(conn, new)
        supersede_chunk(conn, old.chunk_id, new.chunk_id)
        old_result = get_chunk(conn, old.chunk_id, active_only=False)
        assert old_result is not None
        assert old_result.is_active is False
        assert old_result.superseded_by == new.chunk_id


class TestDeactivate:
    def test_deactivates_all_for_source(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        upsert_source(conn, _make_source())
        for i in range(3):
            insert_chunk(conn, _make_chunk(ordinal=i))
        count = deactivate_chunks_by_source(conn, "repo:owner/test:src/foo.py")
        assert count == 3
        active = get_chunks_by_source(conn, "repo:owner/test:src/foo.py", active_only=True)
        assert active == []

    def test_deactivate_different_source_unaffected(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src1 = _make_source("repo:owner/test:a.py")
        src2 = _make_source("repo:owner/test:b.py")
        upsert_source(conn, src1)
        upsert_source(conn, src2)
        insert_chunk(conn, _make_chunk(source_id=src1.source_id, ordinal=0))
        insert_chunk(conn, _make_chunk(source_id=src2.source_id, ordinal=0))
        deactivate_chunks_by_source(conn, src1.source_id)
        active = get_chunks_by_source(conn, src2.source_id, active_only=True)
        assert len(active) == 1


class TestIngestionLog:
    def test_log_and_query(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        log_ingestion_event(
            conn,
            source_id="repo:owner/test:src/foo.py",
            event_type="ingest_start",
            payload={"chunks": 3, "model": "test"},
        )
        rows = conn.execute("SELECT * FROM ingestion_log").fetchall()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "ingest_start"
        payload_parsed = json.loads(rows[0]["payload"])
        assert payload_parsed["chunks"] == 3


class TestFeedback:
    def test_add_feedback(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        upsert_source(conn, _make_source())
        chunk = _make_chunk()
        insert_chunk(conn, chunk)
        add_feedback(conn, chunk.chunk_id, "positive", {"query": "auth flow"})
        rows = conn.execute("SELECT * FROM chunk_feedback").fetchall()
        assert len(rows) == 1
        assert rows[0]["signal"] == "positive"
        meta = json.loads(rows[0]["metadata"])
        assert meta["query"] == "auth flow"


class TestKVCache:
    def test_put_and_get(self) -> None:
        kv_put("chk_abc", {"chunk_id": "chk_abc", "text": "hello"})
        result = kv_get("chk_abc")
        assert result is not None
        assert result["text"] == "hello"

    def test_miss_returns_none(self) -> None:
        assert kv_get("nonexistent_xyz_123") is None

    def test_invalidate(self) -> None:
        kv_put("chk_del", {"chunk_id": "chk_del"})
        kv_invalidate_by_ids(["chk_del"])
        assert kv_get("chk_del") is None
