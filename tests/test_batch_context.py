"""Batch-commit bundling for store.py write operations (#68).

Observable-only tests (no commit-count mocking — sqlite3.Connection
is a C-type and attribute patching fails). We verify what the contract
actually promises:

  - Data written inside batch_context is visible after the block.
  - Exception inside the block → rollback (no partial writes).
  - Nested batch_context works (outer is the commit boundary).
  - Writes outside batch_context still persist (BC preserved).
"""
from __future__ import annotations

import pytest

from src.memory.schema import Chunk, Source
from src.memory.store import (
    batch_context,
    get_chunks_by_source,
    get_source,
    init_memory_db,
    insert_chunk,
    upsert_source,
)


def _mk_source(sid="src:1", path="a.py") -> Source:
    return Source(source_id=sid, source_type="repo_file", repo="r", path=path,
                  content_hash="h1", branch="", commit="")


def _mk_chunk(cid, sid="src:1", text="hello") -> Chunk:
    return Chunk(chunk_id=cid, source_id=sid, text=text,
                 text_hash=Chunk.compute_text_hash(text),
                 chunk_level="file", ordinal=0, dedup_key="k", category_labels=[],
                 created_at="2026-04-22T00:00:00Z")


@pytest.fixture
def conn(tmp_path):
    c = init_memory_db(tmp_path / "m.db")
    yield c
    c.close()


class TestBatchPersistence:
    def test_writes_inside_batch_are_persisted(self, conn):
        with batch_context(conn):
            upsert_source(conn, _mk_source("s:a", "a.py"))
            for i in range(5):
                insert_chunk(conn, _mk_chunk(f"c:{i}", "s:a", f"t{i}"))
        # After the block, everything must be readable
        assert get_source(conn, "s:a") is not None
        assert len(get_chunks_by_source(conn, "s:a")) == 5

    def test_bare_writes_still_persist_without_batch(self, conn):
        upsert_source(conn, _mk_source("s:b", "b.py"))
        insert_chunk(conn, _mk_chunk("c:b0", "s:b"))
        assert get_source(conn, "s:b") is not None
        assert len(get_chunks_by_source(conn, "s:b")) == 1


class TestBatchRollback:
    def test_exception_rolls_back_all_writes(self, conn):
        with pytest.raises(RuntimeError):
            with batch_context(conn):
                upsert_source(conn, _mk_source("s:x", "x.py"))
                insert_chunk(conn, _mk_chunk("c:x0", "s:x"))
                raise RuntimeError("boom")
        # Nothing should have survived
        assert get_source(conn, "s:x") is None
        assert get_chunks_by_source(conn, "s:x") == []

    def test_partial_write_before_raise_is_rolled_back(self, conn):
        # Write outside the batch first — that must survive
        upsert_source(conn, _mk_source("s:keep", "k.py"))
        with pytest.raises(ValueError):
            with batch_context(conn):
                insert_chunk(conn, _mk_chunk("c:lose", "s:keep"))
                raise ValueError("nope")
        assert get_source(conn, "s:keep") is not None  # survived
        assert get_chunks_by_source(conn, "s:keep") == []  # rolled back


class TestNestedBatch:
    def test_nested_commits_only_at_outermost(self, conn):
        with batch_context(conn):
            upsert_source(conn, _mk_source("s:n", "n.py"))
            with batch_context(conn):
                insert_chunk(conn, _mk_chunk("c:n0", "s:n"))
                insert_chunk(conn, _mk_chunk("c:n1", "s:n"))
        assert len(get_chunks_by_source(conn, "s:n")) == 2

    def test_inner_exception_rolls_back_the_outer(self, conn):
        with pytest.raises(RuntimeError):
            with batch_context(conn):
                upsert_source(conn, _mk_source("s:n2", "n2.py"))
                with batch_context(conn):
                    insert_chunk(conn, _mk_chunk("c:n20", "s:n2"))
                    raise RuntimeError("inner boom")
        assert get_source(conn, "s:n2") is None
        assert get_chunks_by_source(conn, "s:n2") == []
