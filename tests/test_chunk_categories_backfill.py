"""Deductive chunk→category backfill: cursor pagination + linking.

Restores reranker-v3 cat_match coverage on the existing corpus (chunks ingested
before Phase 3.2 or while the category Chroma was cold). LLM-free + idempotent.
"""
from __future__ import annotations

import sqlite3

from src.api.routes import reranker_admin as ra


class _FakeChunkCol:
    def get(self, ids, include):  # noqa: A002 - chromadb signature
        return {"ids": list(ids), "embeddings": [[0.1, 0.2] for _ in ids]}


def test_backfill_paginates_and_links(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (chunk_id TEXT)")
    for i in range(3):
        conn.execute("INSERT INTO chunks VALUES (?)", (f"chk_{i}",))
    conn.commit()
    monkeypatch.setattr(ra, "_conn", lambda: conn)
    monkeypatch.setattr("mayring_core.memory.store.get_chroma_collection",
                        lambda name: _FakeChunkCol())
    linked_pairs: list = []

    def _link(c, cats, pairs, **kw):
        linked_pairs.extend(pairs)
        return len(pairs)

    monkeypatch.setattr(
        "mayring_core.memory.ingestion.mayring_process.link_chunks_deductive", _link)

    win1 = ra._backfill_chunk_categories(0, 2)
    assert win1["processed"] == 2
    assert win1["embeddings_found"] == 2
    assert win1["linked"] == 2
    assert win1["has_more"] is True

    win2 = ra._backfill_chunk_categories(win1["next_after"], 2)
    assert win2["processed"] == 1      # only the 3rd chunk left
    assert win2["has_more"] is False

    assert {cid for cid, _ in linked_pairs} == {"chk_0", "chk_1", "chk_2"}


def test_backfill_empty_when_cursor_past_end(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (chunk_id TEXT)")
    conn.execute("INSERT INTO chunks VALUES ('chk_0')")
    conn.commit()
    monkeypatch.setattr(ra, "_conn", lambda: conn)
    out = ra._backfill_chunk_categories(999, 100)
    assert out["processed"] == 0
    assert out["has_more"] is False
