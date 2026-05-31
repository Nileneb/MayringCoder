"""Idempotenter Backfill: visibility/org_id/user_id in Chroma-Metadata stempeln.

Tenancy Phase A — Task 2.
"""
from __future__ import annotations

import pathlib
import tempfile

import pytest

from mayring_core.memory import store
from mayring_core.memory.schema import Chunk, Source
from src.api.admin_backfill_chroma_visibility import backfill_visibility_metadata


class _FakeCollection:
    def __init__(self):
        self.meta: dict[str, dict] = {}

    def update(self, ids: list[str], metadatas: list[dict]) -> None:
        for cid, m in zip(ids, metadatas):
            self.meta.setdefault(cid, {}).update(m)

    def get(self, ids=None, **kwargs):
        if ids is None:
            ids = list(self.meta.keys())
        return {"ids": ids, "metadatas": [self.meta.get(i, {}) for i in ids]}

    def count(self) -> int:
        return len(self.meta)


@pytest.fixture()
def db_with_chunk(tmp_path):
    db_path = tmp_path / "memory.db"
    conn = store.init_memory_db(db_path)
    s = Source(
        source_id="s:1",
        source_type="doc",
        repo="",
        path="p",
        visibility="org",
        org_id="org-x",
    )
    store.upsert_source(conn, s, workspace_id="ws-1")
    ch = Chunk(chunk_id="c1", source_id="s:1", text="t", text_hash="h1")
    store.insert_chunk(conn, ch, workspace_id="ws-1")
    return conn


def test_backfill_stamps_visibility(db_with_chunk):
    coll = _FakeCollection()
    coll.meta["c1"] = {"workspace_id": "ws-1", "source_id": "s:1"}

    updated = backfill_visibility_metadata(db_with_chunk, coll, batch=100)

    assert updated == 1
    assert coll.meta["c1"]["visibility"] == "org"
    assert coll.meta["c1"]["org_id"] == "org-x"


def test_backfill_idempotent(db_with_chunk):
    """Re-running backfill must not raise and must return same count."""
    coll = _FakeCollection()
    coll.meta["c1"] = {"workspace_id": "ws-1", "source_id": "s:1"}

    first = backfill_visibility_metadata(db_with_chunk, coll, batch=100)
    second = backfill_visibility_metadata(db_with_chunk, coll, batch=100)

    assert first == second == 1
    assert coll.meta["c1"]["visibility"] == "org"


def test_backfill_empty_db(tmp_path):
    db_path = tmp_path / "empty.db"
    conn = store.init_memory_db(db_path)
    coll = _FakeCollection()

    updated = backfill_visibility_metadata(conn, coll)

    assert updated == 0


def test_backfill_batching(tmp_path):
    """Batch splitting: 3 chunks mit batch=2 → 2 update-Aufrufe, alle gestempelt."""
    db_path = tmp_path / "batch.db"
    conn = store.init_memory_db(db_path)
    s = Source(
        source_id="s:1",
        source_type="doc",
        repo="",
        path="p",
        visibility="public",
        org_id=None,
    )
    store.upsert_source(conn, s, workspace_id="ws-1")
    for i in range(3):
        ch = Chunk(chunk_id=f"c{i}", source_id="s:1", text=f"t{i}", text_hash=f"h{i}")
        store.insert_chunk(conn, ch, workspace_id="ws-1")

    update_calls: list[list[str]] = []
    original_update = _FakeCollection.update

    coll = _FakeCollection()
    for i in range(3):
        coll.meta[f"c{i}"] = {"workspace_id": "ws-1"}

    def _tracking_update(self, ids, metadatas):
        update_calls.append(list(ids))
        original_update(self, ids, metadatas)

    _FakeCollection.update = _tracking_update  # type: ignore[method-assign]
    try:
        updated = backfill_visibility_metadata(conn, coll, batch=2)
    finally:
        _FakeCollection.update = original_update  # type: ignore[method-assign]

    assert updated == 3
    assert len(update_calls) == 2
    for i in range(3):
        assert coll.meta[f"c{i}"]["visibility"] == "public"
