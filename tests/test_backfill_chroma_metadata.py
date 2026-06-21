"""Idempotenter Backfill: repo + source_class in Chroma-Metadata stempeln.

repo-scoping-hardfilter + reference-doc-layer (2026-06-21).
"""
from __future__ import annotations

import pytest

from mayring_core.memory import store
from mayring_core.memory.schema import Chunk, Source
from src.api.admin_backfill_chroma_metadata import (
    backfill_repo_source_class_metadata,
)


class _FakeCollection:
    def __init__(self):
        self.meta: dict[str, dict] = {}

    def update(self, ids: list[str], metadatas: list[dict]) -> None:
        for cid, m in zip(ids, metadatas):
            self.meta.setdefault(cid, {}).update(m)

    def count(self) -> int:
        return len(self.meta)


@pytest.fixture()
def db_with_chunks(tmp_path):
    conn = store.init_memory_db(tmp_path / "memory.db")
    code = Source(source_id="repo:owner/app:a.py", source_type="repo_file",
                  repo="owner/app", path="a.py", visibility="public")
    ref = Source(source_id="unity-docs:webgl", source_type="note", repo="",
                 path="webgl", visibility="public", source_class="reference")
    store.upsert_source(conn, code, workspace_id="ws-1")
    store.upsert_source(conn, ref, workspace_id="ws-1")
    store.insert_chunk(conn, Chunk(chunk_id="c-code", source_id="repo:owner/app:a.py",
                                   text="t", text_hash="h1"), workspace_id="ws-1")
    store.insert_chunk(conn, Chunk(chunk_id="c-ref", source_id="unity-docs:webgl",
                                   text="t", text_hash="h2"), workspace_id="ws-1")
    return conn


def test_backfill_stamps_repo_and_source_class(db_with_chunks):
    coll = _FakeCollection()
    coll.meta["c-code"] = {"workspace_id": "ws-1"}
    coll.meta["c-ref"] = {"workspace_id": "ws-1"}

    updated = backfill_repo_source_class_metadata(db_with_chunks, coll, batch=100)

    assert updated == 2
    assert coll.meta["c-code"]["repo"] == "owner/app"
    assert coll.meta["c-code"]["source_class"] == "code"
    assert coll.meta["c-ref"]["source_class"] == "reference"
    assert coll.meta["c-ref"]["repo"] == ""


def test_backfill_idempotent(db_with_chunks):
    coll = _FakeCollection()
    coll.meta["c-code"] = {"workspace_id": "ws-1"}
    coll.meta["c-ref"] = {"workspace_id": "ws-1"}
    first = backfill_repo_source_class_metadata(db_with_chunks, coll, batch=100)
    second = backfill_repo_source_class_metadata(db_with_chunks, coll, batch=100)
    assert first == second == 2


def test_backfill_empty_db(tmp_path):
    conn = store.init_memory_db(tmp_path / "empty.db")
    assert backfill_repo_source_class_metadata(conn, _FakeCollection()) == 0
