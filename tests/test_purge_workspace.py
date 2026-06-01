"""Admin workspace purge: remove ALL rows + Chroma vectors for one workspace.

WHY(smoke-noise #253): every smoke run created an ephemeral `<prefix>-<ts>`
workspace and never cleaned it → 1856 junk workspaces accumulated in prod. The
smoke harness (HTTP-only, runs via --ref in CI) needs an admin endpoint to
self-clean its throwaway workspaces. A hard protected-set refuses the real ones.
"""
from __future__ import annotations

import pytest

from mayring_core.memory import store
from mayring_core.memory.schema import Chunk, Source
from src.api.admin_purge_workspace import PROTECTED_WORKSPACES, purge_workspace


class _FakeCollection:
    def __init__(self):
        self.meta: dict[str, dict] = {}

    def add_chunk(self, cid, ws):
        self.meta[cid] = {"workspace_id": ws}

    def delete(self, where=None, **kwargs):
        if not where:
            return
        wanted = where["workspace_id"]["$in"]
        for cid in [c for c, m in self.meta.items() if m.get("workspace_id") in wanted]:
            del self.meta[cid]

    def count(self):
        return len(self.meta)


def _seed(conn, coll, workspace_id, source_id, chunk_id):
    store.upsert_source(conn, Source(source_id=source_id, source_type="doc", repo="", path="p"),
                        workspace_id=workspace_id)
    store.insert_chunk(conn, Chunk(chunk_id=chunk_id, source_id=source_id, text="t", text_hash=chunk_id),
                       workspace_id=workspace_id)
    coll.add_chunk(chunk_id, workspace_id)


def test_purge_removes_rows_and_vectors(tmp_path):
    conn = store.init_memory_db(tmp_path / "m.db")
    coll = _FakeCollection()
    _seed(conn, coll, "oa-1780300000", "s:1", "c1")

    result = purge_workspace(conn, coll, "oa-1780300000")

    assert result["chroma_removed"] == 1
    assert result["rows"]["sources"] == 1
    assert result["rows"]["chunks"] == 1
    assert conn.execute("SELECT COUNT(*) FROM sources WHERE workspace_id=?",
                        ("oa-1780300000",)).fetchone()[0] == 0
    assert coll.count() == 0


def test_purge_leaves_other_workspaces_intact(tmp_path):
    conn = store.init_memory_db(tmp_path / "m.db")
    coll = _FakeCollection()
    _seed(conn, coll, "oa-1780300000", "s:1", "c1")
    _seed(conn, coll, "keep-real", "s:2", "c2")

    purge_workspace(conn, coll, "oa-1780300000")

    assert conn.execute("SELECT COUNT(*) FROM sources WHERE workspace_id=?",
                        ("keep-real",)).fetchone()[0] == 1
    assert "c2" in coll.meta


@pytest.mark.parametrize("ws", sorted(PROTECTED_WORKSPACES))
def test_purge_refuses_protected_workspace(tmp_path, ws):
    conn = store.init_memory_db(tmp_path / "m.db")
    coll = _FakeCollection()
    with pytest.raises(ValueError, match="protected"):
        purge_workspace(conn, coll, ws)
