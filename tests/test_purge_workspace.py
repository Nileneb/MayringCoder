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


# ---------------------------------------------------------------------------
# purge_smoke_projects (2026-06-10): C3-Smoke-Projekte (smoke/repo-c3-<ts>)
# häuften sich in der Projekte-Sicht an — Pattern-gated cross-workspace purge.
# ---------------------------------------------------------------------------

def _seed_project(conn, pid, ws, name, source_ref):
    conn.execute(
        "INSERT INTO projects(id,workspace_id,name,source_type,source_ref,created_at,updated_at) "
        "VALUES (?,?,?,?,?,datetime('now'),datetime('now'))",
        (pid, ws, name, "github", source_ref))
    conn.commit()


def test_purge_smoke_projects_removes_only_smoke_refs(tmp_path):
    from src.api.admin_purge_workspace import purge_smoke_projects
    conn = store.init_memory_db(tmp_path / "m.db")
    # kanonische Slug-Form OHNE führenden Slash (canonical_repo_ref) + URL-Form + echtes Repo
    _seed_project(conn, "p1", "user-ws", "repo-c3-1780000000", "smoke/repo-c3-1780000000")
    _seed_project(conn, "p2", "system", "repo-c3-1780000001", "https://github.com/smoke/repo-c3-1780000001")
    _seed_project(conn, "p3", "user-ws", "mayringcoder", "nileneb/mayringcoder")
    _seed(conn, _FakeCollection(), "user-ws", "s:link", "c1")
    conn.execute(
        "INSERT INTO chunk_project_links(chunk_id,project_id,workspace_id,created_at) "
        "VALUES ('c1','p1','user-ws',datetime('now'))")
    conn.execute(
        "INSERT INTO project_groups(id,workspace_id,name,color,created_at,updated_at) "
        "VALUES ('g1','user-ws','smoke-c1-grp','#fff',datetime('now'),datetime('now'))")
    conn.commit()

    result = purge_smoke_projects(conn)

    assert result["projects"] == 2
    assert result["project_groups"] == 1
    remaining = [r[0] for r in conn.execute("SELECT id FROM projects").fetchall()]
    assert remaining == ["p3"]
    assert conn.execute("SELECT COUNT(*) FROM chunk_project_links").fetchone()[0] == 0


def test_claim_not_smoke_guard_matches_canonical_slug(tmp_path):
    """Der claim-Guard muss die kanonische Form 'smoke/repo-…' (OHNE Slash davor)
    ausschließen — das alte '%/smoke/repo-%' zog 87 Smoke-Projekte in den User-WS."""
    from src.api.routes.projects import canonical_repo_ref
    canon = canonical_repo_ref("https://github.com/smoke/repo-c3-1780000000")
    assert canon == "smoke/repo-c3-1780000000"
    # Pattern-Probe wie im claim-SQL
    conn = store.init_memory_db(tmp_path / "m.db")
    _seed_project(conn, "p1", "system", "repo-c3", canon)
    hit = conn.execute(
        "SELECT COUNT(*) FROM projects WHERE lower(source_ref) NOT LIKE '%smoke/repo-%'"
    ).fetchone()[0]
    assert hit == 0
