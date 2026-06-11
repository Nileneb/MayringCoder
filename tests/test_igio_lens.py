"""GET /stats/igio-lens — workspace-scoped IGIO distribution + drill-down."""
from __future__ import annotations

import asyncio


def _maybe_run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c
import datetime

from mayring_core.memory.store import init_memory_db
import src.api.routes.igio_admin as igio_admin
from src.api.jwt_auth import TokenInfo


def _seed(tmp_path):
    db = init_memory_db(tmp_path / "m.db")
    now = datetime.datetime.utcnow().isoformat()
    rows = [
        ("c1", "ws-a", "issue"), ("c2", "ws-a", "issue"),
        ("c3", "ws-a", "goal"), ("c4", "ws-a", "outcome"),
        ("c5", "ws-a", ""),                      # unclassified
        ("c6", "ws-b", "issue"),                 # other workspace
    ]
    for cid, ws, axis in rows:
        db.execute(
            "INSERT INTO sources (source_id, source_type, captured_at, workspace_id) "
            "VALUES (?,?,?,?)", (f"src:{cid}", "note", now, ws))
        db.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, text_hash, dedup_key, "
            "created_at, workspace_id, igio_axis, igio_confidence, summary) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (cid, f"src:{cid}", f"text {cid}", f"h{cid}", f"d{cid}", now, ws, axis,
             0.9 if axis else 0.0, f"summary {cid}"))
    db.commit()
    return db


def _run(coro):
    return _maybe_run(coro) if asyncio.iscoroutine(coro) else coro


def test_igio_lens_workspace_scoped_counts(tmp_path, monkeypatch):
    db = _seed(tmp_path)
    monkeypatch.setattr(igio_admin, "_conn", lambda: db)
    info = TokenInfo(workspace_id="ws-a", scopes=())
    res = _run(igio_admin.igio_lens(info=info, workspace="ws-a"))
    assert res["workspace_id"] == "ws-a"
    assert res["scope"] == "workspace"
    assert res["axes"]["issue"]["count"] == 2
    assert res["axes"]["goal"]["count"] == 1
    assert res["axes"]["outcome"]["count"] == 1
    assert res["axes"]["intervention"]["count"] == 0
    assert res["unclassified"]["count"] == 1
    issue_chunks = res["axes"]["issue"]["chunks"]
    assert len(issue_chunks) == 2
    assert {"chunk_id", "preview", "source_id", "confidence", "created_at"} <= set(issue_chunks[0])


def test_igio_lens_admin_sees_all_workspaces(tmp_path, monkeypatch):
    db = _seed(tmp_path)
    monkeypatch.setattr(igio_admin, "_conn", lambda: db)
    info = TokenInfo(workspace_id="system", scopes=("*",))
    res = _run(igio_admin.igio_lens(info=info, workspace="system"))
    assert res["scope"] == "all"
    assert res["axes"]["issue"]["count"] == 3
