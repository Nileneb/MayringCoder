"""GET /stats/categories-overview — codebook grouped by Workspace → Goal."""
from __future__ import annotations

import asyncio
import datetime

from mayring_core.memory.store import init_memory_db
import src.api.routes.stats_admin as ia
from src.api.jwt_auth import TokenInfo


def _run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c


def _seed(tmp_path):
    db = init_memory_db(tmp_path / "m.db")
    now = datetime.datetime.utcnow().isoformat()

    # source with a goal anchor
    db.execute(
        "INSERT INTO sources(source_id, source_type, captured_at, workspace_id) "
        "VALUES ('s1', 'note', ?, 'ws-a')",
        (now,),
    )
    db.execute(
        "INSERT INTO source_goals(source_id, goal, updated_at) VALUES ('s1', 'JWT härten', ?)",
        (now,),
    )
    db.execute(
        "INSERT INTO chunks(chunk_id, source_id, text, text_hash, dedup_key, "
        "created_at, workspace_id, is_active, summary) "
        "VALUES ('c1', 's1', 't', 'h', 'd', ?, 'ws-a', 1, 'sum')",
        (now,),
    )

    # linked category
    db.execute(
        "INSERT INTO categories(id, name, status, source, evidence_count) "
        "VALUES (10, 'auth_login', 'active', 'imported', 5)"
    )
    # unlinked category (no chunk_categories row)
    db.execute(
        "INSERT INTO categories(id, name, status, source, evidence_count) "
        "VALUES (20, 'json', 'active', 'induced', 3)"
    )

    db.execute(
        "INSERT INTO chunk_categories(chunk_id, category_id, confidence, source) "
        "VALUES ('c1', 10, 0.9, 'deductive')"
    )
    db.commit()
    return db


def test_nests_by_workspace_then_goal(tmp_path, monkeypatch):
    db = _seed(tmp_path)
    monkeypatch.setattr(ia, "_conn", lambda: db)
    info = TokenInfo(workspace_id="system", scopes=("*",))
    res = _run(ia.categories_overview(info=info))

    assert res["scope"] == "all"
    assert res["total_categories"] == 2

    ws = {w["workspace_id"]: w for w in res["workspaces"]}
    assert "ws-a" in ws

    goals = {g["goal"]: g for g in ws["ws-a"]["goals"]}
    assert "JWT härten" in goals
    cats = goals["JWT härten"]["categories"]
    assert any(c["name"] == "auth_login" and c["chunk_count"] == 1 for c in cats)

    assert any(c["name"] == "json" for c in res["unlinked"])


def test_workspace_scoped(tmp_path, monkeypatch):
    db = _seed(tmp_path)
    monkeypatch.setattr(ia, "_conn", lambda: db)
    info = TokenInfo(workspace_id="ws-other", scopes=())
    res = _run(ia.categories_overview(info=info))

    assert res["scope"] == "workspace"
    # ws-a chunks must NOT appear — caller is ws-other
    assert all(w["workspace_id"] == "ws-other" for w in res["workspaces"])
    # unlinked is always shown regardless of workspace scope
    assert any(c["name"] == "json" for c in res["unlinked"])


def test_admin_sees_linked_fields(tmp_path, monkeypatch):
    db = _seed(tmp_path)
    monkeypatch.setattr(ia, "_conn", lambda: db)
    info = TokenInfo(workspace_id="system", scopes=("*",))
    res = _run(ia.categories_overview(info=info))

    # Every category entry must carry the expected keys
    for ws_entry in res["workspaces"]:
        for goal_entry in ws_entry["goals"]:
            for cat in goal_entry["categories"]:
                assert {"id", "name", "status", "source", "evidence_count", "chunk_count"} <= set(cat)
