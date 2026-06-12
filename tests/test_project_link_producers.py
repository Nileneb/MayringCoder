"""TDD: Project-Link Producers A (repo_events) + B (conversation micro-batch).

TASK 5 — Producer A:
  (a) _repo_event_chunk links its chunk to the project for that repo
  (b) two different repos → two different project_ids, each chunk links only to its own

TASK 6 — Producer B:
  (a) POST with X-Project-Id → conversation chunks get a chunk_project_links entry
  (b) POST without X-Project-Id → no links (chunk stays global)
"""
from __future__ import annotations

import asyncio


def _maybe_run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

_INFO = SimpleNamespace(sub="test-user", org_ids=(), memberships=[])

import pytest

import src.api.dependencies as _deps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path):
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema
    adapter = DBAdapter.create(tmp_path / "test.db", check_same_thread=False)
    _init_schema(adapter)
    return adapter


def _seed_workspace(conn, workspace_id: str) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO workspaces (id, kind, display_name, created_at, updated_at) "
        "VALUES (?, 'user', ?, ?, ?)",
        (workspace_id, workspace_id, now, now),
    )
    conn.commit()


def _seed_project(conn, workspace_id: str, source_ref: str, project_id: str | None = None) -> str:
    """Insert a project row and return its id."""
    import uuid
    from datetime import datetime, timezone
    pid = project_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO projects (id, workspace_id, name, source_type, source_ref, created_at, updated_at) "
        "VALUES (?, ?, ?, 'github', ?, ?, ?)",
        (pid, workspace_id, source_ref.rsplit("/", 1)[-1], source_ref, now, now),
    )
    conn.commit()
    return pid


def _capture_task(coro):
    if inspect.iscoroutine(coro):
        coro.close()
    return MagicMock()


# ---------------------------------------------------------------------------
# TASK 5a — repo_event_chunk links its chunk to the project
# ---------------------------------------------------------------------------

def test_repo_event_chunk_links_to_project(tmp_path, monkeypatch):
    conn = _make_db(tmp_path)
    monkeypatch.setattr(_deps, "_conn", conn)

    # Seed workspace + project
    _seed_workspace(conn, "ws-1")
    pid = _seed_project(conn, "ws-1", "a/b")

    from src.api.routes.models import RepoEventRequest
    req = RepoEventRequest(
        event_type="workflow_run",
        repo="https://github.com/a/b",
        sha="deadbeef",
        conclusion="failure",
        workflow="ci",
    )

    from src.api.routes import repo_events as re_mod
    re_mod._repo_event_chunk(conn, "ws-1", req)

    # Check chunk_project_links table
    rows = conn.execute(
        "SELECT chunk_id, project_id, source FROM chunk_project_links WHERE project_id=?",
        (pid,),
    ).fetchall()
    assert len(rows) == 1, "event chunk must be linked to the project"
    assert rows[0][2] == "repo-analyze", f"source tag should be 'repo-analyze', got {rows[0][2]!r}"


def test_repo_event_chunk_no_project_no_link(tmp_path, monkeypatch):
    """If no matching project exists, the chunk is created but no project link is written."""
    conn = _make_db(tmp_path)
    monkeypatch.setattr(_deps, "_conn", conn)

    _seed_workspace(conn, "ws-x")
    # Do NOT seed a project for this repo

    from src.api.routes.models import RepoEventRequest
    req = RepoEventRequest(
        event_type="security",
        repo="https://github.com/no/project",
        severity="high",
        summary="some vuln",
    )

    from src.api.routes import repo_events as re_mod
    re_mod._repo_event_chunk(conn, "ws-x", req)

    rows = conn.execute("SELECT * FROM chunk_project_links").fetchall()
    assert rows == [], "no project → no chunk_project_links row"


# ---------------------------------------------------------------------------
# TASK 5b — two repos → two distinct project_ids, each chunk links only to its own
# ---------------------------------------------------------------------------

def test_two_repos_two_distinct_project_links(tmp_path, monkeypatch):
    conn = _make_db(tmp_path)
    monkeypatch.setattr(_deps, "_conn", conn)

    _seed_workspace(conn, "ws-2")
    pid_ab = _seed_project(conn, "ws-2", "a/b")
    pid_cd = _seed_project(conn, "ws-2", "c/d")

    from src.api.routes.models import RepoEventRequest
    from src.api.routes import repo_events as re_mod

    req_ab = RepoEventRequest(event_type="workflow_run", repo="https://github.com/a/b",
                               sha="aa", conclusion="failure", workflow="ci")
    req_cd = RepoEventRequest(event_type="workflow_run", repo="https://github.com/c/d",
                               sha="cc", conclusion="failure", workflow="ci")

    re_mod._repo_event_chunk(conn, "ws-2", req_ab)
    re_mod._repo_event_chunk(conn, "ws-2", req_cd)

    rows = conn.execute("SELECT chunk_id, project_id FROM chunk_project_links").fetchall()
    linked_projects = {r[1] for r in rows}
    assert pid_ab in linked_projects, "chunk for a/b must link to project a/b"
    assert pid_cd in linked_projects, "chunk for c/d must link to project c/d"
    assert linked_projects == {pid_ab, pid_cd}, "no cross-linking between projects"


# ---------------------------------------------------------------------------
# TASK 5 — _resolve_project helper
# ---------------------------------------------------------------------------

def test_resolve_project_returns_workspace_and_project_id(tmp_path, monkeypatch):
    conn = _make_db(tmp_path)
    monkeypatch.setattr(_deps, "_conn", conn)

    _seed_workspace(conn, "ws-3")
    pid = _seed_project(conn, "ws-3", "myorg/myrepo")

    from src.api.routes.repo_events import _resolve_project
    ws, project_id = _resolve_project(conn, "https://github.com/myorg/myrepo")
    assert ws == "ws-3"
    assert project_id == pid


def test_resolve_project_creates_project_when_missing(tmp_path, monkeypatch):
    """_resolve_project uses _resolve_workspace which does match-or-CREATE.
    So if no project exists, it creates one and returns the new project_id."""
    conn = _make_db(tmp_path)
    monkeypatch.setattr(_deps, "_conn", conn)

    _seed_workspace(conn, "ws-4")
    # no project pre-seeded — but _resolve_workspace creates one

    from src.api.routes.repo_events import _resolve_project
    ws_id, project_id = _resolve_project(conn, "https://github.com/nobody/norepo")
    # _resolve_workspace match-or-creates the project → project_id must be a string
    assert project_id is not None, "match-or-create should create a project if none exists"
    # verify it's actually in the DB
    row = conn.execute(
        "SELECT id FROM projects WHERE id=?", (project_id,)
    ).fetchone()
    assert row is not None, "created project must be in DB"


# ---------------------------------------------------------------------------
# TASK 6a — conversation micro-batch with X-Project-Id links chunks
# ---------------------------------------------------------------------------

def test_micro_batch_with_project_id_links_chunks(tmp_path, monkeypatch):
    conn = _make_db(tmp_path)
    monkeypatch.setattr(_deps, "_conn", conn)

    _seed_workspace(conn, "ws-5")
    pid = _seed_project(conn, "ws-5", "smoke/project-b")

    from src.api.routes.memory import conversation_micro_batch
    from src.api.routes.models import ConversationMicroBatchRequest, ConversationTurnModel

    chunk_id = "chk_test_conv_linked"

    def _fake_run_ingest(source_dict, content, conn_, chroma, ollama, model, opts, ws):
        # Insert the chunk directly so we can verify the link
        from mayring_core.memory.store import upsert_source, insert_chunk
        from mayring_core.memory.schema import Source, Chunk
        from datetime import datetime, timezone
        src = Source(
            source_id=source_dict["source_id"],
            source_type="conversation_summary",
            repo=ws,
            path=f"{ws}/incremental",
            branch="local",
            commit="",
            content_hash="sha256:aabbcc",
        )
        upsert_source(conn_, src, workspace_id=ws)
        chunk = Chunk(
            chunk_id=chunk_id,
            source_id=source_dict["source_id"],
            text="test conv content",
            text_hash=Chunk.compute_text_hash("test conv content"),
            created_at=datetime.now(timezone.utc).isoformat(),
            workspace_id=ws,
        )
        insert_chunk(conn_, chunk, workspace_id=ws)
        return {"status": "ok", "chunk_ids": [chunk_id], "indexed": True,
                "deduped": 0, "filtered": 0, "superseded": 0}

    request = ConversationMicroBatchRequest(
        turns=[ConversationTurnModel(role="user", content="hello", timestamp="2026-06-04T10:00:00Z")],
        session_id="sess-proj-link",
        presumarized="conversation about project B",
        origin_ref="https://github.com/smoke/project-b",
    )

    with patch("src.api.routes.memory._run_ingest", side_effect=_fake_run_ingest), \
         patch("src.api.routes.memory._get_conn", return_value=conn), \
         patch("src.api.routes.memory._get_chroma", return_value=MagicMock()):
        _maybe_run(conversation_micro_batch(request, workspace_id="ws-5", info=_INFO, x_project_id=pid))

    rows = conn.execute(
        "SELECT chunk_id, project_id, source FROM chunk_project_links WHERE project_id=?",
        (pid,),
    ).fetchall()
    assert len(rows) >= 1, "conversation chunk must be linked when X-Project-Id is provided"
    assert rows[0][2] == "conversation", f"source tag should be 'conversation', got {rows[0][2]!r}"
    assert rows[0][0] == chunk_id


# ---------------------------------------------------------------------------
# TASK 6b — conversation micro-batch WITHOUT X-Project-Id leaves no links
# ---------------------------------------------------------------------------

def test_micro_batch_without_project_id_leaves_no_links(tmp_path, monkeypatch):
    conn = _make_db(tmp_path)
    monkeypatch.setattr(_deps, "_conn", conn)

    _seed_workspace(conn, "ws-6")

    from src.api.routes.memory import conversation_micro_batch
    from src.api.routes.models import ConversationMicroBatchRequest, ConversationTurnModel

    request = ConversationMicroBatchRequest(
        turns=[ConversationTurnModel(role="user", content="hello", timestamp="2026-06-04T10:00:00Z")],
        session_id="sess-no-proj",
        presumarized="no project linked here",
    )

    def _fake_run_ingest(source_dict, content, conn_, chroma, ollama, model, opts, ws):
        return {"status": "ok", "chunk_ids": ["chk_unlinked"], "indexed": True,
                "deduped": 0, "filtered": 0, "superseded": 0}

    with patch("src.api.routes.memory._run_ingest", side_effect=_fake_run_ingest), \
         patch("src.api.routes.memory._get_conn", return_value=conn), \
         patch("src.api.routes.memory._get_chroma", return_value=MagicMock()):
        _maybe_run(conversation_micro_batch(request, workspace_id="ws-6", info=_INFO))

    rows = conn.execute("SELECT * FROM chunk_project_links").fetchall()
    assert rows == [], "without X-Project-Id, no chunk_project_links should be created"
