"""Workspace isolation: tenant tokens must not leak across workspaces."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.api import mcp_auth as mcp_mod
from src.api.jwt_auth import TokenInfo
from mayring_core.memory.retrieval import _scope_filter
from mayring_core.memory.schema import Chunk, Source
from mayring_core.memory.store import init_memory_db, insert_chunk, upsert_source


@pytest.fixture
def memory_db(tmp_path):
    """Isolated DB with seed chunks for two workspaces."""
    conn = init_memory_db(tmp_path / "memory.db")

    for ws in ("tenant-a", "tenant-b"):
        src = Source(
            source_id=f"repo-file:{ws}.py",
            source_type="repo_file",
            repo="acme/demo",
            path=f"{ws}.py",
            content_hash=f"sha256:{ws}",
            captured_at=datetime.now(timezone.utc).isoformat(),
        )
        upsert_source(conn, src, workspace_id=ws)

        chunk = Chunk(
            chunk_id=f"chk_{ws}_1",
            source_id=src.source_id,
            chunk_level="file",
            ordinal=0,
            start_offset=0,
            end_offset=10,
            text=f"content of {ws}",
            text_hash=f"sha256:chk_{ws}",
            embedding_model="nomic-embed-text",
            created_at=datetime.now(timezone.utc).isoformat(),
            is_active=True,
        )
        insert_chunk(conn, chunk, workspace_id=ws)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Retrieval-layer: _scope_filter enforces workspace_id
# ---------------------------------------------------------------------------

def test_tenant_a_sees_only_own_chunks(memory_db):
    ids = _scope_filter(memory_db, workspace_id="tenant-a")
    assert ids == ["chk_tenant-a_1"]


def test_tenant_b_sees_only_own_chunks(memory_db):
    ids = _scope_filter(memory_db, workspace_id="tenant-b")
    assert ids == ["chk_tenant-b_1"]


def test_visibility_user_crosses_workspaces_for_same_user(tmp_path):
    """Issue: claude.ai (workspace-A) and claude-cli (workspace-B) belong to the
    same human user (JWT.sub='42'). A source ingested in workspace-A with
    visibility='user' must surface in a search from workspace-B with the same
    user_id — without falling back to 'public'."""
    conn = init_memory_db(tmp_path / "user_share.db")

    src = Source(
        source_id="text:user-shared-note",
        source_type="note",
        repo="user-42",
        path="note",
        content_hash="sha256:abc",
        captured_at=datetime.now(timezone.utc).isoformat(),
        visibility="user",
        user_id="42",
    )
    upsert_source(conn, src, workspace_id="workspace-A")

    chunk = Chunk(
        chunk_id="chk_user_share",
        source_id=src.source_id,
        chunk_level="file",
        ordinal=0,
        start_offset=0,
        end_offset=10,
        text="cross-workspace user note",
        text_hash="sha256:user_share",
        embedding_model="nomic-embed-text",
        created_at=datetime.now(timezone.utc).isoformat(),
        is_active=True,
    )
    insert_chunk(conn, chunk, workspace_id="workspace-A")
    conn.commit()

    # Search from workspace-B with same user_id → must see the chunk
    same_user_ids = _scope_filter(conn, workspace_id="workspace-B", user_id="42")
    assert "chk_user_share" in same_user_ids

    # Search from workspace-B with DIFFERENT user_id → must NOT see it
    other_user_ids = _scope_filter(conn, workspace_id="workspace-B", user_id="99")
    assert "chk_user_share" not in other_user_ids

    # Search without user_id at all → also must NOT see it (no fallback)
    no_user_ids = _scope_filter(conn, workspace_id="workspace-B", user_id=None)
    assert "chk_user_share" not in no_user_ids


def test_no_workspace_filter_returns_all(memory_db):
    ids = _scope_filter(memory_db, workspace_id=None)
    assert sorted(ids) == ["chk_tenant-a_1", "chk_tenant-b_1"]


# ---------------------------------------------------------------------------
# mcp._enforce_tenant: JWT role decides override-capability
# ---------------------------------------------------------------------------

def test_enforce_tenant_locks_to_own_workspace():
    token = TokenInfo(workspace_id="tenant-a", scopes=())
    mcp_mod._TOKEN_CTX.set(token)
    try:
        # tenant requests cross-workspace — ignored, locked to own
        assert mcp_mod._enforce_tenant(None) == "tenant-a"
        # tenant requests someone else's workspace — ignored, locked to own
        assert mcp_mod._enforce_tenant("tenant-b") == "tenant-a"
    finally:
        mcp_mod._TOKEN_CTX.set(None)


def test_admin_can_query_any_workspace():
    token = TokenInfo(workspace_id="ops", scopes=("admin",))
    mcp_mod._TOKEN_CTX.set(token)
    try:
        # admin may request a specific tenant
        assert mcp_mod._enforce_tenant("tenant-a") == "tenant-a"
        # admin may request cross-workspace (None)
        assert mcp_mod._enforce_tenant(None) is None
    finally:
        mcp_mod._TOKEN_CTX.set(None)


def test_stdio_mode_passes_through_requested():
    mcp_mod._TOKEN_CTX.set(None)
    assert mcp_mod._enforce_tenant("anything") == "anything"
    assert mcp_mod._enforce_tenant(None) is None


def test_effective_workspace_id_falls_back_on_stdio():
    mcp_mod._TOKEN_CTX.set(None)
    assert mcp_mod._effective_workspace_id("default") == "default"


def test_effective_workspace_id_uses_jwt_in_http():
    token = TokenInfo(workspace_id="bene-workspace", scopes=())
    mcp_mod._TOKEN_CTX.set(token)
    try:
        assert mcp_mod._effective_workspace_id("default") == "bene-workspace"
    finally:
        mcp_mod._TOKEN_CTX.set(None)


# ---------------------------------------------------------------------------
# Retrieval-layer: Chroma filter failure must not leak cross-workspace data
# ---------------------------------------------------------------------------

def test_chroma_filter_failure_skips_vector_not_leaks(memory_db, monkeypatch):
    """Chroma filter exception → vector stage skipped entirely, no cross-workspace leak."""
    import mayring_core.memory.retrieval as ret_mod

    monkeypatch.setattr(ret_mod, "_HAS_EMBED", True)
    monkeypatch.setattr(ret_mod, "_embed_texts", lambda texts, url: [[0.1] * 4] * len(texts))

    class FilterFailingChroma:
        def count(self):
            return 5

        def query(self, **kwargs):
            if kwargs.get("where"):
                raise RuntimeError("where filter not supported by this chroma version")
            # Would return cross-workspace data without filter — must NEVER be reached
            return {"ids": [["chk_tenant-b_1"]], "distances": [[0.0]]}

    results = ret_mod.search(
        "content",
        memory_db,
        chroma_collection=FilterFailingChroma(),
        ollama_url="http://fake",
        opts={"workspace_id": "tenant-a", "top_k": 5},
    )
    result_ids = {r.chunk_id for r in results}
    assert "chk_tenant-b_1" not in result_ids, "tenant-b data leaked via chroma fallback"
    assert "chk_tenant-a_1" in result_ids


def test_chroma_workspace_a_cannot_see_workspace_b_data(memory_db, monkeypatch):
    """Stage-1 scope filter blocks cross-workspace chunks even if chroma returns them."""
    import mayring_core.memory.retrieval as ret_mod

    monkeypatch.setattr(ret_mod, "_HAS_EMBED", True)
    monkeypatch.setattr(ret_mod, "_embed_texts", lambda texts, url: [[0.1] * 4] * len(texts))

    class LeakyChroma:
        """Simulates a broken chroma that ignores workspace filter and returns all chunks."""
        def count(self):
            return 2

        def query(self, **kwargs):
            return {"ids": [["chk_tenant-a_1", "chk_tenant-b_1"]], "distances": [[0.0, 0.1]]}

    results = ret_mod.search(
        "content",
        memory_db,
        chroma_collection=LeakyChroma(),
        ollama_url="http://fake",
        opts={"workspace_id": "tenant-a", "top_k": 5},
    )
    result_ids = {r.chunk_id for r in results}
    assert "chk_tenant-b_1" not in result_ids, "tenant-b chunk leaked into tenant-a results"
    assert "chk_tenant-a_1" in result_ids


# ---------------------------------------------------------------------------
# CLI: workspace_id="default" must print a visible warning
# ---------------------------------------------------------------------------

def test_unknown_workspace_id_raises(monkeypatch, tmp_path):
    """Big-Bang: 'default' ist kein Free-Pass mehr. Wer einen unbekannten
    workspace_id übergibt UND keine MAYRING_USER_ID hat → UnknownWorkspaceError.

    Vorher: silent Fallback auf 'default'-Bucket (ergab Datenchaos zwischen
    JWT- und CLI-Pfaden).
    """
    from types import SimpleNamespace
    from mayring_core.identity.cli import resolve_cli_workspace
    from mayring_core.identity.workspace_resolver import UnknownWorkspaceError
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema

    monkeypatch.delenv("MAYRING_USER_ID", raising=False)
    conn = DBAdapter.create(tmp_path / "test.db", check_same_thread=False)
    _init_schema(conn)

    args = SimpleNamespace(workspace_id="default")
    with pytest.raises(UnknownWorkspaceError):
        resolve_cli_workspace(args, conn=conn)


def test_missing_workspace_with_local_email_resolves(monkeypatch, tmp_path):
    """Wenn kein --workspace-id, aber identity.json mit email vorhanden ist:
    Slug aus email-localpart, kein user-N-Fallback."""
    import json
    from types import SimpleNamespace
    from mayring_core.identity.cli import resolve_cli_workspace
    from mayring_core.memory.db_adapter import DBAdapter
    from mayring_core.memory.store import _init_schema

    cfg_dir = tmp_path / ".config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    monkeypatch.setenv("MAYRING_USER_ID", "7")
    (cfg_dir / "mayring").mkdir(parents=True, exist_ok=True)
    (cfg_dir / "mayring" / "identity.json").write_text(json.dumps({
        "user_id": 7, "email": "alice@x.de", "token": None,
    }))
    conn = DBAdapter.create(tmp_path / "test.db", check_same_thread=False)
    _init_schema(conn)

    args = SimpleNamespace(workspace_id=None)
    assert resolve_cli_workspace(args, conn=conn) == "alice"
