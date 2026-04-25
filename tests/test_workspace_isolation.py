"""Workspace isolation: tenant tokens must not leak across workspaces."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.api import mcp_auth as mcp_mod
from src.api.jwt_auth import TokenInfo
from src.memory.retrieval import _scope_filter
from src.memory.schema import Chunk, Source
from src.memory.store import init_memory_db, insert_chunk, upsert_source


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
    import src.memory.retrieval as ret_mod

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
    import src.memory.retrieval as ret_mod

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

def test_default_workspace_warning(capsys, monkeypatch):
    """main() prints Warnung when --workspace-id is not set (falls back to 'default')."""
    import sys
    import types
    import src.cli as cli_mod

    fake_args = types.SimpleNamespace(
        repo="http://fake",
        mode="ingest",
        llm=False,
        resolve_model_only=False,
        model=None,
        run_id=None,
        cache_by_model=False,
        log_training_data=False,
        max_chars=None,
        batch_size=None,
        batch_delay=None,
        ingest_issues=False,
        ingest_images=False,
        pi_task=None,
        generate_wiki=False,
        reset=False,
        history=False,
        compare=None,
        cleanup=None,
        codebook=None,
        prompt=None,
        full=False,
        workspace_id="default",
        populate_memory=True,
    )
    monkeypatch.setattr(cli_mod, "parse_args", lambda: fake_args)
    monkeypatch.setattr(cli_mod, "run_populate_memory", lambda *a, **kw: None)

    with pytest.raises(SystemExit):
        cli_mod.main()

    out = capsys.readouterr().out
    assert "Warnung" in out
    assert "workspace-id" in out
