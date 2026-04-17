"""Multi-tenancy isolation tests.

Verifies that workspace_id properly scopes:
- SQLite writes (sources, chunks)
- Dedup (workspace-scoped: same content in two workspaces = two separate chunks)
- Retrieval scope filter (workspace A cannot see workspace B's data)
- Chunk deactivation (only affects the target workspace)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.memory.schema import Chunk, Source
from src.memory.store import (
    deactivate_chunks_by_source,
    find_by_text_hash,
    get_active_chunk_count,
    get_chunks_by_source,
    init_memory_db,
    insert_chunk,
    upsert_source,
)
from src.memory.retrieval import _scope_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(source_id: str, repo: str = "owner/test", ws: str = "default") -> Source:
    return Source(
        source_id=source_id,
        source_type="repo_file",
        repo=repo,
        path=source_id.split(":", 2)[-1] if ":" in source_id else source_id,
        branch="main",
        commit="abc",
        content_hash="sha256:x",
        captured_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_chunk(source_id: str, ordinal: int = 0, text: str = "def foo(): pass") -> Chunk:
    text_hash = Chunk.compute_text_hash(text)
    return Chunk(
        chunk_id=Chunk.make_id(source_id, ordinal, "function"),
        source_id=source_id,
        chunk_level="function",
        ordinal=ordinal,
        text=text,
        text_hash=text_hash,
        category_labels=["api"],
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Tests: workspace_id stored in DB
# ---------------------------------------------------------------------------

class TestWorkspaceIdPersisted:
    def test_upsert_source_stores_workspace_id(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source("repo:owner/test:foo.py")
        upsert_source(conn, src, workspace_id="ws_alice")

        row = conn.execute(
            "SELECT workspace_id FROM sources WHERE source_id = ?", (src.source_id,)
        ).fetchone()
        assert row is not None
        assert row[0] == "ws_alice"

    def test_insert_chunk_stores_workspace_id(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source("repo:owner/test:foo.py")
        upsert_source(conn, src, workspace_id="ws_alice")
        chunk = _make_chunk(src.source_id, 0)
        insert_chunk(conn, chunk, workspace_id="ws_alice")

        row = conn.execute(
            "SELECT workspace_id FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
        ).fetchone()
        assert row is not None
        assert row[0] == "ws_alice"

    def test_default_workspace_backward_compat(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source("repo:owner/test:foo.py")
        upsert_source(conn, src)  # no workspace_id arg
        chunk = _make_chunk(src.source_id, 0)
        insert_chunk(conn, chunk)  # no workspace_id arg

        row_s = conn.execute(
            "SELECT workspace_id FROM sources WHERE source_id = ?", (src.source_id,)
        ).fetchone()
        row_c = conn.execute(
            "SELECT workspace_id FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
        ).fetchone()
        assert row_s[0] == "default"
        assert row_c[0] == "default"


# ---------------------------------------------------------------------------
# Tests: dedup is workspace-scoped
# ---------------------------------------------------------------------------

class TestWorkspaceScopedDedup:
    def test_same_content_two_workspaces_not_deduped(self, tmp_path: Path) -> None:
        """Same text in ws_a and ws_b should produce separate chunks."""
        conn = init_memory_db(tmp_path / "m.db")

        src_a = _make_source("repo:owner/test:foo.py")
        src_b = _make_source("repo:owner/test:bar.py")
        upsert_source(conn, src_a, workspace_id="ws_a")
        upsert_source(conn, src_b, workspace_id="ws_b")

        text = "def shared_function(): pass"
        chunk_a = _make_chunk(src_a.source_id, 0, text=text)
        chunk_b = _make_chunk(src_b.source_id, 0, text=text)
        # Different chunk_ids (different source_ids) — both should be inserted
        insert_chunk(conn, chunk_a, workspace_id="ws_a")
        insert_chunk(conn, chunk_b, workspace_id="ws_b")

        # find_by_text_hash should be workspace-scoped
        found_a = find_by_text_hash(conn, chunk_a.text_hash, workspace_id="ws_a")
        found_b = find_by_text_hash(conn, chunk_b.text_hash, workspace_id="ws_b")
        not_found = find_by_text_hash(conn, chunk_a.text_hash, workspace_id="ws_c")

        assert found_a is not None
        assert found_b is not None
        assert not_found is None

    def test_dedup_within_workspace(self, tmp_path: Path) -> None:
        """Inserting same text twice in same workspace: second find_by_text_hash hits."""
        conn = init_memory_db(tmp_path / "m.db")
        src = _make_source("repo:owner/test:foo.py")
        upsert_source(conn, src, workspace_id="ws_a")
        chunk = _make_chunk(src.source_id, 0, text="unique text for dedup test")
        insert_chunk(conn, chunk, workspace_id="ws_a")

        found = find_by_text_hash(conn, chunk.text_hash, workspace_id="ws_a")
        assert found is not None
        assert found.chunk_id == chunk.chunk_id


# ---------------------------------------------------------------------------
# Tests: scope filter isolates workspaces
# ---------------------------------------------------------------------------

class TestScopeFilterWorkspaceIsolation:
    def _setup_two_workspaces(self, conn):
        """Insert one chunk in ws_a and one in ws_b, same repo."""
        src_a = _make_source("repo:owner/shared:a.py", repo="owner/shared")
        src_b = _make_source("repo:owner/shared:b.py", repo="owner/shared")
        upsert_source(conn, src_a, workspace_id="ws_a")
        upsert_source(conn, src_b, workspace_id="ws_b")
        insert_chunk(conn, _make_chunk(src_a.source_id, 0, text="workspace a content"), workspace_id="ws_a")
        insert_chunk(conn, _make_chunk(src_b.source_id, 0, text="workspace b content"), workspace_id="ws_b")
        return src_a, src_b

    def test_filter_ws_a_excludes_ws_b(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src_a, src_b = self._setup_two_workspaces(conn)

        ids_a = _scope_filter(conn, workspace_id="ws_a")
        chunk_a = Chunk.make_id(src_a.source_id, 0, "function")
        chunk_b = Chunk.make_id(src_b.source_id, 0, "function")

        assert chunk_a in ids_a
        assert chunk_b not in ids_a

    def test_filter_ws_b_excludes_ws_a(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        src_a, src_b = self._setup_two_workspaces(conn)

        ids_b = _scope_filter(conn, workspace_id="ws_b")
        chunk_a = Chunk.make_id(src_a.source_id, 0, "function")
        chunk_b = Chunk.make_id(src_b.source_id, 0, "function")

        assert chunk_b in ids_b
        assert chunk_a not in ids_b

    def test_no_workspace_filter_returns_all(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        self._setup_two_workspaces(conn)

        all_ids = _scope_filter(conn)  # no workspace_id
        assert len(all_ids) == 2

    def test_unknown_workspace_returns_empty(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")
        self._setup_two_workspaces(conn)

        ids = _scope_filter(conn, workspace_id="ws_unknown")
        assert ids == []


# ---------------------------------------------------------------------------
# Tests: deactivation does not cross workspace boundary
# ---------------------------------------------------------------------------

class TestDeactivationIsolation:
    def test_deactivate_only_affects_given_source(self, tmp_path: Path) -> None:
        conn = init_memory_db(tmp_path / "m.db")

        src_a = _make_source("repo:owner/test:a.py")
        src_b = _make_source("repo:owner/test:b.py")
        upsert_source(conn, src_a, workspace_id="ws_a")
        upsert_source(conn, src_b, workspace_id="ws_b")
        chunk_a = _make_chunk(src_a.source_id, 0)
        chunk_b = _make_chunk(src_b.source_id, 0)
        insert_chunk(conn, chunk_a, workspace_id="ws_a")
        insert_chunk(conn, chunk_b, workspace_id="ws_b")

        # Deactivate source A
        deactivate_chunks_by_source(conn, src_a.source_id)

        # Source B's chunk still active
        remaining = get_active_chunk_count(conn)
        assert remaining == 1

        active_b = get_chunks_by_source(conn, src_b.source_id, active_only=True)
        assert len(active_b) == 1
