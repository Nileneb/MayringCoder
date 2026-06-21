"""Admin purge-by-source-type: deactivate pure-noise chunks (log_event) without
touching code/recall. Safelist-gated so repo_file/note can never be nuked.

WHY(corpus-noise 2026-06-21): /memory/log-event ingested every app logger line as a
searchable chunk → 2193 internal WARNINGs polluted code retrieval. This is the purge
tool for that backlog; the endpoint feeds the same loop tested here."""
from __future__ import annotations

from mayring_core.memory import store
from mayring_core.memory.schema import Chunk, Source
from src.api.routes.purge_admin import _PURGEABLE_NOISE_TYPES


def _seed(conn, source_id, source_type, chunk_id):
    store.upsert_source(conn, Source(source_id=source_id, source_type=source_type, repo="", path="p"),
                        workspace_id="ws")
    store.insert_chunk(conn, Chunk(chunk_id=chunk_id, source_id=source_id, text="t", text_hash=chunk_id),
                       workspace_id="ws")


def _deactivate_source_type(conn, source_type: str) -> int:
    # mirrors purge_source_type_route._run()
    from mayring_core.memory.store import deactivate_chunks_by_source
    rows = conn.execute(
        "SELECT DISTINCT ch.source_id FROM chunks ch JOIN sources s ON ch.source_id = s.source_id "
        "WHERE ch.is_active = 1 AND s.source_type = ?", (source_type,)).fetchall()
    return sum(deactivate_chunks_by_source(conn, r[0]) for r in rows)


def test_purges_only_target_source_type(tmp_path):
    conn = store.init_memory_db(tmp_path / "m.db")
    _seed(conn, "log:svc:sig:1", "log_event", "c-log1")
    _seed(conn, "log:svc:sig:2", "log_event", "c-log2")
    _seed(conn, "repo:x:file.py", "repo_file", "c-code")

    deactivated = _deactivate_source_type(conn, "log_event")

    assert deactivated == 2
    act = lambda cid: conn.execute("SELECT is_active FROM chunks WHERE chunk_id=?", (cid,)).fetchone()[0]
    assert act("c-log1") == 0 and act("c-log2") == 0
    assert act("c-code") == 1  # code untouched


def test_safelist_excludes_code_and_recall_types():
    # the endpoint 422s anything outside this set — guard against a typo nuking real data
    assert "log_event" in _PURGEABLE_NOISE_TYPES
    for protected in ("repo_file", "note", "conversation_summary", "knowledge", "agent_result"):
        assert protected not in _PURGEABLE_NOISE_TYPES
