"""Shared API dependencies — DB and ChromaDB singletons.

Used by both server.py and mcp.py to avoid code duplication.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from mayring_core.memory.db_adapter import DBAdapter
from mayring_core.memory.ingest import get_or_create_chroma_collection
from mayring_core.memory.store import init_memory_db

# Explicit override slot — tests inject a connection via ``dependencies._conn``.
# In prod it stays None (see get_conn) so every thread uses its own connection.
_conn: DBAdapter | None = None
# Per-thread connections (api-concurrency-capacity §5.2): /memory/search runs in
# run_in_threadpool, and a single sqlite connection shared across those worker
# threads deadlocks (8b0ff34, reverted bbabe22). Each thread gets its own.
_conn_local = threading.local()
_chroma = None


def _build_conn() -> DBAdapter:
    local_db = os.environ.get("MAYRING_LOCAL_DB", "")
    if local_db:
        from mayring_core.memory.store import _init_schema
        conn = DBAdapter.create(local_db)
        _init_schema(conn)
        return conn
    return init_memory_db()


def get_conn() -> DBAdapter:
    """Return a SQLite memory DB connection.

    An explicitly injected ``_conn`` (tests) wins on every thread. Otherwise each
    thread gets its OWN connection (§5.2): the search body runs in a threadpool,
    so worker threads must not share one connection (deadlock). Reads parallelise
    via WAL; writes serialise via the file lock (busy_timeout=10s, no corruption).
    """
    if _conn is not None:
        return _conn
    conn = getattr(_conn_local, "conn", None)
    if conn is None:
        conn = _build_conn()
        _conn_local.conn = conn
    return conn


def get_chroma() -> Any:
    """Return the shared ChromaDB 'memory_chunks' collection (lazy singleton)."""
    global _chroma
    if _chroma is None:
        chroma_dir_override = os.environ.get("MAYRING_LOCAL_CHROMA", "")
        if chroma_dir_override:
            from mayring_core.memory.store import get_chroma_collection
            _chroma = get_chroma_collection("memory_chunks", path=Path(chroma_dir_override))
        else:
            _chroma = get_or_create_chroma_collection()
    return _chroma
