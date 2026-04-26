"""Shared API dependencies — DB and ChromaDB singletons.

Used by both server.py and mcp.py to avoid code duplication.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.memory.db_adapter import DBAdapter
from src.memory.ingest import get_or_create_chroma_collection
from src.memory.store import init_memory_db

# Process-scoped lazy singletons
_conn: DBAdapter | None = None
_chroma = None


def get_conn() -> DBAdapter:
    """Return the shared SQLite memory DB connection (lazy singleton)."""
    global _conn
    if _conn is None:
        local_db = os.environ.get("MAYRING_LOCAL_DB", "")
        if local_db:
            from src.memory.store import _init_schema
            _conn = DBAdapter.create(local_db)
            _init_schema(_conn)
        else:
            _conn = init_memory_db()
    return _conn


def get_chroma() -> Any:
    """Return the shared ChromaDB 'memory_chunks' collection (lazy singleton)."""
    global _chroma
    if _chroma is None:
        chroma_dir_override = os.environ.get("MAYRING_LOCAL_CHROMA", "")
        if chroma_dir_override:
            from src.memory.store import get_chroma_collection
            _chroma = get_chroma_collection("memory_chunks", path=Path(chroma_dir_override))
        else:
            _chroma = get_or_create_chroma_collection()
    return _chroma
