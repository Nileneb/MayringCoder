"""Shared API dependencies — DB and ChromaDB singletons.

Used by both server.py and mcp.py to avoid code duplication.
"""

from __future__ import annotations

import sqlite3

from src.memory.ingest import get_or_create_chroma_collection
from src.memory.store import init_memory_db

# Process-scoped lazy singletons
_conn: sqlite3.Connection | None = None
_chroma = None


def get_conn() -> sqlite3.Connection:
    """Return the shared SQLite memory DB connection (lazy singleton)."""
    global _conn
    if _conn is None:
        _conn = init_memory_db()
    return _conn


def get_chroma():
    """Return the shared ChromaDB 'memory_chunks' collection (lazy singleton)."""
    global _chroma
    if _chroma is None:
        _chroma = get_or_create_chroma_collection()
    return _chroma
