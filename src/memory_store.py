"""Memory Storage Layer — SQLite schema for sources, chunks, feedback, and ingestion log.

This module manages a SEPARATE SQLite database at cache/memory.db.
It does NOT touch the per-repo .db files managed by cache.py.

Tables:
    sources         — ingested source records
    chunks          — versioned memory chunks
    chunk_feedback  — user feedback per chunk
    ingestion_log   — audit trail for all ingestion events
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config import CACHE_DIR
from src.memory_schema import Chunk, Source

MEMORY_DB_PATH: Path = CACHE_DIR / "memory.db"

# ---------------------------------------------------------------------------
# In-memory KV cache (session-scoped, evicted on process exit)
# ---------------------------------------------------------------------------
_KV: dict[str, dict] = {}


def kv_get(chunk_id: str) -> dict | None:
    return _KV.get(chunk_id)


def kv_put(chunk_id: str, chunk_dict: dict) -> None:
    _KV[chunk_id] = chunk_dict


def kv_invalidate_by_ids(chunk_ids: list[str]) -> None:
    for cid in chunk_ids:
        _KV.pop(cid, None)


# ---------------------------------------------------------------------------
# DB init
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_memory_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Create or open memory.db and ensure all tables + indexes exist."""
    path = db_path or MEMORY_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            source_id       TEXT PRIMARY KEY,
            source_type     TEXT NOT NULL DEFAULT 'repo_file',
            repo            TEXT NOT NULL DEFAULT '',
            path            TEXT NOT NULL DEFAULT '',
            branch          TEXT NOT NULL DEFAULT 'main',
            "commit"        TEXT NOT NULL DEFAULT '',
            content_hash    TEXT NOT NULL DEFAULT '',
            captured_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id            TEXT PRIMARY KEY,
            source_id           TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
            parent_chunk_id     TEXT,
            chunk_level         TEXT NOT NULL DEFAULT 'file',
            ordinal             INTEGER NOT NULL DEFAULT 0,
            start_offset        INTEGER NOT NULL DEFAULT 0,
            end_offset          INTEGER NOT NULL DEFAULT 0,
            text                TEXT NOT NULL DEFAULT '',
            text_hash           TEXT NOT NULL DEFAULT '',
            summary             TEXT NOT NULL DEFAULT '',
            category_labels     TEXT NOT NULL DEFAULT '',
            category_version    TEXT NOT NULL DEFAULT 'mayring-inductive-v1',
            embedding_model     TEXT NOT NULL DEFAULT 'nomic-embed-text',
            embedding_id        TEXT NOT NULL DEFAULT '',
            quality_score       REAL NOT NULL DEFAULT 0.0,
            dedup_key           TEXT NOT NULL DEFAULT '',
            created_at          TEXT NOT NULL,
            superseded_by       TEXT,
            is_active           INTEGER NOT NULL DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_source_id
            ON chunks(source_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_text_hash
            ON chunks(text_hash);
        CREATE INDEX IF NOT EXISTS idx_chunks_dedup_key
            ON chunks(dedup_key);
        CREATE INDEX IF NOT EXISTS idx_chunks_is_active
            ON chunks(is_active);

        CREATE TABLE IF NOT EXISTS chunk_feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id        TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            signal          TEXT NOT NULL,
            metadata        TEXT NOT NULL DEFAULT '{}',
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_chunk_id
            ON chunk_feedback(chunk_id);

        CREATE TABLE IF NOT EXISTS ingestion_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id       TEXT NOT NULL DEFAULT '',
            event_type      TEXT NOT NULL,
            payload         TEXT NOT NULL DEFAULT '{}',
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_ingestion_log_source_id
            ON ingestion_log(source_id);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Source operations
# ---------------------------------------------------------------------------

def upsert_source(conn: sqlite3.Connection, source: Source) -> None:
    conn.execute(
        """
        INSERT INTO sources
            (source_id, source_type, repo, path, branch, "commit", content_hash, captured_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            source_type  = excluded.source_type,
            repo         = excluded.repo,
            path         = excluded.path,
            branch       = excluded.branch,
            "commit"     = excluded."commit",
            content_hash = excluded.content_hash,
            captured_at  = excluded.captured_at
        """,
        (
            source.source_id, source.source_type, source.repo, source.path,
            source.branch, source.commit, source.content_hash, source.captured_at,
        ),
    )
    conn.commit()


def get_source(conn: sqlite3.Connection, source_id: str) -> Source | None:
    row = conn.execute(
        "SELECT * FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    return Source(
        source_id=d["source_id"],
        source_type=d["source_type"],
        repo=d["repo"],
        path=d["path"],
        branch=d["branch"],
        commit=d["commit"],
        content_hash=d["content_hash"],
        captured_at=d["captured_at"],
    )


# ---------------------------------------------------------------------------
# Chunk operations
# ---------------------------------------------------------------------------

def insert_chunk(conn: sqlite3.Connection, chunk: Chunk) -> None:
    """Insert a new chunk. category_labels stored as comma-joined string."""
    labels_str = ",".join(chunk.category_labels)
    conn.execute(
        """
        INSERT INTO chunks
            (chunk_id, source_id, parent_chunk_id, chunk_level, ordinal,
             start_offset, end_offset, text, text_hash, summary, category_labels,
             category_version, embedding_model, embedding_id, quality_score,
             dedup_key, created_at, superseded_by, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            text=excluded.text, text_hash=excluded.text_hash,
            summary=excluded.summary, category_labels=excluded.category_labels,
            is_active=1, superseded_by=NULL, created_at=excluded.created_at
        """,
        (
            chunk.chunk_id, chunk.source_id, chunk.parent_chunk_id,
            chunk.chunk_level, chunk.ordinal, chunk.start_offset, chunk.end_offset,
            chunk.text, chunk.text_hash, chunk.summary, labels_str,
            chunk.category_version, chunk.embedding_model, chunk.embedding_id,
            chunk.quality_score, chunk.dedup_key, chunk.created_at,
            chunk.superseded_by, int(chunk.is_active),
        ),
    )
    conn.commit()


def get_chunk(conn: sqlite3.Connection, chunk_id: str, active_only: bool = True) -> Chunk | None:
    query = "SELECT * FROM chunks WHERE chunk_id = ?"
    if active_only:
        query += " AND is_active = 1"
    row = conn.execute(query, (chunk_id,)).fetchone()
    if row is None:
        return None
    return Chunk.from_dict(dict(row))


def get_chunks_by_source(
    conn: sqlite3.Connection, source_id: str, active_only: bool = True
) -> list[Chunk]:
    query = "SELECT * FROM chunks WHERE source_id = ?"
    params: list = [source_id]
    if active_only:
        query += " AND is_active = 1"
    query += " ORDER BY ordinal"
    rows = conn.execute(query, params).fetchall()
    return [Chunk.from_dict(dict(r)) for r in rows]


def find_by_text_hash(conn: sqlite3.Connection, text_hash: str) -> Chunk | None:
    """Exact dedup: return first active chunk with this text_hash, or None."""
    row = conn.execute(
        "SELECT * FROM chunks WHERE text_hash = ? AND is_active = 1 LIMIT 1",
        (text_hash,),
    ).fetchone()
    return Chunk.from_dict(dict(row)) if row else None


def supersede_chunk(
    conn: sqlite3.Connection, old_chunk_id: str, new_chunk_id: str
) -> None:
    """Mark old chunk as inactive, point superseded_by to new chunk."""
    conn.execute(
        "UPDATE chunks SET is_active = 0, superseded_by = ? WHERE chunk_id = ?",
        (new_chunk_id, old_chunk_id),
    )
    conn.commit()


def deactivate_chunks_by_source(
    conn: sqlite3.Connection, source_id: str
) -> int:
    """Set is_active=0 for all active chunks of source_id. Returns count."""
    conn.execute(
        "UPDATE chunks SET is_active = 0 WHERE source_id = ? AND is_active = 1",
        (source_id,),
    )
    conn.commit()
    return conn.execute("SELECT changes()").fetchone()[0]


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def add_feedback(
    conn: sqlite3.Connection,
    chunk_id: str,
    signal: str,
    metadata: dict | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO chunk_feedback (chunk_id, signal, metadata, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (chunk_id, signal, json.dumps(metadata or {}), _now_iso()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Ingestion log
# ---------------------------------------------------------------------------

def log_ingestion_event(
    conn: sqlite3.Connection,
    source_id: str,
    event_type: str,
    payload: dict,
) -> None:
    conn.execute(
        """
        INSERT INTO ingestion_log (source_id, event_type, payload, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (source_id, event_type, json.dumps(payload), _now_iso()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_active_chunk_count(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE is_active = 1"
    ).fetchone()[0]


def get_source_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
