"""Memory Storage Layer — SQLite + ChromaDB singletons.

Tables:
    sources, chunks, chunk_feedback, ingestion_log

ChromaDB:
    get_chroma_collection(name, path) — process-scoped singleton
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR

# ---------------------------------------------------------------------------
# ChromaDB process-scoped singleton (replaces chroma_factory.py)
# ---------------------------------------------------------------------------

_chroma_clients: dict[str, Any] = {}
_chroma_collections: dict[str, Any] = {}


def get_chroma_collection(
    name: str = "memory_chunks",
    path: str | Path | None = None,
) -> Any:
    """Return a process-scoped ChromaDB collection singleton.

    The same (name, path) pair always returns the same object, preventing
    multiple PersistentClient instances pointing at the same directory.
    Returns None if chromadb is not installed.
    """
    try:
        import chromadb
    except ImportError:
        return None
    chroma_path = str(path or CACHE_DIR / "memory_chroma")
    key = f"{chroma_path}::{name}"
    if key not in _chroma_collections:
        if chroma_path not in _chroma_clients:
            Path(chroma_path).mkdir(parents=True, exist_ok=True)
            _chroma_clients[chroma_path] = chromadb.PersistentClient(path=chroma_path)
        _chroma_collections[key] = _chroma_clients[chroma_path].get_or_create_collection(name)
    return _chroma_collections[key]
from src.memory.schema import Chunk, Source

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


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add missing columns to existing DBs (idempotent migrations)."""
    migrations = {
        "sources": [
            ("workspace_id", "TEXT NOT NULL DEFAULT 'default'"),
        ],
        "chunks": [
            ("workspace_id", "TEXT NOT NULL DEFAULT 'default'"),
            ("category_source", "TEXT NOT NULL DEFAULT ''"),
            ("category_confidence", "REAL NOT NULL DEFAULT 0.0"),
        ],
    }
    for table, columns in migrations.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col_name, col_def in columns:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")


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

        CREATE TABLE IF NOT EXISTS chunk_source_refs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_chunk_id  TEXT NOT NULL,
            source_id       TEXT NOT NULL,
            workspace_id    TEXT NOT NULL DEFAULT 'default',
            created_at      TEXT NOT NULL,
            UNIQUE(canonical_chunk_id, source_id)
        );

        CREATE INDEX IF NOT EXISTS idx_chunk_source_refs_canonical
            ON chunk_source_refs(canonical_chunk_id);
        CREATE INDEX IF NOT EXISTS idx_chunk_source_refs_source
            ON chunk_source_refs(source_id);

        CREATE TABLE IF NOT EXISTS ingestion_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id       TEXT NOT NULL DEFAULT '',
            event_type      TEXT NOT NULL,
            payload         TEXT NOT NULL DEFAULT '{}',
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_ingestion_log_source_id
            ON ingestion_log(source_id);

        CREATE TABLE IF NOT EXISTS wiki_paper_cache (
            source_id    TEXT NOT NULL,
            rule_name    TEXT NOT NULL,
            extracted    TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            PRIMARY KEY (source_id, rule_name)
        );
    """)

    # Migration: add missing columns to existing DBs
    _migrate_schema(conn)

    # Indexes for workspace_id filtering
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_workspace_id ON chunks(workspace_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sources_workspace_id ON sources(workspace_id)"
    )

    conn.commit()


# ---------------------------------------------------------------------------
# Source operations
# ---------------------------------------------------------------------------

def upsert_source(
    conn: sqlite3.Connection, source: Source, workspace_id: str = "default"
) -> None:
    conn.execute(
        """
        INSERT INTO sources
            (source_id, source_type, repo, path, branch, "commit", content_hash,
             captured_at, workspace_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            source_type  = excluded.source_type,
            repo         = excluded.repo,
            path         = excluded.path,
            branch       = excluded.branch,
            "commit"     = excluded."commit",
            content_hash = excluded.content_hash,
            captured_at  = excluded.captured_at,
            workspace_id = excluded.workspace_id
        """,
        (
            source.source_id, source.source_type, source.repo, source.path,
            source.branch, source.commit, source.content_hash, source.captured_at,
            workspace_id,
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

def insert_chunk(
    conn: sqlite3.Connection, chunk: Chunk, workspace_id: str = "default"
) -> None:
    """Insert a new chunk. category_labels stored as comma-joined string."""
    labels_str = ",".join(chunk.category_labels)
    conn.execute(
        """
        INSERT INTO chunks
            (chunk_id, source_id, parent_chunk_id, chunk_level, ordinal,
             start_offset, end_offset, text, text_hash, summary, category_labels,
             category_version, embedding_model, embedding_id, quality_score,
             dedup_key, category_source, category_confidence,
             created_at, superseded_by, is_active, workspace_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            text=excluded.text, text_hash=excluded.text_hash,
            summary=excluded.summary, category_labels=excluded.category_labels,
            category_source=excluded.category_source,
            category_confidence=excluded.category_confidence,
            is_active=1, superseded_by=NULL, created_at=excluded.created_at,
            workspace_id=excluded.workspace_id
        """,
        (
            chunk.chunk_id, chunk.source_id, chunk.parent_chunk_id,
            chunk.chunk_level, chunk.ordinal, chunk.start_offset, chunk.end_offset,
            chunk.text, chunk.text_hash, chunk.summary, labels_str,
            chunk.category_version, chunk.embedding_model, chunk.embedding_id,
            chunk.quality_score, chunk.dedup_key,
            chunk.category_source, chunk.category_confidence,
            chunk.created_at, chunk.superseded_by, int(chunk.is_active), workspace_id,
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


def find_by_text_hash(
    conn: sqlite3.Connection, text_hash: str, workspace_id: str = "default"
) -> Chunk | None:
    """Exact dedup: return first active chunk with this text_hash in workspace, or None."""
    row = conn.execute(
        "SELECT * FROM chunks WHERE text_hash = ? AND is_active = 1 AND workspace_id = ? LIMIT 1",
        (text_hash, workspace_id),
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


# ---------------------------------------------------------------------------
# Cross-source references (dedup co-occurrence)
# ---------------------------------------------------------------------------

def add_source_ref(
    conn: sqlite3.Connection,
    canonical_chunk_id: str,
    source_id: str,
    workspace_id: str = "default",
) -> None:
    """Record that source_id contains the same text as canonical_chunk_id."""
    conn.execute(
        """
        INSERT OR IGNORE INTO chunk_source_refs
            (canonical_chunk_id, source_id, workspace_id, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (canonical_chunk_id, source_id, workspace_id, _now_iso()),
    )
    conn.commit()


def get_source_refs(
    conn: sqlite3.Connection,
    canonical_chunk_id: str,
) -> list[str]:
    """Return all source_ids that share the same text as canonical_chunk_id."""
    rows = conn.execute(
        "SELECT source_id FROM chunk_source_refs WHERE canonical_chunk_id = ?",
        (canonical_chunk_id,),
    ).fetchall()
    return [r[0] for r in rows]
