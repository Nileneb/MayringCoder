"""SQLite cache with snapshot-based diff detection.

Schema
------
snapshots(id, repo, commit, created_at, raw_source_ref)
file_versions(id, snapshot_id FK, filename, hash, size)
"""

import re
import sqlite3
from datetime import datetime
from urllib.parse import urlparse

from src.config import CACHE_DIR, MAX_FILES_PER_RUN, RISK_CATEGORIES


def _repo_slug(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    slug = parsed.path.strip("/").replace("/", "-").lower()
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    return slug or "repo"


def reset_repo(repo_url: str, run_key: str | None = None) -> str | None:
    """Reset cache for a repo.

    - Without run_key: delete the SQLite DB for the repo (legacy behavior).
    - With run_key: clear analyzed_at stamps only for that run key.
    """
    db_path = CACHE_DIR / f"{_repo_slug(repo_url)}.db"
    if run_key is not None:
        if not db_path.exists():
            return None
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE file_versions SET analyzed_at = NULL WHERE COALESCE(run_key, 'default') = ?",
            (run_key,),
        )
        conn.commit()
        conn.close()
        return f"{db_path} (run_key={run_key})"
    if db_path.exists():
        db_path.unlink()
        return str(db_path)
    return None


def init_db(repo_url: str) -> sqlite3.Connection:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    db_path = CACHE_DIR / f"{_repo_slug(repo_url)}.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            repo           TEXT    NOT NULL,
            commit_hash    TEXT,
            created_at     TEXT    NOT NULL,
            raw_source_ref TEXT
        );
        CREATE TABLE IF NOT EXISTS file_versions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
            filename    TEXT    NOT NULL,
            hash        TEXT    NOT NULL,
            size        INTEGER NOT NULL DEFAULT 0,
            analyzed_at TEXT,
            run_key     TEXT    NOT NULL DEFAULT 'default'
        );
        CREATE INDEX IF NOT EXISTS idx_fv_snapshot
            ON file_versions(snapshot_id);
        CREATE INDEX IF NOT EXISTS idx_fv_filename_hash
            ON file_versions(filename, hash);
    """)
    conn.commit()

    # Migration: add analyzed_at to existing databases that predate this column
    cols = {row[1] for row in conn.execute("PRAGMA table_info(file_versions)").fetchall()}
    if "analyzed_at" not in cols:
        conn.execute("ALTER TABLE file_versions ADD COLUMN analyzed_at TEXT")
        conn.commit()

    if "run_key" not in cols:
        conn.execute("ALTER TABLE file_versions ADD COLUMN run_key TEXT")
        conn.execute("UPDATE file_versions SET run_key = 'default' WHERE run_key IS NULL")
        conn.commit()

    return conn


def _last_snapshot_id(conn: sqlite3.Connection, repo: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM snapshots WHERE repo = ? ORDER BY id DESC LIMIT 1", (repo,)
    ).fetchone()
    return row[0] if row else None


def _create_snapshot(
    conn: sqlite3.Connection,
    repo: str,
    commit_hash: str | None = None,
    raw_source_ref: str | None = None,
) -> int:
    now = datetime.now().isoformat()
    cur = conn.execute(
        "INSERT INTO snapshots (repo, commit_hash, created_at, raw_source_ref) VALUES (?, ?, ?, ?)",
        (repo, commit_hash, now, raw_source_ref),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def _insert_file_versions(
    conn: sqlite3.Connection, snapshot_id: int, files: list[dict], run_key: str = "default"
) -> None:
    """Insert file versions, carrying over analyzed_at from a prior row with same filename+hash."""
    for f in files:
        row = conn.execute(
            "SELECT analyzed_at FROM file_versions"
            " WHERE filename = ? AND hash = ?"
            "   AND COALESCE(run_key, 'default') = ?"
            "   AND analyzed_at IS NOT NULL"
            " ORDER BY id DESC LIMIT 1",
            (f["filename"], f["hash"], run_key),
        ).fetchone()
        analyzed_at = row[0] if row else None
        conn.execute(
            "INSERT INTO file_versions (snapshot_id, filename, hash, size, analyzed_at, run_key)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (snapshot_id, f["filename"], f["hash"], f.get("size", 0), analyzed_at, run_key),
        )
    conn.commit()


def mark_files_analyzed(
    conn: sqlite3.Connection, snapshot_id: int, filenames: list[str], run_key: str = "default"
) -> None:
    """Stamp analyzed_at = now for the given files in this snapshot."""
    now = datetime.now().isoformat()
    conn.executemany(
        "UPDATE file_versions SET analyzed_at = ?"
        " WHERE snapshot_id = ? AND filename = ? AND COALESCE(run_key, 'default') = ?",
        [(now, snapshot_id, fn, run_key) for fn in filenames],
    )
    conn.commit()


def _select_top_k(
    filenames: list[str],
    file_map: dict[str, dict],
    categories: dict[str, str] | None,
    max_files: int | None = None,
) -> tuple[list[str], list[str]]:
    """Prioritize risk categories then by descending file size, cap at max_files.

    max_files=0 or None means no limit.
    """
    limit = max_files if max_files is not None else MAX_FILES_PER_RUN
    if limit <= 0 or len(filenames) <= limit:
        return filenames, []

    def _priority(fn: str) -> tuple[int, int]:
        cat = (categories or {}).get(fn, "uncategorized")
        risk_rank = 0 if cat in RISK_CATEGORIES else 1
        return (risk_rank, -(file_map[fn].get("size", 0)))

    ranked = sorted(filenames, key=_priority)
    return ranked[:limit], ranked[limit:]


def find_changed_files(
    conn: sqlite3.Connection,
    repo: str,
    files: list[dict],
    categories: dict[str, str] | None = None,
    max_files: int | None = None,
    run_key: str = "default",
) -> dict:
    """Diff against last snapshot, persist new snapshot, return diff + selection."""
    last_id = _last_snapshot_id(conn, repo)
    if last_id is None:
        old_map: dict[str, str] = {}
    else:
        rows = conn.execute(
            "SELECT filename, hash FROM file_versions WHERE snapshot_id = ?", (last_id,)
        ).fetchall()
        old_map = {r[0]: r[1] for r in rows}

    new_map: dict[str, dict] = {f["filename"]: f for f in files}

    added = [fn for fn in new_map if fn not in old_map]
    removed = [fn for fn in old_map if fn not in new_map]
    changed = [fn for fn in new_map if fn in old_map and new_map[fn]["hash"] != old_map[fn]]
    unchanged = [fn for fn in new_map if fn in old_map and new_map[fn]["hash"] == old_map[fn]]

    # Persist new snapshot (carry-over analyzed_at for same filename+hash)
    new_snap_id = _create_snapshot(conn, repo)
    _insert_file_versions(conn, new_snap_id, files, run_key)

    # Select all files not yet analyzed in this snapshot
    rows = conn.execute(
        "SELECT filename FROM file_versions"
        " WHERE snapshot_id = ? AND analyzed_at IS NULL AND COALESCE(run_key, 'default') = ?",
        (new_snap_id, run_key),
    ).fetchall()
    unanalyzed = [r[0] for r in rows]
    selected, skipped = _select_top_k(unanalyzed, new_map, categories, max_files)

    return {
        "changed": changed,
        "added": added,
        "removed": removed,
        "unchanged": unchanged,
        "unanalyzed": unanalyzed,
        "selected": selected,
        "skipped": skipped,
        "snapshot_id": new_snap_id,
    }
