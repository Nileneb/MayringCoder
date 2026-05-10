"""V2 Stufe 1 — Identity Single-Source-of-Truth.

Spec: docs/v2-master-audit.md Section 6 Regel 4 + Section 7 Stufe 1.

`resolve_workspace_from_token(info, override_header)` ist der EINZIGE
caller, der workspace_id für eine Request bestimmt. Service-Token-mit-
Override-Header → returns override. User-JWT → returns active workspace.
Mehrdeutige inputs → raise.

Plus: Schema-Invariante stellt sicher, dass jede neue user-data-Tabelle
eine workspace_id-Spalte trägt.
"""
from __future__ import annotations

import pytest

from src.api.jwt_auth import TokenInfo
from src.identity.workspace_resolver import resolve_workspace_from_token


# ---------------------------------------------------------------------------
# 1.1 — resolve_workspace_from_token
# ---------------------------------------------------------------------------


def test_user_jwt_returns_token_workspace():
    """Regular user-JWT → workspace from JWT. Override-Header IGNORIERT."""
    info = TokenInfo(workspace_id="alice", scopes=("mcp:memory",))
    assert resolve_workspace_from_token(info, override_header=None) == "alice"
    # Nicht-Service-Token darf NICHT in fremden Workspace schreiben.
    assert resolve_workspace_from_token(info, override_header="bene") == "alice"


def test_service_token_default_is_system_when_no_override():
    info = TokenInfo(workspace_id="system", scopes=("*",))
    assert resolve_workspace_from_token(info, override_header=None) == "system"


def test_service_token_with_override_returns_override():
    """Service-Token mit X-Workspace-Id-Header → admin-tool case, return override."""
    info = TokenInfo(workspace_id="system", scopes=("*",))
    assert resolve_workspace_from_token(info, override_header="bene") == "bene"


def test_service_token_with_blank_override_returns_default():
    """Whitespace/empty override = nicht gesetzt → default zur token-workspace."""
    info = TokenInfo(workspace_id="system", scopes=("*",))
    assert resolve_workspace_from_token(info, override_header="   ") == "system"
    assert resolve_workspace_from_token(info, override_header="") == "system"


def test_admin_user_jwt_does_not_get_override():
    """Admin-USER-JWT (mcp:memory + admin) ist KEIN Service-Token; darf
    nicht via Header in fremde Workspaces schreiben. Nur scope='*' darf das."""
    info = TokenInfo(workspace_id="alice", scopes=("mcp:memory", "admin"))
    assert resolve_workspace_from_token(info, override_header="bene") == "alice"


def test_none_token_info_raises():
    """Aufruf ohne TokenInfo darf NICHT silent 'system' zurückgeben."""
    with pytest.raises((ValueError, AttributeError, TypeError)):
        resolve_workspace_from_token(None, override_header=None)  # type: ignore[arg-type]


def test_token_without_workspace_id_raises():
    """Token mit leerer workspace_id ist Bug, nicht silent fallback."""
    info = TokenInfo(workspace_id="", scopes=("mcp:memory",))
    with pytest.raises(ValueError):
        resolve_workspace_from_token(info, override_header=None)


# ---------------------------------------------------------------------------
# 1.2 — Schema-Invariante: keine user-data-Tabelle ohne workspace_id
# ---------------------------------------------------------------------------


def test_assert_no_unworkspaced_table(tmp_path, monkeypatch):
    """Jede user-data-Tabelle MUSS workspace_id haben.

    Idempotente Migration: bereits angelegte Tabellen werden auf
    Aufruf von init_memory_db so erweitert, dass sie workspace_id
    enthalten. Sonst CI-fail.
    """
    monkeypatch.setattr(
        "src.config.CACHE_DIR", tmp_path,
        raising=False,
    )
    from src.memory.store import init_memory_db
    db = tmp_path / "memory.db"
    conn = init_memory_db(db)

    REQUIRED = {
        "chunks", "sources", "topic_transitions", "chunk_feedback",
        "wiki_paper_cache", "ingestion_log", "context_feedback_log",
    }
    for table in REQUIRED:
        cols = conn.get_columns(table)
        assert "workspace_id" in cols, (
            f"User-data-Tabelle {table!r} fehlt workspace_id-Spalte. "
            f"Migration in src/memory/store.py::_migrate_schema fehlt."
        )


# ---------------------------------------------------------------------------
# 1.3 — Migration ist idempotent
# ---------------------------------------------------------------------------


def test_migration_idempotent_on_existing_db(tmp_path):
    """Zweiter init_memory_db-Call auf gleicher DB darf NICHT crashen
    weil Spalten schon existieren (DUPLICATE COLUMN)."""
    from src.memory.store import init_memory_db
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    # Zweiter Aufruf — soll silent durchlaufen:
    conn = init_memory_db(db)
    cols = conn.get_columns("topic_transitions")
    assert "workspace_id" in cols


def test_workspace_id_default_value_for_new_rows(tmp_path):
    """Neue Rows in topic_transitions/chunk_feedback/wiki_paper_cache/
    ingestion_log bekommen workspace_id='default' wenn nicht explizit
    gesetzt — backward-compat für legacy Inserts."""
    from src.memory.store import init_memory_db
    from datetime import datetime, timezone
    db = tmp_path / "memory.db"
    conn = init_memory_db(db)

    # legacy-Insert ohne workspace_id (alte API)
    conn.execute(
        "INSERT INTO topic_transitions (from_topic, to_topic, last_seen) "
        "VALUES (?, ?, ?)",
        ("a", "b", datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    row = conn.execute(
        "SELECT workspace_id FROM topic_transitions WHERE from_topic = 'a'"
    ).fetchone()
    assert row is not None
    assert row[0] == "default"


def test_workspace_id_filter_works_on_topic_transitions(tmp_path):
    """workspace_id-WHERE-clause filtert wie erwartet."""
    from src.memory.store import init_memory_db
    from datetime import datetime, timezone
    db = tmp_path / "memory.db"
    conn = init_memory_db(db)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO topic_transitions (from_topic, to_topic, last_seen, workspace_id) "
        "VALUES (?, ?, ?, ?)", ("x", "y", now, "alice"),
    )
    conn.execute(
        "INSERT INTO topic_transitions (from_topic, to_topic, last_seen, workspace_id) "
        "VALUES (?, ?, ?, ?)", ("p", "q", now, "bene"),
    )
    conn.commit()
    rows = conn.execute(
        "SELECT from_topic FROM topic_transitions WHERE workspace_id = ?",
        ("alice",),
    ).fetchall()
    assert [r[0] for r in rows] == ["x"]


def test_workspace_id_filter_works_on_wiki_paper_cache(tmp_path):
    from src.memory.store import init_memory_db
    from datetime import datetime, timezone
    db = tmp_path / "memory.db"
    conn = init_memory_db(db)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO wiki_paper_cache (source_id, rule_name, extracted, created_at, workspace_id) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src1", "rule_a", "[]", now, "alice"),
    )
    conn.execute(
        "INSERT INTO wiki_paper_cache (source_id, rule_name, extracted, created_at, workspace_id) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src2", "rule_a", "[]", now, "bene"),
    )
    conn.commit()
    rows = conn.execute(
        "SELECT source_id FROM wiki_paper_cache WHERE workspace_id = ?",
        ("bene",),
    ).fetchall()
    assert [r[0] for r in rows] == ["src2"]


def test_workspace_id_filter_works_on_ingestion_log(tmp_path):
    from src.memory.store import init_memory_db
    from datetime import datetime, timezone
    db = tmp_path / "memory.db"
    conn = init_memory_db(db)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO ingestion_log (source_id, event_type, payload, created_at, workspace_id) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src1", "added", "{}", now, "alice"),
    )
    conn.execute(
        "INSERT INTO ingestion_log (source_id, event_type, payload, created_at, workspace_id) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src2", "added", "{}", now, "bene"),
    )
    conn.commit()
    rows = conn.execute(
        "SELECT source_id FROM ingestion_log WHERE workspace_id = ?",
        ("alice",),
    ).fetchall()
    assert [r[0] for r in rows] == ["src1"]


def test_workspace_id_filter_works_on_chunk_feedback(tmp_path):
    """chunk_feedback bekommt workspace_id-Spalte (Stufe 1.3)."""
    from src.memory.store import init_memory_db
    from datetime import datetime, timezone
    db = tmp_path / "memory.db"
    conn = init_memory_db(db)
    cols = conn.get_columns("chunk_feedback")
    assert "workspace_id" in cols, (
        "chunk_feedback braucht workspace_id-Spalte (V2 Stufe 1.3)"
    )
