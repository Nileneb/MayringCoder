"""Persistenz für Integrations-Notifications (#270 Phase 1).

Eine Tabelle ``integration_notifications`` sammelt serverseitig Events
(GitHub-Webhooks, Deploy-Probes, Crons). Der UserPromptSubmit-Hook pollt sie
per ``GET /notifications?since=`` (ein Call statt N Poll-Runden).

Lazy-Init via CREATE TABLE IF NOT EXISTS — keine Änderung am zentralen
init_memory_db nötig, kein Risiko für bestehende Schemas.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_table(conn: Any) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS integration_notifications (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            kind         TEXT NOT NULL,
            severity     TEXT NOT NULL DEFAULT 'info',
            title        TEXT NOT NULL DEFAULT '',
            detail       TEXT NOT NULL DEFAULT '',
            repo         TEXT NOT NULL DEFAULT '',
            payload      TEXT NOT NULL DEFAULT '{}',
            acked        INTEGER NOT NULL DEFAULT 0,
            created_at   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_intnotif_created
            ON integration_notifications(created_at);
        """
    )
    conn.commit()


def record_notification(
    conn: Any,
    *,
    kind: str,
    severity: str = "info",
    title: str = "",
    detail: str = "",
    repo: str = "",
    payload: dict | None = None,
) -> int:
    ensure_table(conn)
    cur = conn.execute(
        "INSERT INTO integration_notifications "
        "(kind, severity, title, detail, repo, payload, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (kind, severity, title, detail, repo,
         json.dumps(payload or {}, ensure_ascii=False), _now_iso()),
    )
    conn.commit()
    return int(cur.lastrowid)


def unacked_since(conn: Any, since: str | None = None) -> list[dict]:
    """Return un-acked notifications, optionally only those after ``since`` (ISO)."""
    ensure_table(conn)
    if since:
        rows = conn.execute(
            "SELECT id, kind, severity, title, detail, repo, created_at "
            "FROM integration_notifications "
            "WHERE acked = 0 AND created_at > ? ORDER BY created_at",
            (since,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, kind, severity, title, detail, repo, created_at "
            "FROM integration_notifications "
            "WHERE acked = 0 ORDER BY created_at",
        ).fetchall()
    cols = ("id", "kind", "severity", "title", "detail", "repo", "created_at")
    return [dict(zip(cols, r)) for r in rows]


def ack(conn: Any, ids: list[int]) -> int:
    if not ids:
        return 0
    ensure_table(conn)
    qmarks = ",".join("?" for _ in ids)
    cur = conn.execute(
        f"UPDATE integration_notifications SET acked = 1 WHERE id IN ({qmarks})",
        list(ids),
    )
    conn.commit()
    return cur.rowcount if hasattr(cur, "rowcount") else 0
