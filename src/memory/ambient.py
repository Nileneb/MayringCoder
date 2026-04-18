"""Ambient Context Layer — kompakter Projekt-Snapshot für Pi-Agent."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

_SNAPSHOT_SYSTEM = (
    "Du bist ein präziser Assistent. Erstelle einen kompakten Projekt-Snapshot auf Deutsch. "
    "Keine Begrüßung, kein Nachsatz. Maximal 600 Wörter."
)

_SNAPSHOT_PROMPT = """\
## Aktuelle Conversations (letzte 5 Zusammenfassungen)
{conversation_summaries}

## Offene Issues / bekannte Probleme
{issue_summaries}

## Top-Verbindungen im Wiki (stärkste Abhängigkeiten)
{wiki_connections}

---
Erstelle einen Projekt-Snapshot im Format:

**Aktueller Stand:** <1-2 Sätze was zuletzt implementiert/geändert wurde>

**Architektur-Hotspots:** <die 3-5 wichtigsten Dateien/Module mit kurzer Rolle>

**Offene Punkte:** <max. 5 konkrete TODOs oder bekannte Probleme>

**Wichtige Zusammenhänge:** <max. 5 Datei/Modul-Paare mit Erklärung warum sie zusammengehören>
"""

def _load_recent_conversations(conn: Any, repo_slug: str, limit: int = 5) -> list[str]:
    """Lade die letzten N Conversation-Summaries aus SQLite."""
    rows = conn.execute(
        """SELECT c.text FROM chunks c
           JOIN sources s ON c.source_id = s.source_id
           WHERE s.source_type = 'conversation_summary'
             AND (s.repo = ? OR ? = '')
             AND c.is_active = 1
           ORDER BY s.captured_at DESC
           LIMIT ?""",
        (repo_slug, repo_slug, limit),
    ).fetchall()
    return [r[0][:500] for r in rows]


def _load_recent_issues(conn: Any, repo_slug: str, limit: int = 10) -> list[str]:
    """Lade die letzten N Issue-Summaries aus SQLite."""
    rows = conn.execute(
        """SELECT c.text FROM chunks c
           JOIN sources s ON c.source_id = s.source_id
           WHERE s.source_type = 'github_issue'
             AND (s.repo = ? OR ? = '')
             AND c.is_active = 1
           ORDER BY s.captured_at DESC
           LIMIT ?""",
        (repo_slug, repo_slug, limit),
    ).fetchall()
    return [r[0][:300] for r in rows]


def _load_wiki_top_connections(repo_slug: str, limit: int = 10) -> str:
    """Lade die stärksten Wiki-Verbindungen aus dem Wiki-Markdown."""
    wiki_path = Path("cache") / f"{repo_slug}_wiki.md"
    if not wiki_path.exists():
        return "(kein Wiki vorhanden)"
    content = wiki_path.read_text(encoding="utf-8")
    lines = [line for line in content.splitlines() if line.startswith("- →")]
    return "\n".join(lines[:limit]) or "(keine Verbindungen)"


def generate_ambient_snapshot(
    conn: Any,
    ollama_url: str,
    model: str,
    repo_slug: str = "",
    workspace_id: str = "system",
) -> str | None:
    """Generiere Ambient-Snapshot via LLM und speichere in SQLite.

    Returns snapshot text on success, None if model is empty.
    """
    if not model:
        return None

    from src.analysis.analyzer import _ollama_generate
    from src.memory.schema import Source
    from src.memory.ingest import ingest

    convs = _load_recent_conversations(conn, repo_slug)
    issues = _load_recent_issues(conn, repo_slug)
    wiki_top = _load_wiki_top_connections(repo_slug)

    prompt = _SNAPSHOT_PROMPT.format(
        conversation_summaries="\n".join(f"- {s}" for s in convs) or "(keine)",
        issue_summaries="\n".join(f"- {s}" for s in issues) or "(keine)",
        wiki_connections=wiki_top,
    )

    try:
        snapshot_text = _ollama_generate(
            prompt, ollama_url, model, "ambient_snapshot",
            system_prompt=_SNAPSHOT_SYSTEM,
        )
    except Exception:
        return None

    source_id = f"ambient:{repo_slug or 'global'}:snapshot"
    content_hash = "sha256:" + hashlib.sha256(snapshot_text.encode()).hexdigest()[:16]

    src = Source(
        source_id=source_id,
        source_type="ambient_snapshot",
        repo=repo_slug,
        path="ambient/snapshot",
        branch="local",
        commit="",
        content_hash=content_hash,
    )
    ingest(
        src, snapshot_text, conn, None,
        ollama_url, "",
        opts={"categorize": False, "chunk_level": "ambient_snapshot"},
        workspace_id=workspace_id,
    )
    return snapshot_text


def load_ambient_snapshot(conn: Any, repo_slug: str = "") -> str | None:
    """Lade den letzten Ambient-Snapshot aus SQLite."""
    source_id = f"ambient:{repo_slug or 'global'}:snapshot"
    row = conn.execute(
        """SELECT c.text FROM chunks c
           JOIN sources s ON c.source_id = s.source_id
           WHERE s.source_id = ? AND c.is_active = 1
           ORDER BY s.captured_at DESC LIMIT 1""",
        (source_id,),
    ).fetchone()
    return row[0] if row else None
