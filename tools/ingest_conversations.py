#!/usr/bin/env python3
"""Ingest Claude Code conversation histories into the MCP memory store.

Scans ~/.claude/projects/<workspace>/ for JSONL conversation files,
extracts user+assistant turns, searches memory for related chunks, then
uses Ollama to generate a compact German summary per session.

The summary is ingested as source_type="conversation_summary". Related
source_ids are embedded inline (as a markdown section + HTML comment)
so cross-references survive retrieval without schema changes.

Usage:
    .venv/bin/python tools/ingest_conversations.py \\
        --workspace ~/.claude/projects/-home-nileneb-Desktop-MayringCoder
    .venv/bin/python tools/ingest_conversations.py --all-workspaces
    .venv/bin/python tools/ingest_conversations.py --all-workspaces --dry-run
    .venv/bin/python tools/ingest_conversations.py --all-workspaces --force-reingest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

_PROJECTS_DIR = Path.home() / ".claude" / "projects"

_SUMMARIZE_SYSTEM = (
    "Du bist ein präziser Assistent, der Claude-Code-Gesprächs-Sessions zusammenfasst. "
    "Antworte ausschließlich auf Deutsch. Keine Begrüßung, kein Nachsatz."
)

_SUMMARIZE_PROMPT = """\
Hier ist ein Ausschnitt einer Claude-Code-Session und verwandter Projektkontext.

## Verwandter Projektkontext (aus Memory)
{related_context}

## Session-Turns (Ausschnitt)
{turns_excerpt}

---
Erstelle eine kompakte Zusammenfassung (max. 200 Wörter) im Format:

**Thema:** <1 Satz>

**Entscheidungen:**
- <max. 5 Punkte>

**Ergebnisse:**
- <max. 5 Punkte — was konkret implementiert/geändert wurde>

**Verknüpfte Konzepte:** <Dateien, Module, Issues aus dem Kontext — falls relevant, sonst Zeile weglassen>

**Offene Punkte:** <falls vorhanden, sonst Zeile weglassen>\
"""


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

def _extract_text(content: Any) -> str:
    """Extract plain text from a message content field (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _parse_jsonl(path: Path) -> dict[str, list[dict]]:
    """Parse a JSONL file into {session_id: [turn, ...]}.

    Each turn: {role, content, timestamp}. Tool calls and empty turns are dropped.
    """
    sessions: dict[str, list[dict]] = defaultdict(list)
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if entry.get("type") not in ("user", "assistant"):
                    continue

                msg = entry.get("message", {})
                text = _extract_text(msg.get("content", "")).strip()
                if not text:
                    continue

                sid = entry.get("sessionId") or path.stem
                sessions[sid].append({
                    "role": entry["type"],
                    "content": text[:2000],
                    "timestamp": entry.get("timestamp", ""),
                })
    except Exception:
        pass
    return dict(sessions)


# ---------------------------------------------------------------------------
# Summarization helpers
# ---------------------------------------------------------------------------

def _turns_excerpt(turns: list[dict], max_chars: int = 3000) -> str:
    parts: list[str] = []
    total = 0
    for t in turns:
        prefix = "👤" if t["role"] == "user" else "🤖"
        line = f"{prefix} {t['content'][:500]}\n"
        if total + len(line) > max_chars:
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)


def _keywords(turns: list[dict]) -> str:
    user_texts = [t["content"] for t in turns[:6] if t["role"] == "user"]
    return " ".join(user_texts[:3])[:200]


def _summarize(
    turns: list[dict],
    related_context: str,
    ollama_url: str,
    model: str,
) -> str:
    from src.analysis.analyzer import _ollama_generate

    prompt = _SUMMARIZE_PROMPT.format(
        related_context=related_context or "(kein verwandter Kontext)",
        turns_excerpt=_turns_excerpt(turns),
    )
    try:
        return _ollama_generate(
            prompt, ollama_url, model, "conv_summary",
            system_prompt=_SUMMARIZE_SYSTEM,
        )
    except Exception as exc:
        return f"**Thema:** Zusammenfassung fehlgeschlagen ({exc})\n\n{_turns_excerpt(turns, 500)}"


# ---------------------------------------------------------------------------
# Dedup helper
# ---------------------------------------------------------------------------

def _already_ingested(conn: Any, source_id: str, content_hash: str) -> bool:
    row = conn.execute(
        "SELECT content_hash FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()
    return row is not None and row[0] == content_hash


def _slug(workspace_path: Path) -> str:
    parts = workspace_path.name.split("-")
    return parts[-1] if parts else workspace_path.name


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------

def _ingest_workspace(
    workspace_path: Path,
    conn: Any,
    chroma: Any,
    ollama_url: str,
    model: str,
    workspace_id: str,
    min_messages: int,
    force_reingest: bool,
    dry_run: bool,
    include_subagents: bool,
) -> tuple[int, int, int]:
    from src.memory.ingest import ingest
    from src.memory.retrieval import compress_for_prompt, search
    from src.memory.schema import Source

    slug = _slug(workspace_path)

    jsonl_files = list(workspace_path.glob("*.jsonl"))
    if include_subagents:
        jsonl_files += list(workspace_path.glob("**/agent-*.jsonl"))

    ingested = skipped = errors = 0

    for fpath in sorted(jsonl_files):
        sessions = _parse_jsonl(fpath)

        for session_id, turns in sessions.items():
            if len(turns) < min_messages:
                skipped += 1
                continue

            first_ts = turns[0].get("timestamp", "")[:10]
            source_id = f"conversation:{slug}:{session_id[:16]}"
            raw_key = f"{session_id}:{len(turns)}:{first_ts}"
            content_hash = "sha256:" + hashlib.sha256(raw_key.encode()).hexdigest()[:16]

            if not force_reingest and _already_ingested(conn, source_id, content_hash):
                skipped += 1
                continue

            if dry_run:
                print(f"  [dry] {slug} | {first_ts or '?'} | {len(turns)} turns | {session_id[:8]}")
                ingested += 1
                continue

            try:
                # Step 1: search memory for related chunks
                related: list[Any] = []
                related_context = ""
                kw = _keywords(turns)
                if kw:
                    try:
                        related = search(
                            query=kw,
                            conn=conn,
                            chroma_collection=chroma,
                            ollama_url=ollama_url,
                            opts={"top_k": 5, "workspace_id": None},
                        )
                        related_context = compress_for_prompt(related, char_budget=1500)
                    except Exception:
                        pass

                # Step 2: LLM summary enriched with related context
                summary = _summarize(turns, related_context, ollama_url, model)

                # Step 3: build full content — embed links for searchability
                related_ids = [r.chunk_id for r in related]
                related_sources = list({r.source_id for r in related})
                links_md = ""
                if related_sources:
                    links_md = "\n\n**Verknüpfte Quellen:**\n" + "\n".join(
                        f"- {s}" for s in related_sources[:5]
                    )

                content = (
                    f"# Session {first_ts or 'unbekannt'} | {slug}\n\n"
                    f"{summary}"
                    f"{links_md}\n\n"
                    f"<!-- related_chunk_ids: {json.dumps(related_ids)} -->"
                )

                src = Source(
                    source_id=source_id,
                    source_type="conversation_summary",
                    repo=slug,
                    path=f"{slug}/{fpath.name}",
                    branch="local",
                    commit="",
                    content_hash=content_hash,
                )
                result = ingest(
                    src, content, conn, chroma,
                    ollama_url, model,
                    opts={"categorize": bool(model), "codebook": "social", "mode": "hybrid"},
                    workspace_id=workspace_id,
                )
                chunks = len(result.get("chunk_ids", []))
                print(f"  ✓ {slug} | {first_ts or '?'} | {len(turns)} turns"
                      f" → {chunks} chunks | {len(related_ids)} links")
                ingested += 1

            except Exception as exc:
                print(f"  ✗ {slug} | {session_id[:8]}: {exc}", file=sys.stderr)
                errors += 1

    return ingested, skipped, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Claude Code conversations into MCP memory")
    ap.add_argument("--workspace", metavar="PATH",
                    help="Pfad zum Claude-Workspace-Verzeichnis")
    ap.add_argument("--all-workspaces", action="store_true",
                    help=f"Alle Workspaces unter {_PROJECTS_DIR}")
    ap.add_argument("--workspace-id", default="system", metavar="ID",
                    help="Memory-Workspace-ID (Standard: system)")
    ap.add_argument("--min-messages", type=int, default=5, metavar="N",
                    help="Mindest-Turns pro Session (Standard: 5)")
    ap.add_argument("--force-reingest", action="store_true",
                    help="Bereits ingested Sessions überschreiben")
    ap.add_argument("--dry-run", action="store_true",
                    help="Nur anzeigen, nichts schreiben")
    ap.add_argument("--include-subagents", action="store_true",
                    help="Subagent-Sessions (agent-*.jsonl) einschließen")
    args = ap.parse_args()

    if not args.workspace and not args.all_workspaces:
        ap.error("--workspace PATH oder --all-workspaces angeben")

    from dotenv import load_dotenv
    load_dotenv()

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "")
    if not model:
        print("OLLAMA_MODEL nicht in .env gesetzt.", file=sys.stderr)
        sys.exit(1)

    print(f"Ollama: {ollama_url} | Model: {model} | workspace-id: {args.workspace_id}")
    if args.dry_run:
        print("[dry-run aktiv — keine Änderungen]")

    from src.memory.ingest import get_or_create_chroma_collection
    from src.memory.store import init_memory_db

    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()

    workspaces: list[Path] = []
    if args.all_workspaces:
        if not _PROJECTS_DIR.exists():
            print(f"Nicht gefunden: {_PROJECTS_DIR}", file=sys.stderr)
            sys.exit(1)
        workspaces = sorted(p for p in _PROJECTS_DIR.iterdir() if p.is_dir())
    else:
        wp = Path(args.workspace).expanduser()
        if not wp.exists():
            print(f"Workspace nicht gefunden: {wp}", file=sys.stderr)
            sys.exit(1)
        workspaces = [wp]

    total_i = total_s = total_e = 0
    for ws in workspaces:
        print(f"\n── {_slug(ws)} ({ws.name})")
        i, s, e = _ingest_workspace(
            workspace_path=ws,
            conn=conn,
            chroma=chroma,
            ollama_url=ollama_url,
            model=model,
            workspace_id=args.workspace_id,
            min_messages=args.min_messages,
            force_reingest=args.force_reingest,
            dry_run=args.dry_run,
            include_subagents=args.include_subagents,
        )
        total_i += i
        total_s += s
        total_e += e

    conn.close()
    action = "würde ingested" if args.dry_run else "ingested"
    print(f"\nFertig: {total_i} {action}, {total_s} übersprungen, {total_e} Fehler")

    if not args.dry_run and total_i > 0:
        print("\nKnowledge Graph aktualisieren …")
        from tools.generate_knowledge_graph import generate
        generate()


if __name__ == "__main__":
    main()
