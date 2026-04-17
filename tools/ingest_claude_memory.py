#!/usr/bin/env python3
"""Ingest Claude Code auto-memory files into the MCP memory store.

Scans ~/.claude/projects/<project-slug>/memory/*.md and ingests any
new or changed files. Safe to re-run — deduplication via content_hash.

Usage:
    .venv/bin/python tools/ingest_claude_memory.py
    .venv/bin/python tools/ingest_claude_memory.py --dry-run
    .venv/bin/python tools/ingest_claude_memory.py --force
    .venv/bin/python tools/ingest_claude_memory.py --workspace-id system
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MEMORY_DIR = Path.home() / ".claude/projects/-home-nileneb-Desktop-MayringCoder/memory"
REPO = "nileneb/MayringCoder"


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _already_ingested(conn, source_id: str, content_hash: str) -> bool:
    row = conn.execute(
        "SELECT content_hash FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()
    return row is not None and row[0] == content_hash


def run(dry_run: bool = False, force: bool = False, workspace_id: str = "system") -> None:
    if not MEMORY_DIR.exists():
        print(f"Memory-Verzeichnis nicht gefunden: {MEMORY_DIR}", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(MEMORY_DIR.glob("*.md"))
    if not md_files:
        print("Keine .md Dateien gefunden.")
        return

    from dotenv import load_dotenv
    load_dotenv()

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))

    from src.memory.ingest import get_or_create_chroma_collection, ingest
    from src.memory.schema import Source
    from src.memory.store import init_memory_db

    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()

    ingested = skipped = errors = 0

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        content_hash = _hash(content)
        source_id = f"claude-memory:{md_file.name}"

        if not force and _already_ingested(conn, source_id, content_hash):
            print(f"  skip  {md_file.name}  (unverändert)")
            skipped += 1
            continue

        if dry_run:
            print(f"  would ingest  {md_file.name}  (hash={content_hash})")
            ingested += 1
            continue

        source = Source(
            source_id=source_id,
            source_type="note",
            repo=REPO,
            path=f"claude-memory/{md_file.name}",
            branch="local",
            commit="",
            content_hash=content_hash,
        )
        try:
            result = ingest(
                source, content, conn, chroma, ollama_url, model,
                workspace_id=workspace_id,
                # ingest() handles categorize/codebook/mode via _INGEST_DEFAULTS["note"]
            )
            chunks = len(result.get("chunk_ids", []))
            deduped = result.get("deduped", 0)
            print(f"  ingest  {md_file.name}  →  {chunks} chunks  (deduped={deduped})")
            ingested += 1
        except Exception as exc:
            print(f"  ERROR  {md_file.name}: {exc}", file=sys.stderr)
            errors += 1

    conn.close()
    action = "würde ingestiert" if dry_run else "ingested"
    print(f"\nFertig: {ingested} {action}, {skipped} übersprungen, {errors} Fehler.")

    if not dry_run and ingested > 0:
        print("\nKnowledge Graph aktualisieren …")
        try:
            from tools.generate_knowledge_graph import generate
            generate()
        except Exception as exc:
            print(f"  Knowledge Graph Fehler: {exc}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Claude memory files into MCP store")
    parser.add_argument("--dry-run", action="store_true",
                        help="Zeige was ingested würde, ohne zu schreiben")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingestiere alle Dateien auch wenn unverändert")
    parser.add_argument("--workspace-id", default="system", metavar="ID",
                        help="Memory workspace-ID (Standard: system)")
    args = parser.parse_args()
    run(dry_run=args.dry_run, force=args.force, workspace_id=args.workspace_id)
