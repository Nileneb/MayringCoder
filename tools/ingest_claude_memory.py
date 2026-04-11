#!/usr/bin/env python3
"""
Ingest Claude Code auto-memory files into the MCP memory store.

Scans ~/.claude/projects/<project-slug>/memory/*.md and ingests any
new or changed files. Safe to re-run — deduplication via content_hash.

Usage:
    .venv/bin/python tools/ingest_claude_memory.py
    .venv/bin/python tools/ingest_claude_memory.py --dry-run
    .venv/bin/python tools/ingest_claude_memory.py --force   # re-ingest all
"""

import argparse
import hashlib
import os
import sqlite3
import sys
from pathlib import Path

# Allow running from repo root or tools/
sys.path.insert(0, str(Path(__file__).parent.parent))

MEMORY_DIR = Path.home() / ".claude/projects/-home-nileneb-Desktop-MayringCoder/memory"
DB_PATH = "cache/memory.db"
CHROMA_PATH = "cache/memory_chroma"
REPO = "nileneb/MayringCoder"


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _already_ingested(conn: sqlite3.Connection, source_id: str, content_hash: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT content_hash FROM sources WHERE source_id = ?", (source_id,)
    )
    row = cur.fetchone()
    return row is not None and row[0] == content_hash


def run(dry_run: bool = False, force: bool = False) -> None:
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
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    import chromadb
    from src.memory_ingest import ingest
    from src.memory_schema import Source

    conn = sqlite3.connect(DB_PATH)
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_or_create_collection("memory_chunks")

    ingested = skipped = 0

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
        result = ingest(source, content, conn, collection, ollama_url, embed_model)
        chunks = len(result.get("chunk_ids", []))
        deduped = result.get("deduped", 0)
        print(f"  ingest  {md_file.name}  →  {chunks} chunks  (deduped={deduped})")
        ingested += 1

    conn.close()
    action = "würde ingestiert" if dry_run else "ingested"
    print(f"\nFertig: {ingested} {action}, {skipped} übersprungen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Claude memory files into MCP store")
    parser.add_argument("--dry-run", action="store_true", help="Zeige was ingested würde, ohne zu schreiben")
    parser.add_argument("--force", action="store_true", help="Re-ingestiere alle Dateien auch wenn unverändert")
    args = parser.parse_args()
    run(dry_run=args.dry_run, force=args.force)
