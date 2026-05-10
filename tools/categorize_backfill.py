#!/usr/bin/env python3
"""Backfill chunk-kategorien für alle aktiven chunks ohne category_labels.

User-Auftrag: "bekommt JEDER chunk der ingested wird, KATEGORIEN verpasst???
ABSOLUTES MUSS!!!". Stand 2026-05-10: nur 75.8% (3933/5191) aktiv-chunks
hatten labels — 1258 lücken durch frühere ingest-runs ohne categorize-step
oder fehlgeschlagene LLM-calls.

Pipeline:
  SELECT chunks WHERE is_active=1 AND category_labels IS NULL/leer
  → mayring_categorize() im hybrid-modus (deduktive anker + induktive [neu]-labels)
  → UPDATE chunks SET category_labels=...

Usage:
    # Trockenlauf — zeigt anzahl + 5 sample-prompts ohne UPDATE
    python tools/categorize_backfill.py --dry-run --limit 100

    # Live, in batches von 50 mit qwen3-coder:7b ODER ModelRouter('text')
    python tools/categorize_backfill.py --limit 1500 --batch 50

Run im container:
    ssh nileneb@u-server 'docker exec mayring-mayring-api-1 \\
        python3 tools/categorize_backfill.py --limit 1500 --batch 50'
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _conn() -> sqlite3.Connection:
    """Memory DB connection — sucht /app/cache zuerst (container), dann
    repo-relative cache/memory.db (local dev)."""
    candidates = [
        Path("/app/cache/memory.db"),
        ROOT / "cache" / "memory.db",
    ]
    for p in candidates:
        if p.exists():
            c = sqlite3.connect(str(p))
            c.row_factory = sqlite3.Row
            return c
    sys.exit("memory.db not found in /app/cache or repo cache/")


def load_chunks_without_labels(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """Aktive chunks ohne category_labels — newest first damit ein
    abgebrochener run die aktuellsten chunks zuerst hat."""
    # WHY: source_type liegt in `sources`, nicht in `chunks` → LEFT JOIN.
    return conn.execute(
        """
        SELECT c.chunk_id,
               c.source_id,
               COALESCE(s.source_type, 'repo_file') AS source_type,
               c.text,
               c.chunk_level,
               c.workspace_id
        FROM chunks c
        LEFT JOIN sources s ON s.source_id = c.source_id
        WHERE c.is_active = 1
          AND (c.category_labels IS NULL OR TRIM(c.category_labels) = '')
          AND c.text IS NOT NULL
          AND LENGTH(c.text) >= 20
        ORDER BY c.created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def categorize_one(row: sqlite3.Row, *, ollama_url: str, model: str,
                   mode: str, codebook: str) -> list[str]:
    """Single-chunk categorize. Returns list of label-strings (empty bei error)."""
    # Re-use mayring_categorize() durch ein Chunk-stand-in (vermeidet
    # imports auf Chunk dataclass mit allen feldern).
    from src.memory.ingestion.categorization import mayring_categorize
    from src.memory.schema import Chunk

    chunk = Chunk(
        chunk_id=row["chunk_id"],
        source_id=row["source_id"] or "",
        chunk_level=row["chunk_level"] or "function",
        ordinal=0,
        text=row["text"] or "",
        workspace_id=row["workspace_id"] or "default",
    )
    # source_type wird über das mayring_categorize-arg gepasst,
    # nicht über das Chunk-feld (das hat's nicht).
    result = mayring_categorize(
        [chunk], ollama_url=ollama_url, model=model,
        mode=mode, codebook=codebook,
        source_type=row["source_type"] or "repo_file",
    )
    if not result:
        return []
    cats = result[0].category_labels
    if not cats:
        return []
    # Chunk.category_labels kann list[str] ODER string sein — defensive
    if isinstance(cats, list):
        return [str(c).strip() for c in cats if str(c).strip()]
    return [c.strip() for c in str(cats).split(",") if c.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200,
                    help="max chunks to backfill")
    ap.add_argument("--batch", type=int, default=50,
                    help="batch-size für progress-prints")
    ap.add_argument("--dry-run", action="store_true",
                    help="zeigt was passieren würde, kein UPDATE")
    ap.add_argument("--mode", default="hybrid",
                    choices=("deductive", "inductive", "hybrid"))
    ap.add_argument("--codebook", default="auto",
                    help="auto | code | social | <profile-name>")
    ap.add_argument("--model", default="",
                    help="override ollama model; default ModelRouter('text')")
    ap.add_argument("--ollama-url", default=None)
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="sleep between chunks (rate-limit)")
    args = ap.parse_args()

    ollama_url = args.ollama_url or os.environ.get(
        "OLLAMA_URL", "http://localhost:11434"
    )
    if not args.model:
        try:
            from src.model_router import ModelRouter
            model = ModelRouter(ollama_url).resolve("text") or "mistral:7b-instruct"
        except Exception:
            model = "mistral:7b-instruct"
    else:
        model = args.model

    conn = _conn()
    rows = load_chunks_without_labels(conn, args.limit)
    print(f"backfill: {len(rows)} chunks → model={model}, mode={args.mode}, "
          f"codebook={args.codebook}, dry_run={args.dry_run}")

    successes = 0
    failures = 0
    total_labels = 0

    for i, row in enumerate(rows, start=1):
        cid = row["chunk_id"]
        sid = (row["source_id"] or "")[:60]
        try:
            labels = categorize_one(
                row, ollama_url=ollama_url, model=model,
                mode=args.mode, codebook=args.codebook,
            )
        except Exception as exc:
            print(f"[{i}/{len(rows)}] {cid} FAIL: {type(exc).__name__}: {exc}",
                  flush=True)
            failures += 1
            continue

        if not labels:
            print(f"[{i}/{len(rows)}] {cid} EMPTY (skip) — {sid}", flush=True)
            failures += 1
            continue

        joined = ", ".join(labels)
        if args.dry_run:
            print(f"[{i}/{len(rows)}] {cid} → [{joined}] — {sid}", flush=True)
        else:
            conn.execute(
                "UPDATE chunks SET category_labels=? WHERE chunk_id=?",
                (joined, cid),
            )
            if i % args.batch == 0:
                conn.commit()
            print(f"[{i}/{len(rows)}] {cid} ✓ [{joined}]", flush=True)

        successes += 1
        total_labels += len(labels)
        if args.sleep:
            time.sleep(args.sleep)

    if not args.dry_run:
        conn.commit()

    print(
        f"\n=== SUMMARY ===\n"
        f"  attempted:    {len(rows)}\n"
        f"  labeled:      {successes}\n"
        f"  failed/empty: {failures}\n"
        f"  total labels: {total_labels}\n"
        f"  ø labels:     {total_labels/max(successes,1):.1f}\n"
    )
    return 0 if failures < len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
