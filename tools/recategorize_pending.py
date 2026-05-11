#!/usr/bin/env python3
"""Re-categorize chunks marked ``category_source='cleanup-pending'``.

Hintergrund: ``cleanup_hallucinated_categories.py --strict`` strippt alle
``[neu]X``-labels; chunks die danach gar keine labels mehr haben bekommen
``category_source='cleanup-pending'`` als marker. Dieser job läuft sie
durch und kategorisiert sie neu mit den aktuellen task-anchored prompts
(``prompts/mayring_{mode}.md``).

Task-Kontext: der source-type liefert ein schwaches Selektionskriterium —
``repo_file`` → "die datei <path>"; ``paper`` → titel falls in
wiki_paper_cache; sonst leer (die prompts leiten dann das thema aus dem
chunk selbst ab). Das ist nicht perfekt, aber besser als gar kein anker.

Usage:
    python tools/recategorize_pending.py --dry-run
    python tools/recategorize_pending.py --limit 50
    python tools/recategorize_pending.py --workspace-id <slug>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.store import init_memory_db


def _task_for_source(conn, source_id: str, source_type: str) -> str:
    """Derive a weak task-context from the source. Empty if nothing useful."""
    st = (source_type or "").lower()
    if st == "repo_file" or source_id.startswith("repo-file:") or source_id.startswith("repo:"):
        # path is after the last ':' typically
        path = source_id.split(":", 1)[-1] if ":" in source_id else source_id
        return f"die datei {path[:120]}" if path else ""
    if st == "paper" or source_id.startswith("paper:"):
        # try wiki_paper_cache for a title (best-effort)
        try:
            pid = source_id.split(":", 1)[-1]
            row = conn.execute(
                "SELECT extracted FROM wiki_paper_cache WHERE source_id = ? AND rule_name = 'title' LIMIT 1",
                (f"paper:{pid}",),
            ).fetchone()
            if row and row[0]:
                return f"das paper: {str(row[0])[:160]}"
        except Exception:
            pass
        return "ein wissenschaftliches paper"
    if st in ("conversation", "conversation_summary") or source_id.startswith("conversation:"):
        return "eine arbeits-konversation"
    return ""  # prompts derive topic from the chunk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Preview only, no DB writes / LLM calls")
    ap.add_argument("--limit", type=int, default=0, help="Max chunks to process (0 = all)")
    ap.add_argument("--workspace-id", default=None, help="Limit to one workspace")
    ap.add_argument("--ollama-url", default=None, help="Override OLLAMA_URL")
    args = ap.parse_args()

    import os
    ollama_url = args.ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")

    conn = init_memory_db()

    sql = (
        "SELECT c.chunk_id, c.text, c.source_id, s.source_type, c.workspace_id "
        "FROM chunks c LEFT JOIN sources s ON c.source_id = s.source_id "
        "WHERE c.category_source = 'cleanup-pending' AND c.is_active = 1"
    )
    params: list = []
    if args.workspace_id:
        sql += " AND c.workspace_id = ?"
        params.append(args.workspace_id)
    sql += " ORDER BY c.created_at DESC"
    if args.limit > 0:
        sql += f" LIMIT {int(args.limit)}"

    rows = conn.execute(sql, params).fetchall()
    print(f"Found {len(rows)} cleanup-pending chunks"
          f"{f' in workspace {args.workspace_id}' if args.workspace_id else ''} "
          f"(dry_run={args.dry_run})\n")

    if args.dry_run:
        for chunk_id, text, source_id, source_type, ws in rows[:30]:
            task = _task_for_source(conn, source_id, source_type or "")
            print(f"  {chunk_id[:12]} ws={ws or '-':<20} src_type={source_type or '?':<22} "
                  f"task={task[:60]!r:<62} text={text[:50]!r}")
        if len(rows) > 30:
            print(f"  ... and {len(rows) - 30} more")
        print("\n(dry-run — no LLM calls, no DB writes)")
        return

    # Group by source_type so we resolve the right codebook + mode per chunk.
    from src.memory.ingestion.categorization import mayring_categorize
    from src.memory.schema import Chunk
    from src.model_router import ModelRouter

    router = ModelRouter(ollama_url)
    model = router.resolve("text") or ""
    if not model:
        print("ERROR: no text-model available from ModelRouter — Ollama unreachable?")
        sys.exit(2)

    done = 0
    relabeled = 0
    still_empty = 0
    for chunk_id, text, source_id, source_type, ws in rows:
        task = _task_for_source(conn, source_id, source_type or "")
        ch = Chunk(chunk_id=chunk_id, source_id=source_id, text=text or "")
        result = mayring_categorize(
            [ch], ollama_url=ollama_url, model=model,
            mode="hybrid", source_type=source_type or "repo_file",
            conn=conn, workspace_id=ws or "default", task=task,
        )
        labels = result[0].category_labels if result else []
        labels_csv = ",".join(labels)
        if labels_csv:
            conn.execute(
                "UPDATE chunks SET category_labels = ?, category_source = 'hybrid',"
                " category_confidence = 0.5 WHERE chunk_id = ?",
                (labels_csv, chunk_id),
            )
            relabeled += 1
        else:
            # still nothing — keep the marker so a future run / model can retry
            still_empty += 1
        done += 1
        # WHY(2026-05-12, db-lock): commit after EVERY chunk, not every 25.
        # The old batch-of-25 kept a single write transaction open across ~25
        # mistral calls (~2 min) → any concurrent process running
        # _init_schema DDL hit `database is locked` past the 10s busy_timeout
        # and crashed (workspace:system pipeline errors in JobHistory). The
        # mistral call dominates; one extra COMMIT per chunk is free, and the
        # write lock is now held for milliseconds instead of minutes.
        conn.commit()
        if done % 25 == 0:
            print(f"  ... {done}/{len(rows)} ({relabeled} relabeled, {still_empty} still empty)")

    conn.commit()
    print(f"\nDone: processed={done} relabeled={relabeled} still_empty={still_empty}")


if __name__ == "__main__":
    main()
