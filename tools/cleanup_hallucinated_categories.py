#!/usr/bin/env python3
"""Cleanup-Job: entferne halluzinierte ``[neu]X``-Kategorien aus chunks.category_labels.

Hintergrund: Im hybrid-Mode darf das LLM neue Kategorien mit Prefix ``[neu]``
vorschlagen. Schwächere Modelle (z.B. mistral:7b-instruct) erfinden dabei
sinnfreie oder gibberish-Labels, die später beim symbolischen Scoring noise
erzeugen. Dieser Job filtert pro Chunk:

- Strip-Mode (default): entfernt nur ``[neu]X``-Labels die offensichtlich
  ungültig sind (zu kurz, keine Vokale, Zeichensalat).
- ``--strict``: entfernt ALLE ``[neu]X``-Labels — unabhängig von Validität.
  Sinnvoll wenn man nach einem Modellwechsel komplett neu kategorisieren will.

Chunks ohne valide Labels nach Cleanup bekommen ``category_confidence=0.0``
und ``category_source='cleanup-pending'`` als Marker für künftige
Re-Kategorisierung.

Usage:
    python tools/cleanup_hallucinated_categories.py --dry-run
    python tools/cleanup_hallucinated_categories.py --workspace-id <slug>
    python tools/cleanup_hallucinated_categories.py --strict
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.store import init_memory_db


def is_valid_neu_label(inner: str) -> bool:
    """Return True if the inner part of '[neu]<inner>' looks like a real category.

    Mirrors src/memory/ingestion/categorization.py::_is_plausible_neu_label —
    keep both functions in sync.
    """
    inner = inner.strip().lower()
    if not (2 <= len(inner) <= 30):
        return False
    if not re.fullmatch(r"[a-zäöüß0-9_-]+", inner):
        return False
    if len(inner) >= 5 and max((inner.count(c) for c in set(inner)), default=0) > len(inner) * 0.6:
        return False
    return True


def strip_neu_labels(label_csv: str, strict: bool) -> tuple[str, int]:
    """Remove [neu]X labels from a comma-separated label string.

    Returns (cleaned_csv, removed_count).
    """
    labels = [l.strip() for l in label_csv.split(",") if l.strip()]
    kept = []
    removed = 0
    for lbl in labels:
        low = lbl.lower()
        if low.startswith("[neu]"):
            inner = lbl[len("[neu]"):]
            if strict or not is_valid_neu_label(inner):
                removed += 1
                continue
        kept.append(lbl)
    return ",".join(kept), removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--workspace-id", default=None, help="Limit to one workspace")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no DB writes")
    parser.add_argument("--strict", action="store_true", help="Remove ALL [neu]X labels regardless of validity")
    args = parser.parse_args()

    conn = init_memory_db()

    sql = "SELECT chunk_id, category_labels, workspace_id FROM chunks WHERE category_labels LIKE '%[neu]%' AND is_active = 1"
    params: list = []
    if args.workspace_id:
        sql += " AND workspace_id = ?"
        params.append(args.workspace_id)

    rows = conn.execute(sql, params).fetchall()

    total_chunks = len(rows)
    affected = 0
    total_removed = 0
    pending_recategorize = 0

    print(f"Scanning {total_chunks} chunks with [neu] labels"
          f"{f' in workspace {args.workspace_id}' if args.workspace_id else ''}"
          f" (strict={args.strict}, dry_run={args.dry_run})\n")

    for chunk_id, label_csv, ws in rows:
        cleaned, removed = strip_neu_labels(label_csv or "", args.strict)
        if removed == 0:
            continue
        affected += 1
        total_removed += removed
        recategorize = not cleaned.strip()
        if recategorize:
            pending_recategorize += 1

        print(f"  {chunk_id[:12]} ws={ws or '-':<24} "
              f"removed={removed} pending={recategorize} "
              f"before={(label_csv or '')[:60]!r} after={cleaned[:60]!r}")

        if args.dry_run:
            continue

        if recategorize:
            conn.execute(
                "UPDATE chunks SET category_labels = ?, category_confidence = 0.0,"
                " category_source = 'cleanup-pending' WHERE chunk_id = ?",
                (cleaned, chunk_id),
            )
        else:
            conn.execute(
                "UPDATE chunks SET category_labels = ? WHERE chunk_id = ?",
                (cleaned, chunk_id),
            )

    if not args.dry_run:
        conn.commit()

    print()
    print(f"Summary: scanned={total_chunks} affected={affected} "
          f"labels_removed={total_removed} marked_for_recategorize={pending_recategorize}")
    if args.dry_run:
        print("(dry-run — no DB writes)")


if __name__ == "__main__":
    main()
