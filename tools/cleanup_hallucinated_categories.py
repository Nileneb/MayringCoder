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
    return _process_labels(label_csv, strict=strict, promote_known=None)


def _process_labels(
    label_csv: str, *, strict: bool, promote_known: set[str] | None,
) -> tuple[str, int]:
    """Process a comma-separated label string.

    For each `[neu]X` label:
      - if `promote_known` is given and `X.lower()` is in it → rewrite to `X`
        (drop the prefix — the category is now a real anchor). Counts as 0 removed.
      - else if `strict` OR not a plausible label → drop it. Counts as removed.
      - else → keep `[neu]X` as-is.

    Returns (cleaned_csv, removed_count). Promotions are NOT counted as removed
    (they're rewrites, not deletions), but they DO change the string.
    """
    labels = [l.strip() for l in label_csv.split(",") if l.strip()]
    kept: list[str] = []
    removed = 0
    seen_lower: set[str] = set()
    for lbl in labels:
        low = lbl.lower()
        if low.startswith("[neu]"):
            inner = lbl[len("[neu]"):].strip()
            inner_low = inner.lower()
            if promote_known is not None and inner_low in promote_known:
                # Promote: [neu]X → X (dedup against an existing plain X)
                if inner_low not in seen_lower:
                    kept.append(inner_low)
                    seen_lower.add(inner_low)
                continue
            if strict or not is_valid_neu_label(inner):
                removed += 1
                continue
        else:
            if low in seen_lower:
                continue
            seen_lower.add(low)
        kept.append(lbl)
    return ",".join(kept), removed


def _load_universal_categories() -> set[str]:
    """Return the lowercased category set from codebooks/profiles/universal.yaml."""
    try:
        import yaml
        path = Path(__file__).parent.parent / "codebooks" / "profiles" / "universal.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {str(c).lower().strip() for c in (data or {}).get("categories", [])}
    except Exception:
        return set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--workspace-id", default=None, help="Limit to one workspace")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no DB writes")
    parser.add_argument("--strict", action="store_true", help="Remove ALL [neu]X labels regardless of validity")
    parser.add_argument("--promote-known", action="store_true",
                        help="Rewrite [neu]X → X when X is now in codebooks/profiles/universal.yaml "
                             "(promoted categories). Combine with --strict to also drop the rest.")
    args = parser.parse_args()

    conn = init_memory_db()
    promote_set = _load_universal_categories() if args.promote_known else None
    if args.promote_known:
        print(f"Promote-known: {len(promote_set or [])} codebook categories — "
              f"[neu]X → X for those, {'strip rest' if args.strict else 'keep valid [neu]X'}")

    sql = "SELECT chunk_id, category_labels, workspace_id FROM chunks WHERE category_labels LIKE '%[neu]%' AND is_active = 1"
    params: list = []
    if args.workspace_id:
        sql += " AND workspace_id = ?"
        params.append(args.workspace_id)

    rows = conn.execute(sql, params).fetchall()

    total_chunks = len(rows)
    affected = 0
    total_removed = 0
    total_promoted = 0
    pending_recategorize = 0

    print(f"Scanning {total_chunks} chunks with [neu] labels"
          f"{f' in workspace {args.workspace_id}' if args.workspace_id else ''}"
          f" (strict={args.strict}, promote_known={args.promote_known}, dry_run={args.dry_run})\n")

    for chunk_id, label_csv, ws in rows:
        original = label_csv or ""
        cleaned, removed = _process_labels(original, strict=args.strict, promote_known=promote_set)
        if cleaned == original:
            continue  # no change (e.g. only valid [neu]X kept, no strict, no promote)
        affected += 1
        total_removed += removed
        # promoted = [neu]X labels that became X (delta between original [neu]-count
        # and removed); rough estimate for the summary.
        orig_neu = sum(1 for l in original.split(",") if l.strip().lower().startswith("[neu]"))
        new_neu = sum(1 for l in cleaned.split(",") if l.strip().lower().startswith("[neu]"))
        promoted_here = max(0, orig_neu - new_neu - removed)
        total_promoted += promoted_here
        recategorize = not cleaned.strip()
        if recategorize:
            pending_recategorize += 1

        print(f"  {chunk_id[:12]} ws={ws or '-':<24} "
              f"removed={removed} promoted={promoted_here} pending={recategorize} "
              f"before={original[:60]!r} after={cleaned[:60]!r}")

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
          f"labels_removed={total_removed} labels_promoted={total_promoted} "
          f"marked_for_recategorize={pending_recategorize}")
    if args.dry_run:
        print("(dry-run — no DB writes)")


if __name__ == "__main__":
    main()
