"""Export a Mayring-categorizer fine-tuning dataset (#260).

One JSONL row per hybrid-categorized chunk:

    {
      "input":  {"text": "...", "task": "..."},
      "output": {"kategorien": ["domain", "config"],
                 "belege": [{"kategorie": "domain", "span": [12, 48],
                             "excerpt": "...", "reasoning": "..."}]},
      "workspace_id": "...",
      "chunk_id": "chk_...",
      "category_source": "hybrid"
    }

Source:
  * ``chunks`` (memory.db) — text + comma-separated ``category_labels``,
    filtered to ``category_source='hybrid'`` (the deductive+inductive blend
    that has actually been reviewed; ``fallback`` / ``cleanup-pending`` are
    excluded as low-quality).
  * ``wiki_category_evidence`` (wiki_v2.db, ATTACHed) — per-(chunk, category)
    span/excerpt/reasoning, joined on ``chunk_id``. These supply the
    ``belege``. Empty until ``pi_mark_categories(persist=True)`` has run, so a
    fresh DB emits rows with ``belege: []`` (still usable as a weak label set).

Usage:
    python tools/export_mayring_categorizer_dataset.py \
        --db cache/memory.db --wiki-db cache/wiki_v2.db \
        --out cache/finetuning/mayring_categorizer_dataset.jsonl

No train/test split here — split downstream (see docs/specialist-models.md),
matching tools/export_retrieval_dataset.py.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mayring_core.config import CACHE_DIR

DEFAULT_OUT = CACHE_DIR / "finetuning" / "mayring_categorizer_dataset.jsonl"
DEFAULT_WIKI_DB = CACHE_DIR / "wiki_v2.db"


def _split_labels(raw: str) -> list[str]:
    return [c.strip() for c in (raw or "").split(",") if c.strip()]


def _evidence_map(
    conn: sqlite3.Connection, has_wiki: bool
) -> dict[str, list[dict]]:
    """{chunk_id: [{kategorie, span, excerpt, reasoning}, ...]} from wiki_v2."""
    if not has_wiki:
        return {}
    out: dict[str, list[dict]] = {}
    rows = conn.execute(
        "SELECT chunk_id, category, span_start, span_end, excerpt, reasoning "
        "FROM wikidb.wiki_category_evidence WHERE chunk_id != ''"
    ).fetchall()
    for r in rows:
        out.setdefault(r["chunk_id"], []).append({
            "kategorie": r["category"],
            "span": [r["span_start"], r["span_end"]],
            "excerpt": r["excerpt"],
            "reasoning": r["reasoning"],
        })
    return out


def export(db_path: Path, wiki_db: Path, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        has_wiki = wiki_db.exists()
        if has_wiki:
            conn.execute("ATTACH DATABASE ? AS wikidb", (str(wiki_db),))
        evidence = _evidence_map(conn, has_wiki)
        # task is stored on the evidence rows, not on chunks — fetch separately.
        tasks: dict[str, str] = {}
        if has_wiki:
            for cid, task in conn.execute(
                "SELECT chunk_id, task FROM wikidb.wiki_category_evidence "
                "WHERE chunk_id != '' AND task != ''"
            ):
                tasks.setdefault(cid, task)

        rows = conn.execute(
            "SELECT chunk_id, text, category_labels, workspace_id "
            "FROM chunks "
            "WHERE category_source='hybrid' AND category_labels != '' "
            "AND is_active=1"
        ).fetchall()

        written = 0
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                labels = _split_labels(row["category_labels"])
                if not labels:
                    continue
                belege = evidence.get(row["chunk_id"], [])
                f.write(json.dumps({
                    "input": {
                        "text": row["text"],
                        "task": tasks.get(row["chunk_id"], ""),
                    },
                    "output": {"kategorien": labels, "belege": belege},
                    "workspace_id": row["workspace_id"] or "",
                    "chunk_id": row["chunk_id"],
                    "category_source": "hybrid",
                }, ensure_ascii=False) + "\n")
                written += 1
        return written
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(CACHE_DIR / "memory.db"))
    ap.add_argument("--wiki-db", default=str(DEFAULT_WIKI_DB))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"db not found: {db_path}")
        return 2
    n = export(db_path, Path(args.wiki_db), Path(args.out))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] wrote {n} mayring-categorizer rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
