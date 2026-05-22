"""Export a Forschungsfrage-quality-scorer fine-tuning dataset (#260).

One JSONL row per scored research question:

    {
      "input":  {"forschungsfrage": "Wie beeinflusst X das Y bei Z?"},
      "output": {"score": 78,
                 "warnings": [{"text": "...", "suggestion": "..."}],
                 "pico": {"population": "...", "intervention": "...", ...}},
      "workspace_id": "..."
    }

Source — ⚠ lives in app.linn.games (Laravel PostgreSQL), not in this repo:
  P1 phase-agent outputs (``PhaseAgentResult.result_data.qualitaets_bewertung``)
  plus the newer ``GameState`` evals and ``P1Warnsignal`` / ``P1Komponente``.
  This repo's local ``memory.db`` has no such rows, so a local run emits 0 rows.

The tool reads a logical source with columns
``(forschungsfrage, score, warnings, pico, workspace_id)`` where ``warnings``
and ``pico`` are JSON. Provide that source either as:
  * ``--dsn postgres://…`` — a Postgres DSN to the app.linn.games view that
    flattens ``qualitaets_bewertung`` into those columns (mapping documented in
    docs/specialist-models.md), or
  * ``--db path.sqlite`` with a ``forschungsfrage_quality`` table of the same
    shape (used by the tests).

Usage:
    python tools/export_forschungsfrage_quality_dataset.py \
        --dsn "$APP_LINN_GAMES_DSN" \
        --out cache/finetuning/forschungsfrage_quality_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mayring_core.config import CACHE_DIR

DEFAULT_OUT = CACHE_DIR / "finetuning" / "forschungsfrage_quality_dataset.jsonl"

_SELECT = (
    "SELECT forschungsfrage, score, warnings, pico, workspace_id "
    "FROM forschungsfrage_quality"
)


def _parse_json(raw, default):
    if raw in (None, ""):
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _rows_from_sqlite(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        for r in conn.execute(_SELECT):
            yield (r["forschungsfrage"], r["score"], r["warnings"],
                   r["pico"], r["workspace_id"])
    finally:
        conn.close()


def _rows_from_pg(dsn: str):
    import psycopg2  # WHY: optional — only needed for the app.linn.games source
    conn = psycopg2.connect(dsn)
    try:
        cur = conn.cursor()
        cur.execute(_SELECT)
        yield from cur.fetchall()
    finally:
        conn.close()


def export(out: Path, *, db_path: Path | None = None, dsn: str | None = None) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    if dsn:
        source = _rows_from_pg(dsn)
    elif db_path is not None:
        source = _rows_from_sqlite(db_path)
    else:
        raise ValueError("export() needs either db_path or dsn")

    written = 0
    with out.open("w", encoding="utf-8") as f:
        for forschungsfrage, score, warnings, pico, workspace_id in source:
            if not forschungsfrage:
                continue
            f.write(json.dumps({
                "input": {"forschungsfrage": forschungsfrage},
                "output": {
                    "score": int(score) if score is not None else None,
                    "warnings": _parse_json(warnings, []),
                    "pico": _parse_json(pico, {}),
                },
                "workspace_id": workspace_id or "",
            }, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="sqlite db with a forschungsfrage_quality table")
    ap.add_argument("--dsn", help="Postgres DSN to the app.linn.games quality view")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    if not args.db and not args.dsn:
        print("provide --db (sqlite) or --dsn (postgres); "
              "the production source is app.linn.games PG (see docs/specialist-models.md)")
        return 2
    if args.db and not Path(args.db).exists():
        print(f"db not found: {args.db}")
        return 2
    n = export(
        Path(args.out),
        db_path=Path(args.db) if args.db else None,
        dsn=args.dsn,
    )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] wrote {n} forschungsfrage-quality rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
