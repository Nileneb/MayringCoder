"""One-off prod cleanup of the two data-quality bugs fixed in commit bc110e2.

Both operations are idempotent and DRY-RUN by default — they print exactly what
they would change and write nothing unless `--apply` is passed.

  1. --igio : clear the bogus IGIO axis on `repo_file` (code) chunks. The
              classifier used to force an axis (often 'goal') onto code; the
              go-forward fix skips repo_file, this removes the historical mess.
  2. --rq   : merge near-duplicate research_questions (the dedup bug let every
              prompt mint a new row). Clusters per workspace by the stored
              embedding (cosine >= --sim), keeps the row with the highest
              occurrence_count, sums counts, repoints
              research_question_chunk_links, deletes the collapsed rows.

With no operation flag, both run. Usage inside the prod container:

    python /app/tools/cleanup_rq_and_igio.py            # dry-run, both
    python /app/tools/cleanup_rq_and_igio.py --apply     # write, both
    python /app/tools/cleanup_rq_and_igio.py --igio --apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict

DB_PATH = "/app/cache/memory.db"


def _cos(a: list[float], b: list[float]) -> float:
    s = da = db = 0.0
    for x, y in zip(a, b):
        s += x * y
        da += x * x
        db += y * y
    return s / ((da ** 0.5) * (db ** 0.5) + 1e-9)


def clear_igio_repo_file(db: sqlite3.Connection, apply: bool) -> None:
    print("\n=== IGIO clear (repo_file chunks) ===")
    before = db.execute(
        "SELECT c.igio_axis, COUNT(*) FROM chunks c JOIN sources s ON c.source_id=s.source_id "
        "WHERE s.source_type='repo_file' AND c.igio_axis!='' GROUP BY c.igio_axis ORDER BY 2 DESC"
    ).fetchall()
    total = sum(n for _, n in before)
    print(f"repo_file chunks carrying an IGIO axis: {total}  by axis: {before}")
    if not apply:
        print("DRY-RUN — would clear all of the above (no write).")
        return
    # igio_classified_at is TEXT NOT NULL DEFAULT '' — reset to '' (NOT NULL).
    cur = db.execute(
        "UPDATE chunks SET igio_axis='', igio_confidence=0.0, igio_classified_at='' "
        "WHERE igio_axis!='' AND source_id IN "
        "(SELECT source_id FROM sources WHERE source_type='repo_file')"
    )
    db.commit()
    print(f"APPLIED — cleared {cur.rowcount} rows.")


def merge_rq_dupes(db: sqlite3.Connection, apply: bool, sim: float) -> None:
    print(f"\n=== research_questions merge (cosine >= {sim}) ===")
    workspaces = [r[0] for r in db.execute(
        "SELECT DISTINCT workspace_id FROM research_questions").fetchall()]
    grand_removed = 0
    for ws in workspaces:
        rows = db.execute(
            "SELECT research_question_id, occurrence_count, embedding_id "
            "FROM research_questions WHERE workspace_id=?", (ws,)).fetchall()
        items = []
        for rid, occ, eid in rows:
            try:
                vec = json.loads(eid)
                if isinstance(vec, list) and vec:
                    items.append([rid, occ, vec])
            except Exception:
                pass
        n = len(items)
        if n < 2:
            continue
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if _cos(items[i][2], items[j][2]) >= sim:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)
        multi = [c for c in clusters.values() if len(c) > 1]
        removed = sum(len(c) - 1 for c in multi)
        grand_removed += removed
        print(f"  ws={ws}: {n} RQs → {len(clusters)} clusters, "
              f"{removed} duplicates to remove ({len(multi)} families)")
        if not apply:
            continue
        for c in multi:
            # survivor = highest occurrence_count
            c_sorted = sorted(c, key=lambda i: items[i][1], reverse=True)
            keep = items[c_sorted[0]][0]
            losers = [items[i][0] for i in c_sorted[1:]]
            extra_occ = sum(items[i][1] for i in c_sorted[1:])
            for loser in losers:
                # repoint links, resolving PK collisions (keep,chunk already present)
                for (cid,) in db.execute(
                    "SELECT chunk_id FROM research_question_chunk_links WHERE research_question_id=?",
                    (loser,)).fetchall():
                    exists = db.execute(
                        "SELECT 1 FROM research_question_chunk_links "
                        "WHERE research_question_id=? AND chunk_id=?", (keep, cid)).fetchone()
                    if exists:
                        db.execute("DELETE FROM research_question_chunk_links "
                                   "WHERE research_question_id=? AND chunk_id=?", (loser, cid))
                    else:
                        db.execute("UPDATE research_question_chunk_links SET research_question_id=? "
                                   "WHERE research_question_id=? AND chunk_id=?", (keep, loser, cid))
                db.execute("DELETE FROM research_questions WHERE research_question_id=?", (loser,))
            db.execute("UPDATE research_questions SET occurrence_count=occurrence_count+? "
                       "WHERE research_question_id=?", (extra_occ, keep))
        db.commit()
    print(f"  {'APPLIED' if apply else 'DRY-RUN'} — total duplicate RQs "
          f"{'removed' if apply else 'that would be removed'}: {grand_removed}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--igio", action="store_true", help="run the IGIO repo_file clear")
    ap.add_argument("--rq", action="store_true", help="run the research_question merge")
    ap.add_argument("--sim", type=float, default=0.85)
    ap.add_argument("--apply", action="store_true", help="write (default: dry-run)")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()
    both = not (args.igio or args.rq)
    db = sqlite3.connect(args.db, timeout=30)
    db.execute("PRAGMA busy_timeout=20000")
    if args.igio or both:
        clear_igio_repo_file(db, args.apply)
    if args.rq or both:
        merge_rq_dupes(db, args.apply, args.sim)
    db.close()
    if not args.apply:
        print("\n(DRY-RUN. Re-run with --apply to write.)")


if __name__ == "__main__":
    main()
