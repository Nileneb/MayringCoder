"""Consolidate leftover cruft workspaces into the canonical user workspace.

Merges `default` + `mayringcoder` (and any extra ids passed via CRUFT env, comma-sep)
into the canonical `019e14d6` using the shared repoint_workspace() — moves rows + Chroma
metadata and registers aliases so old tokens keep resolving. KEEPS infra buckets
(system/public/bene:logs).

Read-only DRY-RUN by default. Writes only with APPLY=1. Run in the mayring-api container:
    cat tools/migrate_consolidate_cruft.py | ssh u-server \
      'docker exec -i $(docker ps -qf name=mayring-api|head -1) python -'     # preview
    cat tools/migrate_consolidate_cruft.py | ssh u-server \
      'docker exec -i -e APPLY=1 $(docker ps -qf name=mayring-api|head -1) python -'
"""
import os
import shutil
import sqlite3
import time

DB = os.environ.get("MAYRING_DB", "/app/cache/memory.db")
CANON = os.environ.get("CANON_WS", "019e14d6-0489-7348-bca8-e29c11293cb7")
CRUFT = [w.strip() for w in os.environ.get("CRUFT", "default,mayringcoder").split(",") if w.strip()]
APPLY = os.environ.get("APPLY") == "1"

print(f"# DB={DB}  CANON={CANON}  CRUFT={CRUFT}  mode={'APPLY' if APPLY else 'DRY-RUN'}")
conn = sqlite3.connect(DB)

# Preview counts per cruft ws.
from src.api.workspace_repoint import _tables_with_workspace_id  # noqa: E402
for old in CRUFT:
    total = 0
    for t in _tables_with_workspace_id(conn):
        n = conn.execute(f"SELECT COUNT(*) FROM {t} WHERE workspace_id=?", (old,)).fetchone()[0]
        if n:
            print(f"  {old}: {t}={n}")
            total += n
    print(f"  {old}: TOTAL={total}")

if not APPLY:
    print("\n# DRY-RUN complete — nothing written. APPLY=1 to apply.")
    raise SystemExit(0)

backup = f"{DB}.bak-{int(time.time())}"
shutil.copy2(DB, backup)
print(f"\n# BACKUP: {backup}")

try:
    from mayring_core.memory.store import get_chroma_collection
    chroma = get_chroma_collection("memory_chunks")
except Exception as e:  # noqa: BLE001
    print(f"# WARN: no chroma ({e}) — SQLite rows only")
    chroma = None

from src.api.workspace_repoint import repoint_workspace  # noqa: E402
for old in CRUFT:
    if old == CANON:
        continue
    counts = repoint_workspace(conn, old, CANON, chroma=chroma)
    print(f"# repointed {old} -> {CANON}: {counts}")
print("# consolidation complete")
