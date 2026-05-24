"""Re-point der verwaisten alten Identität 019d6933 → app.linn.games-SoT 019e14d6.

Hintergrund: app.linn.games wurde neu aufgesetzt (user 2→1, workspace
019d6933→019e14d6 "Bene Workspace"). MayringCoder zog nicht mit → die gesamte
Memory liegt unter der verwaisten 019d6933. Dieses Skript re-pointed ALLES auf
die kanonische app.linn.games-Workspace, behält die project_id-Sub-Dimension,
und registriert einen Alias damit alte Tokens (hook.jwt/019d6933) weiter gelten.

Read-only DRY-RUN per Default. Schreiben NUR mit APPLY=1 in der Umgebung.

Lauf (im mayring-api-Container, read-only Vorschau):
    cat tools/migrate_workspace_repoint.py | ssh u-server \
      'docker exec -i $(docker ps -qf name=mayring-api|head -1) python -'
Anwenden:
    cat tools/migrate_workspace_repoint.py | ssh u-server \
      'docker exec -i -e APPLY=1 $(docker ps -qf name=mayring-api|head -1) python -'
"""
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone

DB = "/app/cache/memory.db"
OLD_WS = "019d6933-002e-7153-a7df-f14e4c7d52b4"
NEW_WS = "019e14d6-0489-7348-bca8-e29c11293cb7"  # Bene Workspace (app.linn.games SoT)
OLD_SUB = "2"
NEW_SUB = "1"
CHUNKS_COLLECTION = "memory_chunks"
APPLY = os.environ.get("APPLY") == "1"

mode = "APPLY (writing)" if APPLY else "DRY-RUN (read-only)"
print(f"# workspace re-point  {OLD_WS} -> {NEW_WS}   mode={mode}")

c = sqlite3.connect(DB)
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def has_col(t, col):
    return col in [r[1] for r in c.execute(f"PRAGMA table_info('{t}')")]


ws_tables = [t for t in sorted(tables) if has_col(t, "workspace_id")
             and c.execute(f"SELECT count(*) FROM {t} WHERE workspace_id=?", (OLD_WS,)).fetchone()[0]]

print("\n## SQLite re-point plan (rows under OLD_WS)")
plan = {}
for t in ws_tables:
    n = c.execute(f"SELECT count(*) FROM {t} WHERE workspace_id=?", (OLD_WS,)).fetchone()[0]
    plan[t] = n
    print(f"  {t}: {n}")
print(f"  TOTAL: {sum(plan.values())}")

sub_rows = 0
if has_col("sources", "user_id"):
    sub_rows = c.execute("SELECT count(*) FROM sources WHERE user_id=?", (OLD_SUB,)).fetchone()[0]
print(f"\n## user_id remap sources {OLD_SUB} -> {NEW_SUB}: {sub_rows} rows")

ws_row_exists = c.execute("SELECT count(*) FROM workspaces WHERE id=?", (NEW_WS,)).fetchone()[0]
print(f"## workspaces row for NEW_WS exists: {bool(ws_row_exists)} (insert if missing)")

if "workspace_aliases" in tables:
    acols = [r[1] for r in c.execute("PRAGMA table_info('workspace_aliases')")]
    print(f"## workspace_aliases columns: {acols}")


def _chunks_collection():
    """Same chroma the app uses: MAYRING_LOCAL_CHROMA env, else default path."""
    from mayring_core.memory.store import get_chroma_collection
    path = os.environ.get("MAYRING_LOCAL_CHROMA") or None
    return get_chroma_collection(CHUNKS_COLLECTION, path=path), \
        (path or "<default cache/memory_chroma>")


print("\n## Chroma preview (memory_chunks vectors under OLD_WS)")
try:
    _col, _cpath = _chunks_collection()
    _got = _col.get(where={"workspace_id": OLD_WS}, include=[])
    print(f"  path={_cpath}  vectors_to_update={len(_got.get('ids') or [])}")
except Exception as e:  # noqa: BLE001 — preview only
    print(f"  chroma preview error: {e}")

if not APPLY:
    print("\n# DRY-RUN complete — nichts geschrieben. APPLY=1 zum Anwenden.")
    c.close()
    raise SystemExit(0)

# ---- APPLY ----------------------------------------------------------------
backup = f"{DB}.bak-{int(time.time())}"
shutil.copy2(DB, backup)
print(f"\n# BACKUP: {backup}")

try:
    now = datetime.now(timezone.utc).isoformat()
    c.execute("BEGIN")
    if not ws_row_exists:
        # workspaces: id, kind CHECK(user/team/project/system), display_name, created_at, updated_at
        c.execute("INSERT INTO workspaces(id, kind, display_name, created_at, updated_at) "
                  "VALUES (?, 'user', ?, ?, ?)", (NEW_WS, "Bene Workspace", now, now))
    for t in ws_tables:
        c.execute(f"UPDATE {t} SET workspace_id=? WHERE workspace_id=?", (NEW_WS, OLD_WS))
    if sub_rows:
        c.execute("UPDATE sources SET user_id=? WHERE user_id=?", (NEW_SUB, OLD_SUB))
    # Alias 019d6933 -> 019e14d6 (Tabelle: alias PK, workspace_id FK, created_at NOT NULL).
    # ACHTUNG: der API-Pfad resolve_workspace_from_token konsultiert Aliases NICHT —
    # alte API-Tokens (019d6933) müssen neu ausgestellt werden. Der Alias deckt nur
    # den 6-Schritt-CLI-Resolver + Zukunftssicherheit ab.
    c.execute("INSERT OR IGNORE INTO workspace_aliases(alias, workspace_id, created_at) "
              "VALUES (?,?,?)", (OLD_WS, NEW_WS, now))
    c.execute("COMMIT")
    print("# SQLite re-point committed")
except Exception as e:
    c.execute("ROLLBACK")
    print(f"# SQLITE ERROR -> ROLLBACK: {e}  (restore: cp {backup} {DB})")
    raise

# ---- Chroma metadata re-point (memory_chunks) -----------------------------
try:
    col, cpath = _chunks_collection()
    print(f"# Chroma path: {cpath}")
    got = col.get(where={"workspace_id": OLD_WS}, include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    print(f"\n# Chroma {CHUNKS_COLLECTION}: {len(ids)} vectors with OLD_WS metadata")
    B = 256
    for i in range(0, len(ids), B):
        chunk_ids = ids[i:i + B]
        chunk_metas = [{**(m or {}), "workspace_id": NEW_WS} for m in metas[i:i + B]]
        col.update(ids=chunk_ids, metadatas=chunk_metas)
    print(f"# Chroma metadata updated: {len(ids)} vectors -> NEW_WS")
except Exception as e:
    print(f"# CHROMA ERROR: {e}  (SQLite already committed; re-run only the chroma block)")
    raise

# ---- verify ---------------------------------------------------------------
print("\n## verify (rows now under NEW_WS)")
for t in ws_tables:
    n = c.execute(f"SELECT count(*) FROM {t} WHERE workspace_id=?", (NEW_WS,)).fetchone()[0]
    print(f"  {t}: {n}")
left = sum(c.execute(f"SELECT count(*) FROM {t} WHERE workspace_id=?", (OLD_WS,)).fetchone()[0]
           for t in ws_tables)
print(f"# remaining under OLD_WS (should be 0): {left}")
c.close()
print("# DONE")
