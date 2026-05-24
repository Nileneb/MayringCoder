"""Read-only Inventur für die Workspace-Re-Point-Migration (019d6933 → app.linn.games-SoT).

Zählt pro workspace-scoped Tabelle die Zeilen unter der verwaisten alten Identität
(workspace_id=019d6933, user_id/sub=2) — Grundlage für die Migrations-Spec. NUR SELECTs.

Lauf auf Prod (read-only):
    cat tools/ws_inventory.py | ssh u-server \
      'cd ~/app.linn.games && docker compose exec -T mayring-api python -'
"""
import sqlite3

DB = "/app/cache/memory.db"
OLD_WS = "019d6933-002e-7153-a7df-f14e4c7d52b4"
OLD_SUB = "2"

c = sqlite3.connect(DB)
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def cols(t):
    return [r[1] for r in c.execute(f"PRAGMA table_info('{t}')")]


print(f"# DB={DB}  OLD_WS={OLD_WS}  OLD_SUB={OLD_SUB}")
print(f"# tables: {len(tables)}")

print("\n## workspace_id rows under OLD_WS (per table)")
total = 0
for t in sorted(tables):
    if "workspace_id" not in cols(t):
        continue
    try:
        n = c.execute(f"SELECT count(*) FROM {t} WHERE workspace_id=?", (OLD_WS,)).fetchone()[0]
    except sqlite3.Error as e:
        print(f"  {t}: ERROR {e}")
        continue
    if n:
        total += n
        print(f"  {t}: {n}")
print(f"  TOTAL rows to re-point (workspace_id): {total}")

print("\n## distinct workspace_id in chunks (landscape)")
try:
    for ws, n in c.execute("SELECT workspace_id, count(*) FROM chunks GROUP BY workspace_id "
                           "ORDER BY 2 DESC LIMIT 15"):
        print(f"  {ws!r}: {n}")
except sqlite3.Error as e:
    print(f"  ERROR {e}")

print("\n## user_id distribution (chunks/sources) — sub=2 drift")
for t in ("chunks", "sources"):
    if t in tables and "user_id" in cols(t):
        for uid, n in c.execute(f"SELECT user_id, count(*) FROM {t} GROUP BY user_id "
                                f"ORDER BY 2 DESC LIMIT 8"):
            print(f"  {t}.user_id={uid!r}: {n}")

print("\n## projects (id, workspace_id, owner_id, name)")
if "projects" in tables:
    pc = cols("projects")
    sel = "id, workspace_id" + (", owner_id" if "owner_id" in pc else "") + \
          (", name" if "name" in pc else "")
    for row in c.execute(f"SELECT {sel} FROM projects LIMIT 30"):
        print(f"  {row}")

print("\n## codebook_categories.project_id (Phase 3.2 — projekt-scoped induced)")
if "codebook_categories" in tables and "project_id" in cols("codebook_categories"):
    for pid, n in c.execute("SELECT project_id, count(*) FROM codebook_categories "
                            "GROUP BY project_id ORDER BY 2 DESC LIMIT 10"):
        print(f"  project_id={pid!r}: {n}")

print("\n## workspaces table (id, owner_user_id)")
if "workspaces" in tables:
    wc = cols("workspaces")
    sel = "id" + (", owner_user_id" if "owner_user_id" in wc else "") + \
          (", name" if "name" in wc else "")
    for row in c.execute(f"SELECT {sel} FROM workspaces LIMIT 30"):
        print(f"  {row}")
c.close()
