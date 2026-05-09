# Workspace-Identity-Migration

**Status:** Foundation merged (commit folgt), Callsite-Rollout in Folge-PRs.

## Problem

Vor dieser Migration gab es zwei nicht synchronisierte Workspace-Auflösungs-Pfade:

```
JWT-Pfad  (claude web, MCP, /api/*):    f"user-{sub}"
CLI-Pfad  (python checker.py, ...):     getattr(args, "workspace_id", "default")
```

Derselbe physische User landete in zwei verschiedenen Buckets, je nachdem welchen
Eingangsweg eine Anfrage genommen hat. Effekt: ein User ingestete via CLI in
`default`, suchte via Web-UI in `user-2` und sah seine eigenen Daten nie.

## Architektur

Workspace ist nicht mehr ein freier String, sondern Erst-Klassen-DB-Entität mit
Typ und Hierarchie:

```sql
workspaces (
  id            TEXT PRIMARY KEY,        -- z.B. "user-2", "user-2:mayringcoder"
  kind          TEXT NOT NULL,           -- 'user' | 'team' | 'project' | 'system'
  parent_id     TEXT REFERENCES workspaces(id),
  owner_user_id INTEGER,                 -- app.linn.games User.id
  display_name, created_at, updated_at
)

workspace_aliases (
  alias        TEXT PRIMARY KEY,         -- z.B. "default", "nileneb-mayringcoder"
  workspace_id TEXT REFERENCES workspaces(id)
)
```

### Hierarchie-Beispiel

```
user-2 (kind=user, owner_user_id=2)
├── user-2:mayringcoder  (kind=project, parent=user-2)
└── user-2:applinngames  (kind=project, parent=user-2)
team-acme (kind=team)
└── team-acme:projectx   (kind=project, parent=team-acme)
```

Sub-Workspaces erlauben Repo-Granularität ohne den User-Bucket aufzubrechen.
Multi-Tenant via `kind=team` ist vorbereitet, aber nicht umgesetzt — eine Tabelle
`team_memberships(team_id, user_id, role)` kommt sobald der erste Team-Use-Case
auftritt.

## API

```python
from src.identity.workspace_resolver import resolve_workspace

# CLI mit explicit input
ws = resolve_workspace(conn, args.workspace_id, default_user_id=user_id)

# JWT-flow (input == f"user-{sub}", kein default nötig)
ws = resolve_workspace(conn, jwt_workspace_id)

# Read-only check, kein auto-create
ws = resolve_workspace(conn, candidate, auto_create_user_workspace=False)
```

Auflösungsreihenfolge:

1. `input is None or ""` → fallback auf `default_user_id` → `user-{id}` (sonst raise)
2. `input` matcht `user-N` oder `user-N:slug` → ensure-row + return
3. `input` ist canonical workspace (in `workspaces`) → return as-is
4. `input` ist alias (in `workspace_aliases`) → return aliased canonical
5. else: `UnknownWorkspaceError`

### Local Identity (CLI-Pfad)

```python
from src.identity.local_identity import get_local_user_id

uid = get_local_user_id()
# 1. ENV MAYRING_USER_ID
# 2. ~/.config/mayring/identity.json (geschrieben von `mayring login`)
# 3. None → caller MUSS error-out, kein 'default'-Silent-Fallback
```

## Callsite-Rollout-Plan

17 Stellen verwenden den alten `getattr(args, "workspace_id", "default")`-Pattern.
Reihenfolge nach Risiko, je PR ein Logical-Bundle:

### PR 2.1 — CLI-Entry-Points (5 Callsites)
- `src/cli.py:41,58,89,213,235,252,268,310,432`

### PR 2.2 — Workflow-Module (6 Callsites)
- `src/workflows/issue_ingest.py:75`
- `src/workflows/image_ingest.py:30`
- `src/workflows/paper_ingest.py:18`
- `src/workflows/analysis_overview.py:114`
- `src/workflows/memory_ingest.py:141,233`
- `src/workflows/analysis_main.py:185,294,443,472,479`

### PR 2.3 — API-Routen (1 Callsite)
- `src/api/routes/wiki.py:275`

### PR 2.4 — `mayring login` Command + OAuth-Flow
- Neuer CLI-Subcommand mit Browser-OAuth gegen app.linn.games
- Schreibt `~/.config/mayring/identity.json`

### PR 2.5 — Legacy-Aliases registrieren (Datenmigration)
- Pre-launch: keine echten User → optional. Falls doch:
  ```python
  add_alias(conn, "default", "user-1")  # Initial-Owner
  add_alias(conn, "nileneb-mayringcoder", "user-2:mayringcoder")
  ```

## Rollback

Foundation lässt das alte Verhalten unverändert: solange kein Callsite den
Resolver aufruft, läuft alles wie vor. Tabellen `workspaces` und
`workspace_aliases` sind reine ADD ohne Constraint-Änderungen an existing tables.

## Tests

- `tests/test_workspace_resolver.py` — 12 Tests (kanonisch, alias, error-paths)
- `tests/test_local_identity.py` — 15 Tests (env-vs-file, atomicity, expiry)

27/27 grün. Foundation deployed bevor erste Callsite umgestellt wird.
