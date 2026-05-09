# Workspace-Identity-Migration

**Status:** Big-Bang abgeschlossen. `default`-Bucket als Eingang ist entfernt.

## Problem (vor der Migration)

Zwei nicht-synchronisierte Workspace-Auflösungs-Pfade:

```
JWT-Pfad  (claude web, MCP, /api/*):    f"user-{sub}"
CLI-Pfad  (python checker.py, ...):     getattr(args, "workspace_id", "default")
```

Derselbe physische User landete in zwei verschiedenen Buckets, je nachdem welchen
Eingangsweg eine Anfrage genommen hat. Effekt: ein User ingestete via CLI in
`default`, suchte via Web-UI in `user-2` und sah seine eigenen Daten nie.

## Lösung — kanonisches Schema

Workspace ist Erst-Klassen-DB-Entität mit Typ und Hierarchie:

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

### Hierarchie

```
user-2 (kind=user, owner_user_id=2)
├── user-2:mayringcoder  (kind=project, parent=user-2)
└── user-2:applinngames  (kind=project, parent=user-2)
team-acme (kind=team)
└── team-acme:projectx   (kind=project, parent=team-acme)
```

`team_memberships(team_id, user_id, role)` kommt sobald der erste Team-Use-Case
auftritt. Schema lässt das ohne Breaking-Change zu.

## API

```python
from src.identity.workspace_resolver import resolve_workspace
from src.identity.cli import resolve_cli_workspace

# CLI-Pfad — der Standard für ALLE workflows + cli.py
ws = resolve_cli_workspace(args)

# JWT-Pfad — input == f"user-{sub}", kein default nötig
ws = resolve_workspace(conn, jwt_workspace_id)

# Read-only check, kein auto-create (z.B. classify-igio)
ws = resolve_cli_workspace(args, auto_create=False)
```

Auflösungsreihenfolge in `resolve_cli_workspace`:

1. `args.workspace_id` → Resolver
2. `MAYRING_USER_ID` env → Resolver mit default
3. `~/.config/mayring/identity.json` (von `mayring login`)
4. `IdentityRequiredError` — kein silent `default`-Fallback

## Big-Bang-Rollout (umgesetzt)

Pre-Launch-Beta ohne echte User → kein Grund für inkrementellen Rollout.
**Alle 21 Callsites** in einem PR umgestellt:

| Datei | Pattern |
|-------|---------|
| `src/cli.py` × 9 | `getattr(args, "workspace_id", "default")` etc. → `resolve_cli_workspace(args)` |
| `src/workflows/analysis_main.py` × 5 | dito |
| `src/workflows/memory_ingest.py` × 2 | dito |
| `src/workflows/issue_ingest.py` × 1 | dito |
| `src/workflows/analysis_overview.py` × 1 | dito |
| `src/workflows/image_ingest.py` × 1 | dito |
| `src/workflows/paper_ingest.py` | `workspace_id: str = "default"` → `str \| None = None` + raise |

`src/api/routes/wiki.py:275` ist nicht betroffen — workspace_id kommt dort aus
JWT-Auth (Dependency-Injection), nicht aus args.

## Tests

- `tests/test_workspace_resolver.py` — 12 Tests (Pattern-Match, Alias, Errors)
- `tests/test_local_identity.py` — 15 Tests (env-vs-file, atomicity, expiry)
- `tests/test_workspace_isolation.py` — 13 Tests (inkl. neue:
  `test_unknown_workspace_id_raises`,
  `test_missing_workspace_with_local_user_id_resolves`)
- `tests/conftest.py` autouse-Fixture: setzt `MAYRING_USER_ID=1` für
  Mock-Tests, damit MagicMock-args nicht `IdentityRequiredError` werfen.

**1242 Tests grün, 7 skipped.**

## Was noch offen

- **`mayring login` CLI-Subcommand** mit OAuth-Browser-Flow gegen
  app.linn.games. Cache-Layer (`src/identity/local_identity.py`) ist
  fertig — fehlt nur das Frontend.
- **Legacy-Aliases registrieren**: bei Pre-Launch nicht nötig. Wenn
  irgendwann doch: `add_alias(conn, "default", "user-1")` oder analog.
