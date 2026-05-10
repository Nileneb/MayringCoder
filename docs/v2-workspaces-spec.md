# V2.0 Workspaces, Organisations & Public Memory — Spec

**Stand:** 2026-05-10
**Status:** Draft — wartet auf User-Approval
**Verbunden mit:** Audit-Report (parallel zu diesem Doc), Issue #104 (MayringCoder), Issue #225 (app.linn.games)

## Problem (User-Aussage)

> Wir haben IMMER WIEDER workspace probleme. Wir wollen im MayringCoder ein Organisationsgedächtnis sowie ein Public Gedächtnis. Akzeptanzkriterien wurden mehrfach nicht durchgehalten — Symptome werden gepatched, Pipeline bricht silent woanders. Ziel: Version 2.0 Userverwaltung mit Organisationsverwaltung. Safety + Usability nicht vernachlässigen. Keine Quick-Fixes.

## Audit-Befunde — Kurzfassung

8 konkrete Bugs mit File:Line, in Audit-Report dokumentiert:

| # | Repo | Datei:Line | Symptom |
|---|---|---|---|
| L1 | app.linn.games | `database/schema/pgsql-schema.sql:1779-1810` | `workspaces` ohne `type`-Spalte → kein personal/org-Unterschied |
| L2 | app.linn.games | `app/Services/JwtIssuer.php:44-75` | JWT trägt nur 1 ws → Multi-Membership unsichtbar |
| L3 | beide | — | `org_id` als JWT-Claim wird nirgends gesetzt → org-visibility tot |
| L4 | MayringCoder | — | Membership-Revoke in Laravel ⇒ orphan chunks in Mayring |
| L5 | beide | — | JWT-Refresh bei ws-Switch nicht implementiert |
| L6 | MayringCoder | `src/api/routes/sync.py:33-74` | Sync droppt `org`/`user`-visibility silent |
| L7 | MayringCoder | `src/api/routes/memory.py:90-114` | REST-search passt `org_id`/`user_id` nicht → Hook + Laravel sehen nur private+public |
| L8 | MayringCoder | `src/api/routes/memory.py:559-579` | `PATCH /sources/{id}/visibility` ohne Owner-Check → cross-tenant-vandalism |

## V2-Architektur

### Workspace-Typ-Modell

```
personal      — auto-erstellt pro User. owner + optional members.
organization  — explizit erstellt. mehrere members, roles.
public        — virtuell, kein eigener ws. visibility='public' auf source-level.
```

KISS: 2 echte ws-Typen (personal, organization). `public` ist ein Visibility-Flag, kein eigener Bucket. Sub-Workspaces / Projects (vom MayringCoder-Schema teilweise vorbereitet) sind **out of scope V2** — kann später ergänzt werden ohne Breaking Change.

### Visibility-Semantik (verbindlich)

| Visibility | Sichtbar für |
|---|---|
| `private` | nur Caller wo `workspace_id == active_ws` |
| `user` | nur Caller wo `sub == source.user_id` (cross-device des same human) |
| `org` | nur Caller wo Mitgliedschaft im `org_id` der Source |
| `public` | jeder gauthed JWT-Holder |

`_scope_filter` in `src/memory/retrieval.py:66-122` macht diese Semantik bereits korrekt. Bug ist auf der **Caller-Seite** (REST-routes passen `org_id`/`user_id` nicht durch).

### JWT-Claim-Vertrag V2

```json
{
  "sub": "42",
  "email": "bene@linn.games",
  "workspace_id": "<active_ws_uuid>",
  "memberships": [
    { "id": "<uuid_personal>",  "type": "personal",     "role": "owner" },
    { "id": "<uuid_team_acme>", "type": "organization", "role": "editor" },
    { "id": "<uuid_team_lab>",  "type": "organization", "role": "viewer" }
  ],
  "scope": ["mcp:memory"],
  "iat": 1234567890,
  "exp": 1234571490,
  "iss": "https://app.linn.games",
  "aud": "mayringcoder"
}
```

`memberships` ist die **single source of truth für visibility-checks**. MayringCoder berechnet aus `memberships`:
- `caller.workspace_id` = `active_ws_uuid` (für `private`)
- `caller.sub` = `sub` (für `user`)
- `caller.org_ids` = `[m.id for m in memberships if m.type=='organization']` (für `org`)

### Schema-Änderungen

**Laravel** (`workspaces`):
```sql
ALTER TABLE workspaces
  ADD COLUMN type VARCHAR(32) NOT NULL DEFAULT 'personal'
  CHECK (type IN ('personal', 'organization'));
```

**MayringCoder** — Schema bereits V2-ready:
- `sources.visibility` CHECK in (`private`,`org`,`public`,`user`) ✓ (`store.py:218,254`)
- `sources.org_id` ✓
- `sources.user_id` ✓
- `workspaces.kind` ✓ (wird mit Laravel-`type` synchronisiert)

Keine destruktiven Migrationen nötig — alles additive.

### Defaults für die 5 Open-Questions (aus Audit)

| Frage | Default V2 | Begründung |
|---|---|---|
| **Workspace-Typ-Modell** | nur `personal` + `organization` (kein `project`-Sub-WS) | KISS. project-Sub-WS bringt UI-Komplexität ohne klaren Use-case heute. |
| **Membership-Source-of-Truth** | Laravel (`workspace_users`). MayringCoder liest aus JWT-Claim `memberships[]`, kein eigener Cache. | Keine Sync-Drift. JWT-TTL = 1h; bei Membership-Änderung in Laravel braucht es 1h bis neuer JWT alle Mayring-clients erreicht. |
| **`'user'` + `'org'` Visibility** | beide behalten. | `user` = cross-device (Hook von claude-cli + claude.ai-web sehen dieselben chunks). `org` = team-share. Verschiedene Use-cases. |
| **PATCH /sources/{id}/visibility Authz** | Source-Owner (= ws-owner) bei `private`/`user`. Org-Admin bei `org`-shares. **Plus** neuer Endpunkt `POST /sources/{id}/share` mit `target=public|org:<id>` für expliziten Share-Workflow. | Zwei Wege, beide enforced. PATCH bleibt für admin-tools, share ist user-facing. |
| **Public-Read ungeauthed** | JWT bleibt Pflicht. | Tracking + abuse-prevention. Public bedeutet "alle JWTs sehen", nicht "anonym readable". |

### API V2

**MayringCoder** (zusätzlich zu existierenden):
- `POST /memory/orgs` — create org-workspace (synchronisiert mit Laravel via service-token-RPC)
- `GET /memory/orgs/{org_id}/members` — list members (verifiziert via JWT-membership-claim)
- `POST /memory/orgs/{org_id}/invite` — invite by email (proxy zu Laravel)
- `POST /sources/{id}/share` — explicit share (`target=public` | `org:<id>`)
- **Fix bestehende:** `/memory/search` (REST) passt `org_id`/`user_id` (siehe MCP-Pfad als Vorbild)
- **Fix bestehende:** `/sources/{id}/visibility` PATCH mit owner-check

**app.linn.games**:
- `POST /api/workspaces` — create workspace (mit `type`)
- `POST /api/workspaces/{id}/members` — invite
- `DELETE /api/workspaces/{id}/members/{user_id}` — remove
- `PATCH /api/workspaces/{id}` — update name/type
- workspace-switcher in Livewire-Layout (active_ws-Select im Header)

### Implementations-Reihenfolge (Iterationen)

**Iter 1 — JWT-Vertrag (Foundation):**
1. Laravel: `workspaces.type`-column migration
2. Laravel: `JwtIssuer::issueForUser` setzt `memberships[]` aus `workspace_users`
3. MayringCoder: `TokenInfo` parst `memberships[]` (backward-compat: alte JWTs ohne field → 1-element array aus workspace_id)
4. MayringCoder: `_scope_filter` nutzt `caller.org_ids` (List-IN-Query) statt `caller.org_id` (Single)
5. **Smoke-test:** User in 2 orgs sieht beide org-buckets

**Iter 2 — Bug-Fixes (Hardening):**
1. L7 — REST `/memory/search` passes `org_id`/`user_id` aus TokenInfo
2. L8 — PATCH visibility mit owner/admin-check
3. L6 — `sync.py` respektiert org/user
4. L3 — JWT-`memberships[]` durchgereicht überall

**Iter 3 — Org-Verwaltung (UI/API):**
1. Laravel: `/api/workspaces*`-routes (create/invite/remove/update)
2. Livewire: workspace-switcher + org-Verwaltungs-UI
3. MayringCoder: `/memory/orgs*`-Endpunkte (proxies)

**Iter 4 — Public-Share (UI):**
1. `POST /sources/{id}/share`
2. Livewire: "Share with Org/Public"-Button im Memory-Dashboard
3. Re-Audit: 0 chunks im `system`-bucket nach Migration; alle `paper:`-sources im richtigen ws

### Test-Matrix (Smoke + Pest, alle MUST pass)

| Test | Pfad | Erwartung |
|---|---|---|
| `private_isolation` | User A ingest private → User B search | 0 hits |
| `user_cross_device` | User claude-cli ingest user → User claude-web search | 1 hit |
| `org_member_visibility` | A in org-X ingest org → B in org-X search | 1 hit |
| `org_non_member_blocked` | A in org-X ingest org → C nicht-member search | 0 hits |
| `public_visibility` | A ingest public → B (different org) search | 1 hit |
| `org_revoke_isolation` | A in org-X ingest → A removed from org → A search | 0 hits NACH JWT-refresh |
| `patch_visibility_authz` | A ingest source → B PATCH visibility | 403 |
| `share_endpoint` | A `POST /sources/{id}/share target=public` | 200, B sieht source |
| `multi_org_membership` | A in org-X+org-Y ingest in beiden → A search | beide hits |
| `stats_workspaces_lists_all` | User A in 3 ws → `GET /stats/workspaces` | 3 rows |

### Akzeptanz-Kriterien (V2.0 erfüllt wenn ALLE 10 Tests grün)

1. ✓ `private_isolation` — keine cross-tenant-leaks bei `private`
2. ✓ `user_cross_device` — `'user'`-visibility funktioniert
3. ✓ `org_member_visibility` — org-share funktioniert für members
4. ✓ `org_non_member_blocked` — non-members sehen nichts
5. ✓ `public_visibility` — public ist global lesbar
6. ✓ `org_revoke_isolation` — JWT-refresh nach revoke killt access
7. ✓ `patch_visibility_authz` — 403 für fremde
8. ✓ `share_endpoint` — share-API funktioniert
9. ✓ `multi_org_membership` — Multi-org-User sieht alle seine orgs
10. ✓ `stats_workspaces_lists_all` — `/stats/workspaces` listet alle
11. ✓ `Bug L1-L8 fixed` — alle 8 Audit-Bugs durch ihre eigenen Tests abgedeckt

## Open Questions (für User-Entscheidung)

Defaults oben sind sinnvolle KISS-Wahlen. User-Entscheidung wenn anderes gewünscht:

1. **Sub-Workspaces (project)** — drinhaben oder Iter 5+? **Default: Iter 5+.**
2. **Workspace-Tier-Differenzierung** — Personal-tier kostenlos, Org-tier kostet? **Default: heutige Stripe-tier-Logik bleibt unverändert.**
3. **Org-Branding** — eigenes Logo/Color für Orgs? **Default: nur `name`, kein Branding V2.**
4. **Soft-delete** für Orgs — leave statt delete? **Default: hard-delete blockt wenn andere members; soft-delete bei single-owner-leave.**
5. **Backfill der existierenden 671 system-bucket-sources** — wohin? **Default: SQL-migration mit `paper_id`→`p5_treffer.projekt_id`→`projekt.user_id`-Lookup; sources die das nicht resolven können bleiben in `system`.**

## Estimated Effort

- Iter 1: 1 Tag (TDD JWT-Erweiterung)
- Iter 2: 1 Tag (4 Bug-Fixes)
- Iter 3: 2 Tage (UI + API)
- Iter 4: 0.5 Tage (Share-Endpoint + Button)
- **Gesamt: ~4-5 Tage**

## Wenn fertig: Beweis

Smoke-Suite mit allen 10 Tests grün auf production. `GET /stats/workspaces` zeigt für `bene`:
```json
{
  "workspaces": [
    {"id": "<bene-uuid>", "type": "personal", "chunks": 1500+, ...},
    {"id": "<acme-uuid>", "type": "organization", "chunks": 200+, ...}
  ]
}
```

Plus dokumentierte Reproduktions-Kommandos pro Test in `docs/v2-workspaces-runbook.md`.
