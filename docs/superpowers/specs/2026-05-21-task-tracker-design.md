# Design: Task-Tracker + Begriffs-Konsolidierung (`task_categories` → `research_questions`)

**Status:** Draft for review · **Datum:** 2026-05-21 · **Repos:** MayringCoder (Backend + MCP), app.linn.games (Frontend, via PR)

## Context & Problem

Drei Befunde aus dem Audit (2026-05-21):

1. **„task" ist im Code überladen.** Drei verschiedene Konzepte heißen „task": der `task=`-Parameter beim Kategorisieren (Mayring-Selektionskriterium, nicht persistiert), die Tabelle `task_categories` (automatisch derivierte, geclusterte Forschungsfragen) und `pi_jobs.task_text` (Agent-Job-Queue). Das verwirrt.
2. **Es gibt keine Aufgaben-Anzeige.** Das Memory-Dashboard zeigt Stats/Jobs/Feedback, aber keine Task/Goal-Tabelle. `/memory/goals` existiert, wird vom Frontend nie aufgerufen; `task_categories` hat gar keinen Endpoint.
3. **Es fehlt ein echtes Aufgaben-Konzept.** Heute existiert kein To-do-/Work-Item-Entity mit Status — nur Klassifikations-Flags (`igio_axis='goal'`) und Analytics (`task_categories`).

**Outcome dieses Specs:** Ein echter, workspace-isolierter **Task-Tracker** in MayringCoder (Mensch *und* Agent bespielen ihn via MCP/API), mit eigener Frontend-Seite in app.linn.games. Gleichzeitig wird die `task`-Kollision im Storage-Layer beseitigt, indem `task_categories` → `research_questions` umbenannt wird.

## Goals / Non-Goals

**Goals**
- Neues first-class Entity `task` (Work-Item mit Status) in MayringCoder, workspace-isoliert.
- CRUD via REST + MCP-Tools (Agents können Tasks anlegen/abhaken).
- Dedizierte interaktive „Tasks"-Seite in app.linn.games.
- `task_categories`/`task_chunk_links` → `research_questions`/`research_question_chunk_links` umbenennen (Storage + Derivation-Layer), um die Code-Kollision mit dem neuen `task`-Entity zu vermeiden.

**Non-Goals (v1, YAGNI)**
- Kein Rename des `ingest(task=...)`-API-Parameters (stabiler Client-Vertrag, inkl. Harness/Laravel) — bleibt, in Doku als „Mayring-Selektionskriterium" abgegrenzt. Future.
- Kein Rename von `pi_jobs.task_text` (separate Agent-Queue). Future.
- Keine org-/visibility-Sharing-Schicht für Tasks (Tasks sind v1 strikt an den aufgelösten `workspace_id` gebunden — der aktive-Workspace-Mechanismus greift dadurch automatisch). Future.
- Kein Assignee/Mehrnutzer-Zuweisung, keine Subtasks, keine Wiederholungen.

## Concept Model & Naming

| Konzept | Bedeutung | Bleibt/Neu |
|---|---|---|
| **task** (neu) | umsetzbares Work-Item mit Status — *das* User-facing „Task" | NEU (`tasks`-Tabelle) |
| **research_question** | automatisch derivierte, geclusterte Forschungsfrage (vormals `task_categories`) | RENAME |
| `igio_axis='goal'` | aus Memory extrahiertes Ziel (Chunk-Klassifikation) | unverändert |
| `ingest(task=...)` | Mayring-Selektionskriterium beim Kategorisieren | unverändert (dokumentiert) |
| `pi_jobs.task_text` | Agent-Job-Queue-Eintrag | unverändert (Future-Rename) |

Ein `task` kann **optional** verknüpft werden mit: einem Goal-Chunk (`linked_chunk_id` → `chunks`, typ. `igio_axis='goal'`) und/oder einem `scope_key` (`repo:`/`project:`). Lose Kopplung — keine erzwungene Hierarchie.

## Data Model

Neue Tabelle in `memory.db` (`src/memory/store.py` `_init_schema`):

```sql
CREATE TABLE IF NOT EXISTS tasks (
    task_id        TEXT PRIMARY KEY,              -- "tsk_" + uuid4 hex[:16]
    workspace_id   TEXT NOT NULL,                 -- Tenant-Isolation (wie chunks)
    title          TEXT NOT NULL,
    description    TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'open'
                     CHECK(status IN ('open','in_progress','done')),
    priority       TEXT NOT NULL DEFAULT 'medium'
                     CHECK(priority IN ('low','medium','high')),
    due_date       TEXT,                          -- ISO-8601 date oder NULL
    tags           TEXT NOT NULL DEFAULT '',      -- CSV
    created_by     TEXT,                          -- JWT sub (Mensch) | 'agent:<name>'
    linked_chunk_id TEXT REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    scope_key      TEXT,                          -- optional repo:/project:
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    completed_at   TEXT                            -- gesetzt bei status→done
);
CREATE INDEX IF NOT EXISTS idx_tasks_workspace_status ON tasks(workspace_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_workspace_due    ON tasks(workspace_id, due_date);
```

Pure CRUD-Helper in neuem Modul **`src/memory/tasks.py`** (kein DB-Zugriff in Routen):
`create_task`, `list_tasks(workspace_id, *, status?, tag?, priority?)`, `get_task`, `update_task`, `complete_task`, `delete_task`. `scope_key` wird gegen `schema.is_valid_scope_key` validiert. `completed_at` wird beim Übergang nach `done` gesetzt (und beim Reopen genullt).

### Rename `task_categories` → `research_questions`
Begrenzt auf **3 Dateien / 24 Referenzen**: `src/memory/store.py`, `src/memory/task_derivation.py`, `tests/test_task_derivation.py`.
- Tabellen: `task_categories` → `research_questions`; `task_chunk_links` → `research_question_chunk_links`; FK/PK-Spalte `task_id` → `research_question_id`. Indizes analog.
- Funktionen in `task_derivation.py`: `derive_task*` → `derive_research_question*`, `link_chunk_to_task` → `link_chunk_to_research_question`, `get_task_boost_for_chunks` → `get_research_question_boost`. **Caller-Sweep** (ingest/categorization/ambient) im Plan: `grep -rn "derive_task\|link_chunk_to_task\|get_task_boost"`.
- **Migration** (idempotent, in `_migrate_schema`): `ALTER TABLE task_categories RENAME TO research_questions` etc., guarded auf Existenz der Alt-Tabelle (SQLite unterstützt `RENAME TO` + `RENAME COLUMN`). Bestehende Prod-DB wird migriert, nicht neu angelegt.

## API + MCP

**REST — neue `src/api/routes/tasks.py`** (registriert via `app.include_router` in `server.py`), alle Endpoints mit `workspace_id = Depends(get_workspace)` (→ automatisch tenant-isoliert + aktiver-Workspace-fähig):
- `POST /tasks` — {title, description?, priority?, due_date?, tags?, linked_chunk_id?, scope_key?} → erstellt; `created_by` aus `get_token_info().sub`.
- `GET /tasks?status=&tag=&priority=` — Liste des Workspaces.
- `PATCH /tasks/{id}` — Teil-Update (title/description/status/priority/due_date/tags/links). Owner = Workspace; cross-workspace 404 (kein Leak).
- `POST /tasks/{id}/complete` — Shortcut status→done.
- `DELETE /tasks/{id}`.
- Response-Modelle in `src/api/routes/models.py`.

**MCP — neues `src/api/mcp_task_tools.py` (`register_task_tools`)**, registriert im MCP-Server analog `register_memory_tools`:
- `task_create(title, description?, priority?, due_date?, tags?, linked_chunk_id?, scope_key?, workspace_id?)`
- `task_list(status?, tag?, priority?, workspace_id?)`
- `task_update(task_id, ...)` / `task_complete(task_id)`
- Workspace via `_effective_workspace_id()`/`_enforce_tenant`; `created_by='agent:'+<tool-caller>` wenn kein menschlicher sub (sonst JWT sub). Spiegelt bewusst die Harness-Task-Tools (TaskCreate/TaskUpdate), die der User bereits nutzt.

## Frontend (app.linn.games, via PR)

- **`app/Services/Mcp/MayringTasksClient.php`** — analog `MayringStatsClient`: `list/create/update/complete/delete`, ruft die `/tasks`-API mit dem User-JWT.
- **Dedizierte Seite** (eigene Route + Livewire-Komponente `App\Livewire\Mayring\TaskBoard` + Blade), getrennt vom read-only Memory-Dashboard: Tabelle (Titel, Status, Priorität, Fällig, Tags, Ersteller, Goal-Link), Anlege-Formular, Inline-Status-Wechsel/Complete, Filter (Status/Priorität/Tag). Stil = bestehendes Dashboard. Nav-Eintrag „Tasks".

## Workspace-Isolation & Auth
Tasks gehören dem via `get_workspace` aufgelösten `workspace_id`. Dadurch greift automatisch das in `feat/active-workspace-org-sharing` gebaute Modell (personal vs. aktiver Org-Workspace). v1 ohne separate `visibility`-Spalte — ein Task ist sichtbar genau im Workspace, dem er gehört. Cross-Workspace-Zugriff → 404.

## Testing & Verification
- **MayringCoder (pytest, TDD):** `tests/test_tasks.py` — Store-CRUD; Status-Transition setzt/nullt `completed_at`; `scope_key`-Validierung; **Workspace-Isolation** (Workspace A sieht/ändert B's Tasks nicht → leere Liste/404); API-Endpoints via TestClient (dependency-override-Pattern wie test_memory_endpoints); MCP-`created_by`-Stamping. `tests/test_task_derivation.py` nach Rename grün. Volle Suite grün.
- **app.linn.games (Pest):** `MayringTasksClient` (Request-Shape) + Livewire-Komponente (Render + Create/Complete-Aktion).
- **Manuell (E2E):** Agent legt Task via `task_create` (MCP) an → erscheint in Laravel-„Tasks"-Tabelle → abhaken im UI → `status=done`/`completed_at` gesetzt; zweiter Workspace sieht ihn nicht.

## Affected files
- **Neu:** `src/memory/tasks.py`, `src/api/routes/tasks.py`, `src/api/mcp_task_tools.py`, `tests/test_tasks.py`; Laravel: `MayringTasksClient.php`, `TaskBoard` Livewire + Blade + Route.
- **Geändert:** `src/memory/store.py` (tasks-DDL + Rename-Migration), `src/api/server.py` (`include_router`), MCP-Server-Registrierung, `src/api/routes/models.py`; Rename: `src/memory/task_derivation.py` (+ Caller-Sweep), `tests/test_task_derivation.py`.

## Out of scope / Future
- Rename `ingest(task=...)`-Param und `pi_jobs.task_text`.
- Tasks org-sharing (`visibility`/Team-Sicht), Assignee, Subtasks, Recurrence.
- Strang 3: Verwaltung der auto-ingesteten/überwachten Repos (eigene Spec).
- Verdrahtung von `/memory/goals` ins Frontend (eigene kleine Spec) — Goal-Anzeige.
