# Design: Captured Task-Lists in the IGIO-Lens Intervention Column

**Status:** Approved (2026-05-25, User „ok") · **Repos:** mayring-claude-plugin (hook), MayringCoder (backend), app.linn.games (lens frontend)
**Verbunden mit:** [[project_igio_lens]], [[project_org_public_memory_acceptance]], `docs/superpowers/specs/2026-05-22-prompt-derived-goals-todos.md` (Vorgänger — Backend großteils gebaut), Subsystem B der Org/Public-Memory-Decomposition.

## Problem (User-Aussage)

> „aktuell kommen bei 'interVentions' keine interventionen dazu. Aber eigentlich solltest du dir doch immer Todo Listen erstellen == die sollten da gelistet werden."

Prod-Befund (Workspace `bene`, 2026-05-24): die IGIO-Lens-Spalte `intervention` ist quasi leer (6 chunk-igio-Treffer), und die Todo-Listen, die der Agent während der Arbeit anlegt, landen **nirgends sichtbar**. Der 2026-05-22-Spec (`derive_todo` aus Prompts + Tasks-Seite) ist backend-seitig gebaut (`todo_derivation.py`, `/tasks`, `/tasks/goals`), aber: (a) `GET /tasks` liefert 0 Todos in Prod (derive_todo tot/proxy), und (b) der Frontend-Surface wurde diese Session entfernt — die Tasks-Seite wurde zur IGIO-Lens umgebaut, `MayringTasksClient` gelöscht.

## Entscheidungen (User, 2026-05-25)

1. **Surface:** Die Intervention-Spalte der IGIO-Lens **zeigt die echten Todos** (statt der sparse chunk-igio-Interventions). Vereint die gelöschte Tasks-View wieder in der Lens.
2. **Quelle:** Die Todos kommen aus den **tatsächlichen Task-Listen des Agenten** — erfasst via neuem **PostToolUse-Hook** auf `TaskCreate`/`TaskUpdate`. (Nicht `derive_todo`-aus-Prompts; das bleibt dormant, um Doppel-Quellen zu vermeiden.)

## Architektur

```
Agent ruft TaskCreate/TaskUpdate
      │  (PostToolUse)
      ▼
plugin hook  ──HTTP(hook.jwt)──▶  POST /tasks (external_id, created_by='agent')
                                  PATCH /tasks/{id} / complete  (Status)
      │
      ▼
tasks-Tabelle (workspace=bene)
      │
      ▼
GET /stats/igio-lens → { issue, goal, outcome: chunk-igio;  intervention: { todos: [...] } }
      │
      ▼
IgioLens (Livewire) — Intervention-Spalte rendert Todos (Titel + Status-Badge)
```

### Komponente 1 — PostToolUse-Hook (mayring-claude-plugin, neu)
- Neue Datei `hooks/task_capture.py`, registriert als `PostToolUse` in `hooks/hooks.json`.
- Feuert auf die Todo-Tools (`tool_name` in `{TaskCreate, TaskUpdate, TodoWrite}` — die exakte Menge wird vom Spike bestätigt). Andere Tools → sofortiger No-op-Exit.
- Extrahiert aus dem PostToolUse-Payload: `harness_task_id` (für Idempotenz), `title`, `status`. Für `TodoWrite` (List-Tool): iteriert die Todos-Liste.
- Schreibt via HTTP an `MAYRING_API_URL` mit `hook.jwt` (workspace-scoped):
  - neu → `POST /tasks` mit `{title, created_by:'agent', external_id, tags:'agent', status}`
  - Status-/Titel-Änderung → `PATCH /tasks/{id}` (resolved über external_id) bzw. `POST /tasks/{id}/complete` bei status=completed.
- **Best-effort, nicht-blockierend, raised NIE** (folgt dem Muster der bestehenden Hooks: `_read_token`-Skip ohne JWT, stderr-Log bei Fehler, kein Tool-Call-Bruch). Kurzer Timeout (≤3s), kein Retry-Sturm.

### Komponente 2 — Backend-Idempotenz (MayringCoder)
- Additive Migration in `core/mayring_core/memory/store.py`: nullable Spalte `tasks.external_id TEXT` + Index `idx_tasks_external_id (workspace_id, external_id)`.
- `create_task(... external_id=None)`: wenn `external_id` gesetzt und eine offene/bestehende Zeile mit (workspace_id, external_id) existiert → **Update statt Insert** (Upsert). Sonst Insert.
- `POST /tasks` akzeptiert `external_id` im `TaskCreateRequest`. So ist Re-Capture desselben Tool-Calls idempotent (1 Zeile).

### Komponente 3 — Lens-Backend (MayringCoder)
- `GET /stats/igio-lens` (`src/api/routes/igio_admin.py`) erweitern: das Response-Objekt bekommt für die Achse `intervention` zusätzlich `todos`: die offenen + die zuletzt (≤7 Tage) erledigten Tasks des Workspace via `list_tasks(conn, ws, status=...)`, neueste/offene zuerst, je `{task_id, title, status, created_by, created_at, completed_at}`. Die chunk-igio-Zählung für `intervention` bleibt als `count` erhalten (Rückwärtskompat), aber das Frontend nutzt `todos` für die Spalte.
- Workspace-Scoping wie der bestehende Endpoint (admin=alle / JWT=eigener ws).

### Komponente 4 — Lens-Frontend (app.linn.games)
- `MayringStatsClient::getIgioLens()` reicht das erweiterte Response (inkl. `intervention.todos`) durch (defensiv: fehlt `todos` → leere Liste).
- `IgioLens` Livewire-Component + `igio-lens.blade.php`: die **Intervention-Spalte** rendert die `todos` (Titel + Status-Badge: open/in_progress/done; offene zuerst, erledigte gedimmt). Die Spalten issue/goal/outcome unverändert (chunk-Preview). Nur Skalar-Blätter rendern (htmlspecialchars-500-Lektion).

## De-Risking — Spike zuerst (Plan-Task 1)

Es ist **unverifiziert**, ob PostToolUse für die `Task*`-Tools feuert und welches Payload-Schema vorliegt (wo `harness_task_id`/`status` stehen). Plan-Task 1 ist daher ein **Spike**: einen minimalen `task_capture.py` deployen, der NUR das rohe PostToolUse-Payload nach stderr/eine Datei loggt; ein `TaskCreate`+`TaskUpdate` auslösen; das echte Schema festhalten. Erst danach den Rest bauen. **Fallback** falls PostToolUse nicht feuert: `TodoWrite`-Pfad, sonst `derive_todo` reaktivieren (der dann die Quelle wird).

## Error Handling
- Hook: kein JWT → still skip; HTTP-Fehler/Timeout → stderr-Log, kein Raise, kein Tool-Bruch. Idempotenz schützt vor Doppel-Posts bei Hook-Re-Runs.
- Backend: `external_id`-Upsert in einer Transaktion; fehlender Task bei PATCH → 404 (vom Hook geschluckt).
- Lens: fehlt `intervention.todos` → leere Spalte, kein Crash.

## Verification
- **Hook-Unit (pytest):** TaskCreate-Payload → korrekter `POST /tasks`-Body (external_id, title, created_by='agent'); TaskUpdate(completed) → complete-Call; non-todo tool_name → no-op; fehlendes JWT → skip; HTTP-Fehler → kein Raise.
- **Backend-Unit (TDD):** `create_task` mit bestehendem `external_id` → Update (1 Zeile, kein Duplikat); `GET /stats/igio-lens` enthält `intervention.todos` (workspace-scoped, eigene Tasks).
- **Smoke (prod):** ein via act-as/JWT angelegter Task erscheint in `GET /stats/igio-lens` unter `intervention.todos` (neuer Check, red-green).
- **Pest (Laravel):** `getIgioLens()` reicht `todos` durch; `IgioLens` rendert die Intervention-Spalte mit einem Todo.
- **Manuell (User):** `/mayring/igio` eingeloggt — die Intervention-Spalte zeigt die in der Session angelegten Task-Listen.

## Out of Scope (YAGNI)
- `derive_todo`-aus-Prompts als zweite Quelle (bleibt dormant; spätere Option falls Auto-Todos gewünscht).
- Auto-Status-Progression über die Tool-Capture hinaus (kein „erledigt"-Inferenz aus Prompts).
- Manuelle Todo-Erstellung-UI (das bestehende `POST /tasks` reicht; eine Eingabe-UI ist späterer Komfort).
- Priorität/Fälligkeit der erfassten Todos.
- Die anderen 3 igio-Achsen umbauen.
