# Design: Saubere Goals + Todos aus Prompts, sichtbar im Tasks-Tracker

**Status:** Draft for review · **Datum:** 2026-05-22 · **Repos:** MayringCoder (Backend/MCP), app.linn.games (Frontend, PR)

## Context

Diagnose (Prod-belegt, Workspace `bene`): Die IGIO-Klassifikation *läuft* (Konversationen → outcome 23, issue 21, **goal 8**, intervention 6), aber das vom User erwartete Verhalten fehlt:
1. **Goals sind verrauscht** — von 161 `goal`-Chunks dominieren `ambient_snapshot` + `paper:*` (Paper-Sätze wie „This study aims to…"); nur 8 stammen aus echten Konversationen. Die *eigenen* Ziele ersaufen.
2. **Todos existieren nicht** — IGIO kennt nur issue/goal/intervention/outcome, kein actionable Todo. Aus Prompts werden nie Todos abgeleitet.
3. **Nichts ist sichtbar** — `goal`-Chunks erreicht man nur über den manuellen `/goal`-Skill bzw. `/memory/goals` (vom Frontend nie aufgerufen). Der `tasks`-Tracker ist separat → wirkt leer.

Bezug: Issue #141 (O-Ton User: „WAS ist aus der Goal/Intervention/Outcome-Sortierung im WikiV2 geworden?"). Diese Sortierung sollte sichtbar sein, wurde es nie.

**Outcome:** Aus den Prompts werden **saubere Todos** (actionable) und **Goals** (entrauscht) abgeleitet und in der **Tasks-Seite** sichtbar — Todos editierbar, Ziele read-only.

## Goals / Non-Goals

**Goals**
- Server-seitige, **LLM-gestützte** Ableitung eines Todos pro User-Prompt (actionable? + sauberer Titel), Dedup, als `tasks`-Zeile (status=open, `created_by='derived'`).
- Goal-Ansicht: `igio_axis='goal'` des Workspaces, **gefiltert auf eigene Quellen** (conversation/note/session/task), ambient/paper raus.
- Tasks-Seite zeigt beides: **Todos** (auto+manuell) + **Ziele** (read-only).

**Non-Goals (v1, YAGNI)**
- Kein neuer IGIO-Axis „todo" — Todos sind `tasks`-Zeilen (vorhandene Tabelle).
- Goal-Classifier nicht umbauen — nur die *User-Ansicht* filtert (Papers behalten ihre Achse für Recherche-Kontext).
- Keine Auto-Status-Progression (open→done) der derivierten Todos; nur Anlegen. Abhaken bleibt manuell.
- Kein LLM im 9s-Hook-Budget — die Ableitung läuft serverseitig async im micro-batch-Pfad.

## Design

### Teil A — Goal-Ansicht (entrauscht, read-only)
Neuer Store-Helper `list_workspace_goals(conn, workspace_id, *, limit=100)` in `src/memory/tasks.py`: `chunks WHERE igio_axis='goal' AND is_active=1 AND workspace_id=? AND source_type IN ('conversation_summary','note','session_knowledge','session','task','user_context','knowledge')` (NICHT ambient/paper/repo_file), neueste zuerst, als read-only task-förmige dicts (`source='goal'`, `read_only=True`). Neuer Endpoint `GET /tasks/goals` (workspace-scoped) liefert sie. (Bewusst getrennt von `GET /tasks`, das nach dem Revert nur echte Tasks liefert.)

### Teil B — Todo-Ableitung aus Prompts (LLM, serverseitig)
- Neue Funktion `derive_todo(prompt, conn, ollama_url, workspace_id, *, model)` in neuem Modul `src/memory/todo_derivation.py`:
  1. LLM (JSON-Mode, mistral/Pi via ModelRouter) auf den Prompt: `{"actionable": bool, "title": "<imperativ, <=120 Zeichen>"}`. Prompt-Vorlage: „Ist das eine konkrete Arbeits-Aufgabe (Todo), die der User erledigt haben will? Wenn ja, prägnanter imperativer Titel."
  2. Wenn `actionable=false` → None (kein Todo).
  3. **Dedup**: Embedding des Prompts; gegen offene Todos des Workspaces (deren gespeichertes Prompt-Embedding) cosine≥0.85 → kein neues Todo (skip). Speichert das Prompt-Embedding am Task (neue Spalte `tasks.derive_embedding TEXT` o. wiederverwendbar) — analog zum gefixten RQ-Dedup (prompt-vs-prompt!).
  4. Sonst: `tasks.create_task(workspace_id, title, created_by='derived', tags='derived')`, status=open.
- **Einhängen** in `POST /conversation/micro-batch` (`src/api/routes/memory.py:369`): nach dem Chunk-Ingest, in einem **daemon-Thread** (nicht blockierend, wie `derive_research_question_background`), `derive_todo(user_prompt, …)` aufrufen. Nur für echte User-Prompts (nicht system/smoke — Workspace ≠ 'system').

### Teil C — Frontend (app.linn.games, PR)
Tasks-Seite (`TaskBoard`): zwei Sektionen — **„Todos"** (bestehende Tabelle: auto-derivierte + manuelle, editierbar; auto-Todos mit Badge `source=derived`) und **„Ziele"** (neue read-only Tabelle aus `GET /tasks/goals`, Quelle-Badge). `MayringTasksClient.goals()` → `GET /tasks/goals`.

## Architecture / Files
- **Neu:** `src/memory/todo_derivation.py` (`derive_todo` + LLM + dedup), `tests/test_todo_derivation.py`; Endpoint `GET /tasks/goals` + `list_workspace_goals` in tasks.py; ggf. `tasks.derive_embedding`-Spalte (Migration in store.py).
- **Geändert:** `src/api/routes/memory.py` (micro-batch → background derive_todo), `src/api/routes/tasks.py` (+`/tasks/goals`); Laravel `MayringTasksClient` + `TaskBoard`/Blade.
- **Wiederverwenden:** `tasks.create_task`, das RQ-Dedup-Embedding-Muster (prompt-vs-prompt), `ModelRouter`, der micro-batch-Daemon-Thread-Pattern.

## Verification
- **Unit (pytest, TDD):** `derive_todo`: actionable=true → Task angelegt; actionable=false → None; near-duplicate Prompt → skip (kein 2. Todo). `list_workspace_goals`: liefert nur eigene Quellen, NICHT ambient/paper. API: `GET /tasks/goals` workspace-scoped.
- **Integration:** micro-batch mit einem actionable Prompt (workspace=bene) → ein offenes 'derived' Todo erscheint in `GET /tasks`; ein non-actionable Prompt → keins; zweiter ähnlicher Prompt → kein Duplikat.
- **Prod-Verify (read-only via SSH):** nach Deploy ein paar Turns, dann `GET /tasks` (bene) zeigt derived Todos; `GET /tasks/goals` zeigt nur Konversations-/Note-Goals (kein paper/ambient).
- **Pest (Laravel):** Client `goals()` + TaskBoard rendert beide Sektionen.

## Out of scope / Future
- Auto-Abschluss derivierter Todos (Erkennung „erledigt" aus späteren Prompts).
- Todo-Priorität/Fälligkeit-Ableitung.
- IGIO-„goal"-Präzision generell verbessern (separates Thema; hier nur View-Filter).
