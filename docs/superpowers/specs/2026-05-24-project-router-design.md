# Project Router — Slice 1 Design

**Datum:** 2026-05-24
**Status:** Approved (Design) — wartet auf Spec-Review
**Repos:** MayringCoder (Server-Endpoint) + mayring-claude-plugin (Hook)
**Verwandt:** Pipeline v2.0 (`~/.claude/plans/hashed-sparking-crescent.md`),
session_ctx.json (Phase 2, Commit 81509c0), Workspace-UUID-SoT.

## Problem

Memory-Retrieval läuft heute workspace-weit. Die Projekt-Dimension existiert im
Schema (`chunks.project_id`, `projects`-Tabelle) und `retrieval.py:136-143`
filtert bereits `AND c.project_id = ?` — aber **niemand setzt `active_project_id`**.
Folge: kein projekt-scharfer Kontext, keine projekt-bezogene Observability.

Ziel: ab dem ersten substanziellen Prompt einer Session deterministisch ein
`active_project_id` (oder `null`) anheften, ohne Projekt-Proliferation und ohne
schwere Seiteneffekte.

## Bestätigte Entscheidungen (User)

1. **Research-Modell:** Projekt = Container (`source_type ∈ {github, papers,
   research}`); Research-Question = versioniertes Goal *innerhalb* des Projekts.
   Wiederverwendet das existierende RQ-Subsystem (`parent_research_question_id`,
   `research_question_chunk_links`, `embedding_id`). KEINE neue RQ-Versionierung.
2. **No-Match-Verhalten:** Kein Auto-Create aus vagem Prompt. Unsicher → `null`
   → workspace-weite Suche (heutiges Verhalten). Projekte entstehen nur aus
   harten, eindeutigen Signalen (Repo-Remote) oder explizit.

## Scope

**In Slice 1:**
- Deterministisches Coding-Routing über das **cwd-git-remote** (stärkstes Signal,
  funktioniert ohne Repo-URL im Prompt, weil der Hook IM Repo läuft).
- Server-Endpoint `POST /projects/route`.
- Hook ruft Router **einmal** pro Session, cached in `session_ctx.json`, reicht
  `project` an die `/memory/search`-Calls durch.
- Observability-Zeile im injizierten Memory-Block.

**Explizit zurückgestellt:**
- Semantischer Projekt-Match ohne Repo-Signal (braucht Projekt-Embeddings) → Slice 1.5
- `derive_task` / `task_context = {project}:{intent}:{artifact}` → Slice 1.5
- CI/Security/Issue-Auto-Aktivierung (Coding) → Slice 2
- Research `scope_lock` + p1–p8-Aktivierung + `research_questions.project_id`-Link → Slice 3

## Architektur

Ein dünner Server-Endpoint (Logik + Daten co-lokalisiert, gerätunabhängig, testbar)
+ dünner Hook (kennt nur die lokale Umgebung).

### Komponente A — `POST /projects/route` (MayringCoder, `src/api/routes/projects.py`)

**Input** (JSON): `{ "cwd_remote": str|null, "prompt": str, "signals": {…}|null }`
- `cwd_remote`: normalisierte Git-Remote-URL des Arbeitsverzeichnisses (Hook liefert sie).

**Logik (deterministisch, retrieval-frei in Slice 1):**
1. Wenn `cwd_remote` gesetzt: normalisieren (ssh↔https, `.git`-Suffix, Owner/Name).
   `SELECT id FROM projects WHERE workspace_id=? AND source_type='github' AND
   source_ref` matcht Owner/Name. Treffer → `project_id`. Kein Treffer →
   **CREATE** `projects(source_type='github', source_ref=<remote>, name=<repo>)`
   (Repo-Remote ist ein harter, eindeutiger Trigger → Create erlaubt).
   → `mode='coding'`, `confidence=0.9`, `reason='cwd-remote'`.
2. Sonst (kein hartes Signal): `project_id=null`, `mode` aus Prompt-Regex
   (coding/research/mixed/unknown), `confidence=0.0`, `reason='no-hard-signal'`.

**Output:** `{ "project_id": str|null, "mode": str, "confidence": float, "reason": str }`

**Auth:** Standard-Token-Dependency (`get_workspace`); Projekte sind
workspace-gescopt (single-workspace UUID).

### Komponente B — Hook-Integration (mayring-claude-plugin)

- `_session_ctx.py`: neue `route_project(token, cwd_remote, prompt) -> dict`
  (POST /projects/route, fail-soft → `{project_id: null}`). Ergebnis wird in
  `session_ctx.json` unter `active_project` gecached (gilt für die Session).
- `_git_remote()`-Helper: `git -C <cwd> remote get-url origin` (subprocess,
  timeout 2s, fail-soft → None).
- `memory_inject.main()`: beim ersten substanziellen Prompt `active_project` aus
  `session_ctx.json` lesen; fehlt es → `route_project(...)` aufrufen + cachen.
  `project_id` (als Feld `project`) an die 3 `_search`-Calls durchreichen
  (`_search` bekommt Parameter `project_id`, schreibt `body["project"]`).
- Observability: erste Zeile des injizierten Blocks
  `📁 Projekt: <name> (<mode>, conf=<c> · <reason>)` bzw.
  `📁 Projekt: — (workspace-weit, <mode>)` bei null.

## Datenfluss

```
SessionStart  → (nichts entscheiden; kein Prompt vorhanden)
1. UserPrompt → memory_inject:
     cwd_remote = _git_remote()
     active = session_ctx.active_project  ?? route_project(token, cwd_remote, prompt)
     session_ctx.active_project = active        (cache)
     _multi_lens_search(prompt, token, project_id=active.project_id)
     inject: "📁 Projekt: …" + Memory-Kontext
2..n UserPrompt → active aus Cache (kein Re-Route, kein Latenz-Aufschlag)
```

## Schema-Änderungen

- Slice 1 braucht **keine** neuen Spalten (kein draft/status, kein
  Projekt-Embedding). Nur ein Index für den Match:
  `CREATE INDEX IF NOT EXISTS idx_projects_source ON projects(workspace_id, source_type, source_ref);`
- `session_ctx.json`: zusätzlicher Block `active_project: {project_id, mode,
  confidence, reason, name}` (Plugin-seitig, kein DB-Schema).

## Error Handling (fail-soft, nie blockieren)

- `_git_remote()` Fehler/kein Git → None → Router bekommt `cwd_remote=null` → null-Route.
- `/projects/route` 5xx/timeout → Hook behandelt wie heutige Search-5xx
  (Retry/silent-skip im Deploy-Window); `active_project=null` → workspace-weite Suche.
- Router-Create schlägt fehl → loggt laut (kein silent), gibt `project_id=null` zurück.

## Verifikation (end-to-end)

- Coding-Prompt im Repo-cwd (z.B. MayringCoder) → `/projects/route` liefert das
  passende `project_id`, `reason='cwd-remote'`; projekt-gescopte Suche gibt nur
  Chunks dieses Projekts (verglichen mit workspace-weit).
- Prompt ohne Git-cwd → `project_id=null`, workspace-weite Suche wie heute.
- Zweiter Prompt derselben Session → kein zweiter Router-Call (Cache-Hit).
- Smoke: neuer Check `projects_route_cwd_remote` (POST mit bekanntem Remote →
  200 + project_id gesetzt; POST ohne → 200 + project_id null).

## Risiken

- **Remote-Normalisierung** (ssh↔https, Mono-Repos, Forks): bei Fehl-Normalisierung
  Fehl-Match. Mitigation: konservativ auf Owner/Name matchen, im Zweifel null.
- **Mehrere Projekte pro Repo** (z.B. App + Submodule): Slice 1 matcht das erste;
  Disambiguierung erst mit semantischem Match (Slice 1.5).
