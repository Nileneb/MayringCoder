# Project Router — Slice 1 Design

**Datum:** 2026-05-24
**Status:** Approved (Design + Spec-Review) — bereit für Implementierungsplan
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
   → workspace-weite Suche (heutiges Verhalten).
3. **Repo-Remote ist der EINZIGE Auto-Create-Trigger.** Alles andere matcht nur
   *existierende* Projekte oder fällt auf `null`.
4. **Slice 1 enthält den semantischen Match** (vormals Slice 1.5) + `derive_task`.
5. **Keine neuen SQLite-Spalten in Slice 1** — siehe „Schema-Entscheidung".

## Scope

**In Slice 1:**
- Deterministisches Coding-Routing über das **cwd-git-remote** (stärkstes Signal,
  funktioniert ohne Repo-URL im Prompt, weil der Hook IM Repo läuft).
- **Semantischer Projekt-Match**: ohne hartes Signal → Prompt-Embedding gegen
  Projekt-Embeddings (Chroma-Collection `projects`), Top-1 mit Margin → match
  *existierendes* Projekt (KEIN Create; unsicher → null).
- **`derive_task` + `task_context`**: mit Projekt im Kontext ist
  `task_context = {project}:{intent}:{artifact}` *nicht* mehr redundant zur Query
  → echter Mehrwert (Query-Expansion + Advisor, `retrieval.py:755`). Regex-first
  (Imperative DE/EN), Projekt/Goal-Titel als Kontext, phi3-Fallback nur wenn <3
  sinnvolle Tokens.
- Server-Endpoint `POST /projects/route`.
- Hook ruft Router **einmal** pro Session, cached in `session_ctx.json`, reicht
  `project` + `task_context` an die `/memory/search`-Calls durch.
- Observability-Zeile im injizierten Memory-Block.

**Explizit zurückgestellt:**
- CI/Security/Issue-Auto-Aktivierung (Coding) → Slice 2
- Research `scope_lock` + p1–p8-Aktivierung + `research_questions.project_id`-Link → Slice 3
- Goal→Todo→Task-*Management* (anlegen/abhaken) → Slice 3 (Slice 1 nutzt Goals nur
  lesend als Kontext für `derive_task`)

## Architektur

Ein dünner Server-Endpoint (Logik + Daten + Embeddings co-lokalisiert,
gerätunabhängig, testbar) + dünner Hook (kennt nur die lokale Umgebung).

### Komponente A — `POST /projects/route` (MayringCoder, `src/api/routes/projects.py`)

**Input** (JSON): `{ "cwd_remote": str|null, "prompt": str }`
- `cwd_remote`: normalisierte Git-Remote-URL des Arbeitsverzeichnisses (Hook liefert sie).

**Logik (deterministisch zuerst, semantisch als Fallback):**
1. **Hartes Signal — `cwd_remote`:** normalisieren (ssh↔https, `.git`-Suffix,
   Owner/Name). `SELECT id FROM projects WHERE workspace_id=? AND
   source_type='github'` + Owner/Name-Match. Treffer → `project_id`. Kein Treffer
   → **CREATE** `projects(source_type='github', source_ref=<remote>, name=<repo>)`
   + Projekt-Embedding upserten (s.u.). → `mode='coding'`, `confidence=0.9`,
   `reason='cwd-remote'`.
2. **Semantischer Match (nur ohne hartes Signal):** Prompt embedden → Chroma-
   Collection `projects` queren → Top-1. Wenn `score ≥ 0.55` UND
   `score_top1 − score_top2 ≥ 0.08` (Margin) → match `project_id`, `confidence=score`,
   `reason='semantic'`. KEIN Create.
3. **Sonst:** `project_id=null`, `mode` aus Prompt-Regex (coding/research/mixed/
   unknown), `confidence=0.0`, `reason='no-match'`.

**Output:** `{ "project_id": str|null, "name": str|null, "mode": str,
"confidence": float, "reason": str }`

**Auth:** Standard-Token-Dependency (`get_workspace`); Projekte workspace-gescopt
(single-workspace UUID). Create setzt `workspace_id` = resolved workspace.

### Komponente A2 — Projekt-Embeddings (Chroma-Collection `projects`)

- Embedding-Text pro Projekt: `"<name> <source_ref> <source_type>"`.
- **embedding_id deterministisch = `proj:<project_id>`** → KEINE neue SQLite-Spalte.
- Sync: bei Projekt-Create im Router → sofort upserten. Einmaliger Backfill aller
  bestehenden Projekte via `tools/embed_projects.py` (Muster wie
  `import_codebooks_to_db.py`, Ollama `nomic-embed-text` über `three.linn.games`).

### Komponente B — Hook-Integration (mayring-claude-plugin)

- `_session_ctx.py`: neue `route_project(token, cwd_remote, prompt) -> dict`
  (POST /projects/route, fail-soft → `{project_id: null}`). Ergebnis → `session_ctx.json`
  unter `active_project` (gilt für die Session).
- `_git_remote()`-Helper: `git -C <cwd> remote get-url origin` (subprocess,
  timeout 2s, fail-soft → None).
- `derive_task(prompt, project_name, goal_title) -> str` (regex-first, phi3-Fallback).
- `memory_inject.main()`: beim ersten substanziellen Prompt `active_project` aus
  `session_ctx.json` lesen; fehlt → `route_project(...)` + cachen. `project_id`
  (Body-Feld `project`) und `task_context` an die 3 `_search`-Calls durchreichen
  (`_search` bekommt Parameter `project_id`, `task_context`).
- Observability: erste Zeile des injizierten Blocks
  `📁 Projekt: <name> (<mode>, conf=<c> · <reason>) · task=<task>` bzw.
  `📁 Projekt: — (workspace-weit, <mode>)` bei null.

## Datenfluss

```
SessionStart  → (nichts entscheiden; kein Prompt vorhanden)
1. UserPrompt → memory_inject:
     cwd_remote = _git_remote()
     active = session_ctx.active_project
              ?? route_project(token, cwd_remote, prompt)   # hart → semantisch → null
     task   = derive_task(prompt, active.name, best_goal)
     session_ctx.active_project = active                    # cache (Session)
     _multi_lens_search(prompt, token, project_id=active.project_id, task_context=task)
     inject: "📁 Projekt: …" + Memory-Kontext
2..n UserPrompt → active aus Cache (kein Re-Route, kein Latenz-Aufschlag)
```

## Schema-Entscheidung (dokumentiert — Punkt 3 des Spec-Reviews)

Slice 1 fügt **keine** SQLite-Spalten hinzu. Begründung + was stattdessen:
- **Kein `status`/`draft`:** No-Auto-Create-aus-vagem-Prompt (Entscheidung 2) macht
  draft-States überflüssig — es gibt keine provisorischen Projekte zu tracken.
  (Falls Slice 3 doch draft-Research-Projekte braucht → dann Spalte, nicht jetzt.)
- **Kein `embedding_id` auf `projects`:** deterministische Konvention
  `proj:<project_id>` als Chroma-Doc-ID (gleiches Muster wie codebook_categories) →
  Spalte unnötig.
- **Einziges DDL:** ein Index für den Remote-Match:
  `CREATE INDEX IF NOT EXISTS idx_projects_source ON projects(workspace_id, source_type, source_ref);`
  (idempotent, additiv, `_migrate_schema`).
- **Neue Chroma-Collection** `projects` (kein SQLite-Schema).
- `session_ctx.json`: zusätzlicher Block `active_project: {project_id, name, mode,
  confidence, reason}` (Plugin-seitig, kein DB-Schema).

## Error Handling (fail-soft, nie blockieren)

- `_git_remote()` Fehler/kein Git → None → Router → semantisch/null.
- `/projects/route` 5xx/timeout → Hook wie heutige Search-5xx (Retry/silent-skip
  im Deploy-Window); `active_project=null` → workspace-weite Suche.
- Router-Create / Embedding-Upsert schlägt fehl → loggt laut (kein silent),
  gibt `project_id=null` zurück (Routing scheitert sicher, blockiert nie).
- Embedding-Query gegen leere/fehlende Collection → null-Route (kein Crash).

## Verifikation (end-to-end)

- Coding-Prompt im Repo-cwd (MayringCoder) → `/projects/route` liefert passendes
  `project_id`, `reason='cwd-remote'`; projekt-gescopte Suche gibt nur Chunks
  dieses Projekts (vs. workspace-weit).
- Neuer cwd-Remote ohne Projekt → Create + Embedding upserted; zweiter Call matcht
  per cwd-remote (nicht nochmal create).
- Prompt ohne Git-cwd, der semantisch klar zu einem Projekt gehört → `reason='semantic'`,
  korrektes `project_id`; vager Prompt → `null` + workspace-weit.
- `task_context` erscheint im injizierten Block + im Search-Body (Advisor nutzt ihn).
- Zweiter Prompt derselben Session → kein zweiter Router-Call (Cache-Hit).
- Smoke: `projects_route_cwd_remote` (POST mit bekanntem Remote → 200 + project_id;
  ohne → 200 + project_id null) und `projects_semantic_match` (Prompt → erwartetes Projekt).

## Risiken

- **Remote-Normalisierung** (ssh↔https, Mono-Repos, Forks): Fehl-Normalisierung →
  Fehl-Match. Mitigation: konservativ auf Owner/Name matchen, im Zweifel null.
- **Semantischer Fehl-Match:** Margin-Schwelle (0.08) + Mindest-Score (0.55) gegen
  Rauschen; bei eng beieinanderliegenden Kandidaten → null (lieber workspace-weit
  als falsches Projekt).
- **Mehrere Projekte pro Repo** (App + Submodule): Slice 1 matcht das erste per
  Owner/Name; feinere Disambiguierung erst mit Pfad-Signalen (späterer Slice).
