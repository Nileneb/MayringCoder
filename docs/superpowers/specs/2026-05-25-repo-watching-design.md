# Design: Repo-Watching — Auto-Ingest + CI/Security Events (Subsystem C+D)

**Status:** Approved (2026-05-25, User „ok") · **Repos:** MayringCoder (endpoint + event→chunk), a reusable GitHub Action (central + per-repo caller)
**Verbunden mit:** Audit G18 (Auto-Ingest PARTIAL), [[project_device_hook_events_pipeline]], [[project_v2_deferred_infra_ideas]] (Pi-Agent/CI-Security-Hooks pro Repo), [[project_igio_intervention_todos]] (Lens-Konsument), Subsystem C+D der Org/Public-Memory-Decomposition.

## Problem (User-Aussage)

> „wenn ich sämtliche Events einer Repo überwachen will, ist es ja vernünftig, immer die neueste Version zu ingesten. Irgendwo in der Tabellen-Struktur speichern wir Repos doch auch als Sources?"

**Befund:** Ein Repo = eine `projects`-Zeile (`source_type='github'`, `source_ref=<repo-url>`, `workspace_id`); seine Dateien = `sources` (`source_type='repo_file'`). 8 Projekte unter `bene`. Auto-Ingest feuert heute NUR für MayringCoder (`post-deploy-ingest.yml`) — die anderen 7 Repos re-ingesten nie auf Push (Audit G18 „PARTIAL"). CI/Security-Events fließen nirgends ins Memory.

## Entscheidungen (User, 2026-05-25)

1. **C+D = ein Subsystem „Repo-Watching"** (nicht zwei): ein „dieses Repo hat sich bewegt"-Signal löst beides aus.
2. **Trigger = reusable GitHub Action pro Repo** (explizit, repo-owned, CI+Security nativ; opt-in = Repo bindet den Caller ein).
3. **Event-Handling = Hybrid:** Push → Re-Ingest; CI/Security → `hook_events`-Log **und** ein leichter durchsuchbarer Memory-Chunk.

## Architektur

```
Repo X (.github/workflows/mayring-watch.yml, 3-Zeilen-Caller)
   │  on: push / workflow_run / (security advisory|code-scanning)
   ▼
reusable Action (Nileneb/mayring-claude-plugin/.github/workflows/repo-watch.yml)
   │  POST {event_type, repo, sha, conclusion, workflow, severity, summary, url}
   │  Authorization: Bearer <MAYRING_TOKEN repo-secret>   (continue-on-error)
   ▼
POST /repo-events  (MayringCoder, service-token auth)
   │  resolve repo-url → projects row → workspace_id (match-or-create)
   ├─ push          → trigger /populate {repo} (debounce: skip if a populate job runs)
   ├─ workflow_run  → hook_events row + repo_event chunk (igio: fail→issue / success→outcome)
   └─ security      → hook_events row + repo_event chunk (igio: issue)
   ▼
chunks (source_type='repo_event') → recall (hook) + IGIO-Lens
hook_events (hook_type='repo_*', payload JSON) → timeline
```

### Komponente 1 — Reusable GitHub Action
- **Central** workflow in `mayring-claude-plugin/.github/workflows/repo-watch.yml` (a `workflow_call` reusable workflow). Inputs: the event context (GitHub provides `github.event_name`, `event.workflow_run.conclusion`, `github.sha`, etc.). Secret: `MAYRING_TOKEN`.
- Builds a JSON body `{event_type, repo, sha, ref, conclusion, workflow, severity, summary, url}` from the GitHub context and `curl -m 10 --fail-with-body` POSTs it to `${MAYRING_API_URL:-https://mcp.linn.games}/repo-events`. **`continue-on-error: true`** on the step — a MayringCoder hiccup must NEVER fail the watched repo's CI/security run.
- **Per-repo caller** (`.github/workflows/mayring-watch.yml`, ~10 lines) in each watched repo:
  ```yaml
  on:
    push: { branches: ["**"] }
    workflow_run: { workflows: ["*"], types: [completed] }
  jobs:
    watch:
      uses: Nileneb/mayring-claude-plugin/.github/workflows/repo-watch.yml@main
      secrets: { MAYRING_TOKEN: ${{ secrets.MAYRING_TOKEN }} }
  ```
  (Security events — `security_advisory` / code-scanning alerts — added where the repo has them; push + workflow_run are the baseline.)
- **Loop guard (critical):** the watch workflow listens to `workflow_run` of other workflows — it MUST exclude its OWN runs, else it triggers itself infinitely. The reusable workflow's first step exits early when `github.event.workflow_run.name == 'mayring-watch'` (or the caller filters `workflow_run.workflows` to exclude the watcher). Verified in the plan with a unit-level assertion on the guard.

### Komponente 2 — `POST /repo-events` (MayringCoder)
- New route `src/api/routes/repo_events.py`, service-token auth (`get_token_info`, requires privileged scope — reuses `_is_privileged`). Body: `RepoEventRequest {event_type: str, repo: str, sha: str|None, ref: str|None, conclusion: str|None, workflow: str|None, severity: str|None, summary: str|None, url: str|None}`.
- **Workspace resolution:** look up `projects WHERE source_type='github' AND source_ref=<repo>` → `workspace_id`. If none → match-or-create a project (reuse the `src/api/routes/projects.py` match-or-create pattern) under a resolvable workspace; if still unresolvable → land under `system` (never reject — return 200 so the Action never breaks).
- **Dispatch by `event_type`:**
  - `push` → enqueue `/populate` for `repo` via the existing populate job path (`src/api/routes/jobs.py`), workspace-scoped. **Debounce:** skip if a populate job for the same repo is already `started`/running (check `_JOBS` / job_queue).
  - `workflow_run` → `_record_event("repo_ci", ...)` + `_repo_event_chunk(...)`.
  - `security` → `_record_event("repo_security", ...)` + `_repo_event_chunk(...)`.
- Idempotent: dedup on `(repo, event_type, sha, workflow)` — a re-delivered event must not double-insert (the Action/CI can retry).

### Komponente 3 — Event storage (no schema migration)
- **`hook_events` (reused as-is — NO migration):** `hook_type ∈ {'repo_push','repo_ci','repo_security'}`, `payload` = the full event JSON, `workspace_id`, `fired_at`. The existing flexible `payload TEXT` holds conclusion/severity/url/sha — **deliberately no new columns** (the v14-migration incident showed column+index migrations are risky; reuse the JSON payload instead).
- **`repo_event` chunk (the lightweight searchable entry):** a `memory.put`-style ingest with `source_type='repo_event'`, `repo=<repo>`, a one-line `content` (`"CI <workflow> <conclusion> @<sha>"` / `"Security <severity>: <summary>"`), `visibility` inherited from the project's workspace. **A fast IGIO hint (NO LLM in the event path)** sets `igio_axis`: ci `failure`→`issue`, ci `success`→`outcome`, security→`issue`. So a failed CI / security alert surfaces in recall AND the IGIO-Lens (connecting to Subsystem B's lens). `categorize=False` to keep the endpoint fast.

## Outcome guarantee (gegen „gebaut aber nie genutzt")
The plan MUST wire at least ONE real repo (recommend **app.linn.games**) with the caller workflow + `MAYRING_TOKEN` secret, and verify a real push/CI event flows end-to-end: `/populate` fires, a `hook_events` row + `repo_event` chunk appear, and the chunk shows in the IGIO-Lens. Not just the machinery.

## Error handling
- Endpoint: best-effort — the `hook_events` write and the chunk write are independent (a chunk failure still logs the event). Unknown repo → 200 + match-or-create or `system` (never reject). Idempotent on re-delivery.
- Reusable Action: `continue-on-error: true` + `curl -m 10` — a MayringCoder outage never breaks a watched repo's CI/security.
- Push debounce prevents a populate-storm from rapid pushes.

## Verification
- **Unit (pytest, TDD):** `/repo-events` push → populate-job enqueued (mocked); `workflow_run` failure → `hook_events` row (`hook_type='repo_ci'`) + `repo_event` chunk with `igio_axis='issue'`; `workflow_run` success → `igio_axis='outcome'`; `security` → `hook_type='repo_security'` + chunk `igio_axis='issue'`; repo→workspace resolution via projects; unknown repo → match-or-create, 200; re-delivered event → no duplicate.
- **Smoke (prod):** POST a synthetic `workflow_run` event to `/repo-events` → assert a `hook_events` row + a `repo_event` chunk surfaces (igio-lens / search), workspace-scoped.
- **Live outcome:** the wired repo (app.linn.games) — a real push triggers a populate job; a real CI completion writes an event + chunk; visible in the IGIO-Lens.

## Out of Scope (YAGNI)
- GitHub webhooks / polling (rejected in favour of the reusable Action).
- Per-repo `projects.watch_enabled` flag (opt-in = adding the caller workflow; the endpoint match-or-creates).
- LLM classification of events in the hot path (fast igio-hint only; a background re-classify can refine later).
- Backfilling historical CI/security runs.
- Schema migration for hook_events (reuse `payload` JSON).
