# v2.0-Architektur — Konsolidierungs-Status (#270)

Tracking-Epic über #265, #266, #267, #268, #211 (+ #261). Dieses Dokument hält
fest, was in MayringCoder umgesetzt ist und was cross-repo / Cloud-Deploy bleibt.

## Done (in-repo, verifiziert)

| Teil | Issue | Status |
|---|---|---|
| claude-plugin ausgelagert → mayring-claude-plugin | #268 | ✅ gemerged (origin/master) |
| mayring-core Package | #267 | ✅ PR #271 (dev/v2-bigbang) |
| Pi-Agent web_fetch + plan (read-only-safe) | #211 | ✅ dev/v2-bigbang |
| Pi-Task Suchstring-Optimierung (über Distributor) | #261 | ✅ dev/v2-bigbang |
| Pi-Agent-Extraktion vorbereitet | #266 | ✅ Branch `pi-agent-split`; Push blockiert (User-Aktion) |
| **integrations/github + /notifications** (Phase 1 Kern) | #270 | ✅ dev/v2-bigbang (Classifier + Store + Router + 9 Tests) |

## Phase 1 — Cloud-Side Integrations-Module

- [x] **`integrations/github`** Webhook-Empfang + Event-Klassifikation
  (`src/api/integrations/github_events.py`) — löst ci_security_warner serverseitig ab.
- [x] **`integration_notifications`**-Tabelle + Store (`notifications_store.py`).
- [x] **`GET /notifications?since=`** + `POST /integrations/github/webhook` +
  `/notifications/ack` (`src/api/routes/integrations.py`, in server.py eingehängt).
- [ ] **Extern:** GitHub-Webhook auf den Repos konfigurieren (Settings → Webhooks
  → `https://mcp.linn.games/integrations/github/webhook`, Events: workflow_run,
  security alerts, push, pull_request).
- [ ] **Cross-repo:** UserPromptSubmit-Hook (mayring-claude-plugin) auf
  `GET /notifications?since=` umstellen (1 Call statt N Polls).
- [ ] **`integrations/deploy`**: _silent_skip_counter-Logik (liegt im Plugin-Repo)
  serverseitig via /health-Probe-Cron ablösen.

## Phase 2 — Workflow-Orchestrator

- [ ] **`POST /workflow/start`** — Mayring-Pipeline expliziter Endpoint:
  Pi-categorize → IGIO-Resolver (Embedding-Match gegen Goals) →
  Memory-Search scoped to goal_id → `{context_block, chunk_ids, goal_id, notifications}`.
  Bausteine existieren bereits (`pi_categorize`, IGIO-Classifier, `search`) —
  Arbeit ist die Verdrahtung als ein Endpoint.
- [ ] **`POST /workflow/feedback`** — Auto-Rating + Outcome (migriert stop_hook-Logik,
  die im Plugin-Repo liegt) → Outcome in IGIO-Achse.

## Phase 3 — Pi-Agent als Microservice

= #266. Extraktion vorbereitet (`pi-agent-split`), Push blockiert (User-Aktion),
Cutover-Plan in `docs/pi-agent-extraction-266.md`.

## Was bleibt cross-repo / extern (nicht in MayringCoder lösbar)

- GitHub-Webhook-Konfiguration (GitHub-Settings).
- Plugin-Hooks (`UserPromptSubmit`/`Stop`) → mayring-claude-plugin-Repo.
- Cloud-Deploy von mcp.linn.games + dem neuen mayring-pi-agent-Service.
- `mayring-core` als installierbare Dependency veröffentlichen (für den Service).

## Big-Bang-Cutover (wenn alle Teile bereit)

dev/v2-bigbang + PR #271 zusammen nach master mergen, dann production-Deploy.
Reihenfolge: core publish → pi-agent deploy → webhook config → plugin-hook-Umstellung
→ src/agents-Removal (Cutover). Bis dahin laufen alle In-Repo-Teile mit
in-process-Fallback weiter (nichts bricht vor dem Deploy).
