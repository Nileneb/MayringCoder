# Device-Worker-Pool — Broker/Worker-Modell für verteilte Job-Ausführung

**Stand:** 2026-05-29 · **Status:** Design (Part 1 gebaut, Part 2+3 Spec)
**Verbunden:** [[project-central-llama-queue-2-plan]], [[project-issue-274-device-registry]],
[[project-repo-watching-live]], `docs/superpowers/specs/2026-05-25-repo-watching-design.md`

## Problem (User-Vision 2026-05-29)

> „An EINER zentralen Stelle in mcp.linn.games die Jobs sammeln + über die Cloud verteilen."
> Basic-Worker `three.linn.games` + `ollama.com` (50/50 LLM-Split), Embeddings immer
> three.linn.games. Mittelfristig: wer das Game installiert, stellt sein Gerät als Cloud-Worker
> bereit. Separat: Pi-Agenten als Docker-Worker (Dev-Env) für A2A-Entwicklung.

## Broker/Worker-Modell

**mcp.linn.games = Broker**, nicht zwingend Executor. `POST /pi/run` ist der eine Funnel; zwei Lanes:

| Lane | Mechanik | Caller-Bedarf | Worker |
|------|----------|---------------|--------|
| **Sync** (heute, Part 1) | in-process `PiQueue` (`server.py::_handler`) | Ergebnis **inline** (Ingest categorize→speichern, Hooks) | keiner — API-Prozess rechnet selbst |
| **Async** (Part 2/3) | cloud-pull-queue `pi_jobs` (scope=cloud) | Fire-and-forget / verteilbar | externe Worker pullen |

Der LLM-Cloud-Split (50/50 three↔ollama.com) sitzt unabhängig davon in
`mayring_core.ollama_client` (`OLLAMA_CLOUD_PRIMARY_RATIO`, prod=0.5) und gilt für **jeden**
generate-Call eines Workers. Embeddings sind dort bewusst ausgenommen → immer three.linn.games.

### Worker-Topologie

- **Baseline (always-on):** Prod-Container `mayring-pi` (app.linn.games-Host). Garantiert
  Abarbeitung der Async-Lane auch ohne User/Game-Worker. **Konsistenz-Fix (offen):** nur noch die
  zentrale cloud-pull-queue ziehen, lokalen `pi_jobs`-DB-Poll (`pi_worker._loop`) abschalten →
  eine Source-of-Truth = mcp.linn.games. (Antwort auf „prod-Worker vs. User-Pi-Agent?": Baseline
  ist der Container; auf den User-Pi-Agenten kann die Baseline NICHT bauen — er ist per Default
  nicht installiert.)
- **Additiv (opt-in):** User-Pi-Agent-Dev-Container + Game-Geräte. Reine Kapazität, kein Ersatz.

Bestehende Infra (verifiziert, #274, c04365c): Device-Registry
(`vendor/mayring-core/mayring_core/memory/devices.py`), `POST /pi_task_claim_cloud` +
`/pi_task_complete_cloud` (`src/api/routes/devices.py`), registry-authoritative caps
(`effective_capabilities` — write nur per Registrierung, nicht self-report), nginx-Allowlist deckt
`devices|pi_task|pi-jobs` ab.

## Part 2 — Game-Geräte als Worker

**Identifizierte Lücken (aus Exploration):**

1. **JWT-Workspace-Binding (Blocker):** Game-Player-JWT hat `aud=battlefield`, KEIN
   `workspace_id`-Claim (`GameAuthController::issueForGamePlayer`). Die Claim-Endpoints scopen aber
   per `get_workspace`. → Lösung: `user.id → activeWorkspaceId`-Lookup in den Claim-Endpoints ODER
   `workspace_id`/`scope:["worker:claim"]` in den Game-Player-Token aufnehmen.
2. **Worker-Lifecycle:** crashed worker → Job bleibt `running`. → Timeout via `last_seen`-Heartbeat
   + Requeue (lease-Pattern): `claimed_at` + TTL, ein Reaper setzt abgelaufene Claims auf `queued`.
3. **Capability-Modell:** Game-Installation = `read-only`-Worker (nur generate/categorize, kein
   write/exec). Default-Caps bei Game-Registrierung leer-write.
4. **Client-SDK:** Unity-Client braucht Pull-Loop (`claim → run gegen lokales/cloud-Ollama →
   complete`). Capability `local-gpu` self-reported, write registry-gated.

## Part 3 — Pi-Agent Docker-Worker / A2A (separater Track)

- Container `mayring-pi` = Referenz-Worker-Blaupause (keep & repurpose; Q2 überschreibt die alte
  „redundant→raus"-Notiz aus [[project-central-llama-queue-2-plan]]).
- **Dev-Env-Deploy:** Pi-Agent-Container mit `PI_ALLOW_WRITE`/`PI_ALLOW_EXEC` + `X-Device-Id`-
  Registrierung (write-cap → in Registry eintragen), pullt von mcp.linn.games (Async-Lane).
- **A2A (nicht vorhanden, künftig):** Agent-Cards (Capability-Advertisement pro Worker) +
  gerichtetes Message-Passing zwischen Workern. Aktuell ist die Topologie Star (Hub=Broker). A2A =
  eigener Track, hier nur skizziert, nicht gebaut.

## Observability-Voraussetzung (heute gefixt)

core + pi-agent waren NICHT unter Repo-Watching (nur app.linn.games hatte den Caller). pi-agent
caller direkt gepusht; core via PR. **Aktivierung:** `MCP_SERVICE_TOKEN`-Secret in beiden Repos
(gemappt als `MAYRING_TOKEN` an die reusable `repo-watch.yml`) — sonst stiller No-op.

## Verifikation Part 1 (gebaut)

- `tests/test_generate_queue_routing.py` (5) + `test_provider_seam.py` (3) grün; 41 Nachbar-Tests grün.
- Prod: `GET /pi-jobs/stats` background-lane wächst beim populate; categorize bleibt deterministisch;
  Ingest-Embeddings treffen three.linn.games, nie ollama.com.
