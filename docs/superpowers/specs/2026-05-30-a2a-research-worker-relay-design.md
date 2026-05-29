# Design: Research-Worker via A2A-Relay

**Datum:** 2026-05-30 · **Status:** Approved-pending-review
**Verbunden mit:** [[project_dev_environment_pi_agent]] (M2b A2A-Server), #274 Device-Registry,
[[project_central_llama_queue]], `spec:app.linn.games:llm-endpoints-implementation`, M2b-Commit
mayring-pi-agent 40395cc.

## Problem / Vision (User 2026-05-30)

> „Ein Agent auf meinem Laptop, der tiefergehende Recherchen macht, direkt in Langdock connectbar.
> Kaum Geld zahlen (nur Laptop-Strom), trotzdem die Tools — optimal für lange Aufträge."

Langdock ist Cloud-SaaS, der Laptop hängt hinter NAT. Es braucht einen Pfad von Langdock zum
Laptop, der ohne Port-Forwarding/Tunnel auskommt und lange Laufzeiten verträgt.

## Entscheidungen (brainstorming 2026-05-30)

- **Topologie:** Cloud-Queue-Relay, Laptop wählt nur ausgehend. Kein Tailscale, kein Tunnel.
- **Tools:** Cloud-Memory (read), Web-Suche + Fetch, Memory-write (ingest). KEINE Paper-Suche (YAGNI).
- **web_search-Backend:** SearXNG self-host (Docker auf u-server) — kostenlos, keine Keys, privat.
- **Worker-Modell:** `qwen3.5:9b` (lokal, RTX 3060/12GB).
- **Auth:** RS256-JWT/Bearer wie der bestehende MCP wiederverwenden.

## Architektur

```
Langdock ──A2A message/send──▶ mcp.linn.games/a2a ──enqueue(scope=cloud, cap=research)──▶ pi_jobs
   ▲                                                                                          │
   └──A2A get_task (poll) ──── status/result ◀── complete_cloud ◀── Laptop-Worker ───claim───┘
                              (run_task_with_memory + web_search + web_fetch + ingest, lokales Ollama qwen3.5:9b)
```

Der öffentliche A2A-Server lebt in der **Cloud** (mayring-api). `message/send` legt einen
cloud-scoped pi-job an und gibt sofort einen **async A2A-Task** (state WORKING) zurück. Der Laptop
zieht den Job über den vorhandenen cloud-pull-Worker, rechnet lokal, meldet zurück. Langdock pollt
`get_task` bis COMPLETED.

## Wiederverwendung (existiert bereits)

- `pi_worker` CLOUD-Modus: `claim_cloud_next(worker_id, capabilities)` + `complete_cloud`,
  persistentes `_worker_id()`, `_capabilities()` aus `PI_WORKER_CAPABILITIES`.
- `pi_jobs`: Spalten `scope` ('local'|'cloud') + `capability_required`; `_capability_match` =
  `required ⊆ worker-caps`. **Scoping-Garantie:** Research-Jobs mit `capability_required="research"`
  zieht NUR ein Worker der „research" advertised → Prod-Pi bleibt unberührt (löst die M2a-Sorge).
- Device-Registry (#274), `pi_task_*_cloud`-Endpoints (commit c04365c).
- Agent-Tools in `pi.py::_TOOLS`: `search_memory`, `web_fetch` (allow-list), `read_file`,
  `write_file`, `bash`, `plan`, `search_wiki`.
- M2b A2A-Server (`mayring_pi_agent/a2a_agent.py`): `build_agent_card`, `PiAgentExecutor`,
  `register_a2a` — Executor-Pattern + Card-Bau wiederverwendbar.

## Komponenten (neu zu bauen)

### 1. Cloud-A2A-Gateway (mayring-api, public via mcp.linn.games)
- **Was:** AgentCard unter well-known-Pfad, Skill `deep-research`. `message/send` → `create_cloud_job`
  (scope=cloud, capability_required="research", task=Recherche-Prompt) → A2A-Task (id = job_id,
  state WORKING). `get_task(id)` → pi-job-status → A2A-Status-Mapping (queued/running→WORKING,
  done→COMPLETED+result, error→FAILED).
- **Wie:** Neuer `RelayAgentExecutor` (statt M2b-`PiAgentExecutor`): `execute()` enqueuet + emittiert
  WORKING; ein A2A-`TaskStore`-Adapter liest pi-job-Status für `get_task`. AgentCard-Bau aus
  `a2a_agent.build_agent_card` wiederverwenden (Skill-Liste anpassen).
- **Interface:** A2A JSON-RPC (`message/send`, `tasks/get`) hinter JWT. Card-URL = öffentlich
  (mit `default_input_modes=["text/plain"]`).
- **Abhängigkeit:** `a2a-sdk>=1.1.0` (schon Dep in mayring-pi-agent; in mayring-api ergänzen),
  cloud-job-create + status-read (pi_jobs cloud-API).

### 2. Laptop-Research-Worker (Konfig + Run)
- **Was:** `mayring-pi-agent` im cloud-pull-Modus, `PI_WORKER_CAPABILITIES=research`,
  `OLLAMA_URL=localhost:11434`, Modell `qwen3.5:9b`, großzügiges `PI_NUM_PREDICT`.
- **Wie:** systemd-user-Service / `mayring-pi-agent`-CLI-Flag für cloud-Modus; persistenter
  `worker_id`; web_fetch-Allow-List für Worker geweitet (lokaler Worker darf breiter fetchen —
  `pi.py` hat dafür schon den Hook-Kommentar).
- **Interface:** zieht nur `cap=research`-Jobs; meldet via `complete_cloud`.
- **Abhängigkeit:** SearXNG-URL (Komponente 4), Cloud-Claim-Endpoint + JWT (`~/.config/mayring/hook.jwt`).

### 3. Zwei neue Agent-Tools (`pi.py::_TOOLS`)
- **`web_search(query)`** → SearXNG-JSON über `mcp.linn.games/searxng/search?format=json` (Bearer-JWT),
  gibt Top-N Titel+URL+Snippet zurück. Ergänzt das vorhandene `web_fetch` (search → fetch → read-Workflow).
- **`ingest(title, text)`** → Memory-write-back über die vorhandene cloud-ingest-API, damit
  Rechercheergebnisse persistent + durchsuchbar werden (Tool ruft `MAYRING_API_URL/ingest`,
  Bearer-JWT). Fail-soft mit Fehlertext (kein stilles Schlucken).
- **Interface:** beide als `_execute_web_search`/`_execute_ingest`, registriert in `_TOOLS` + im
  Agent-Loop-Dispatch (`pi.py` ~Z740).

### 4. SearXNG (Docker, u-server)
- **Was:** SearXNG-Container mit `json`-Format aktiviert. Der Laptop-Worker ist extern (hinter NAT)
  und erreicht SearXNG deshalb AUSSCHLIESSLICH über public `mcp.linn.games` — nicht intern.
- **Wie:** nginx-location `/searxng/` → proxy_pass `http://searxng:8080/`, hinter JWT (gleicher
  Bearer wie sonst). `web_search`-Tool ruft `https://mcp.linn.games/searxng/search?format=json&q=...`
  mit `Authorization: Bearer <hook.jwt>`.
- **Interface:** GET JSON, Top-N results[].(title,url,content).

### 5. nginx / Reachability
- A2A-Routes in `app.linn.games/docker/mayring/nginx/mcp.conf`-Allowlist aufnehmen (neuer Pfad
  `a2a` oder bestehende Regex-location erweitern) — sonst 401-body=None via MCP-upstream
  (bekannte Falle [[project_nginx_mcp_conf_sot]]).

## Data Flow (End-to-End)

1. Langdock: A2A `message/send` { task: "Recherchiere X tief" } + JWT → `mcp.linn.games/a2a`.
2. Gateway: `create_cloud_job(scope=cloud, cap=research, task=...)` → returns A2A-Task(WORKING, id=job_id).
3. Laptop-Worker (Poll-Loop): `claim_cloud_next(worker_id, caps=["research"])` → job.
4. Worker: `run_task_with_memory(task, model=qwen3.5:9b, session_id=job.context)` → Agent-Loop nutzt
   `search_memory` (cloud), `web_search` (SearXNG), `web_fetch`, `ingest` → Ergebnis-Text.
5. Worker: `complete_cloud(job_id, {text: result})`.
6. Langdock: `tasks/get(id)` (Poll) → COMPLETED + result.

## Fehlerbehandlung

- Worker offline / kein Claim: Job bleibt `queued`; A2A-Task bleibt WORKING (Langdock pollt weiter).
  Optional: Job-TTL → FAILED nach N Minuten ohne Claim (Plan-Phase).
- Tool-Fehler (SearXNG down, ingest 401): Tool gibt Fehlertext an den Agent zurück (kein stilles
  Schlucken, CLAUDE.md-Regel) — Agent kann es im Result vermerken.
- Worker-Crash mid-job: bestehende claim-Lease/Requeue-Semantik von pi_jobs (Plan-Phase prüfen).

## Testing

- **Unit:** `_execute_web_search` (SearXNG-JSON gemockt), `_execute_ingest` (HTTP gemockt),
  `RelayAgentExecutor.execute` (enqueue gemockt → WORKING-Task), status-mapping (pi-job→A2A-state).
- **Integration:** A2AClient → Cloud-Gateway (TestClient) → in-memory pi_jobs → fake worker
  completes → `tasks/get` liefert COMPLETED.
- **Live-Proof (PFLICHT, kein „gebaut aber nie genutzt"):** echter Laptop-Worker (cloud-Modus,
  cap=research) + echter Langdock-/A2AClient-Call gegen mcp.linn.games → reale Recherche mit
  SearXNG-Hits + Cloud-Memory, end-to-end. Belegen wie M2b.

## Scope-Grenzen (YAGNI)

- EIN Research-Worker zuerst (Design generalisiert auf Pool via capability-tags, aber nicht jetzt bauen).
- Keine Paper-Suche, kein SSE-Streaming (Polling reicht für lange Aufträge), kein Push-Notification.
- Kein Multi-Tenant-Routing über die capability hinaus.

## Offene Punkte für die Plan-Phase

- Exakte Namen der cloud-job-create + status-read-Endpoints (pi_task_*_cloud vs /pi/run) verifizieren.
- A2A-Task-Persistenz cloud-seitig (TaskStore-Adapter auf pi_jobs vs InMemory + job-id-Mapping).
- Worker-Run als systemd-user-Service vs manueller Start.
- SearXNG-Image + settings.yml (json-Format, limiter aus für interne Nutzung).
