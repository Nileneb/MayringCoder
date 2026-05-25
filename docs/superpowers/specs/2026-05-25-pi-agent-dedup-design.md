# Pi-Agent Dedup — #266 abschließen (`src/agents`-Duplikat eliminieren)

**Datum:** 2026-05-25
**Issue:** #266 (Stufe 3 von #265), Parent #270
**Status:** Approved (User „passt", 2026-05-25) — bereit für writing-plans
**Verbunden:** [[architecture:2026-05-25-modularisierung-pi-agent-duplikat]], #267 (mayring-core), #268 (plugin)

## Problem

Die Pi-Agent-Auslagerung (#266) ist **halb fertig und genau deshalb gefährlich**:

- `Nileneb/mayring-pi-agent` ist ein echtes, aktives, **privates** Repo: paketiert
  (`mayring_pi_agent/`, `pyproject.toml`, depend auf `mayring-core` via
  git-subdirectory), eigenes Image `nileneb/mayring-pi-agent:0.1.1`, läuft als
  eigener Prod-Container `mayring-pi` (RS256-Auth, FS-Sandbox, `web_fetch` #211/#224).
- **ABER** `MayringCoder/src/agents/` existiert weiter und wird **live in-process**
  genutzt: `src/api/server.py` (= der `mayring-api`-Container selbst),
  `routes/{memory,devices,duel,pi_stats}.py`, `analysis/analyzer.py`,
  `workflows/pi_task.py`, `provider_setup.py`, `local_mcp.py`, `main.py:57`
  (`pi_server`-Thread auf :8091). Nur `mcp_agent_tools.py` spricht HTTP via
  `PI_AGENT_URL` gegen den Container (mit in-process-Fallback).

**Folge:** Zwei Runtimes, zwei Codebases, **aktiver Drift** — `pi_worker.py` ist
um 41 Zeilen divergiert, das Paket hat Features (`web_fetch`-Allowlist,
`write/exec`-FS-Sandbox, `_execute_bash`), die `src/agents` **nicht** hat.
Bugfixes müssen 2× gemacht werden — exakt das Frustrations-Anti-Pattern
„doppelt gefixt".

## Ziel

MayringCoder konsumiert `mayring-pi-agent` als **Package** (Git-Submodule),
`src/agents/` wird **gelöscht**. Eine Codebase, kein Drift. Der `mayring-pi`-Container
und die HTTP-Boundary bleiben unverändert. Damit ist #266 erfüllt (eigenes
Repo/Image/Deploy existiert ja schon; es fehlt nur die Eliminierung des Duplikats).

**Entscheidung (User-approved):** In-process Package-Dependency, **nicht**
Full-HTTP-Boundary. Begründung: die Paket-API ist 1:1-kompatibel
(`run_task_with_memory`, `analyze_with_memory`, `get_pi_queue`,
`classify_pi_job`, `PiJob`), `pi_queue`/`classify_pi_job` sind tief in der
Async-Job-Infrastruktur verankert; ein HTTP-Umbau wäre #270-Scope mit
Latenz/Auth-Overhead ohne akuten Mehrwert.

## Architektur & Build

### Bezug: Git-Submodule
- Submodule `vendor/mayring-pi-agent`, gepinnt auf Tag `0.1.1`.
- Reproduzierbar via Submodule-Commit-Pin, **kein Runtime-Token** im Image
  (passt 1:1 zum bestehenden `core/`-COPY-Muster).

### Dockerfile (nach dem bestehenden `pip install -e ./core`)
```dockerfile
COPY vendor/mayring-pi-agent ./vendor/mayring-pi-agent
RUN pip install --no-cache-dir -e ./vendor/mayring-pi-agent --no-deps
```
`--no-deps` ist der **Schlüssel gegen die zirkuläre Core-Dep**: das Paket würde
sonst `mayring-core` von Git ziehen (`git+...MayringCoder.git@master#subdirectory=core`),
obwohl `mayring-core` schon via `core/` editable installiert ist. Die übrigen
Paket-Deps (`httpx`, `fastapi`, `uvicorn[standard]`) sind bereits in
`requirements.txt` vorhanden — verifiziert.

### CI / Deploy
- `build-and-push.yml`: `actions/checkout` → `submodules: true` +
  `token: ${{ secrets.GH_PAT }}` (Secret existiert bereits im Workflow).
  **Voraussetzung:** der PAT muss `mayring-pi-agent` lesen dürfen.
- Deploy (app.linn.games `deploy-mayring.sh`) unverändert — zieht nur das
  fertige Image, baut nicht.

## Code-Swap (~12 Stellen, API 1:1)

`from src.agents.X` → `from mayring_pi_agent.X` in:

| Datei | Import |
|---|---|
| `src/api/server.py` | `pi_queue.get_pi_queue`, `pi_jobs.PiJob`, `pi.run_task_with_memory` |
| `src/api/routes/memory.py` | `pi_queue.get_pi_queue`, `pi_jobs.{PiJob,classify_pi_job}` |
| `src/api/routes/devices.py` | `pi_jobs` |
| `src/api/routes/duel.py` | `pi.run_task_with_memory` (2×) |
| `src/api/routes/pi_stats.py` | `pi_queue.get_pi_queue` |
| `src/api/mcp_agent_tools.py` | Fallback `pi.run_task_with_memory`, `pi_jobs`; Hint-String anpassen |
| `src/analysis/analyzer.py` | `pi.analyze_with_memory` |
| `src/workflows/pi_task.py` | `pi.run_task_with_memory` |
| `src/provider_setup.py` | `vision.{caption_image,get_image_metadata}` |
| `src/api/local_mcp.py` | `pi_worker` |
| `src/main.py:57` | Thread-Target `src.agents.pi_server:app` → `mayring_pi_agent.pi_server:app` |

Danach: **`src/agents/` löschen.**

## Risiken & Mitigation

1. **API-Drift einzelner Funktionen** (nicht nur Namen) — größtes Restrisiko.
   Mitigation: pro Call-Site die Signatur gegen das Paket prüfen; bestehende
   Tests + Roundtrip-Tests fangen Reste.
2. **Submodule-Pin-Disziplin** — Pi-Agent-Updates brauchen künftig einen
   Submodule-Bump in MayringCoder. Bewusster Trade-off der gewählten Option.
3. **GH_PAT-Scope** — muss das private Repo lesen können, sonst rotes CI.
4. **Atomarer Übergang** — Prod läuft bis zum Deploy auf `src/agents`; der Swap
   ist atomar (ein Image-Build).

## Teststrategie

- Bestehende Tests, die `src.agents` importieren/mocken → auf `mayring_pi_agent`
  umstellen.
- In-process `pi_queue`/`classify_pi_job`-Roundtrip aus dem Paket.
- HTTP-Pfad (`mcp_agent_tools`) bleibt grün.
- **Grep-Gate:** nach dem Swap kein `from src.agents` / `import src.agents` mehr.
- Smoke: Pi-Task-Roundtrip (#266-Akzeptanz).

## Akzeptanzkriterien (#266)

- [ ] `src/agents/` gelöscht, kein `src.agents`-Import im Repo
- [ ] `vendor/mayring-pi-agent`-Submodule auf `0.1.1` gepinnt
- [ ] Dockerfile installiert Paket via `-e --no-deps`
- [ ] `build-and-push.yml` checkt Submodule mit `GH_PAT` aus
- [ ] Alle bestehenden Tests grün
- [ ] Pi-Task funktioniert (in-process Queue + HTTP-Boundary)
- [ ] Drift eliminiert (eine Codebase)

## Nicht-Ziele (entkoppelt, #270)

- Full-HTTP-Boundary für `pi_queue`/`classify`
- MCP-Server als eigenes Paket
- `local_mcp.py`-Deprecation / Cloud-only
- Backup-Strategie (Azure/Laravel) — orthogonales Thema
