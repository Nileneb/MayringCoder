# Pi-Agent-Auslagerung → `mayring-pi-agent` (#266)

Stufe 3 der Modularisierung (#265), nach #267 (`mayring-core`). Pi-Agent wird
als eigenständiger Microservice mit eigenem Deploy-Zyklus ausgelagert.

## Status (Stand dev/v2-bigbang)

| Schritt | Status |
|---|---|
| HTTP-API-Boundary in `mcp_agent_tools.py` | ✅ existiert bereits (`PI_AGENT_URL`) |
| Extraktion `src/agents/` mit History | ✅ vorbereitet (lokaler Branch `pi-agent-split`, 42 Commits) |
| Neues Repo `Nileneb/mayring-pi-agent` anlegen + Push | ⛔ **blockiert** durch Auto-Mode-Classifier (Push zu neuem externem Repo = Data-Exfiltration-Schutz) — **braucht User-Aktion** |
| Removal `src/agents/` aus MayringCoder + HTTP-only | ⏸ Cutover-Schritt (10+ Importer, erst nach Service-Deploy) |
| `mayring-core` als Dependency im neuen Repo | ⏸ Cutover (core ist nur `pip install -e ./core`, nicht published) |

## HTTP-Boundary (bereits vorhanden)

`src/api/mcp_agent_tools.py::pi_task` routet schon über
`PI_AGENT_URL` (default `http://localhost:8091`): ist der Wert ≠ `direct`,
geht ein `httpx.post` an den Pi-Service; sonst in-process-Fallback
(`run_task_with_memory`). Damit ist der Roundtrip über die HTTP-Grenze für
`pi_task` abgedeckt.

**Offen (Cutover):** `pi_categorize`, `pi_judge_relevance`,
`pi_summarize_for_memory` rufen aktuell in-process bzw. gegen die MayringCoder-API;
sie müssen analog hinter `PI_AGENT_URL` gelegt werden, bevor `src/agents/`
entfernt wird.

## Extraktion reproduzieren (wenn Push autorisiert)

Der Branch `pi-agent-split` ist bereits via Subtree-Split erzeugt:

```bash
# (bereits geschehen auf dev/v2-bigbang)
git subtree split --prefix=src/agents -b pi-agent-split

# Vom User auszuführen (Push zu neuem Repo braucht Freigabe):
gh repo create Nileneb/mayring-pi-agent --private \
  --description "Pi-Agent microservice (extracted from MayringCoder #266)"
git push https://github.com/Nileneb/mayring-pi-agent.git pi-agent-split:main
```

Danach im neuen Repo: `pyproject.toml` (depends on `mayring-core`),
`Dockerfile`, `server.py` (← `pi_server.py`), CI `build-and-push`.

> Der Auto-Mode-Classifier verweigert den `git push` zu einem neuen externen
> Repo (Exfiltrationsschutz). Zum Freigeben: Bash-Permission-Rule in den
> Settings ergänzen ODER den Push manuell ausführen (`! git push …` in der
> Session).

## Cutover-Reihenfolge (Big-Bang)

1. `mayring-core` als installierbare Dependency bereitstellen (git-dep oder
   published) — Voraussetzung, dass der neue Service `import mayring_core` kann.
2. Restliche Pi-Tools (`pi_categorize` etc.) hinter `PI_AGENT_URL` legen.
3. `mayring-pi-agent` deployen (eigener Container, Healthcheck, `/pi-jobs/stats`).
4. Erst dann in MayringCoder `src/agents/` entfernen + alle Importer auf
   HTTP-Client umstellen (`provider_setup`, `workflows/pi_task`, `routes/*`,
   `local_mcp`, `server`). Latenz-Delta < 50ms gegen In-Process messen (AC #266).
5. Smoke: Pi-Task-Roundtrip über HTTP-Boundary.

Bis zum Cutover bleibt `src/agents/` in MayringCoder lauffähig (in-process
Default), damit nichts vor dem Service-Deploy bricht.
