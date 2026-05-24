# API Concurrency & Capacity — Design Spec

**Datum:** 2026-05-24
**Status:** Draft — wartet auf User-Review
**Auslöser:** Memory-Hook timeoutet bei jedem Prompt (`TIMEOUT after 9.0s`). Ganztägiger Outage-Firefight (2026-05-24) hat als Endbefund die single-worker-Kapazität als Wurzel isoliert.
**Verbunden mit:** [[project_mayring_api_saturation_outage]], Phase-5-Trace-View (separat, gebaut).

---

## 1. Problem

Der Cloud-API (`mcp.linn.games`) kann **gleichzeitige Requests nicht bedienen**. Der `session_start`/UserPromptSubmit-Hook feuert pro Prompt **3 Suchen** (`primary`/`ambient`/`conversation`). Eine idle-Suche dauert ~3.8s (embed + Chroma + Rerank). Auf der aktuellen Architektur:

- **`async def` + synchroner Body auf dem Event-Loop** → 3 gleichzeitige Suchen serialisieren auf dem einen Loop → 16-30s → alle 3 überschreiten das 9s-Hook-Timeout → **null injizierte Memory**.
- **Threadpool-Offload (versucht, `8b0ff34`, reverted `bbabe22`)** → die 3 Threads contenden auf der EINEN geteilten SQLite-Connection + EINEM Chroma-Client → **Deadlock** (alle 35s) — schlimmer als serialisiert.

`/health` (trivial, kein DB) bleibt schnell und **täuscht „API ok" vor** — der eigentliche Schaden ist auf dem Such-Pfad.

### Messung (Beweis)
| Szenario | Ergebnis |
|---|---|
| 1 idle Suche | ~3.8s |
| 3 nacheinander (async-loop) | 2.8s / 16s / 30s-Timeout |
| 3 gleichzeitig (threadpool) | alle 35s-Timeout (Deadlock) |

## 2. Root Causes (4 Schichten)

1. **1 uvicorn-Worker** (1 Prozess, 1 Event-Loop). `async def`-Endpoints mit synchronem Body (embed/Chroma/SQLite) blockieren den Loop für die Request-Dauer → keine echte Nebenläufigkeit.
2. **1 geteilte SQLite-Connection** (`get_conn()`-Singleton, `check_same_thread=False`). Threadpool-Threads deadlocken darauf + auf dem Chroma-Client.
3. **Embedded Chroma `PersistentClient`** (1 Prozess, NICHT multi-prozess-sicher) → blockiert den naheliegenden `--workers`-Fix.
4. **In-memory per-Prozess-State**: `_JOBS` (job-registry), `_DASH_CACHE`, `_RECENT_ACTIVATIONS`, `_STATS_CACHE` → würde unter Multi-Worker fragmentieren (Dashboards je nach Prozess inkonsistent).

## 3. Ziel

Der Hook (3 gleichzeitige Suchen) UND allgemeine Last (Dashboards, post-deploy-ingest+smoke) ohne Timeout/Deadlock bedienen. **Akzeptanzkriterium:** 3 gleichzeitige `/memory/search` je < 5s (p95 < 8s); `/health` < 200ms unter Such-Last; keine Chroma-Corruption.

## 4. Betrachtete Ansätze

| # | Ansatz | Urteil |
|---|---|---|
| A | `uvicorn --workers` + embedded Chroma | **Blockiert** — embedded Chroma ist nicht multi-prozess-sicher (Daten-Corruption-Risiko bei Writes). |
| B | **Chroma-Server-Modus + `uvicorn --workers` + Redis-für-Shared-State** | **EMPFOHLEN** — löst die Such-Concurrency direkt, inkrementell + lasttestbar. |
| C | Voller async-Rewrite (async-SQLite via aiosqlite + async-Chroma) | Zu groß/riskant; berührt den ganzen Hot-Path. Später evtl. |
| D | Celery + Redis Job-Queue | **Abgelehnt** — löst das FALSCHE Problem (asynchrone Jobs, nicht synchrone Suchen). Der Hook wartet synchron auf das Ergebnis; eine Job-Queue passt nicht. MayringCoder hat zudem schon `pi_jobs`+`pi_worker` (Flood gefixt). Celery = drittes Job-System = „brutal vermixxen". |
| E | Threadpool-Offload (`run_in_threadpool`) auf shared State | **Versucht + reverted** — Deadlock (s. §1). Nur tragfähig MIT per-thread-Connections + concurrency-safe Chroma. |

## 5. Empfohlenes Design (B)

### 5.1 Chroma als Service
- Eigener Container (`chromadb/chroma` o. `chroma run`), zeigt auf das **bestehende** Persist-Dir (`/app/cache/memory_chroma`) → keine Daten-Migration, nur „wer öffnet die Files".
- API: `get_chroma_collection()` nutzt `chromadb.HttpClient(host, port)` statt `PersistentClient(path)`. Der Chroma-Server serialisiert/locked intern → multi-prozess-sicher.
- Healthcheck + `restart: unless-stopped`; neuer SPOF → bewusst mit Monitoring.

### 5.2 Multi-Worker API
- `uvicorn ... --workers N` (Start N=2-3; Container hat 7.4GB RAM, 1 Core war gepegt → mehr Cores nutzen). Gleichzeitige Suchen landen in verschiedenen Prozessen → echte Parallelität.
- SQLite: WAL ist an (`db_wal_journal_active` ✅). Jeder Worker hat seinen eigenen `get_conn()`-Singleton (eigener Prozess) + eigenes `ATTACH wikidb`. Writes serialisieren via SQLite-Lock (kurz), Reads parallel (WAL).
- Endpoints bleiben `async def` — aber da jetzt N Worker existieren, blockiert ein langsamer Such-Request nur SEINEN Worker, nicht alle. (Optionaler späterer Schliff: heavy sync-Bodies via `run_in_threadpool` MIT per-thread-Connection — erst nach C-Bewertung.)

### 5.3 Redis für Cross-Worker-State (Phase 2, nur wenn nötig)
- `_DASH_CACHE` / `_STATS_CACHE` / `_RECENT_ACTIVATIONS` / `_JOBS` → Redis, damit alle Worker konsistent sehen.
- **Dedizierter Redis-Container** (NICHT app.linn.games' Redis → kein cross-service-Coupling) ODER separate DB-Nummer mit Namespace.
- **Phase-1-Vereinfachung:** Die Dashboard-Caches tolerieren kurzfristig per-Worker-Inkonsistenz (kosmetisch; echte Daten liegen in SQLite). `_JOBS` ist bereits nach `cache/jobs_state.json` persistiert. → Redis kann Phase 2 sein, falls die Inkonsistenz stört.

## 6. Rollout (inkrementell, jede Phase einzeln verifiziert)

1. **Phase 1a:** Chroma-Server-Container hoch, API per `HttpClient` anbinden, **noch 1 Worker**. Verify: `/memory/search` liefert dieselben Treffer wie embedded (Collections intakt, max_score vergleichbar). Smoke grün.
2. **Phase 1b:** `--workers 2` → Lasttest (s. §7). Verify: 3 gleichzeitige Suchen je < 5s; `/health` < 200ms unter Last; keine Chroma-Fehler.
3. **Phase 1c:** `--workers 3` falls Phase 1b sauber + mehr Headroom nötig.
4. **Phase 2 (optional):** Redis-Shared-State, falls Dashboard-Inkonsistenz unter Multi-Worker auffällt.

## 7. Lasttest-Plan (PFLICHT vor Prod-Rollout)

- Benchmark-Skript: feuert 3 (dann 6, 10) gleichzeitige `/memory/search` (Hook-Muster) + misst je-Request-Latenz + p95 + `/health`-Latenz parallel.
- Baseline (heute) vs nach jeder Phase. Akzeptanz: §3-Kriterien.
- Gegen einen **Canary/Staging** ODER mit Ankündigung gegen Prod in einem ruhigen Fenster (NICHT während post-deploy-ingest).
- Chroma-Integritäts-Check nach Multi-Worker-Last (`count()` + ein paar Known-Queries).

## 8. Risiken & Mitigationen

| Risiko | Mitigation |
|---|---|
| Chroma-Daten beim Umzug embedded→Server | Server zeigt auf dasselbe Dir; vorher Backup; Collections-Count verifizieren. |
| Chroma-Server = neuer SPOF | Healthcheck + restart-policy + Monitoring; API fail-soft wenn Chroma weg. |
| SQLite-Write-Contention über Worker | WAL an; Writes kurz + serialisiert (kein Corruption); ggf. Write-Endpoints drosseln. |
| Multi-Worker fragmentiert in-memory-State | Phase 2 Redis; Phase 1 akzeptiert kosmetische Dashboard-Inkonsistenz. |
| Mehr Worker → mehr gleichzeitige Embed-Calls an `three.linn.games` | Ollama-Proxy verträgt das (war nie der Engpass); überwachen. |

## 9. Bonus-Effekt
Multi-Worker behebt **auch** den Deploy-Fenster-Wedge: während post-deploy-ingest einen Worker belastet, bedienen die anderen Suchen/Health → keine 5-8-min-Mini-Outages + keine false-alarm-Smokes mehr.

## 10. Out of Scope
- Celery / Job-Queue-Umbau (Ansatz D — abgelehnt; `pi_jobs`/`pi_worker` bleibt).
- Voller async-Rewrite (Ansatz C — später, falls Multi-Worker nicht reicht).
- Per-Prompt-Trace-View (Phase-5-Feature; Backend `/stats/prompt-trace` + UI bereits gebaut+getestet, separat zu mergen).
