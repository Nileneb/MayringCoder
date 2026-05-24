# API Concurrency & Capacity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans. Steps use `- [ ]`.
> **Run as a DEDICATED, deploy-quiet session** — not bundled with feature work. Each phase is independently deployable + verified by load test before the next.

**Goal:** Let the API serve concurrent `/memory/search` (the hook fires 3) without timeout/deadlock, by moving Chroma to a service + running multiple uvicorn workers.

**Architecture:** Chroma-as-service (HttpClient) unblocks multi-process; `uvicorn --workers N` gives real parallelism; (phase 3) Redis for cross-worker state. See spec `docs/superpowers/specs/2026-05-24-api-concurrency-capacity-design.md`.

**Tech Stack:** FastAPI/uvicorn, ChromaDB (embedded → server), SQLite (WAL), docker-compose (`app.linn.games/docker-compose.mayring.yml`), optional Redis.

---

## Phase 0 — Load-test harness (baseline before any change)

### Task 0.1: Concurrent-search benchmark tool

**Files:** Create `tools/loadtest_search.py`

- [ ] **Step 1:** Write the tool — fires N concurrent `/memory/search` with a hook.jwt, plus parallel `/health` probes; prints per-request latency, p50/p95, and worst `/health`.

```python
"""Concurrency load test for /memory/search (the session-start hook fires 3).
Usage: python tools/loadtest_search.py --n 3 --api https://mcp.linn.games"""
import argparse, concurrent.futures as cf, json, os, statistics, time, urllib.request
from pathlib import Path

def _jwt() -> str:
    return Path(os.getenv("MAYRING_HOOK_JWT", str(Path.home()/".config/mayring/hook.jwt"))).read_text().strip()

def _search(api, jwt, q):
    body = json.dumps({"query": q, "top_k": 8, "include_text": False}).encode()
    req = urllib.request.Request(f"{api}/memory/search", data=body,
        headers={"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=40) as r: r.read(); return time.monotonic()-t0, 200
    except Exception as e: return time.monotonic()-t0, getattr(e, "code", 0)

def _health(api):
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(f"{api}/health", timeout=10) as r: r.read(); return time.monotonic()-t0
    except Exception: return time.monotonic()-t0

def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=3)
    p.add_argument("--api", default=os.getenv("MAYRING_API_URL", "https://mcp.linn.games"))
    a = p.parse_args(); jwt = _jwt(); api = a.api.rstrip("/")
    qs = [f"loadtest lens {i}" for i in range(a.n)]
    with cf.ThreadPoolExecutor(max_workers=a.n+1) as ex:
        h = ex.submit(_health, api)
        futs = [ex.submit(_search, api, jwt, q) for q in qs]
        results = [f.result() for f in futs]; health = h.result()
    lat = [t for t, _ in results]; codes = [c for _, c in results]
    print(f"n={a.n}  codes={codes}")
    print(f"  per-req: {[round(t,2) for t in lat]}")
    print(f"  p50={statistics.median(lat):.2f}s  max={max(lat):.2f}s  health-during={health:.2f}s")
    ok = all(c == 200 for c in codes) and max(lat) < 5.0 and health < 0.2
    print("  PASS" if ok else "  FAIL (target: all 200, max<5s, health<0.2s)")

if __name__ == "__main__": main()
```

- [ ] **Step 2:** Run baseline (current single-worker): `python tools/loadtest_search.py --n 3`
  Expected: FAIL (current: 16-30s / timeouts). Record the number — this is the "before".
- [ ] **Step 3:** Commit. `git add tools/loadtest_search.py && git commit -m "test: concurrent-search load harness (capacity baseline)"`

## Phase 1 — Chroma as a service

### Task 1.1: Chroma server container

**Files:** Modify `app.linn.games/docker-compose.mayring.yml`

- [ ] **Step 1:** Add a `chroma` service (project `mayring`), mounting the EXISTING persist dir so no data migration:

```yaml
  mayring-chroma:
    image: chromadb/chroma:latest
    volumes:
      - ../cache/memory_chroma:/chroma/chroma   # existing embedded data
    environment:
      IS_PERSISTENT: "TRUE"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v2/heartbeat"]
      interval: 15s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks: [mayring]
```
(Confirm the host path of the current `memory_chroma` dir on u-server first; mount the SAME one.)

- [ ] **Step 2:** Add `MAYRING_CHROMA_HOST: mayring-chroma` + `MAYRING_CHROMA_PORT: "8000"` to the `environment:` of `mayring-api`, `mayring-mcp`, `mayring-webui`.

### Task 1.2: get_chroma_collection → HttpClient when configured

**Files:** Modify `core/mayring_core/memory/store.py:31-51`

- [ ] **Step 1:** Write a failing test `tests/test_chroma_httpclient.py`: with `MAYRING_CHROMA_HOST` set, `get_chroma_collection` builds an `HttpClient`, not `PersistentClient` (monkeypatch `chromadb.HttpClient` to a sentinel + assert it's used).
- [ ] **Step 2:** In `get_chroma_collection`, branch:

```python
host = os.getenv("MAYRING_CHROMA_HOST")
if host:
    port = int(os.getenv("MAYRING_CHROMA_PORT", "8000"))
    ckey = f"http://{host}:{port}"
    if ckey not in _chroma_clients:
        _chroma_clients[ckey] = chromadb.HttpClient(host=host, port=port)
    client = _chroma_clients[ckey]
else:
    # back-compat (tests, standalone) — embedded PersistentClient
    chroma_path = str(path or CACHE_DIR / "memory_chroma")
    ...existing PersistentClient path...
```
Keep the embedded path for tests/standalone (`MAYRING_CHROMA_HOST` unset).

- [ ] **Step 3:** Run the test + full suite (`pytest -q -n auto`). Expected: PASS (embedded default keeps tests green).
- [ ] **Step 4:** Commit.

### Task 1.3: Deploy Phase 1 with STILL 1 worker — verify Chroma-server path

- [ ] **Step 1:** Deploy (push MayringCoder + the compose change in app.linn.games).
- [ ] **Step 2:** Verify search works against the Chroma server (collections intact via shared volume): `python tools/loadtest_search.py --n 1` → 200, max_score sane. Post-deploy smoke green.
- [ ] **Step 3:** Chroma integrity: a known query returns expected source_ids; collection count matches pre-migration.

**STOP/verify gate:** Do not proceed to multi-worker until search-via-Chroma-server is confirmed equivalent to embedded.

## Phase 2 — Multi-worker

### Task 2.1: Bump uvicorn workers

**Files:** Modify `app.linn.games/docker-compose.mayring.yml:22`

- [ ] **Step 1:** `command: python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8090 --workers 2`
- [ ] **Step 2:** Deploy.
- [ ] **Step 3:** Load test: `python tools/loadtest_search.py --n 3` → **target: all 200, max<5s, health<0.2s**. Then `--n 6`.
- [ ] **Step 4:** Watch `/health` during a real deploy (post-deploy ingest+smoke) — should stay responsive (the deploy-window wedge is gone).
- [ ] **Step 5:** Verify the hook in a live session: the `TIMEOUT after 9.0s` block is gone; memory loads.

### Task 2.2: Scale to 3 if needed

- [ ] **Step 1:** If Phase 2 clean + headroom needed: `--workers 3`, re-run load test `--n 10`.

**Known caveats to watch (from spec §8):** SQLite write-contention (WAL on; writes serialise, no corruption); in-memory dashboard state fragments per-worker (cosmetic until Phase 3); more concurrent embed calls to three.linn.games (monitor).

## Phase 3 — Redis shared state (ONLY if Phase 2 surfaces inconsistency)

### Task 3.1: Dedicated Redis + move process-local caches

**Files:** `docker-compose.mayring.yml` (+ a small `redis` service, NOT app.linn.games'); `src/api/routes/dashboard.py` (`_DASH_CACHE`), `src/api/server.py` (`_STATS_CACHE`), `src/api/memory_service.py` (`_RECENT_ACTIVATIONS`), `src/api/job_queue.py` (`_JOBS`).

- [ ] **Step 1:** Add a dedicated `mayring-redis` service (own container / separate DB number — no cross-coupling to app.linn.games' Redis).
- [ ] **Step 2:** Replace the in-memory dicts with a tiny Redis-backed TTL cache helper (keep the same keys; workspace-scoped). One module, reused — no per-call duplication.
- [ ] **Step 3:** Deploy + verify dashboards consistent across workers (hit repeatedly, same numbers).

## Verification (end-to-end, per phase)
- Phase 0: baseline FAIL recorded.
- Phase 1: search-via-Chroma-server == embedded; smoke green; 1 worker.
- Phase 2: `loadtest_search --n 3/6` PASS (all 200, max<5s); hook timeout gone; deploy-window wedge gone.
- Phase 3: dashboards consistent across workers.

## Rollback
- Each phase is a compose/env revert: unset `MAYRING_CHROMA_HOST` → back to embedded; `--workers 1` → back to single. Chroma data is the same volume throughout (no destructive migration).

## Out of scope
Celery/job-queue rework; full async rewrite; per-prompt-trace UI (separate, built).

---

## Umsetzung & Ergebnis (2026-05-24, deployed)

**Status: Phase 0–2 umgesetzt + auf Prod verifiziert. Phase 3 (Redis) NICHT — Bedingung „nur falls Multi-Worker Inkonsistenz zeigt" nicht eingetreten.**

### Was geliefert wurde
- **Phase 0:** `tools/loadtest_search.py` (MayringCoder, commit c5eed5f).
- **Phase 1.2:** `get_chroma_collection` HttpClient-Branch + `tests/test_chroma_httpclient.py` (commit ec5ed5c). Volle Suite 1679 passed.
- **Phase 1.1 + 2.1:** `mayring-chroma`-Service + `MAYRING_CHROMA_HOST` (api/mcp/webui) + `--workers 4` in `app.linn.games/docker-compose.mayring.yml`.

### Korrekturen am Plan-Snippet (waren falsch für dieses Setup)
1. **Volume:** Bestehende Daten liegen im **named volume `linn-mayring-cache`** (`/app/cache/memory_chroma`), NICHT in einem `../cache`-bind. Chroma-Service mountet das named volume.
2. **Netz:** `mayring-internal` (nicht `mayring`).
3. **Image-Tag:** auf **`chromadb/chroma:1.5.9`** gepinnt (== client-lib → identisches Persist-Format, keine Migration).
4. **KRITISCH — Persist-Pfad:** `chromadb/chroma:1.x` ignoriert `PERSIST_DIRECTORY`/`IS_PERSISTENT` (das war der 0.4/0.5-Env-Contract). Entrypoint ist `chroma run [CONFIG]`. Mit nur den Env-Vars startete der Server LEER → `score_vector=0` für alle Treffer (symbolisches Fallback maskierte es). **Fix:** `command: ["run","--path","/app/cache/memory_chroma","--host","0.0.0.0","--port","8000"]`.
5. **Healthcheck:** entfernt — 1.x ist ein Rust-Binary-Slim-Image ohne garantiertes curl/python für eine in-container-Probe.

### Messungen
| Szenario | Baseline (1 Worker) | Nach Fix (4 Worker, warm) |
|---|---|---|
| 3 gleichzeitige Suchen (Hook) | max 6.8s, alle 200 | max ~2.6s, alle 200 |
| `/health` unter Such-Last | **6.79s** | **~0.06–0.08s** (gelegentl. 0.24s-Blip) |
| `score_vector` | (n/a) | echt & variiert (0.15–0.31) ✓ |

→ **Akzeptanzkriterium (Hook-Muster: 3 Suchen <5s, `/health` <0.2s) erfüllt.** n=6 (Stress > Worker-Zahl) degradiert erwartungsgemäß (max ~8s) — bräuchte den deferrten threadpool/async-Schliff (§5.2).

### Worker-Tuning
`--workers 2` löste bereits den Hook-Timeout (3 Suchen ~4s), ließ aber `/health` ~2.5s hinter einer Suche warten. `--workers 4` gibt der Hook-Last einen freien Worker → `/health` <0.2s. (Zwischenmessung „alle 502 bei workers=4" war das Deploy-Cutover-/Boot-Fenster, kein OOM — 4 Worker laufen stabil.)

### Offen / separat
- `coverage_map_complete` Smoke rot: `missing=[429]` — Coverage-Map-Docs-Lücke, unabhängig von dieser Arbeit (#429).
- Phase 3 (Redis Cross-Worker-State): deferred bis Dashboard-Inkonsistenz unter Multi-Worker tatsächlich auffällt.
- `/health <0.2s` auch unter n>Worker-Last: bräuchte `run_in_threadpool` mit per-thread-Connections (§5.2, bewusst deferred).
