# A2A Research-Worker Relay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) or
> superpowers:subagent-driven-development to implement task-by-task. Steps use checkbox (`- [ ]`).

**Goal:** Ein Laptop-Recherche-Agent, den Langdock über `mcp.linn.games` (A2A) anstößt; der Laptop
zieht den Job aus der Cloud-Queue, recherchiert lokal (Ollama + SearXNG + Cloud-Memory) und meldet
das Ergebnis zurück — async, NAT-sicher, „kaum Geld".

**Architecture:** Cloud-A2A-Gateway (mayring-api) legt cloud-scoped `pi_jobs` mit
`capability_required="research"` an und gibt einen async A2A-Task zurück (task_id == job_id). Der
Laptop läuft als pi_worker im cloud-pull-Modus (cap=research), claimt nur eigene Jobs, ruft
`run_task_with_memory` mit den Tools `search_memory`/`web_search`/`web_fetch`/`ingest`. Langdock
pollt `tasks/get`; ein `PiJobsTaskStore` mappt pi_jobs-Status → A2A-Status.

**Tech Stack:** Python, FastAPI, `a2a-sdk>=1.1.0`, SQLite (pi_jobs), Ollama (qwen3.5:9b), SearXNG
(Docker), nginx. Spec: `docs/superpowers/specs/2026-05-30-a2a-research-worker-relay-design.md`.

## File Structure

| Datei | Repo | Verantwortung |
|---|---|---|
| `mayring_pi_agent/pi.py` | mayring-pi-agent | +`_execute_web_search`, `_execute_ingest`, _TOOLS-Einträge, Dispatch |
| `tests/test_research_tools.py` | mayring-pi-agent | Unit-Tests für die zwei Tools |
| `mayring_pi_agent/pi_worker.py` | mayring-pi-agent | (nur Doku/Konfig — cloud-Modus existiert) |
| `deploy/mayring-research-worker.service` | mayring-pi-agent | systemd-user-Unit |
| `docs/RESEARCH_WORKER.md` | mayring-pi-agent | Run-Anleitung |
| `src/api/a2a_relay.py` | MayringCoder | `PiJobsTaskStore`, `RelayAgentExecutor`, `register_a2a_relay(app)` |
| `tests/test_a2a_relay.py` | MayringCoder | Unit + Integration |
| `src/api/server.py` | MayringCoder | `register_a2a_relay(app)` aufrufen |
| `src/api/routes/devices.py` | MayringCoder | +TTL-Sweep für unclaimed cloud-jobs |
| `docker/mayring/searxng/` | app.linn.games | SearXNG compose + settings.yml |
| `docker/mayring/nginx/mcp.conf` | app.linn.games | Allowlist `a2a`+`searxng`, `/searxng/`-location |

**Reuse (nicht neu bauen):** `pi_jobs.insert_cloud_job/get_job/fail_job`, `claim_cloud_next`,
`/pi_task_claim_cloud`+`/pi_task_complete_cloud` (devices.py), pi_worker cloud-loop,
`a2a_agent.build_agent_card`, REST `/ingest` (routes/memory.py:929).

---

## Phase 1 — Worker-Tools (mayring-pi-agent)

### Task 1: `web_search`-Tool (SearXNG)

**Files:**
- Modify: `mayring_pi_agent/pi.py` (neben `_execute_web_fetch` ~Z49; `_TOOLS` ~Z276; Dispatch ~Z762)
- Test: `tests/test_research_tools.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_research_tools.py
import json
from unittest.mock import patch
from mayring_pi_agent import pi


def test_web_search_returns_formatted_results(monkeypatch):
    monkeypatch.setenv("MAYRING_API_URL", "https://mcp.linn.games")
    fake = {"results": [
        {"title": "A2A spec", "url": "https://example.org/a2a", "content": "Agent2Agent protocol"},
        {"title": "SearXNG", "url": "https://example.org/sx", "content": "meta search"},
    ]}

    class _Resp:
        status = 200
        def read(self, *_): return json.dumps(fake).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch("mayring_pi_agent.pi.urllib.request.urlopen", return_value=_Resp()):
        out = pi._execute_web_search("a2a protocol")
    assert "A2A spec" in out and "https://example.org/a2a" in out
    assert "Agent2Agent protocol" in out
```

- [ ] **Step 2: Run, verify FAIL**

Run: `cd ~/Desktop/mayring-pi-agent && python3 -m pytest tests/test_research_tools.py::test_web_search_returns_formatted_results -q`
Expected: FAIL `AttributeError: module 'mayring_pi_agent.pi' has no attribute '_execute_web_search'`

- [ ] **Step 3: Implement `_execute_web_search`** (in pi.py, nach `_execute_web_fetch`)

```python
def _read_jwt() -> str:
    """DRY helper — selber JWT wie _cloud_search (Z447: Path(_MEMORY_JWT_FILE).read_text())."""
    try:
        return Path(_MEMORY_JWT_FILE).read_text().strip()
    except Exception:
        return ""


_SEARXNG_TIMEOUT = float(os.getenv("PI_WEB_SEARCH_TIMEOUT", "20"))
_SEARXNG_MAX_RESULTS = int(os.getenv("PI_WEB_SEARCH_MAX_RESULTS", "8"))


def _searxng_url() -> str:
    api = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
    return f"{api}/searxng/search"


def _execute_web_search(query: str) -> str:
    if not query.strip():
        return "web_search Fehler: leere Query"
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    url = f"{_searxng_url()}?{params}"
    headers = {"User-Agent": "mayring-pi-agent/1.0"}
    token = _read_jwt()  # existierender JWT-Reader in pi.py (Bearer für Cloud)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=_SEARXNG_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        return f"web_search Fehler: {exc}"
    results = (data.get("results") or [])[:_SEARXNG_MAX_RESULTS]
    if not results:
        return f"web_search: keine Treffer für {query!r}"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title','')}\n   {r.get('url','')}\n   {r.get('content','')}")
    return "\n".join(lines)
```

> **Hinweis:** `_read_jwt()` ist der vorhandene JWT-Leser (suche in pi.py nach `hook.jwt`/Bearer;
> Z19 `MAYRING_JWT_FILE`). Falls die Funktion anders heißt, denselben Leser wie `_cloud_search`
> verwenden. `urllib.parse` ist bereits importiert (web_fetch). `json` ggf. oben importieren.

- [ ] **Step 4: Register in `_TOOLS`** (nach dem `web_fetch`-Eintrag)

```python
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Durchsucht das Web (SearXNG) und liefert Top-Treffer als "
                "Titel + URL + Snippet. Danach mit web_fetch die beste URL laden."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Suchbegriff"},
                },
                "required": ["query"],
            },
        },
    },
```

- [ ] **Step 5: Dispatch** (im Agent-Loop, neben `elif func_name == "web_fetch"`)

```python
            elif func_name == "web_search":
                result_text = _execute_web_search(args.get("query", ""))
```

- [ ] **Step 6: Run, verify PASS**

Run: `python3 -m pytest tests/test_research_tools.py::test_web_search_returns_formatted_results -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add mayring_pi_agent/pi.py tests/test_research_tools.py
git commit -m "feat(tools): web_search via SearXNG"
```

### Task 2: `ingest`-Tool (Memory-write-back)

**Files:**
- Modify: `mayring_pi_agent/pi.py`
- Test: `tests/test_research_tools.py`

- [ ] **Step 1: Failing test**

```python
def test_ingest_posts_to_cloud_and_reports_ok(monkeypatch):
    monkeypatch.setenv("MAYRING_API_URL", "https://mcp.linn.games")
    captured = {}

    class _Resp:
        status = 200
        def read(self, *_): return b'{"ingested": 1}'
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["body"] = req.data
        return _Resp()

    with patch("mayring_pi_agent.pi.urllib.request.urlopen", side_effect=_fake_urlopen):
        out = pi._execute_ingest("Research X", "Findings: ...")
    assert "/ingest" in captured["url"]
    assert b"Research X" in captured["body"]
    assert "ok" in out.lower() or "ingest" in out.lower()
```

- [ ] **Step 2: Run, verify FAIL** — `has no attribute '_execute_ingest'`

- [ ] **Step 3: Implement `_execute_ingest`**

```python
def _execute_ingest(title: str, text: str) -> str:
    if not text.strip():
        return "ingest Fehler: leerer Text"
    api = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
    body = json.dumps({
        "text": text,
        "source_id": f"research:{title}"[:200],
        "source_type": "knowledge",
    }).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "mayring-pi-agent/1.0"}
    token = _read_jwt()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(f"{api}/ingest", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except Exception as exc:
        return f"ingest Fehler: {exc}"
    return f"ingest ok: '{title}' ins Memory geschrieben"
```

> **Verify in Plan-Step:** `routes/memory.py:929 @router.post("/ingest")` — Feldnamen des
> Request-Models prüfen (`text`/`source_id`/`source_type`). Falls abweichend, body anpassen.

- [ ] **Step 4: Register in `_TOOLS`**

```python
    {
        "type": "function",
        "function": {
            "name": "ingest",
            "description": (
                "Schreibt ein Recherche-Ergebnis dauerhaft ins Memory (durchsuchbar). "
                "Nutze dies am Ende, um wichtige Findings zu sichern."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Kurzer Titel"},
                    "text": {"type": "string", "description": "Der zu speichernde Inhalt"},
                },
                "required": ["title", "text"],
            },
        },
    },
```

- [ ] **Step 5: Dispatch**

```python
            elif func_name == "ingest":
                result_text = _execute_ingest(args.get("title", ""), args.get("text", ""))
```

- [ ] **Step 6: Run full file** — `python3 -m pytest tests/test_research_tools.py -q` → PASS

- [ ] **Step 7: Commit** — `git commit -am "feat(tools): ingest memory-write-back"`

---

## Phase 2 — Worker cloud-mode run (mayring-pi-agent)

### Task 3: Worker-Run-Konfig + systemd-Unit + Doku

**Files:**
- Create: `deploy/mayring-research-worker.service`
- Create: `docs/RESEARCH_WORKER.md`

> Der cloud-pull-Modus existiert bereits (`pi_worker._cloud_loop`, opt-in via `MAYRING_API_URL`).
> Hier nur Run-Konfiguration + Persistenz.

- [ ] **Step 1: systemd-user-Unit**

```ini
# deploy/mayring-research-worker.service  → ~/.config/systemd/user/
[Unit]
Description=MayringCoder Research Worker (A2A cloud-pull)
After=network-online.target

[Service]
Environment=MAYRING_API_URL=https://mcp.linn.games
Environment=OLLAMA_URL=http://localhost:11434
Environment=PI_WORKER_CAPABILITIES=research
Environment=PI_WORKER_MODEL=qwen3.5:9b
Environment=PI_NUM_PREDICT=6000
Environment=PI_WEB_FETCH_ALLOWLIST=*
ExecStart=%h/miniconda3/bin/python -m mayring_pi_agent.pi_worker
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

> **Verify:** `python -m mayring_pi_agent.pi_worker` als Entry-Point — falls keiner existiert,
> kleinen `__main__`-Block in pi_worker.py ergänzen, der `start_cloud_worker()` ruft + blockt.
> `PI_WORKER_MODEL`/`PI_WEB_FETCH_ALLOWLIST=*` (unrestricted für lokalen Worker, Hook-Kommentar Z109)
> ggf. in pi_worker/pi.py respektieren — prüfen, sonst dort verdrahten.

- [ ] **Step 2: Run-Doku** `docs/RESEARCH_WORKER.md` (Install: Unit kopieren, `systemctl --user
  daemon-reload && systemctl --user enable --now mayring-research-worker`; manueller Test:
  `MAYRING_API_URL=... PI_WORKER_CAPABILITIES=research python -m mayring_pi_agent.pi_worker`).

- [ ] **Step 3: Commit** — `git commit -m "feat(worker): research cloud-worker run config + systemd"`

---

## Phase 3 — Cloud-A2A-Gateway (MayringCoder)

### Task 4: `PiJobsTaskStore` (pi_jobs → A2A-Task-Mapping)

**Files:**
- Create: `src/api/a2a_relay.py`
- Test: `tests/test_a2a_relay.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_a2a_relay.py
import asyncio
from pathlib import Path
from mayring_pi_agent import pi_jobs
from src.api.a2a_relay import PiJobsTaskStore


def test_taskstore_maps_completed_job(tmp_path):
    db = tmp_path / "jobs.db"
    job = pi_jobs.insert_cloud_job("recherchiere X", workspace_id="ws1",
                                   capability_required="research", db_path=db)
    pi_jobs.complete_job(job.job_id, {"text": "ERGEBNIS 42"}, db_path=db)
    store = PiJobsTaskStore(db_path=db)
    task = asyncio.run(store.get(job.job_id, context=None))
    assert task is not None
    assert task.id == job.job_id
    from a2a.types import TaskState
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert "ERGEBNIS 42" in str(task)
```

- [ ] **Step 2: Run, verify FAIL** — `ModuleNotFoundError: src.api.a2a_relay`

- [ ] **Step 3: Implement `PiJobsTaskStore`**

```python
# src/api/a2a_relay.py
from __future__ import annotations
import json
from pathlib import Path

from a2a.helpers import new_task, new_text_part
from a2a.server.tasks import TaskStore
from a2a.types import Message, Task, TaskState
from mayring_pi_agent import pi_jobs

_STATE = {
    "queued": TaskState.TASK_STATE_SUBMITTED,
    "running": TaskState.TASK_STATE_WORKING,
    "completed": TaskState.TASK_STATE_COMPLETED,
    "failed": TaskState.TASK_STATE_FAILED,
}
_USER = Message.DESCRIPTOR.fields_by_name["role"].enum_type.values_by_name["ROLE_AGENT"].number


class PiJobsTaskStore(TaskStore):
    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path

    async def get(self, task_id: str, context=None) -> Task | None:
        job = pi_jobs.get_job(task_id, db_path=self._db_path)
        if job is None:
            return None
        state = _STATE.get(job.status, TaskState.TASK_STATE_WORKING)
        task = new_task(job.job_id, job.job_id, state)
        if job.status == "completed" and job.result_json:
            try:
                text = json.loads(job.result_json).get("text", job.result_json)
            except Exception:
                text = job.result_json
            msg = Message(message_id=job.job_id, context_id=job.job_id,
                          task_id=job.job_id, role=_USER, parts=[new_text_part(text)])
            task.status.message.CopyFrom(msg)
        elif job.status == "failed" and job.error:
            task.status.message.CopyFrom(Message(
                message_id=job.job_id, context_id=job.job_id, task_id=job.job_id,
                role=_USER, parts=[new_text_part(job.error)]))
        return task

    async def save(self, task: Task, context=None) -> None:
        return None  # pi_jobs ist die authoritative Quelle

    async def delete(self, task_id: str, context=None) -> None:
        return None

    async def list(self, context=None):
        return []
```

> **Verify:** `new_task(task_id, context_id, state)` + `task.status.message.CopyFrom` gegen
> a2a-sdk 1.1.0 prüfen (Task.status ist `TaskStatus` mit `message`-Feld). Bei Bedarf
> `task.status.MergeFrom(...)`.

- [ ] **Step 4: Run, verify PASS** — `cd ~/Desktop/MayringCoder && python3 -m pytest tests/test_a2a_relay.py::test_taskstore_maps_completed_job -q`

- [ ] **Step 5: Commit** — `git commit -m "feat(a2a-relay): PiJobsTaskStore status mapping"`

### Task 5: `RelayAgentExecutor` (enqueuet cloud-job statt inline)

**Files:** Modify `src/api/a2a_relay.py`; Test `tests/test_a2a_relay.py`

- [ ] **Step 1: Failing test**

```python
def test_relay_executor_enqueues_cloud_job(tmp_path):
    import asyncio
    from src.api.a2a_relay import RelayAgentExecutor
    from mayring_pi_agent import pi_jobs

    db = tmp_path / "jobs.db"

    class _Q:
        def __init__(self): self.events = []
        async def enqueue_event(self, e): self.events.append(e)

    class _Ctx:
        task_id = "ignored"; context_id = "ctx"
        current_task = None
        def get_user_input(self, d="\n"): return "recherchiere Quantencomputing"

    ex = RelayAgentExecutor(workspace_id="ws1", model="qwen3.5:9b",
                            capability="research", db_path=db)
    asyncio.run(ex.execute(_Ctx(), _Q()))
    recent = pi_jobs.list_recent(db_path=db)
    assert any(j.scope == "cloud" and j.capability_required == "research"
               and "Quantencomputing" in j.task_text for j in recent)
```

- [ ] **Step 2: Run, verify FAIL** — `cannot import name 'RelayAgentExecutor'`

- [ ] **Step 3: Implement `RelayAgentExecutor`** (in a2a_relay.py)

```python
from a2a.server.agent_execution import AgentExecutor
from a2a.server.tasks import TaskUpdater


class RelayAgentExecutor(AgentExecutor):
    def __init__(self, workspace_id: str, model: str, capability: str = "research",
                 db_path=None):
        self._ws = workspace_id
        self._model = model
        self._cap = capability
        self._db_path = db_path

    async def execute(self, context, event_queue) -> None:
        text = context.get_user_input()
        job = pi_jobs.insert_cloud_job(
            text, workspace_id=self._ws, model=self._model,
            capability_required=self._cap, db_path=self._db_path,
        )
        # Task-id == job_id, damit der Client mit dieser id pollt.
        await event_queue.enqueue_event(
            new_task(job.job_id, job.job_id, TaskState.TASK_STATE_SUBMITTED)
        )
        updater = TaskUpdater(event_queue, job.job_id, job.job_id)
        await updater.start_work()  # WORKING — Worker rechnet async, kein complete hier

    async def cancel(self, context, event_queue) -> None:
        return None
```

> **Verify:** `pi_jobs.list_recent` Signatur (Z366). `insert_cloud_job` akzeptiert `model=` (Z211 ja).
> Workspace-Default: in Prod `workspace_id` aus dem JWT (`get_workspace`), hier injiziert.

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit** — `git commit -m "feat(a2a-relay): RelayAgentExecutor enqueues cloud research job"`

### Task 6: A2A-Routes auf die App mounten + Live-Card

**Files:** Modify `src/api/a2a_relay.py` (+`register_a2a_relay`), `src/api/server.py`; Test `tests/test_a2a_relay.py`

- [ ] **Step 1: Failing integration test**

```python
def test_agent_card_served_with_research_skill():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.a2a_relay import register_a2a_relay

    app = FastAPI()
    register_a2a_relay(app, base_url="http://testserver", model="qwen3.5:9b")
    r = TestClient(app).get("/.well-known/agent-card.json")
    assert r.status_code == 200
    body = r.json()
    assert any(s["id"] == "deep-research" for s in body["skills"])
```

- [ ] **Step 2: Run, verify FAIL** — `cannot import name 'register_a2a_relay'`

- [ ] **Step 3: Implement `register_a2a_relay`** (reuse a2a-sdk routes + a research AgentCard)

```python
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (add_a2a_routes_to_fastapi, create_agent_card_routes,
                               create_jsonrpc_routes)
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils import DEFAULT_RPC_URL, TransportProtocol


def _research_card(base_url: str, model: str) -> AgentCard:
    url = base_url.rstrip("/") + "/"
    return AgentCard(
        name="MayringCoder Research Worker",
        description=f"Deep-research agent ({model}) — web search + cloud memory, laptop-powered.",
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"], default_output_modes=["text/plain"],
        supported_interfaces=[AgentInterface(url=url, protocol_binding=TransportProtocol.JSONRPC)],
        skills=[AgentSkill(id="deep-research", name="Deep Research",
                           description="Mehrstufige Web- + Memory-Recherche, async (lange Aufträge).",
                           tags=["research", "web", "memory"])],
    )


def register_a2a_relay(app, *, base_url: str, model: str, workspace_id: str = "default",
                       db_path=None) -> AgentCard:
    card = _research_card(base_url, model)
    executor = RelayAgentExecutor(workspace_id=workspace_id, model=model, db_path=db_path)
    handler = DefaultRequestHandler(agent_executor=executor,
                                    task_store=PiJobsTaskStore(db_path=db_path), agent_card=card)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, DEFAULT_RPC_URL),
    )
    return card
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Wire into server.py** (nach den `include_router`-Zeilen ~Z81)

```python
# A2A research-relay (Langdock → cloud queue → laptop worker)
if os.getenv("MAYRING_A2A_RELAY_ENABLED", "1") == "1":
    from src.api.a2a_relay import register_a2a_relay
    from src.api.routes.devices import _job_db_path
    register_a2a_relay(
        app,
        base_url=os.getenv("MAYRING_A2A_BASE_URL", "https://mcp.linn.games"),
        model=os.getenv("MAYRING_A2A_MODEL", "qwen3.5:9b"),
        workspace_id=os.getenv("MAYRING_A2A_WORKSPACE_ID", "019e14d6"),  # = Worker-JWT-Workspace
        db_path=_job_db_path(),  # MUSS = devices.py claim-DB (sonst findet Worker den Job nie)
    )
```

> **KRITISCH (Workspace-Konsistenz):** Der Worker claimt mit der workspace_id seines JWT
> (`hook.jwt` → 019e14d6, [[project_mayringcoder_workspace_model]]). Der Gateway MUSS Jobs unter
> DERSELBEN workspace_id anlegen, sonst matcht `claim_cloud_next` (workspace-gefiltert) nie. Beide
> per Env auf 019e14d6 pinnen. (Per-Request-Workspace aus Langdocks JWT = Multi-Tenant-Erweiterung
> später; MVP = single-user, fix gepinnt.)

> **Verify:** JSON-RPC liegt auf `/` (DEFAULT_RPC_URL) — kollidiert mit dem MCP-default-upstream
> in nginx (`location /`). LÖSUNG: rpc_url auf `/a2a` setzen (`create_jsonrpc_routes(handler, "/a2a")`)
> UND `AgentInterface(url=base_url+"/a2a")` — damit Langdock JSON-RPC an `/a2a` schickt, nicht an `/`.
> Card bleibt unter `/.well-known/agent-card.json`. Test + base_url entsprechend anpassen.

- [ ] **Step 6: Run full file** — `python3 -m pytest tests/test_a2a_relay.py -q` → PASS

- [ ] **Step 7: Commit** — `git commit -m "feat(a2a-relay): mount research agent-card + jsonrpc on /a2a"`

### Task 7: TTL-Sweep für unclaimed cloud-jobs

**Files:** Modify `mayring_pi_agent/pi_jobs.py` (+`fail_stale_cloud_jobs`), `src/api/routes/devices.py`; Test `tests/test_pi_jobs.py`

- [ ] **Step 1: Failing test** (in mayring-pi-agent tests)

```python
def test_fail_stale_cloud_jobs_marks_old_queued(db):
    j = pi_jobs.insert_cloud_job("alt", capability_required="research", db_path=db)
    # created_at künstlich in die Vergangenheit setzen:
    import sqlite3
    with sqlite3.connect(db) as c:
        c.execute("UPDATE pi_jobs SET created_at=? WHERE job_id=?",
                  ("2000-01-01T00:00:00+00:00", j.job_id))
    n = pi_jobs.fail_stale_cloud_jobs(max_age_s=600, db_path=db)
    assert n == 1
    assert pi_jobs.get_job(j.job_id, db_path=db).status == "failed"
```

- [ ] **Step 2: Run, verify FAIL** — `has no attribute 'fail_stale_cloud_jobs'`

- [ ] **Step 3: Implement `fail_stale_cloud_jobs`** in pi_jobs.py

```python
def fail_stale_cloud_jobs(max_age_s: float = 1800, *, db_path: Path | None = None) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_s)).isoformat()
    with _conn(db_path) as c:
        cur = c.execute(
            "UPDATE pi_jobs SET status='failed', error='ttl: no worker claimed in time', "
            "finished_at=? WHERE status='queued' AND scope='cloud' AND created_at < ?",
            (_now_iso(), cutoff),
        )
        return cur.rowcount
```

> **Verify:** `datetime/timezone/timedelta` import (oben in pi_jobs.py ergänzen falls fehlt).

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Call from claim endpoint** (devices.py `pi_task_claim_cloud`, vor dem claim — billig, kein Cron nötig)

```python
    pi_jobs.fail_stale_cloud_jobs(
        max_age_s=float(os.getenv("MAYRING_CLOUD_JOB_TTL_S", "1800")),
        db_path=_job_db_path(),
    )
```

- [ ] **Step 6: Commit** — `git commit -m "feat(queue): TTL-fail unclaimed cloud jobs"`

---

## Phase 4 — Infra (app.linn.games)

### Task 8: SearXNG-Container

**Files:**
- Create: `docker/mayring/searxng/settings.yml`
- Modify: `docker/mayring/docker-compose*.yml` (mayring-Stack)

- [ ] **Step 1: settings.yml** (json-Format + limiter aus für interne Nutzung)

```yaml
# docker/mayring/searxng/settings.yml
use_default_settings: true
server:
  secret_key: "${SEARXNG_SECRET}"
  limiter: false
search:
  formats: [html, json]
```

- [ ] **Step 2: compose-Service** (im mayring-Stack, gleiches Docker-Netz wie nginx/mayring-api)

```yaml
  searxng:
    image: searxng/searxng:latest
    container_name: mayring-searxng
    environment:
      - SEARXNG_SECRET=${SEARXNG_SECRET}
    volumes:
      - ./searxng/settings.yml:/etc/searxng/settings.yml:ro
    restart: unless-stopped
    expose:
      - "8080"
```

- [ ] **Step 3: `.env`** — `SEARXNG_SECRET=<random>` ergänzen (u-server `~/app.linn.games/.env`).

- [ ] **Step 4: Commit** — `git commit -m "feat(infra): SearXNG container for research worker"`

### Task 9: nginx-Allowlist + `/searxng/`-location

**Files:** Modify `docker/mayring/nginx/mcp.conf`

- [ ] **Step 1: `/searxng/`-location** (vor der `location /`-default, hinter JWT)

```nginx
    location /searxng/ {
        set $searxng_upstream http://searxng:8080/;
        proxy_pass $searxng_upstream;
        proxy_set_header Host $host;
    }
```

- [ ] **Step 2: A2A-Pfad in die Regex-Allowlist** (Z64) — `a2a` ergänzen, damit `/a2a` zu
  mayring-api proxyt (nicht zum MCP-default):

```nginx
    location ~ ^/(health|stats|search|ingest|...|repo-events|a2a)(/|_|$) {
        set $mayring_api_upstream http://mayring-api:8090;
        proxy_pass $mayring_api_upstream;
    }
```

> **Falle [[project_nginx_mcp_conf_sot]]:** Diese Datei ist die Prod-SoT; Deploy via
> `docker/mayring/**` → „Deploy MayringCoder" → nginx recreated. `/.well-known/agent-card.json`
> wird von der `location /`-default an den MCP-upstream geroutet → FALSCH. Eigene location
> `location = /.well-known/agent-card.json { proxy_pass mayring-api; }` ergänzen.

- [ ] **Step 3: location für die Agent-Card** ergänzen

```nginx
    location = /.well-known/agent-card.json {
        set $mayring_api_upstream http://mayring-api:8090;
        proxy_pass $mayring_api_upstream;
    }
```

- [ ] **Step 4: Commit** — `git commit -m "feat(nginx): expose /a2a + /searxng + agent-card"`

---

## Phase 5 — Deploy + End-to-End-Live-Proof

### Task 10: Deploy + realer Langdock-Round-trip (PFLICHT — kein „gebaut aber nie genutzt")

- [ ] **Step 1:** mayring-pi-agent pushen (main) + MayringCoder vendor-Submodul bumpen + pushen.
- [ ] **Step 2:** app.linn.games `docker/mayring/**` pushen → „Deploy MayringCoder" → nginx+mayring-api+searxng recreated. `/health`=200 abwarten.
- [ ] **Step 3:** Laptop-Worker starten: `systemctl --user enable --now mayring-research-worker`; `journalctl --user -u mayring-research-worker -f` zeigt cloud-poll.
- [ ] **Step 4:** A2A-Smoke (lokales Script wie M2b, aber gegen Prod):
  `create_client("https://mcp.linn.games")` mit Bearer-JWT → `message/send` „Recherchiere die
  aktuellen A2A-Protokoll-Implementierungen" → `tasks/get`-Poll bis COMPLETED.
- [ ] **Step 5:** VERIFY: Worker-Log zeigt `web_search`-Calls (SearXNG-Treffer) + `search_memory`;
  A2A-Antwort enthält echte Recherche. Beweis dokumentieren (wie M2b).
- [ ] **Step 6:** Langdock: AgentCard-URL `https://mcp.linn.games/.well-known/agent-card.json` +
  Bearer eintragen → realer Recherche-Auftrag aus Langdock → Ergebnis. (User-Schritt; Anleitung in `docs/RESEARCH_WORKER.md`.)
- [ ] **Step 7:** Memory-Update: project_dev_environment_pi_agent.md „M3 Research-Relay live".

---

## Risiken / Fallen (aus Recon)

- **JSON-RPC auf `/` kollidiert** mit nginx MCP-default → rpc_url `/a2a` (Task 6 Verify).
- **agent-card well-known** wird sonst zum MCP-upstream geroutet → eigene nginx-location (Task 9).
- **`insert_cloud_job` hat aktuell 0 Caller** — Gateway ist der erste Ersteller; DB-Pfad MUSS der
  sein, den `devices.py::claim_cloud_next` liest (`_job_db_path()` → `MEMORY_DB_PATH`). In Prod also
  `register_a2a_relay(db_path=_job_db_path())` übergeben (server.py), NICHT default.
- **Worker-Job-Steal:** cap=research + registry-authoritative caps (devices.py) → Prod-Pi unberührt.
- **`PI_WEB_FETCH_ALLOWLIST=*`** nur für den lokalen Research-Worker (unrestricted), NICHT in Prod-Pi.
- **a2a-sdk = Protobuf** (siehe M2b-Memory): supported_interfaces/protocol_binding, new_task vor
  StatusUpdate, role-Enum-Ints, Antwort unter task.status.message.
```
