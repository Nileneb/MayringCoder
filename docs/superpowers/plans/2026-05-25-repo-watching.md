# Repo-Watching Implementation Plan (Subsystem C+D)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reusable GitHub Action per repo POSTs push/CI/security events to a new `POST /repo-events`, which re-ingests the repo on push and logs CI/security events as `hook_events` rows + lightweight searchable `repo_event` chunks.

**Architecture:** New `src/api/routes/repo_events.py` (service-token auth) resolves repo→`projects`→workspace, then dispatches by event_type: push→enqueue `/populate`; workflow_run/security→`hook_events` (reuse `hook_type`+`payload`, NO schema migration) + a deterministically-igio-classified `repo_event` chunk. A reusable workflow in `mayring-claude-plugin` is invoked by each watched repo and `continue-on-error`-POSTs the event.

**Tech Stack:** Python 3.13, FastAPI, SQLite (DBAdapter), pytest; GitHub Actions (reusable `workflow_call`).

**Spec:** `docs/superpowers/specs/2026-05-25-repo-watching-design.md`

**Repos:** MayringCoder `/home/nileneb/Desktop/MayringCoder` (master) · plugin `/home/nileneb/Desktop/mayring-claude-plugin` (main, reusable Action) · app.linn.games `/home/nileneb/Desktop/WebDev/app.linn.games` (main, first watched repo).

---

## File Structure

- `src/api/routes/repo_events.py` — **create**: `POST /repo-events` + `RepoEventRequest` + workspace resolution + dispatch + `_record_repo_event` + `_repo_event_chunk`.
- `src/api/server.py` — **modify**: `include_router(repo_events.router)`.
- `src/api/routes/jobs.py` — **modify**: extract `enqueue_populate(repo, workspace_id) -> str` (DRY: `/populate` + `/repo-events` both call it); debounce.
- `tools/smoke_test_production.py` — **modify**: `check_repo_event_surfaces` + register.
- `mayring-claude-plugin/.github/workflows/repo-watch.yml` — **create**: reusable workflow.
- `app.linn.games/.github/workflows/mayring-watch.yml` — **create**: the first per-repo caller (outcome guarantee).
- Tests: `tests/test_repo_events.py`.

---

## Task 1: `enqueue_populate` helper (DRY the populate trigger)

**Files:**
- Modify: `src/api/routes/jobs.py` (the `trigger_populate` handler ~line 248)
- Test: `tests/test_repo_events.py` (create)

**Context:** `trigger_populate` does `job_id = _make_job(workspace_id)` then builds `args` (with `--populate-memory`, `--memory-categorize`, `--repo`, `--workspace-id`) and `asyncio.create_task(_run_with_v2_postingest(job_id, args, workspace_id, request.repo))`. Extract that into a reusable helper so `/repo-events` can trigger the same re-ingest, with a debounce.

- [ ] **Step 1: Write the failing test** `tests/test_repo_events.py`:

```python
from __future__ import annotations
import asyncio
from unittest.mock import patch


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_enqueue_populate_starts_a_job(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from src.api.routes import jobs
    with patch("src.api.routes.jobs._run_with_v2_postingest") as m:
        jid = jobs.enqueue_populate("https://github.com/a/b", "ws-1")
    assert jid and jq.get_job(jid) is not None
    assert m.called  # the v2-chain background task was scheduled


def test_enqueue_populate_debounces_running_repo(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json")
    jq._JOBS.clear()
    from src.api.routes import jobs
    with patch("src.api.routes.jobs._run_with_v2_postingest"):
        jid1 = jobs.enqueue_populate("https://github.com/a/b", "ws-1")
        jid2 = jobs.enqueue_populate("https://github.com/a/b", "ws-1")  # same repo, still running
    assert jid2 == jid1, "a populate already running for this repo must be reused, not duplicated"
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=.:core python3 -m pytest tests/test_repo_events.py -q`
Expected: FAIL — `module 'src.api.routes.jobs' has no attribute 'enqueue_populate'`.

- [ ] **Step 3: Implement `enqueue_populate`** in `src/api/routes/jobs.py` (add near `trigger_populate`):

```python
def enqueue_populate(repo: str, workspace_id: str) -> str:
    """Enqueue a repo re-ingest (populate + v2-chain) and return the job id.
    Debounce: if a populate job for the same repo is still running in this
    workspace, reuse it instead of spawning a storm (rapid pushes).

    WHY(repo-watching): shared by POST /populate and POST /repo-events so a
    push event re-ingests exactly like a manual populate."""
    from src.api.job_queue import _load_jobs, _JOBS
    merged = {**_load_jobs(), **_JOBS}
    for j in merged.values():
        if (j.get("workspace_id") == workspace_id
                and j.get("repo") == repo
                and j.get("status") in ("started", "running")):
            return j["job_id"]
    job_id = _make_job(workspace_id)
    _JOBS[job_id]["repo"] = repo  # tag for debounce + UI
    args = [
        "--repo", repo, "--populate-memory", "--memory-categorize",
        "--workspace-id", workspace_id,
    ]
    asyncio.create_task(_run_with_v2_postingest(job_id, args, workspace_id, repo))
    return job_id
```

Then refactor `trigger_populate` to call it (keep its response shape):
```python
    job_id = enqueue_populate(request.repo, workspace_id)
    return {"job_id": job_id, "status": "started"}
```
(Match the existing return shape — read `trigger_populate`'s current return and preserve any extra fields.)

- [ ] **Step 4: Run to verify PASS** — `PYTHONPATH=.:core python3 -m pytest tests/test_repo_events.py -q` → 2 passed.
- [ ] **Step 5: Run the jobs/populate regression** — `PYTHONPATH=.:core python3 -m pytest tests/test_v2_ops_endpoints.py -q` → all pass (trigger_populate still works).
- [ ] **Step 6: Commit** — `git add src/api/routes/jobs.py tests/test_repo_events.py && git commit -m "feat(jobs): enqueue_populate helper with per-repo debounce"`

---

## Task 2: `POST /repo-events` — model, auth, workspace resolution, push dispatch

**Files:**
- Create: `src/api/routes/repo_events.py`
- Modify: `src/api/server.py` (register router); `src/api/routes/models.py` (RepoEventRequest)
- Test: `tests/test_repo_events.py` (extend)

**Context:** Service-token/admin auth (reuse `get_token_info` + `_is_privileged` from `src/api/auth.py`). Resolve repo→workspace via `projects WHERE source_type='github' AND source_ref=<repo>`; match-or-create if absent (mirror `src/api/routes/projects.py`). Reuse `get_conn as _conn`.

- [ ] **Step 1: Add `RepoEventRequest`** to `src/api/routes/models.py`:

```python
class RepoEventRequest(BaseModel):
    event_type: str               # 'push' | 'workflow_run' | 'security'
    repo: str                     # repo URL (matches projects.source_ref)
    sha: str | None = None
    ref: str | None = None
    conclusion: str | None = None  # workflow_run: success|failure|...
    workflow: str | None = None
    severity: str | None = None    # security: low|moderate|high|critical
    summary: str | None = None
    url: str | None = None
```

- [ ] **Step 2: Write the failing test** (extend `tests/test_repo_events.py`):

```python
def _client(monkeypatch, tmp_path):
    import src.api.job_queue as jq
    monkeypatch.setattr(jq, "_JOBS_STATE_FILE", tmp_path / "jobs.json"); jq._JOBS.clear()
    from fastapi.testclient import TestClient
    from src.api import server as srv
    from src.api import auth as auth_module
    from src.api.jwt_auth import TokenInfo
    async def _svc():
        return TokenInfo(workspace_id="system", scopes=("*",))
    srv.app.dependency_overrides[auth_module.get_token_info] = _svc
    return TestClient(srv.app)


def test_repo_events_push_enqueues_populate(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    try:
        with patch("src.api.routes.repo_events.enqueue_populate", return_value="job-1") as m:
            r = client.post("/repo-events",
                            json={"event_type": "push", "repo": "https://github.com/a/b",
                                  "sha": "abc"},
                            headers={"Authorization": "Bearer t"})
        assert r.status_code == 200
        assert m.called
        assert m.call_args.args[0] == "https://github.com/a/b"
    finally:
        from src.api import server as srv
        srv.app.dependency_overrides.clear()
```

- [ ] **Step 3: Run to verify it fails** — `pytest tests/test_repo_events.py::test_repo_events_push_enqueues_populate -q` → FAIL (404/no route).

- [ ] **Step 4: Implement** `src/api/routes/repo_events.py`:

```python
"""POST /repo-events — the reusable GitHub Action posts repo push/CI/security
events here. Push re-ingests the newest version; CI/security are logged in
hook_events + a lightweight searchable repo_event chunk (recall + IGIO-Lens).

WHY(repo-watching C+D): closes Audit G18 (only MayringCoder auto-ingested) and
gives every watched repo's CI/security a memory presence."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import get_token_info
from src.api.dependencies import get_conn as _conn
from src.api.jwt_auth import TokenInfo
from src.api.routes.jobs import enqueue_populate
from src.api.routes.models import RepoEventRequest

router = APIRouter()


def _is_privileged(info: TokenInfo) -> bool:
    return info.is_admin or "*" in info.scopes


def _resolve_workspace(conn, repo: str) -> str:
    """repo-url → projects.workspace_id; match-or-create under 'system' if unknown."""
    row = conn.execute(
        "SELECT workspace_id FROM projects WHERE source_type='github' AND source_ref=?",
        (repo,),
    ).fetchone()
    if row is not None:
        return row[0]
    # match-or-create: register the repo as a system project so future events
    # (and a later workspace re-assignment) have a home. Never reject.
    now = datetime.now(timezone.utc).isoformat()
    pid = "prj_" + __import__("uuid").uuid4().hex[:16]
    conn.execute(
        "INSERT INTO projects (id, workspace_id, name, source_type, source_ref, created_at, updated_at) "
        "VALUES (?, 'system', ?, 'github', ?, ?, ?)",
        (pid, repo.rsplit("/", 1)[-1], repo, now, now),
    )
    conn.commit()
    return "system"


_AXIS = {  # deterministic, NO LLM in the hot path
    ("workflow_run", "failure"): "issue",
    ("workflow_run", "success"): "outcome",
    ("security", None): "issue",
}


@router.post("/repo-events")
async def repo_events(req: RepoEventRequest, info: TokenInfo = Depends(get_token_info)) -> dict:
    if not _is_privileged(info):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="repo-events requires a service/admin token")
    conn = _conn()
    workspace_id = _resolve_workspace(conn, req.repo)

    if req.event_type == "push":
        job_id = enqueue_populate(req.repo, workspace_id)
        return {"ok": True, "action": "populate", "job_id": job_id, "workspace_id": workspace_id}

    # workflow_run | security → hook_events log + repo_event chunk
    hook_type = "repo_ci" if req.event_type == "workflow_run" else "repo_security"
    _record_repo_event(conn, workspace_id, hook_type, req)
    axis = _AXIS.get((req.event_type, req.conclusion)) or _AXIS.get((req.event_type, None)) or ""
    _repo_event_chunk(conn, workspace_id, req, axis)
    return {"ok": True, "action": hook_type, "workspace_id": workspace_id, "igio_axis": axis}
```

(The `_record_repo_event` + `_repo_event_chunk` helpers are added in Tasks 3 and 4 — for now stub them as `def _record_repo_event(*a): pass` / `def _repo_event_chunk(*a): pass` so push works; Tasks 3/4 fill them with their tests.)

- [ ] **Step 5: Register the router** in `src/api/server.py` (next to the other `include_router` calls ~line 77):
```python
from src.api.routes import repo_events as _repo_events
app.include_router(_repo_events.router)
```

- [ ] **Step 6: Run to verify PASS** — `pytest tests/test_repo_events.py -q` → all pass.
- [ ] **Step 7: Commit** — `git add src/api/routes/repo_events.py src/api/routes/models.py src/api/server.py tests/test_repo_events.py && git commit -m "feat(repo-events): POST /repo-events — push→populate + workspace resolve"`

---

## Task 3: `_record_repo_event` — hook_events log (no migration)

**Files:**
- Modify: `src/api/routes/repo_events.py`
- Test: `tests/test_repo_events.py` (extend)

**Context:** `hook_events(id, workspace_id, device_id, hook_type, fired_at, payload)` — store the event JSON in `payload`, NO new columns. Idempotency: skip if a row with the same `(hook_type, payload.sha, payload.workflow)` already exists for the workspace.

- [ ] **Step 1: Write the failing test:**

```python
def test_workflow_run_records_hook_event(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    from src.api.dependencies import get_conn
    try:
        r = client.post("/repo-events", headers={"Authorization": "Bearer t"},
            json={"event_type": "workflow_run", "repo": "https://github.com/a/b",
                  "sha": "deadbeef", "conclusion": "failure", "workflow": "ci"})
        assert r.status_code == 200 and r.json()["action"] == "repo_ci"
        rows = get_conn().execute(
            "SELECT hook_type, payload FROM hook_events WHERE hook_type='repo_ci'").fetchall()
        assert any('"sha": "deadbeef"' in row[1] for row in rows)
        # idempotent: same event again → still one row
        client.post("/repo-events", headers={"Authorization": "Bearer t"},
            json={"event_type": "workflow_run", "repo": "https://github.com/a/b",
                  "sha": "deadbeef", "conclusion": "failure", "workflow": "ci"})
        rows2 = get_conn().execute(
            "SELECT id FROM hook_events WHERE hook_type='repo_ci' AND payload LIKE '%deadbeef%'").fetchall()
        assert len(rows2) == 1, "re-delivered event must not duplicate"
    finally:
        from src.api import server as srv; srv.app.dependency_overrides.clear()
```

- [ ] **Step 2: Run → fails** (the stub does nothing → no row).

- [ ] **Step 3: Implement** `_record_repo_event` in `repo_events.py` (replace the stub):

```python
def _record_repo_event(conn, workspace_id: str, hook_type: str, req: RepoEventRequest) -> None:
    payload = json.dumps({
        "repo": req.repo, "sha": req.sha, "ref": req.ref,
        "conclusion": req.conclusion, "workflow": req.workflow,
        "severity": req.severity, "summary": req.summary, "url": req.url,
    }, default=str)
    # idempotency: same (hook_type, sha, workflow) for this workspace → skip
    existing = conn.execute(
        "SELECT 1 FROM hook_events WHERE workspace_id=? AND hook_type=? "
        "AND payload LIKE ? AND payload LIKE ? LIMIT 1",
        (workspace_id, hook_type, f'%"sha": "{req.sha}"%', f'%"workflow": "{req.workflow}"%'),
    ).fetchone()
    if existing is not None:
        return
    conn.execute(
        "INSERT INTO hook_events (workspace_id, device_id, hook_type, fired_at, payload) "
        "VALUES (?, 'github-action', ?, ?, ?)",
        (workspace_id, hook_type, datetime.now(timezone.utc).isoformat(), payload),
    )
    conn.commit()
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat(repo-events): log CI/security to hook_events (idempotent, no migration)"`

---

## Task 4: `_repo_event_chunk` — lightweight searchable chunk

**Files:**
- Modify: `src/api/routes/repo_events.py`
- Test: `tests/test_repo_events.py` (extend)

**Context:** Insert a `source` (`source_type='repo_event'`) + a `chunk` with the one-line text + `igio_axis` set directly, workspace-scoped, NO LLM. Use `mayring_core.memory.store` `upsert_source` + `insert_chunk` and `mayring_core.memory.schema` `Source`/`Chunk` (the canonical low-level path — see `tests/test_memory_retrieval.py` `_make_source`/`_make_chunk` for the exact constructor fields).

- [ ] **Step 1: Write the failing test:**

```python
def test_workflow_run_failure_creates_issue_chunk(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    from src.api.dependencies import get_conn
    try:
        client.post("/repo-events", headers={"Authorization": "Bearer t"},
            json={"event_type": "workflow_run", "repo": "https://github.com/a/b",
                  "sha": "c0ffee", "conclusion": "failure", "workflow": "tests"})
        rows = get_conn().execute(
            "SELECT text, igio_axis FROM chunks c JOIN sources s ON c.source_id=s.source_id "
            "WHERE s.source_type='repo_event'").fetchall()
        assert rows, "a repo_event chunk must be created"
        assert any(r[1] == "issue" and "tests" in r[0] for r in rows)
    finally:
        from src.api import server as srv; srv.app.dependency_overrides.clear()
```

- [ ] **Step 2: Run → fails** (stub).

- [ ] **Step 3: Implement** `_repo_event_chunk` (replace the stub). **First read** `tests/test_memory_retrieval.py` `_make_source`/`_make_chunk` + `core/mayring_core/memory/schema.py` for the exact `Source`/`Chunk` required fields and `store.upsert_source`/`insert_chunk` signatures, then:

```python
def _repo_event_chunk(conn, workspace_id: str, req: RepoEventRequest, axis: str) -> None:
    from mayring_core.memory.store import upsert_source, insert_chunk
    from mayring_core.memory.schema import Source, Chunk
    now = datetime.now(timezone.utc).isoformat()
    if req.event_type == "workflow_run":
        text = f"CI {req.workflow or ''} {req.conclusion or ''} on {req.repo}@{(req.sha or '')[:8]}"
    else:
        text = f"Security {req.severity or ''}: {req.summary or ''} in {req.repo}"
    sid = f"repo_event:{req.repo}:{req.event_type}:{(req.sha or now)[:12]}"
    src = Source(
        source_id=sid, source_type="repo_event", repo=req.repo,
        path=req.url or "", branch=req.ref or "", commit=req.sha or "",
        content_hash="sha256:" + Chunk.compute_text_hash(text), captured_at=now,
    )
    upsert_source(conn, src, workspace_id=workspace_id)   # match the real upsert_source signature
    chunk = Chunk(
        chunk_id=Chunk.make_id(sid, 0, "event"), source_id=sid, chunk_level="event",
        ordinal=0, text=text, text_hash=Chunk.compute_text_hash(text),
        created_at=now, igio_axis=axis, igio_confidence=0.9 if axis else 0.0,
    )
    insert_chunk(conn, chunk, workspace_id=workspace_id)  # match the real insert_chunk signature
```

> If `upsert_source`/`insert_chunk` take `workspace_id` differently (some set it on the dataclass), follow what `tests/test_memory_retrieval.py` does — adjust the calls to the real signatures (do NOT guess; read them). The behavioural assertion (a `repo_event` chunk with the right `igio_axis`) stays.

- [ ] **Step 4: Run → PASS** (issue chunk created). Also add `test_workflow_run_success_creates_outcome_chunk` (conclusion='success' → axis='outcome') and `test_security_event_creates_issue_chunk`.
- [ ] **Step 5: Run the broader suite** — `PYTHONPATH=.:core python3 -m pytest tests/test_repo_events.py tests/test_memory_retrieval.py -q` → all pass.
- [ ] **Step 6: Commit** — `git commit -am "feat(repo-events): lightweight igio-classified repo_event chunk"`

---

## Task 5: Reusable GitHub Action + per-repo caller

**Files:**
- Create: `mayring-claude-plugin/.github/workflows/repo-watch.yml` (reusable `workflow_call`)
- Create: `app.linn.games/.github/workflows/mayring-watch.yml` (first caller — outcome)

- [ ] **Step 1: Create the reusable workflow** `mayring-claude-plugin/.github/workflows/repo-watch.yml`:

```yaml
name: repo-watch
on:
  workflow_call:
    secrets:
      MAYRING_TOKEN: { required: true }
jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      # Loop guard: never react to our own run.
      - if: ${{ github.event_name == 'workflow_run' && github.event.workflow_run.name == 'mayring-watch' }}
        run: echo "self-run, skipping" && exit 0
      - name: POST event to MayringCoder (best-effort)
        continue-on-error: true
        env:
          MAYRING_TOKEN: ${{ secrets.MAYRING_TOKEN }}
          API: ${{ vars.MAYRING_API_URL || 'https://mcp.linn.games' }}
          EV: ${{ github.event_name }}
          REPO: ${{ github.server_url }}/${{ github.repository }}
          SHA: ${{ github.sha }}
          REF: ${{ github.ref }}
          CONCL: ${{ github.event.workflow_run.conclusion }}
          WF: ${{ github.event.workflow_run.name }}
        run: |
          case "$EV" in
            push) ETYPE=push ;;
            workflow_run) ETYPE=workflow_run ;;
            *) ETYPE="$EV" ;;
          esac
          curl -sS -m 10 -X POST "$API/repo-events" \
            -H "Authorization: Bearer $MAYRING_TOKEN" -H 'Content-Type: application/json' \
            -d "$(jq -nc --arg e "$ETYPE" --arg r "$REPO" --arg s "$SHA" --arg ref "$REF" \
                     --arg c "$CONCL" --arg w "$WF" \
                     '{event_type:$e, repo:$r, sha:$s, ref:$ref, conclusion:$c, workflow:$w}')" \
            || echo "mayring repo-watch POST failed (non-fatal)"
```

- [ ] **Step 2: Create the per-repo caller** `app.linn.games/.github/workflows/mayring-watch.yml`:

```yaml
name: mayring-watch
on:
  push: { branches: ["**"] }
  workflow_run:
    workflows: ["*"]
    types: [completed]
jobs:
  watch:
    uses: Nileneb/mayring-claude-plugin/.github/workflows/repo-watch.yml@main
    secrets:
      MAYRING_TOKEN: ${{ secrets.MAYRING_TOKEN }}
```

- [ ] **Step 3: Lint both YAMLs** — `python3 -c "import yaml,sys; [yaml.safe_load(open(p)) for p in sys.argv[1:]]; print('yaml ok')" mayring-claude-plugin/.github/workflows/repo-watch.yml app.linn.games/.github/workflows/mayring-watch.yml`
- [ ] **Step 4: Commit** both repos (plugin + app.linn.games) locally:
```bash
cd /home/nileneb/Desktop/mayring-claude-plugin && git add .github/workflows/repo-watch.yml && git commit -m "feat(ci): reusable repo-watch workflow → MayringCoder /repo-events"
cd /home/nileneb/Desktop/WebDev/app.linn.games && git add .github/workflows/mayring-watch.yml && git commit -m "feat(ci): watch this repo via mayring repo-watch"
```

---

## Task 6: Smoke check — a repo_event surfaces

**Files:** Modify `tools/smoke_test_production.py` (add + register `check_repo_event_surfaces`).

- [ ] **Step 1: Add the check:**
```python
def check_repo_event_surfaces(api: str, token: str) -> CheckResult:
    """POST a synthetic workflow_run failure to /repo-events → a hook_events row
    + a repo_event chunk should be created (workspace-scoped)."""
    suffix = int(time.time())
    repo = f"https://github.com/smoke/repo-{suffix}"
    code1, body1, _ = _http("POST", f"{api}/repo-events", token,
        body={"event_type": "workflow_run", "repo": repo, "sha": f"s{suffix}",
              "conclusion": "failure", "workflow": "smoke-ci"}, timeout=12.0)
    if code1 != 200:
        return CheckResult("repo_event_surfaces", False, f"post failed http={code1}: {body1}")
    ok = isinstance(body1, dict) and body1.get("action") == "repo_ci" and body1.get("igio_axis") == "issue"
    return CheckResult("repo_event_surfaces", ok,
        f"action={body1.get('action')} igio_axis={body1.get('igio_axis')} (want repo_ci/issue) marker={suffix}")
```
- [ ] **Step 2: Register** `("repo_event_surfaces", check_repo_event_surfaces),` in `ALL_CHECKS`.
- [ ] **Step 3: Verify import** (`python3 -c "import tools.smoke_test_production as s; print(len(s.ALL_CHECKS))"`, +1). Live run deferred to Task 7.
- [ ] **Step 4: Commit** — `git commit -am "test(smoke): repo_event_surfaces check"`

---

## Task 7: Deploy + wire the first real repo (outcome guarantee)

- [ ] **Step 1: Push** MayringCoder master (Tasks 1–4, 6) → build+deploy. Push plugin main (Task 5 reusable workflow). Push app.linn.games main (Task 5 caller).
- [ ] **Step 2: Add the `MAYRING_TOKEN` secret** to the app.linn.games GitHub repo: `gh secret set MAYRING_TOKEN --repo Nileneb/app.linn.games --body "<MCP_SERVICE_TOKEN value>"` (the human runs this with the real service-token value, or confirms it via `gh secret list`).
- [ ] **Step 3: Live backend verify** — `POST /repo-events` (workflow_run, failure) against prod → `action=repo_ci`, `igio_axis=issue`; a push event → `action=populate` + a job id; confirm a `repo_event` chunk shows in `/stats/igio-lens` (issue column) for the resolved workspace.
- [ ] **Step 4: Real event** — push a trivial commit to app.linn.games → the `mayring-watch` workflow runs → `/repo-events` receives `push` → a populate job appears in `/stats/jobs-history`; when its CI completes → a `repo_ci` event + chunk appears. Verify in the IGIO-Lens.
- [ ] **Step 5: Smoke** — the auto post-deploy-smoke `repo_event_surfaces` is green.
- [ ] **Step 6: Memory** — record that Repo-Watching is live, which repo is wired, and the endpoint/Action.

---

## Out of Scope (YAGNI)
- GitHub webhooks/polling; `projects.watch_enabled` flag; LLM event classification in the hot path; historical backfill; hook_events schema migration (reuse `payload`).
