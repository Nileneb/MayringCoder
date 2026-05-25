# IGIO-Lens Intervention Todos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the agent's `TaskCreate`/`TaskUpdate` task-lists via a PostToolUse hook into `/tasks`, and surface them as the IGIO-Lens intervention column.

**Architecture:** A new plugin PostToolUse hook posts captured todos (idempotent via a new `tasks.external_id`) to MayringCoder `/tasks`. `GET /stats/igio-lens` gains an `intervention.todos` list. The `IgioLens` Livewire view renders that list in the intervention column. `derive_todo` stays dormant (single source).

**Tech Stack:** Python 3.13, FastAPI, SQLite (DBAdapter), pytest; Claude-Code plugin hooks (stdin-JSON, `hooks.json`); Laravel/Livewire (Pest); `tools/smoke_test_production.py`.

**Spec:** `docs/superpowers/specs/2026-05-25-igio-intervention-todos-design.md`

**Repos / canonical clones:** MayringCoder `/home/nileneb/Desktop/MayringCoder` (master) · plugin `/home/nileneb/Desktop/mayring-claude-plugin` (main) · Laravel `/home/nileneb/Desktop/WebDev/app.linn.games` (main).

---

## File Structure

- `mayring-claude-plugin/hooks/task_capture.py` — **create**: PostToolUse hook (spike→full).
- `mayring-claude-plugin/hooks/hooks.json` — **modify**: register PostToolUse matcher.
- `core/mayring_core/memory/store.py` — **modify**: `tasks.external_id` column + index, schema v13→v14.
- `core/mayring_core/memory/tasks.py` — **modify**: `create_task(external_id=...)` upsert.
- `src/api/routes/models.py` — **modify**: `TaskCreateRequest.external_id`.
- `src/api/routes/igio_admin.py` — **modify**: `intervention.todos` in `GET /stats/igio-lens`.
- `tools/smoke_test_production.py` — **modify**: new check `check_intervention_todos_surface`.
- `app.linn.games/app/Services/Mcp/MayringStatsClient.php` + `app/Livewire/Mayring/IgioLens.php` + `resources/views/livewire/mayring/igio-lens.blade.php` — **modify**: render `intervention.todos`.
- Tests: `tests/test_task_external_id.py`, `tests/test_igio_lens_todos.py`, `mayring-claude-plugin/hooks/test_task_capture.py`, Laravel `tests/Feature/Mayring/IgioLensTodosTest.php`.

---

## Task 1: SPIKE — confirm PostToolUse fires for Task* tools + capture payload schema

**Files:**
- Create: `mayring-claude-plugin/hooks/task_capture.py` (spike version)
- Modify: `mayring-claude-plugin/hooks/hooks.json`

This task is exploratory (deploy + observe), not TDD. Its output is the documented payload schema that Task 4 consumes.

- [ ] **Step 1: Write the spike hook** `mayring-claude-plugin/hooks/task_capture.py`:

```python
"""PostToolUse capture (SPIKE): log the raw payload for the agent's todo tools
so we learn the exact schema (tool_name + where id/title/status live) before
building the real capture. Replaced by the full implementation in Task 4."""
import datetime
import json
import os
import sys

_TODO_TOOLS = {"TaskCreate", "TaskUpdate", "TaskGet", "TaskList", "TodoWrite"}
_LOG = os.path.expanduser("~/.config/mayring/task_capture_spike.log")


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        return
    if payload.get("tool_name") not in _TODO_TOOLS:
        return
    try:
        os.makedirs(os.path.dirname(_LOG), exist_ok=True)
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(datetime.datetime.now().isoformat() + " " + json.dumps(payload) + "\n")
    except OSError as e:
        sys.stderr.write(f"[task_capture spike] log failed: {e}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Register the PostToolUse hook** in `mayring-claude-plugin/hooks/hooks.json`. Read the existing file for the exact JSON shape (it already registers `UserPromptSubmit`/`Stop`/`PostCompact`/`SessionStart` with `{"type":"command","command":"python3 ${CLAUDE_PLUGIN_ROOT}/hooks/<file>.py"}`). Add:

```json
"PostToolUse": [
  {
    "matcher": "TaskCreate|TaskUpdate|TaskGet|TaskList|TodoWrite",
    "hooks": [
      {"type": "command", "command": "python3 ${CLAUDE_PLUGIN_ROOT}/hooks/task_capture.py"}
    ]
  }
]
```

- [ ] **Step 3: Commit + reload** (plugin runs from the installed cache — bump version like the prior mistral fix did):

```bash
cd /home/nileneb/Desktop/mayring-claude-plugin
# bump .claude-plugin/plugin.json version (e.g. 1.1.1 → 1.1.2)
git add hooks/task_capture.py hooks/hooks.json .claude-plugin/plugin.json
git commit -m "spike(hooks): PostToolUse task_capture — log raw payload for Task* tools"
git push origin main
```
Then the human runs `/reload-plugins` in a Claude Code session.

- [ ] **Step 4: Trigger + capture.** In a Claude Code session, create and update a task (TaskCreate then TaskUpdate). Then read the log:

Run: `cat ~/.config/mayring/task_capture_spike.log`
Expected: one JSON line per tool call. **Record** for Task 4: the exact `tool_name` values that fired, and the JSON paths for the task's id (for idempotency), `title`/content, and `status`. If `TodoWrite` is the actual mechanism, record its `tool_input.todos[]` shape.

- [ ] **Step 5: Decision gate.**
  - **Fires + id/title/status present** → proceed to Task 2; the controller passes the confirmed schema to Task 4.
  - **Does NOT fire for Task\* tools** → fallback: if `TodoWrite` fires, target it instead; else STOP and escalate — the source must change to reactivating `derive_todo` (re-scope Task 4 accordingly).

---

## Task 2: Backend — `tasks.external_id` column + idempotent `create_task` upsert

**Files:**
- Modify: `core/mayring_core/memory/store.py` (tasks DDL ~line 732, `CURRENT_SCHEMA_VERSION` line 123, migration block)
- Modify: `core/mayring_core/memory/tasks.py` (`create_task` ~line 54)
- Modify: `src/api/routes/models.py` (`TaskCreateRequest` ~line 255)
- Test: `tests/test_task_external_id.py` (create)

- [ ] **Step 1: Write the failing test** `tests/test_task_external_id.py`:

```python
from pathlib import Path

from mayring_core.memory.store import init_memory_db
from mayring_core.memory.tasks import create_task, list_tasks


def test_external_id_upserts_not_duplicates(tmp_path: Path):
    conn = init_memory_db(tmp_path / "m.db")
    ws = "ws-1"
    t1 = create_task(conn, workspace_id=ws, title="do X", created_by="agent",
                     external_id="harness-42")
    t2 = create_task(conn, workspace_id=ws, title="do X (renamed)", created_by="agent",
                     external_id="harness-42")
    rows = list_tasks(conn, ws)
    assert len(rows) == 1, "same external_id must update, not duplicate"
    assert t1["task_id"] == t2["task_id"]
    assert t2["title"] == "do X (renamed)"


def test_no_external_id_always_inserts(tmp_path: Path):
    conn = init_memory_db(tmp_path / "m.db")
    create_task(conn, workspace_id="ws", title="a", created_by="agent")
    create_task(conn, workspace_id="ws", title="a", created_by="agent")
    assert len(list_tasks(conn, "ws")) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=.:core python3 -m pytest tests/test_task_external_id.py -q`
Expected: FAIL — `create_task() got an unexpected keyword argument 'external_id'`.

- [ ] **Step 3: Add the column + index + version bump** in `core/mayring_core/memory/store.py`. In the `tasks` CREATE TABLE (line ~732) add `external_id TEXT` to the column list. After the existing tasks indexes (line ~750) add:

```sql
CREATE INDEX IF NOT EXISTS idx_tasks_external_id ON tasks(workspace_id, external_id);
```

Bump `CURRENT_SCHEMA_VERSION = 13` → `14` (line 123) and add the migration for existing DBs following the SAME idiom used for the v12/v13 idx additions earlier (locate that migration block — it ALTERs/creates idx when `current_version < N`). The ALTER:

```python
# v14: tasks.external_id for idempotent agent-task capture (PostToolUse hook)
conn.execute("ALTER TABLE tasks ADD COLUMN external_id TEXT")
conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_external_id ON tasks(workspace_id, external_id)")
```
(Wrap exactly as the neighbouring version-gated migrations do — match their `if current_version < 14:` structure.)

- [ ] **Step 4: Add `external_id` to `create_task`** (`core/mayring_core/memory/tasks.py`). Add the param and upsert logic:

```python
def create_task(
    conn: DBAdapter,
    *,
    workspace_id: str,
    title: str,
    description: str = "",
    priority: str = "medium",
    due_date: str | None = None,
    tags: str = "",
    created_by: str | None = None,
    linked_chunk_id: str | None = None,
    scope_key: str | None = None,
    external_id: str | None = None,
) -> dict:
    if not title or not title.strip():
        raise ValueError("title must not be empty")
    if priority not in _PRIORITY:
        raise ValueError(f"priority must be one of {_PRIORITY}, got {priority!r}")
    if not is_valid_scope_key(scope_key):
        raise ValueError(f"invalid scope_key: {scope_key!r}")

    # WHY(igio-todos): idempotent capture — a re-fired PostToolUse hook for the
    # same harness task must update the existing row, not create a duplicate.
    if external_id:
        existing = conn.execute(
            "SELECT task_id FROM tasks WHERE workspace_id=? AND external_id=?",
            (workspace_id, external_id),
        ).fetchone()
        if existing is not None:
            existing_id = existing[0]
            conn.execute(
                "UPDATE tasks SET title=?, description=?, priority=?, tags=?, updated_at=? "
                "WHERE task_id=? AND workspace_id=?",
                (title, description, priority, tags, _now(), existing_id, workspace_id),
            )
            conn.commit()
            return get_task(conn, workspace_id, existing_id)  # type: ignore[return-value]

    task_id = "tsk_" + uuid.uuid4().hex[:16]
    now = _now()
    conn.execute(
        """
        INSERT INTO tasks
            (task_id, workspace_id, title, description, status, priority,
             due_date, tags, created_by, linked_chunk_id, scope_key,
             external_id, created_at, updated_at, completed_at)
        VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            task_id, workspace_id, title, description, priority,
            due_date, tags, created_by, linked_chunk_id, scope_key,
            external_id, now, now,
        ),
    )
    conn.commit()
    return get_task(conn, workspace_id, task_id)  # type: ignore[return-value]
```

- [ ] **Step 5: Add `external_id` to the request model** (`src/api/routes/models.py`, `TaskCreateRequest`): add `external_id: str | None = None`. Then in `src/api/routes/tasks.py` `create_task` route, forward `external_id=req.external_id` to `_t.create_task(...)`.

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=.:core python3 -m pytest tests/test_task_external_id.py tests/ -k "task" -q`
Expected: PASS (new tests + existing task tests green).

- [ ] **Step 7: Commit**

```bash
git add core/mayring_core/memory/store.py core/mayring_core/memory/tasks.py src/api/routes/models.py src/api/routes/tasks.py tests/test_task_external_id.py
git commit -m "feat(tasks): external_id for idempotent agent-task capture (schema v14)"
```

---

## Task 3: Backend — `intervention.todos` in `GET /stats/igio-lens`

**Files:**
- Modify: `src/api/routes/igio_admin.py` (`igio_lens` handler ~line 83-138)
- Test: `tests/test_igio_lens_todos.py` (create)

**Context:** the handler builds `axes[axis] = {"count": ..., "chunks": [...]}` for each `VALID_AXES` and returns `{..., "axes-or-flat", "unclassified": {...}}`. Read the exact return shape (lines 113-138) and the workspace-scoping helper it uses (`_conn`, `_is_admin`, `get_token_info` — same as the existing endpoint). Use `list_tasks` from `mayring_core.memory.tasks`.

- [ ] **Step 1: Write the failing test** `tests/test_igio_lens_todos.py`:

```python
from fastapi.testclient import TestClient


def test_igio_lens_includes_intervention_todos(monkeypatch):
    """GET /stats/igio-lens must expose the workspace's tasks under
    intervention.todos so the lens column can render them."""
    # Build app + a seeded task in the caller's workspace, then assert the
    # response's intervention block carries a `todos` list containing it.
    # (Follow the existing igio-lens test setup — find it via:
    #   grep -rn "stats/igio-lens" tests/ )
    from tests._helpers import make_app_with_seeded_task  # see Step 1a
    client, ws, task_id = make_app_with_seeded_task(monkeypatch, title="capture me")
    r = client.get("/stats/igio-lens", headers={"X-Workspace-Id": ws})
    assert r.status_code == 200
    body = r.json()
    todos = body["intervention"]["todos"]
    assert any(t["task_id"] == task_id and t["title"] == "capture me" for t in todos)
```

- [ ] **Step 1a:** If no reusable harness exists, instead write the test against the store + a direct call to the handler's logic (mirror how `tests/` already tests `igio_admin` endpoints — `grep -rn "igio_lens\|igio-lens" tests/`). Use whichever pattern the repo already uses for route tests (TestClient with a temp DB + token override). The assertion stays: `body["intervention"]["todos"]` contains the seeded task.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=.:core python3 -m pytest tests/test_igio_lens_todos.py -q`
Expected: FAIL — `KeyError: 'todos'`.

- [ ] **Step 3: Implement** in `src/api/routes/igio_admin.py`. After the `axes` loop builds the intervention entry, attach the workspace's tasks:

```python
from mayring_core.memory.tasks import list_tasks
# ... inside igio_lens(), after axes are built, before the return:
open_todos = list_tasks(_conn(), workspace_id, status="open")
done_todos = list_tasks(_conn(), workspace_id, status="done")  # recently completed
axes["intervention"]["todos"] = [
    {
        "task_id": t["task_id"], "title": t["title"], "status": t["status"],
        "created_by": t.get("created_by"), "created_at": t.get("created_at"),
        "completed_at": t.get("completed_at"),
    }
    for t in ([*open_todos, *done_todos][:limit])
]
```
Use the same `workspace_id` the handler already resolves (admin=all / JWT=own). If admin scope means cross-workspace, scope todos to the requested/own workspace consistently with the chunk query. Keep `intervention.count` (chunk-igio) for backward-compat.

- [ ] **Step 4: Run test to verify PASS**

Run: `PYTHONPATH=.:core python3 -m pytest tests/test_igio_lens_todos.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/igio_admin.py tests/test_igio_lens_todos.py
git commit -m "feat(igio-lens): expose intervention.todos from the tasks table"
```

---

## Task 4: The real PostToolUse capture hook (uses Task 1's confirmed schema)

**Files:**
- Modify: `mayring-claude-plugin/hooks/task_capture.py` (replace spike with full impl)
- Test: `mayring-claude-plugin/hooks/test_task_capture.py` (create)

**Context:** Task 1 confirmed which `tool_name`s fire and the JSON paths for `id`/`title`/`status`. The code below assumes the standard Claude-Code PostToolUse contract: stdin JSON with `tool_name`, `tool_input` (the call args), `tool_response` (the result). **Adjust the four `_extract_*` accessors to the exact paths Task 1 recorded.** Reuse the JWT/HTTP pattern from `hooks/stop_hook.py` (`_read_token`, `~/.config/mayring/hook.jwt`, `MAYRING_API_URL`).

- [ ] **Step 1: Write the failing test** `mayring-claude-plugin/hooks/test_task_capture.py`:

```python
import importlib.util
from pathlib import Path
from unittest.mock import patch

_mod_path = Path(__file__).parent / "task_capture.py"
spec = importlib.util.spec_from_file_location("task_capture", _mod_path)
tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)


def test_taskcreate_posts_to_tasks():
    payload = {"tool_name": "TaskCreate",
               "tool_input": {"description": "fix the widget"},
               "tool_response": {"id": "harness-7", "status": "open"}}
    calls = []
    with patch.object(tc, "_read_token", return_value="jwt"), \
         patch.object(tc, "_post", side_effect=lambda *a, **k: calls.append((a, k))):
        tc.handle(payload)
    assert calls, "TaskCreate must POST /tasks"
    method, path, body = calls[0][0][0], calls[0][0][1], calls[0][0][2]
    assert method == "POST" and path == "/tasks"
    assert body["external_id"] == "harness-7"
    assert body["title"] == "fix the widget"
    assert body["created_by"] == "agent"


def test_non_todo_tool_is_noop():
    calls = []
    with patch.object(tc, "_read_token", return_value="jwt"), \
         patch.object(tc, "_post", side_effect=lambda *a, **k: calls.append(a)):
        tc.handle({"tool_name": "Bash", "tool_input": {}, "tool_response": {}})
    assert calls == []


def test_completed_update_completes_task():
    payload = {"tool_name": "TaskUpdate",
               "tool_input": {"status": "completed"},
               "tool_response": {"id": "harness-7", "status": "completed"}}
    calls = []
    with patch.object(tc, "_read_token", return_value="jwt"), \
         patch.object(tc, "_post", side_effect=lambda *a, **k: calls.append(a)):
        tc.handle(payload)
    assert any(c[1] == "/tasks/harness-7/complete" or "complete" in c[1] for c in calls)


def test_no_token_is_silent_skip():
    with patch.object(tc, "_read_token", return_value=""):
        tc.handle({"tool_name": "TaskCreate", "tool_input": {"description": "x"},
                   "tool_response": {"id": "1"}})  # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/nileneb/Desktop/mayring-claude-plugin && python3 -m pytest hooks/test_task_capture.py -q`
Expected: FAIL — `module 'task_capture' has no attribute 'handle'`.

- [ ] **Step 3: Implement** `hooks/task_capture.py` (replace the spike). Adjust `_extract_*` to Task 1's schema:

```python
"""PostToolUse capture: mirror the agent's Task* tool calls into MayringCoder
/tasks (idempotent via external_id) so the IGIO-Lens intervention column shows
the real work todos. Best-effort, never blocks the tool call."""
import json
import os
import sys
import urllib.error
import urllib.request

_API = os.getenv("MAYRING_API_URL", "https://mcp.linn.games").rstrip("/")
_JWT_FILE = os.path.expanduser("~/.config/mayring/hook.jwt")
_TODO_TOOLS = {"TaskCreate", "TaskUpdate"}  # confirmed by Task 1; add TodoWrite if it fired
_TIMEOUT = 3.0


def _read_token() -> str:
    try:
        with open(_JWT_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _post(method: str, path: str, body: dict, token: str) -> None:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{_API}{path}", data=data, method=method,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT):
            pass
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        sys.stderr.write(f"[task_capture] {method} {path} failed: {e}\n")


# --- schema accessors — set to the exact paths Task 1 recorded ---
def _extract_id(p: dict) -> str:
    return str((p.get("tool_response") or {}).get("id")
               or (p.get("tool_input") or {}).get("id") or "")


def _extract_title(p: dict) -> str:
    ti = p.get("tool_input") or {}
    return (ti.get("description") or ti.get("title") or ti.get("prompt") or "").strip()


def _extract_status(p: dict) -> str:
    return ((p.get("tool_response") or {}).get("status")
            or (p.get("tool_input") or {}).get("status") or "").strip()


def handle(payload: dict) -> None:
    if payload.get("tool_name") not in _TODO_TOOLS:
        return
    token = _read_token()
    if not token:
        return
    ext = _extract_id(payload)
    title = _extract_title(payload)
    status = _extract_status(payload)
    if status in ("completed", "done") and ext:
        _post("POST", f"/tasks/{ext}/complete", {}, token)
        return
    if not title:
        return
    _post("POST", "/tasks", {
        "title": title[:200], "created_by": "agent", "tags": "agent",
        "external_id": ext or None,
    }, token)


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        return
    try:
        handle(payload)
    except Exception as e:  # never break the tool call
        sys.stderr.write(f"[task_capture] crashed: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
```

Note: the `/complete` path uses `external_id`, but `POST /tasks/{task_id}/complete` expects the internal `task_id`. If Task 1 shows updates can't be matched to the internal id, resolve via `GET /tasks?tag=agent` then complete by the row whose `external_id` matches — or add a `POST /tasks/by-external/{ext}/complete` convenience route in Task 2. Pick the simpler given the confirmed schema; record the choice.

- [ ] **Step 4: Run tests to verify PASS**

Run: `cd /home/nileneb/Desktop/mayring-claude-plugin && python3 -m pytest hooks/test_task_capture.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit + bump plugin version**

```bash
cd /home/nileneb/Desktop/mayring-claude-plugin
# bump .claude-plugin/plugin.json version
git add hooks/task_capture.py hooks/test_task_capture.py .claude-plugin/plugin.json
git commit -m "feat(hooks): capture Task* tool calls into /tasks (PostToolUse)"
git push origin main
```

---

## Task 5: Smoke check — captured task surfaces in intervention.todos

**Files:**
- Modify: `tools/smoke_test_production.py` (add `check_intervention_todos_surface`, register in `ALL_CHECKS`)

- [ ] **Step 1: Add the check** (mirrors the `_act_as`/`_http` patterns from the org-memory acceptance checks):

```python
def check_intervention_todos_surface(api: str, token: str) -> CheckResult:
    """A task created via POST /tasks must appear in GET /stats/igio-lens under
    intervention.todos (the lens intervention column source)."""
    suffix = int(time.time())
    ws = f"todo-{suffix}"
    title = f"SMOKE-TODO {suffix}"
    code1, body1, _ = _http("POST", f"{api}/tasks", token,
        body={"title": title, "created_by": "agent", "tags": "agent",
              "external_id": f"smoke-{suffix}"},
        extra_headers=_act_as("A", workspace=ws), timeout=12.0)
    if code1 != 200:
        return CheckResult("intervention_todos_surface", False, f"create failed http={code1}: {body1}")
    code2, body2, _ = _http("GET", f"{api}/stats/igio-lens?limit=20", token,
        extra_headers=_act_as("A", workspace=ws), timeout=12.0)
    todos = (((body2 or {}).get("intervention") or {}).get("todos")) or []
    found = any(t.get("title") == title for t in todos)
    return CheckResult("intervention_todos_surface", found,
        f"task_in_intervention_todos={found} (must be True)  todos={len(todos)}  marker={suffix}")
```

- [ ] **Step 2: Register** `("intervention_todos_surface", check_intervention_todos_surface),` in `ALL_CHECKS`.
- [ ] **Step 3: Verify import** `python3 -c "import tools.smoke_test_production as s; print(len(s.ALL_CHECKS))"` (count +1). Live run deferred to Task 7.
- [ ] **Step 4: Commit** `git commit -am "test(smoke): intervention_todos_surface check"`

---

## Task 6: Frontend — render intervention.todos in the IGIO-Lens

**Files:**
- Modify: `app.linn.games/app/Services/Mcp/MayringStatsClient.php` (`getIgioLens` already returns the raw response — confirm it passes `intervention.todos` through; defensive default `[]`)
- Modify: `app.linn.games/app/Livewire/Mayring/IgioLens.php` + `resources/views/livewire/mayring/igio-lens.blade.php`
- Test: `app.linn.games/tests/Feature/Mayring/IgioLensTodosTest.php` (create)

- [ ] **Step 1: Write the failing Pest test** `tests/Feature/Mayring/IgioLensTodosTest.php`: assert the `IgioLens` component, given a fake `getIgioLens()` response whose `intervention.todos` contains `{title:'render me', status:'open'}`, renders `render me` in the intervention column. (Mirror the existing `IgioLens` Livewire test if one exists — `grep -rn IgioLens app.linn.games/tests`.)

- [ ] **Step 2: Run it red** — `cd /home/nileneb/Desktop/WebDev/app.linn.games && ./vendor/bin/pest --filter IgioLensTodos` → FAIL.

- [ ] **Step 3: Implement.** In `IgioLens.php` `fetchAxes()`, keep the existing 4-axis structure but expose `intervention.todos` to the view (defensive: `$axes['intervention']['todos'] ?? []`). In `igio-lens.blade.php`, the intervention column iterates `todos` (title + status badge; open first, done dimmed) instead of/above the chunk previews. Render only scalar leaves (htmlspecialchars-500 lesson). Other columns unchanged.

- [ ] **Step 4: Run it green** — `./vendor/bin/pest --filter IgioLensTodos` → PASS.

- [ ] **Step 5: Commit** (app.linn.games):
```bash
git add app/Services/Mcp/MayringStatsClient.php app/Livewire/Mayring/IgioLens.php resources/views/livewire/mayring/igio-lens.blade.php tests/Feature/Mayring/IgioLensTodosTest.php
git commit -m "feat(igio-lens): render intervention.todos in the intervention column"
```

---

## Task 7: Deploy + verify end-to-end

- [ ] **Step 1: Push** MayringCoder master (Tasks 2,3,5) → build+deploy. Push app.linn.games main (Task 6). Plugin main already pushed (Tasks 1,4).
- [ ] **Step 2: Reload plugin** — human runs `/reload-plugins` (picks up the bumped task_capture version).
- [ ] **Step 3: Live capture** — in a Claude Code session, create a task with `TaskCreate` (and complete one with `TaskUpdate`).
- [ ] **Step 4: Verify backend** — `GET /stats/igio-lens` for workspace `bene` (`X-Workspace-Id`/own JWT) shows the created task under `intervention.todos`; the completed one shows `status=done`.
- [ ] **Step 5: Verify smoke** — the auto post-deploy-smoke `intervention_todos_surface` check is green.
- [ ] **Step 6: Verify UI (human)** — `/mayring/igio` logged-in shows the captured task-lists in the intervention column.
- [ ] **Step 7: Memory** — record (MEMORY.md + project memory) that the IGIO-Lens intervention column is now fed by captured agent task-lists; note the confirmed PostToolUse schema from Task 1.

---

## Out of Scope (YAGNI)
- `derive_todo`-from-prompts as a second source (stays dormant).
- Manual todo-entry UI (the `POST /tasks` API suffices; add later if wanted).
- Priority/due-date inference for captured todos.
- Reworking the issue/goal/outcome columns.
