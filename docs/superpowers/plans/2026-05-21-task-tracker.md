# Task-Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a workspace-isolated task tracker to MayringCoder (REST + MCP so humans and agents manage work items), expose it as a dedicated Tasks page in app.linn.games, and rename the overloaded `task_categories` storage layer to `research_questions`.

**Architecture:** New `tasks` table in `memory.db` with pure CRUD helpers (`src/memory/tasks.py`), a FastAPI router (`src/api/routes/tasks.py`) reusing the `get_workspace` auth dependency, and MCP tools (`src/api/mcp_task_tools.py`) mirroring the harness Task tools. The legacy `task_categories`/`task_chunk_links` tables + `task_derivation.py` functions are renamed (idempotent SQLite migration). Frontend follows the existing `MayringStatsClient` + Livewire pattern, shipped via PR.

**Tech Stack:** Python 3.13, FastAPI, SQLite (via `DBAdapter`), FastMCP, pytest; Laravel 11, Livewire, Pest.

**Spec:** `docs/superpowers/specs/2026-05-21-task-tracker-design.md`

**Branch:** `feat/task-tracker` (already created off `master`, spec committed).

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `src/memory/store.py` | `tasks` DDL in `_init_schema`; rename migration in `_migrate_schema` | Modify |
| `src/memory/tasks.py` | pure CRUD helpers for `tasks` | Create |
| `src/memory/task_derivation.py` | rename functions/queries `task_*` → `research_question_*` | Modify |
| `src/api/routes/tasks.py` | REST endpoints | Create |
| `src/api/routes/models.py` | Pydantic request/response models | Modify |
| `src/api/server.py` | `include_router(tasks.router)` | Modify |
| `src/api/mcp_task_tools.py` | `register_task_tools(mcp)` | Create |
| `src/api/mcp.py`, `src/api/local_mcp.py` | register task tools | Modify |
| `tests/test_tasks.py` | store + API + MCP tests | Create |
| `tests/test_task_derivation.py` | update renamed identifiers | Modify |
| app.linn.games `app/Services/Mcp/MayringTasksClient.php` | API client | Create |
| app.linn.games `app/Livewire/Mayring/TaskBoard.php` + blade + route | Tasks page | Create |
| app.linn.games `tests/Feature/Mayring/TaskBoardTest.php` | Pest tests | Create |

DBAdapter API (from `src/memory/db_adapter.py`): `execute(sql, params)->cursor`, `commit()`, `get_columns(table)->set[str]`, `changes()->int`; cursor rows are `sqlite3.Row` (access via `row["col"]` or `row[0]`). Test DB: `DBAdapter.memory()` then `_init_schema(db)`.

---

## Phase 0 — Rename `task_categories` → `research_questions`

### Task 0.1: Idempotent rename migration

**Files:**
- Modify: `src/memory/store.py` (`_migrate_schema`, ~line 125-212)
- Test: `tests/test_task_derivation.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_task_derivation.py`)

```python
def test_migration_renames_task_categories_to_research_questions(tmp_path):
    from src.memory.db_adapter import DBAdapter
    from src.memory.store import _migrate_schema
    db = DBAdapter.create(tmp_path / "legacy.db")
    # Simulate a pre-rename production DB.
    db.execute("CREATE TABLE task_categories (task_id TEXT PRIMARY KEY, title TEXT, "
               "workspace_id TEXT)")
    db.execute("CREATE TABLE task_chunk_links (task_id TEXT, chunk_id TEXT, "
               "relevance_score REAL, created_at TEXT, PRIMARY KEY(task_id, chunk_id))")
    db.commit()
    _migrate_schema(db)
    names = {r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "research_questions" in names
    assert "research_question_chunk_links" in names
    assert "task_categories" not in names
    assert "research_question_id" in db.get_columns("research_question_chunk_links")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_task_derivation.py::test_migration_renames_task_categories_to_research_questions -v`
Expected: FAIL (`research_questions` not in names — migration not implemented).

- [ ] **Step 3: Implement migration** — add to `_migrate_schema` in `src/memory/store.py` BEFORE `_migrate_visibility_check(conn)`:

```python
    _migrate_rename_research_questions(conn)
```

Then add the function (next to `_migrate_visibility_check`):

```python
def _migrate_rename_research_questions(conn: DBAdapter) -> None:
    """Rename the legacy task_categories layer to research_questions.

    WHY(task-tracker): 'task' was overloaded — a new first-class `tasks`
    tracker now owns that word, so the derived-research-question storage is
    renamed to disambiguate. Idempotent: only fires when the old tables exist.
    """
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "task_categories" in tables and "research_questions" not in tables:
        conn.execute("ALTER TABLE task_categories RENAME TO research_questions")
        conn.execute("ALTER TABLE research_questions RENAME COLUMN task_id TO research_question_id")
    if "task_chunk_links" in tables and "research_question_chunk_links" not in tables:
        conn.execute("ALTER TABLE task_chunk_links RENAME TO research_question_chunk_links")
        conn.execute("ALTER TABLE research_question_chunk_links "
                     "RENAME COLUMN task_id TO research_question_id")
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_task_derivation.py::test_migration_renames_task_categories_to_research_questions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/store.py tests/test_task_derivation.py
git commit -m "feat(rename): idempotent migration task_categories -> research_questions"
```

### Task 0.2: Rename DDL, derivation functions, and callers

**Files:**
- Modify: `src/memory/store.py` (`_init_schema` DDL, ~line 479-507)
- Modify: `src/memory/task_derivation.py` (all queries + function names)
- Modify: `tests/test_task_derivation.py` (identifiers)
- Modify callers found by sweep

- [ ] **Step 1: Map callers (run the sweep, record results)**

Run: `grep -rn "task_categories\|task_chunk_links\|derive_task\|link_chunk_to_task\|get_task_boost_for_chunks" src/ tools/ tests/`
Expected: references in `store.py`, `task_derivation.py`, `test_task_derivation.py`, and the categorization/ambient callers of `derive_task*`/`get_task_boost_for_chunks`. Note each file:line.

- [ ] **Step 2: Update the `_init_schema` DDL** in `src/memory/store.py` — replace the `task_categories`/`task_chunk_links` CREATE blocks with:

```sql
        CREATE TABLE IF NOT EXISTS research_questions (
            research_question_id TEXT PRIMARY KEY,
            title             TEXT NOT NULL,
            embedding_id      TEXT NOT NULL DEFAULT '',
            parent_research_question_id TEXT REFERENCES research_questions(research_question_id) ON DELETE SET NULL,
            occurrence_count  INTEGER NOT NULL DEFAULT 1,
            first_seen_at     TEXT NOT NULL,
            last_used_at      TEXT NOT NULL,
            workspace_id      TEXT NOT NULL DEFAULT 'default'
        );
        CREATE INDEX IF NOT EXISTS idx_research_questions_workspace ON research_questions(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_research_questions_last_used ON research_questions(last_used_at);

        CREATE TABLE IF NOT EXISTS research_question_chunk_links (
            research_question_id TEXT NOT NULL REFERENCES research_questions(research_question_id) ON DELETE CASCADE,
            chunk_id          TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            relevance_score   REAL NOT NULL DEFAULT 1.0,
            created_at        TEXT NOT NULL,
            PRIMARY KEY (research_question_id, chunk_id)
        );
        CREATE INDEX IF NOT EXISTS idx_rq_chunk_links_chunk ON research_question_chunk_links(chunk_id);
```

- [ ] **Step 3: Rename functions + queries in `src/memory/task_derivation.py`**

Apply these renames (function defs + their bodies' SQL + internal callers):
- `derive_task` → `derive_research_question`, `derive_task_fast` → `derive_research_question_fast`, `derive_task_background` → `derive_research_question_background`
- `link_chunk_to_task` → `link_chunk_to_research_question`
- `get_task_boost_for_chunks` → `get_research_question_boost`
- `_load_task_embeddings` → `_load_research_question_embeddings`
- All SQL: `task_categories`→`research_questions`, `task_chunk_links`→`research_question_chunk_links`, column `task_id`→`research_question_id`.

- [ ] **Step 4: Update callers found in Step 1**

For each caller file (e.g. `src/memory/ingestion/categorization.py`, `src/memory/ambient.py`, `src/memory/retrieval.py`), replace the old function names with the new ones. Use exact replacements from Step 1's line list.

- [ ] **Step 5: Update `tests/test_task_derivation.py`** — replace all `derive_task`/`task_categories`/`task_chunk_links`/`task_id`/`link_chunk_to_task`/`get_task_boost_for_chunks` references with the renamed identifiers.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/test_task_derivation.py -v && python -m pytest -q`
Expected: PASS (no `task_categories` references remain). Confirm `grep -rn "task_categories\|task_chunk_links\|derive_task\b\|get_task_boost_for_chunks" src/ tools/` returns nothing.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(rename): task_categories layer -> research_questions (DDL, fns, callers)"
```

---

## Phase 1 — `tasks` store (`src/memory/tasks.py`)

### Task 1.1: tasks DDL + create_task

**Files:**
- Modify: `src/memory/store.py` (`_init_schema`)
- Create: `src/memory/tasks.py`
- Test: `tests/test_tasks.py`

- [ ] **Step 1: Write the failing test** (create `tests/test_tasks.py`)

```python
from __future__ import annotations
import pytest
from src.memory.db_adapter import DBAdapter
from src.memory.store import _init_schema
from src.memory import tasks as t


def _db():
    db = DBAdapter.memory()
    _init_schema(db)
    return db


def test_create_task_persists_and_returns_row():
    db = _db()
    row = t.create_task(db, workspace_id="ws1", title="Fix auth",
                        description="JWT bug", priority="high",
                        tags="auth,bug", created_by="42")
    assert row["task_id"].startswith("tsk_")
    assert row["status"] == "open"
    assert row["priority"] == "high"
    assert row["workspace_id"] == "ws1"
    assert row["completed_at"] is None
    stored = db.execute("SELECT title, tags FROM tasks WHERE task_id=?",
                        (row["task_id"],)).fetchone()
    assert stored["title"] == "Fix auth"
    assert stored["tags"] == "auth,bug"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tasks.py::test_create_task_persists_and_returns_row -v`
Expected: FAIL (`tasks` table missing / module `tasks` has no `create_task`).

- [ ] **Step 3a: Add the DDL** to `_init_schema` in `src/memory/store.py`:

```sql
        CREATE TABLE IF NOT EXISTS tasks (
            task_id        TEXT PRIMARY KEY,
            workspace_id   TEXT NOT NULL,
            title          TEXT NOT NULL,
            description    TEXT NOT NULL DEFAULT '',
            status         TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','in_progress','done')),
            priority       TEXT NOT NULL DEFAULT 'medium' CHECK(priority IN ('low','medium','high')),
            due_date       TEXT,
            tags           TEXT NOT NULL DEFAULT '',
            created_by     TEXT,
            linked_chunk_id TEXT REFERENCES chunks(chunk_id) ON DELETE SET NULL,
            scope_key      TEXT,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            completed_at   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_workspace_status ON tasks(workspace_id, status);
        CREATE INDEX IF NOT EXISTS idx_tasks_workspace_due ON tasks(workspace_id, due_date);
```

- [ ] **Step 3b: Create `src/memory/tasks.py`**

```python
"""CRUD helpers for the workspace-isolated task tracker.

Pure functions over a DBAdapter — routes/MCP tools call these, never SQL
directly. A task is workspace-scoped: every read/write is filtered by
workspace_id, so multi-tenant isolation holds without a visibility column.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.memory.db_adapter import DBAdapter
from src.memory.schema import is_valid_scope_key

_STATUS = ("open", "in_progress", "done")
_PRIORITY = ("low", "medium", "high")
_COLS = ("task_id", "workspace_id", "title", "description", "status", "priority",
         "due_date", "tags", "created_by", "linked_chunk_id", "scope_key",
         "created_at", "updated_at", "completed_at")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict:
    return {k: row[k] for k in _COLS}


def create_task(conn: DBAdapter, *, workspace_id: str, title: str,
                description: str = "", priority: str = "medium",
                due_date: str | None = None, tags: str = "",
                created_by: str | None = None,
                linked_chunk_id: str | None = None,
                scope_key: str | None = None) -> dict:
    if not title or not title.strip():
        raise ValueError("task title is required")
    if priority not in _PRIORITY:
        raise ValueError(f"priority must be one of {_PRIORITY}")
    if not is_valid_scope_key(scope_key):
        raise ValueError("scope_key must be type-prefixed (repo:/project:/campaign:)")
    task_id = "tsk_" + uuid.uuid4().hex[:16]
    now = _now()
    conn.execute(
        "INSERT INTO tasks (task_id, workspace_id, title, description, status, "
        "priority, due_date, tags, created_by, linked_chunk_id, scope_key, "
        "created_at, updated_at, completed_at) "
        "VALUES (?,?,?,?,'open',?,?,?,?,?,?,?,?,NULL)",
        (task_id, workspace_id, title.strip(), description, priority, due_date,
         tags, created_by, linked_chunk_id, scope_key, now, now),
    )
    conn.commit()
    return get_task(conn, workspace_id, task_id)


def get_task(conn: DBAdapter, workspace_id: str, task_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM tasks WHERE task_id=? AND workspace_id=?",
        (task_id, workspace_id),
    ).fetchone()
    return _row_to_dict(row) if row else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tasks.py::test_create_task_persists_and_returns_row -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/store.py src/memory/tasks.py tests/test_tasks.py
git commit -m "feat(tasks): tasks table + create_task/get_task"
```

### Task 1.2: list_tasks with filters

**Files:** Modify `src/memory/tasks.py`; Test `tests/test_tasks.py`

- [ ] **Step 1: Write the failing test**

```python
def test_list_tasks_filters_by_status_and_workspace():
    db = _db()
    t.create_task(db, workspace_id="ws1", title="a")
    done = t.create_task(db, workspace_id="ws1", title="b")
    t.complete_task(db, "ws1", done["task_id"])
    t.create_task(db, workspace_id="ws2", title="other-ws")

    open_ws1 = t.list_tasks(db, "ws1", status="open")
    assert [r["title"] for r in open_ws1] == ["a"]
    assert all(r["workspace_id"] == "ws1" for r in t.list_tasks(db, "ws1"))
    assert t.list_tasks(db, "ws2")[0]["title"] == "other-ws"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tasks.py::test_list_tasks_filters_by_status_and_workspace -v`
Expected: FAIL (`list_tasks`/`complete_task` not defined).

- [ ] **Step 3: Implement `list_tasks`** (append to `src/memory/tasks.py`)

```python
def list_tasks(conn: DBAdapter, workspace_id: str, *, status: str | None = None,
               tag: str | None = None, priority: str | None = None) -> list[dict]:
    sql = "SELECT * FROM tasks WHERE workspace_id=?"
    params: list = [workspace_id]
    if status:
        sql += " AND status=?"
        params.append(status)
    if priority:
        sql += " AND priority=?"
        params.append(priority)
    if tag:
        sql += " AND (',' || tags || ',') LIKE ?"
        params.append(f"%,{tag},%")
    sql += " ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'in_progress' THEN 1 ELSE 2 END, " \
           "CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, created_at DESC"
    return [_row_to_dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]
```

(Tag stored CSV without surrounding commas; the `',' || tags || ','` wrap makes the LIKE match whole tags.)

- [ ] **Step 4: Run test** (will still fail on `complete_task` — implemented in Task 1.3; run after 1.3). For now run create/list only:

Run: `python -m pytest tests/test_tasks.py -k "list_tasks" -v`
Expected: FAIL referencing `complete_task` — proceed to Task 1.3, then re-run.

- [ ] **Step 5: Commit**

```bash
git add src/memory/tasks.py tests/test_tasks.py
git commit -m "feat(tasks): list_tasks with status/tag/priority filters"
```

### Task 1.3: update_task / complete_task / delete_task (status sets completed_at)

**Files:** Modify `src/memory/tasks.py`; Test `tests/test_tasks.py`

- [ ] **Step 1: Write the failing test**

```python
def test_complete_sets_completed_at_and_reopen_clears_it():
    db = _db()
    task = t.create_task(db, workspace_id="ws1", title="x")
    done = t.complete_task(db, "ws1", task["task_id"])
    assert done["status"] == "done"
    assert done["completed_at"] is not None
    reopened = t.update_task(db, "ws1", task["task_id"], status="open")
    assert reopened["completed_at"] is None


def test_update_and_delete_are_workspace_scoped():
    db = _db()
    task = t.create_task(db, workspace_id="ws1", title="x")
    # foreign workspace cannot touch it
    assert t.update_task(db, "ws2", task["task_id"], title="hijack") is None
    assert t.delete_task(db, "ws2", task["task_id"]) is False
    assert t.get_task(db, "ws1", task["task_id"])["title"] == "x"
    assert t.delete_task(db, "ws1", task["task_id"]) is True
    assert t.get_task(db, "ws1", task["task_id"]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tasks.py -k "completed_at or workspace_scoped" -v`
Expected: FAIL (`update_task`/`complete_task`/`delete_task` not defined).

- [ ] **Step 3: Implement** (append to `src/memory/tasks.py`)

```python
_UPDATABLE = ("title", "description", "status", "priority", "due_date",
              "tags", "linked_chunk_id", "scope_key")


def update_task(conn: DBAdapter, workspace_id: str, task_id: str, **fields) -> dict | None:
    current = get_task(conn, workspace_id, task_id)
    if current is None:
        return None
    sets, params = [], []
    for k, v in fields.items():
        if k not in _UPDATABLE:
            raise ValueError(f"field {k!r} is not updatable")
        if k == "status" and v not in _STATUS:
            raise ValueError(f"status must be one of {_STATUS}")
        if k == "priority" and v not in _PRIORITY:
            raise ValueError(f"priority must be one of {_PRIORITY}")
        sets.append(f"{k}=?")
        params.append(v)
    if "status" in fields:
        sets.append("completed_at=?")
        params.append(_now() if fields["status"] == "done" else None)
    sets.append("updated_at=?")
    params.append(_now())
    params.extend([task_id, workspace_id])
    conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE task_id=? AND workspace_id=?",
                 tuple(params))
    conn.commit()
    return get_task(conn, workspace_id, task_id)


def complete_task(conn: DBAdapter, workspace_id: str, task_id: str) -> dict | None:
    return update_task(conn, workspace_id, task_id, status="done")


def delete_task(conn: DBAdapter, workspace_id: str, task_id: str) -> bool:
    conn.execute("DELETE FROM tasks WHERE task_id=? AND workspace_id=?",
                 (task_id, workspace_id))
    conn.commit()
    return conn.changes() > 0
```

- [ ] **Step 4: Run the whole tasks store test file**

Run: `python -m pytest tests/test_tasks.py -v`
Expected: PASS (all store tests incl. Task 1.2's list test).

- [ ] **Step 5: Commit**

```bash
git add src/memory/tasks.py tests/test_tasks.py
git commit -m "feat(tasks): update/complete/delete, workspace-scoped, completed_at lifecycle"
```

---

## Phase 2 — REST API (`src/api/routes/tasks.py`)

### Task 2.1: models + POST/GET endpoints

**Files:**
- Modify: `src/api/routes/models.py`
- Create: `src/api/routes/tasks.py`
- Modify: `src/api/server.py`
- Test: `tests/test_tasks.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_tasks.py`)

```python
def _client_with(ws="ws1", sub="42"):
    from fastapi.testclient import TestClient
    from src.api.server import app
    from src.api.auth import get_workspace, get_token_info
    from src.api.jwt_auth import TokenInfo
    import src.api.dependencies as _deps
    db = _db()
    app.dependency_overrides[get_workspace] = lambda: ws
    app.dependency_overrides[get_token_info] = lambda: TokenInfo(workspace_id=ws, sub=sub, scopes=("mcp:memory",))
    _deps._conn = db
    return TestClient(app), db


def test_post_and_get_tasks_endpoint():
    client, db = _client_with()
    try:
        r = client.post("/tasks", json={"title": "API task", "priority": "high"},
                        headers={"Authorization": "Bearer t"})
        assert r.status_code == 200, r.text
        tid = r.json()["task_id"]
        assert r.json()["created_by"] == "42"
        lst = client.get("/tasks?status=open", headers={"Authorization": "Bearer t"})
        assert lst.status_code == 200
        assert any(x["task_id"] == tid for x in lst.json()["tasks"])
    finally:
        from src.api.server import app
        import src.api.dependencies as _deps
        app.dependency_overrides.clear(); _deps._conn = None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tasks.py::test_post_and_get_tasks_endpoint -v`
Expected: FAIL (404 — `/tasks` route not registered).

- [ ] **Step 3a: Add models** to `src/api/routes/models.py`:

```python
class TaskCreateRequest(BaseModel):
    title: str
    description: str = ""
    priority: str = "medium"
    due_date: str | None = None
    tags: str = ""
    linked_chunk_id: str | None = None
    scope_key: str | None = None


class TaskUpdateRequest(BaseModel):
    title: str | None = None
    description: str | None = None
    status: str | None = None
    priority: str | None = None
    due_date: str | None = None
    tags: str | None = None
    linked_chunk_id: str | None = None
    scope_key: str | None = None
```

- [ ] **Step 3b: Create `src/api/routes/tasks.py`**

```python
"""REST endpoints for the task tracker — workspace-scoped via get_workspace."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_workspace, get_token_info
from src.api.jwt_auth import TokenInfo
from src.api.dependencies import get_conn as _get_conn
from src.api.routes.models import TaskCreateRequest, TaskUpdateRequest
from src.memory import tasks as _t

router = APIRouter()


@router.post("/tasks")
async def create_task(req: TaskCreateRequest,
                      workspace_id: str = Depends(get_workspace),
                      info: TokenInfo = Depends(get_token_info)) -> dict:
    try:
        return _t.create_task(
            _get_conn(), workspace_id=workspace_id, title=req.title,
            description=req.description, priority=req.priority,
            due_date=req.due_date, tags=req.tags,
            created_by=info.sub, linked_chunk_id=req.linked_chunk_id,
            scope_key=req.scope_key)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/tasks")
async def list_tasks(status: str | None = None, tag: str | None = None,
                     priority: str | None = None,
                     workspace_id: str = Depends(get_workspace)) -> dict:
    return {"workspace_id": workspace_id,
            "tasks": _t.list_tasks(_get_conn(), workspace_id,
                                   status=status, tag=tag, priority=priority)}
```

- [ ] **Step 3c: Register the router** in `src/api/server.py` (next to the other `include_router` calls, ~line 62):

```python
from src.api.routes import tasks as _tasks
app.include_router(_tasks.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tasks.py::test_post_and_get_tasks_endpoint -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/models.py src/api/routes/tasks.py src/api/server.py tests/test_tasks.py
git commit -m "feat(api): POST/GET /tasks"
```

### Task 2.2: PATCH / complete / DELETE + cross-workspace 404

**Files:** Modify `src/api/routes/tasks.py`; Test `tests/test_tasks.py`

- [ ] **Step 1: Write the failing test**

```python
def test_patch_complete_delete_endpoints_and_cross_ws_404():
    client, db = _client_with(ws="ws1")
    try:
        tid = client.post("/tasks", json={"title": "x"},
                          headers={"Authorization": "Bearer t"}).json()["task_id"]
        pc = client.post(f"/tasks/{tid}/complete", headers={"Authorization": "Bearer t"})
        assert pc.status_code == 200 and pc.json()["status"] == "done"
        pa = client.patch(f"/tasks/{tid}", json={"priority": "low"},
                         headers={"Authorization": "Bearer t"})
        assert pa.status_code == 200 and pa.json()["priority"] == "low"
        miss = client.patch("/tasks/tsk_nope", json={"title": "y"},
                           headers={"Authorization": "Bearer t"})
        assert miss.status_code == 404
        assert client.delete(f"/tasks/{tid}", headers={"Authorization": "Bearer t"}).status_code == 200
    finally:
        from src.api.server import app
        import src.api.dependencies as _deps
        app.dependency_overrides.clear(); _deps._conn = None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tasks.py::test_patch_complete_delete_endpoints_and_cross_ws_404 -v`
Expected: FAIL (405/404 — endpoints not defined).

- [ ] **Step 3: Implement** (append to `src/api/routes/tasks.py`)

```python
@router.patch("/tasks/{task_id}")
async def update_task(task_id: str, req: TaskUpdateRequest,
                      workspace_id: str = Depends(get_workspace)) -> dict:
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        updated = _t.update_task(_get_conn(), workspace_id, task_id, **fields)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail="task not found")
    return updated


@router.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str,
                        workspace_id: str = Depends(get_workspace)) -> dict:
    done = _t.complete_task(_get_conn(), workspace_id, task_id)
    if done is None:
        raise HTTPException(status_code=404, detail="task not found")
    return done


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str,
                      workspace_id: str = Depends(get_workspace)) -> dict:
    if not _t.delete_task(_get_conn(), workspace_id, task_id):
        raise HTTPException(status_code=404, detail="task not found")
    return {"deleted": task_id}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tasks.py::test_patch_complete_delete_endpoints_and_cross_ws_404 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/tasks.py tests/test_tasks.py
git commit -m "feat(api): PATCH/complete/DELETE /tasks with 404 isolation"
```

---

## Phase 3 — MCP tools (`src/api/mcp_task_tools.py`)

### Task 3.1: register_task_tools

**Files:**
- Create: `src/api/mcp_task_tools.py`
- Modify: `src/api/mcp.py`, `src/api/local_mcp.py`
- Test: `tests/test_tasks.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_tasks.py`)

```python
def test_register_task_tools_create_and_list(monkeypatch):
    from unittest.mock import MagicMock
    import src.api.mcp_task_tools as mt
    import src.api.dependencies as _deps
    db = _db()
    _deps._conn = db
    monkeypatch.setattr(mt, "_effective_workspace_id", lambda: "ws1")
    monkeypatch.setattr(mt, "_enforce_tenant", lambda w: w or "ws1")
    monkeypatch.setattr(mt, "_effective_user_id", lambda: None)  # agent path

    captured = {}
    class FakeMCP:
        def tool(self):
            def deco(fn): captured[fn.__name__] = fn; return fn
            return deco
    mt.register_task_tools(FakeMCP())

    created = captured["task_create"](title="agent task")
    assert created["created_by"] == "agent"
    listed = captured["task_list"]()
    assert any(t["task_id"] == created["task_id"] for t in listed["tasks"])
    _deps._conn = None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tasks.py::test_register_task_tools_create_and_list -v`
Expected: FAIL (module `mcp_task_tools` does not exist).

- [ ] **Step 3: Create `src/api/mcp_task_tools.py`**

```python
"""MCP tools for the task tracker — agents & humans manage work items.

Mirrors the harness Task tools (create/list/update/complete). Workspace is
resolved exactly like the memory tools (active-workspace aware); created_by
is the JWT sub for humans, 'agent' when no human identity is present.
"""
from __future__ import annotations

from src.api.mcp_auth import (
    _enforce_tenant, _effective_workspace_id, _effective_user_id,
)
from src.api.dependencies import get_conn as _get_conn
from src.memory import tasks as _t


def register_task_tools(mcp) -> None:
    @mcp.tool()
    def task_create(title: str, description: str = "", priority: str = "medium",
                    due_date: str | None = None, tags: str = "",
                    linked_chunk_id: str | None = None, scope_key: str | None = None,
                    workspace_id: str | None = None) -> dict:
        """Create a work item in the current workspace's task tracker."""
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        return _t.create_task(_get_conn(), workspace_id=ws, title=title,
                              description=description, priority=priority,
                              due_date=due_date, tags=tags,
                              created_by=_effective_user_id() or "agent",
                              linked_chunk_id=linked_chunk_id, scope_key=scope_key)

    @mcp.tool()
    def task_list(status: str | None = None, tag: str | None = None,
                  priority: str | None = None, workspace_id: str | None = None) -> dict:
        """List work items in the current workspace (filter status/tag/priority)."""
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        return {"workspace_id": ws,
                "tasks": _t.list_tasks(_get_conn(), ws, status=status, tag=tag,
                                       priority=priority)}

    @mcp.tool()
    def task_update(task_id: str, title: str | None = None, description: str | None = None,
                    status: str | None = None, priority: str | None = None,
                    due_date: str | None = None, tags: str | None = None,
                    workspace_id: str | None = None) -> dict:
        """Update fields of a work item (status, priority, etc.)."""
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        fields = {k: v for k, v in dict(
            title=title, description=description, status=status, priority=priority,
            due_date=due_date, tags=tags).items() if v is not None}
        updated = _t.update_task(_get_conn(), ws, task_id, **fields)
        return updated or {"error": "task not found", "task_id": task_id}

    @mcp.tool()
    def task_complete(task_id: str, workspace_id: str | None = None) -> dict:
        """Mark a work item as done."""
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        done = _t.complete_task(_get_conn(), ws, task_id)
        return done or {"error": "task not found", "task_id": task_id}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tasks.py::test_register_task_tools_create_and_list -v`
Expected: PASS

- [ ] **Step 5: Register in the MCP servers** — add to `src/api/mcp.py` (after line 65 `register_memory_tools(mcp)`):

```python
from src.api.mcp_task_tools import register_task_tools
register_task_tools(mcp)
```

And the same two lines in `src/api/local_mcp.py` (after `register_agent_tools(mcp)`).

- [ ] **Step 6: Run full suite + commit**

Run: `python -m pytest -q`
Expected: PASS (only any pre-existing unrelated failures, if any — confirm none new).

```bash
git add src/api/mcp_task_tools.py src/api/mcp.py src/api/local_mcp.py tests/test_tasks.py
git commit -m "feat(mcp): task_create/list/update/complete tools"
```

---

## Phase 4 — app.linn.games Tasks page (separate PR)

> Branch off `main`: `git checkout -b feat/mayring-tasks-page origin/main`. PR-review required per repo convention. Run tests in CI (local vendor may be incomplete).

### Task 4.1: MayringTasksClient

**Files:**
- Create: `app/Services/Mcp/MayringTasksClient.php`
- Test: `tests/Feature/Mayring/TaskBoardTest.php`

- [ ] **Step 1: Write the failing Pest test** — `tests/Feature/Mayring/TaskBoardTest.php`

```php
<?php
use App\Services\Mcp\MayringTasksClient;
use Illuminate\Support\Facades\Http;

test('client lists tasks from the mayring API', function () {
    Http::fake(['*/tasks*' => Http::response(['tasks' => [
        ['task_id' => 'tsk_1', 'title' => 'A', 'status' => 'open', 'priority' => 'high'],
    ]], 200)]);
    $tasks = app(MayringTasksClient::class)->list(status: 'open');
    expect($tasks)->toHaveCount(1);
    expect($tasks[0]['title'])->toBe('A');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `php vendor/bin/pest tests/Feature/Mayring/TaskBoardTest.php` (in CI)
Expected: FAIL (`MayringTasksClient` not found).

- [ ] **Step 3: Implement `MayringTasksClient`** — mirror `app/Services/Mcp/MayringStatsClient.php`'s base-URL + JWT header pattern; methods: `list(?string $status=null, ?string $tag=null, ?string $priority=null): array` (`GET /tasks`), `create(array $data): array` (`POST /tasks`), `update(string $id, array $data): array` (`PATCH /tasks/{id}`), `complete(string $id): array` (`POST /tasks/{id}/complete`), `delete(string $id): void`. Read the existing client first for the exact base-url config key + auth header helper, and reuse them.

- [ ] **Step 4: Run test to verify it passes** — `php vendor/bin/pest tests/Feature/Mayring/TaskBoardTest.php` → PASS

- [ ] **Step 5: Commit** — `git commit -m "feat(mayring): MayringTasksClient"`

### Task 4.2: TaskBoard Livewire page

**Files:**
- Create: `app/Livewire/Mayring/TaskBoard.php`, `resources/views/livewire/mayring/task-board.blade.php`
- Modify: route file + nav (mirror how `MemoryDashboard` is routed/linked)
- Test: `tests/Feature/Mayring/TaskBoardTest.php`

- [ ] **Step 1: Write the failing Livewire test** (append)

```php
use App\Livewire\Mayring\TaskBoard;
use Livewire\Livewire;

test('task board renders tasks and can complete one', function () {
    $user = \App\Models\User::factory()->withoutTwoFactor()->create();
    Http::fake([
        '*/tasks*' => Http::response(['tasks' => [
            ['task_id' => 'tsk_1', 'title' => 'Do X', 'status' => 'open',
             'priority' => 'high', 'tags' => '', 'created_by' => '1', 'due_date' => null],
        ]], 200),
        '*/tasks/tsk_1/complete' => Http::response(['task_id' => 'tsk_1', 'status' => 'done'], 200),
    ]);
    Livewire::actingAs($user)->test(TaskBoard::class)
        ->assertSee('Do X')
        ->call('complete', 'tsk_1')
        ->assertOk();
});
```

- [ ] **Step 2: Run test to verify it fails** — `php vendor/bin/pest tests/Feature/Mayring/TaskBoardTest.php` → FAIL (component missing).

- [ ] **Step 3: Implement the component + view + route.** `TaskBoard.php`: inject `MayringTasksClient`; properties `$status`, `$priority`, `$tag` filters; `render()` calls `->list(...)` and passes `$tasks` to the blade; actions `create(array $data)`, `complete($id)`, `updateStatus($id, $status)`, `delete($id)`. Blade: a table (Titel, Status, Priorität, Fällig, Tags, Ersteller, Goal-Link) + a create form + per-row complete/status controls, styled like `memory-dashboard.blade.php`. Add a route (e.g. `/mayring/tasks`) and a nav entry next to the Memory-Dashboard link.

- [ ] **Step 4: Run test to verify it passes** — `php vendor/bin/pest tests/Feature/Mayring/TaskBoardTest.php` → PASS

- [ ] **Step 5: Commit + push + open PR**

```bash
git add -A && git commit -m "feat(mayring): Tasks page (Livewire TaskBoard)"
git push -u origin feat/mayring-tasks-page
gh pr create --repo Nileneb/app.linn.games --base main --title "feat(mayring): Tasks tracker page" --body "Consumes MayringCoder /tasks API. Pairs with feat/task-tracker."
```

---

## Self-Review

- **Spec coverage:** Data model → Task 1.1; CRUD → 1.1-1.3; REST → 2.1-2.2; MCP → 3.1; rename (full layer + migration) → 0.1-0.2; Laravel Tasks page → 4.1-4.2; workspace isolation → tests in 1.3/2.1. ✓
- **Out-of-scope honored:** no `ingest(task=...)` or `pi_jobs.task_text` rename; no task visibility/org-sharing. ✓
- **Type consistency:** `create_task`/`list_tasks`/`get_task`/`update_task`/`complete_task`/`delete_task` signatures match across store, routes, and MCP tasks; `research_question_id` used consistently after rename.
- **Verification gate:** after Phase 3, run full `python -m pytest -q` (expect green except any pre-existing unrelated failures); Phase 4 verified in CI.
