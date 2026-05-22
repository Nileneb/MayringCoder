# Prompt-derived Goals + Todos — Implementation Plan

> REQUIRED SUB-SKILL: subagent-driven-development. TDD per task.

**Goal:** Auto-derive clean actionable todos from user prompts (server-side LLM) into the `tasks` table, and surface the workspace's clean IGIO goals — both visible in the Tasks page.

**Architecture:** A new `derive_todo()` (LLM actionable?+title, prompt-vs-prompt embedding dedup, → `tasks` row) runs in a daemon thread off `POST /conversation/micro-batch`. A `GET /tasks/goals` returns `igio_axis='goal'` chunks filtered to the user's own source types. Frontend renders Todos + Ziele sections.

**Spec:** `docs/superpowers/specs/2026-05-22-prompt-derived-goals-todos.md`. **Branch:** `feat/prompt-derived-goals-todos`.

---

## Task 1: `tasks.derive_embedding` column (dedup support)
**Files:** `src/memory/store.py` (`_init_schema` tasks DDL + a migration), `tests/test_tasks.py`

- [ ] **Step 1 (test):** add to `tests/test_tasks.py`:
```python
def test_tasks_has_derive_embedding_column():
    db = _db()
    assert "derive_embedding" in db.get_columns("tasks")
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** In `_init_schema` add `derive_embedding TEXT` to the `tasks` CREATE (nullable, after `scope_key`). Add an idempotent migration in `_migrate_schema` (follow the `migrations` dict pattern that already adds columns): add `("derive_embedding", "TEXT")` under a `"tasks"` key. Bump `CURRENT_SCHEMA_VERSION`.
- [ ] **Step 4:** run → PASS; `python -m pytest tests/test_tasks.py -q` green.
- [ ] **Step 5:** commit `feat(tasks): derive_embedding column for todo dedup`.

## Task 2: `derive_todo()` — LLM actionable+title + prompt-dedup → task
**Files:** Create `src/memory/todo_derivation.py`; Test `tests/test_todo_derivation.py`

- [ ] **Step 1 (test):** mirror `tests/test_task_derivation.py` mocking (`patch("src.memory.todo_derivation._embed_text")` + patch the LLM call). Tests:
```python
# actionable prompt → creates an open task (todo)
def test_derive_todo_creates_task_when_actionable(conn):
    import src.memory.todo_derivation as td
    with patch.object(td, "_embed_text", return_value=[0.1]*768), \
         patch.object(td, "_llm_todo", return_value={"actionable": True, "title": "Fix the auth bug"}):
        r = td.derive_todo("please fix the auth bug in jwt_auth", conn, "http://x", "ws1")
    assert r and r["created"] is True
    rows = conn.execute("SELECT title,status,created_by,tags FROM tasks WHERE workspace_id='ws1'").fetchall()
    assert rows == [("Fix the auth bug","open","derived","derived")]

# non-actionable → no task
def test_derive_todo_skips_when_not_actionable(conn):
    import src.memory.todo_derivation as td
    with patch.object(td,"_embed_text",return_value=[0.1]*768), \
         patch.object(td,"_llm_todo",return_value={"actionable": False, "title": ""}):
        r = td.derive_todo("what does this function do?", conn, "http://x", "ws1")
    assert r is None
    assert conn.execute("SELECT COUNT(*) FROM tasks WHERE workspace_id='ws1'").fetchone()[0] == 0

# near-duplicate prompt → no second todo (dedup prompt-vs-prompt)
def test_derive_todo_dedups_near_identical_open_todo(conn):
    import src.memory.todo_derivation as td
    emb=[0.2]*768
    with patch.object(td,"_embed_text",return_value=emb), \
         patch.object(td,"_llm_todo",return_value={"actionable":True,"title":"Fix auth"}):
        td.derive_todo("fix the auth bug", conn, "http://x", "ws1")
        r2 = td.derive_todo("please fix that auth bug", conn, "http://x", "ws1")
    assert r2 is None
    assert conn.execute("SELECT COUNT(*) FROM tasks WHERE workspace_id='ws1'").fetchone()[0] == 1
```
- [ ] **Step 2:** run → FAIL (module missing).
- [ ] **Step 3:** Create `src/memory/todo_derivation.py`:
```python
"""Derive an actionable todo from a user prompt (server-side LLM).

Mirrors task_derivation's prompt-embedding dedup (prompt-vs-prompt — the bug
we fixed). NOT for the hot path; run in a daemon thread off micro-batch.
"""
from __future__ import annotations
import json, logging
from typing import Optional
from src.memory.db_adapter import DBAdapter
from src.memory.task_derivation import _embed_text, _cosine  # reuse
from src.memory import tasks as _t

_log = logging.getLogger(__name__)
_SIM = 0.85
_PROMPT = (
    "Du bekommst einen User-Prompt aus einer Coding-Session. Entscheide, ob er eine "
    "konkrete, umsetzbare Arbeits-Aufgabe (Todo) ausdrückt, die der User erledigt haben will "
    "(z.B. 'implementiere X', 'fixe Y', 'baue Z'). Reine Fragen, Smalltalk oder Status-Checks "
    "sind KEINE Todos. Antworte NUR mit JSON: "
    '{"actionable": true|false, "title": "<imperativer Titel, <=120 Zeichen, leer wenn nicht actionable>"}\n\nPrompt:\n'
)

def _llm_todo(prompt: str, ollama_url: str, model: str) -> Optional[dict]:
    try:
        import requests
        resp = requests.post(ollama_url.rstrip("/") + "/api/generate",
            json={"model": model, "prompt": _PROMPT + prompt.strip()[:1500],
                  "format": "json", "stream": False,
                  "options": {"temperature": 0.1, "num_predict": 120}}, timeout=30)
        if resp.status_code != 200:
            return None
        data = json.loads(resp.json().get("response", "").strip())
        if not isinstance(data, dict):
            return None
        return {"actionable": bool(data.get("actionable")),
                "title": str(data.get("title") or "").strip()[:120]}
    except Exception as e:
        _log.warning("derive_todo LLM fail: %s", e)
        return None

def derive_todo(prompt: str, conn: DBAdapter, ollama_url: str, workspace_id: str,
                *, model: Optional[str] = None) -> Optional[dict]:
    prompt = (prompt or "").strip()
    if len(prompt) < 8:
        return None
    if model is None:
        try:
            from src.model_router import ModelRouter
            model = ModelRouter(ollama_url=ollama_url).resolve("text")
        except Exception:
            model = "mistral:7b-instruct"
    verdict = _llm_todo(prompt, ollama_url, model)
    if not verdict or not verdict["actionable"] or not verdict["title"]:
        return None
    prompt_emb = _embed_text(prompt, ollama_url)
    if prompt_emb is None:
        return None
    # dedup: prompt-vs-prompt against this workspace's OPEN derived todos
    for (eid,) in conn.execute(
        "SELECT derive_embedding FROM tasks WHERE workspace_id=? AND status!='done' "
        "AND derive_embedding IS NOT NULL", (workspace_id,)).fetchall():
        try:
            if _cosine(prompt_emb, json.loads(eid)) >= _SIM:
                return None
        except Exception:
            continue
    row = _t.create_task(conn, workspace_id=workspace_id, title=verdict["title"],
                         created_by="derived", tags="derived")
    conn.execute("UPDATE tasks SET derive_embedding=? WHERE task_id=?",
                 (json.dumps(prompt_emb), row["task_id"]))
    conn.commit()
    return {"task_id": row["task_id"], "title": verdict["title"], "created": True}
```
- [ ] **Step 4:** run the 3 tests → PASS; full suite green.
- [ ] **Step 5:** commit `feat(todo): derive_todo — LLM actionable+title, prompt dedup, creates task`.

## Task 3: hook `derive_todo` into the micro-batch handler (background)
**Files:** `src/api/routes/memory.py` (`conversation_micro_batch`, ~line 451 after igio_hint block); Test `tests/test_*` (light)

- [ ] **Step 1:** After the igio_hint tagging block, before the predictive-memory block, add a non-blocking derive: extract the **last user-role turn** text from `turns_dicts` (`[t for t in turns_dicts if t.get("role")=="user"]`, take last `.get("content","")`); skip when `workspace_id == "system"` (no smoke/cron noise). Fire a daemon thread like `derive_research_question_background` does:
```python
        # Prompt → actionable todo (background, never blocks the response;
        # not for system/smoke workspaces).
        if workspace_id != "system":
            user_turns = [t.get("content", "") for t in turns_dicts if t.get("role") == "user"]
            last_user = (user_turns[-1] if user_turns else "").strip()
            if last_user:
                import threading
                from src.memory.store import MEMORY_DB_PATH
                def _derive_todo_bg(p=last_user, ws=workspace_id):
                    try:
                        from src.memory.store import init_memory_db
                        from src.memory.todo_derivation import derive_todo
                        c = init_memory_db(MEMORY_DB_PATH)
                        try:
                            derive_todo(p, c, _OLLAMA_URL, ws)
                        finally:
                            c.close()
                    except Exception as exc:
                        logging.getLogger(__name__).warning("derive_todo_bg failed: %s", exc)
                threading.Thread(target=_derive_todo_bg, daemon=True).start()
```
- [ ] **Step 2:** sanity test: `python -c "import src.api.routes.memory"` imports clean. Add a light test that posting a micro-batch with a user turn doesn't error (the derive runs in a thread; assert 200). Run full suite green.
- [ ] **Step 3:** commit `feat(todo): derive todos from micro-batch prompts (background)`.

## Task 4: `list_workspace_goals` + `GET /tasks/goals`
**Files:** `src/memory/tasks.py`, `src/api/routes/tasks.py`, `tests/test_tasks.py`

- [ ] **Step 1 (test):** seed a goal chunk from a conversation source and one from `paper` + one from `ambient_snapshot`; assert `list_workspace_goals` returns ONLY the conversation one; assert `GET /tasks/goals` returns it and is workspace-scoped. (Reuse a seed helper inserting into sources+chunks with igio_axis='goal'.)
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3a:** `list_workspace_goals(conn, workspace_id, *, limit=100)` in tasks.py: SELECT chunks JOIN sources WHERE igio_axis='goal' AND is_active=1 AND workspace_id=? AND `s.source_type IN ('conversation_summary','note','session_knowledge','session','task','user_context','knowledge')` ORDER BY created_at DESC LIMIT ?; return read-only dicts (`source='goal'`, `read_only=True`, title from summary/text[:120]).
- [ ] **Step 3b:** `GET /tasks/goals` in routes/tasks.py: `{"workspace_id": ws, "goals": list_workspace_goals(_get_conn(), ws)}` with `Depends(get_workspace)`.
- [ ] **Step 4:** run → PASS; full suite green.
- [ ] **Step 5:** commit `feat(tasks): GET /tasks/goals — workspace goals from own sources only`.

## Task 5: Frontend (app.linn.games, separate PR off main)
**Files:** `MayringTasksClient.php` (+`goals()` → `GET /tasks/goals`), `TaskBoard.php` + blade, `tests/Feature/Mayring/TaskBoardTest.php`
- [ ] Add `goals(): array` to MayringTasksClient (same per-user JWT auth). TaskBoard `render()` also fetches goals; blade gets a second read-only **„Ziele"** table (Titel + Quelle-Badge), above/below the Todos table. Auto-todos show a `derived` badge (already have source/read_only handling). Pest: `goals()` request shape + TaskBoard renders both sections. `php -l`; CI verifies. PR.

## Self-Review
- Spec coverage: A=Task4+5, B=Task1+2+3, C=Task5. Todos=tasks (no new axis) ✓. Dedup prompt-vs-prompt ✓. No LLM in hook (background thread) ✓. system workspace skipped ✓.
- Types: `derive_todo` returns `{task_id,title,created}` or None; `_llm_todo` returns `{actionable,title}` or None; `list_workspace_goals` returns list[dict] with source/read_only.

## Verification (after build)
`python -m pytest -q` green; deploy MayringCoder; post a micro-batch with an actionable prompt for `bene` → `GET /tasks` shows a `derived` open todo; `GET /tasks/goals` shows only conversation/note goals (no paper/ambient). Frontend PR via CI.
