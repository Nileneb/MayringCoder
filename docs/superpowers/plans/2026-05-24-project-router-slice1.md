# Project Router Slice 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beim ersten substanziellen Prompt einer Session ein `active_project_id` (oder null) deterministisch anheften (cwd-git-remote hart, semantisch als Fallback) und projekt-scharf suchen.

**Architecture:** Dünner Server-Endpoint `POST /projects/route` (MayringCoder) entscheidet (hart→semantisch→null), persistiert Projekt-Embeddings in Chroma-Collection `projects`. Dünner Hook (mayring-claude-plugin) liefert das cwd-Remote, cached die Entscheidung in `session_ctx.json` und reicht `project`+`task_context` an `/memory/search` durch. Consumer (`retrieval.py:136` filtert `project_id`) existiert bereits.

**Tech Stack:** Python/FastAPI, SQLite (`mayring_core.memory.store`), ChromaDB (`get_chroma_collection`), Ollama-Embeddings (`nomic-embed-text` über `three.linn.games`), pytest (MayringCoder), Plugin-Hooks (stdlib-only).

**Spec:** `docs/superpowers/specs/2026-05-24-project-router-design.md`

---

## File Structure

**MayringCoder:**
- Create `src/api/routes/projects.py` — Endpoint + reine Helfer (`_normalize_remote`, `_cosine`, `project_embed_text`, `route()` mit injizierbarem `embed_fn`).
- Modify `core/mayring_core/memory/store.py` — Index `idx_projects_source` in `_migrate_schema`.
- Modify `src/api/server.py` — Router registrieren.
- Create `tools/embed_projects.py` — Backfill bestehender Projekte → Chroma `projects`.
- Create `tests/test_project_router.py` — Unit-Tests (reine Fn + `route()` mit Fake-embed/Fake-chroma).
- Modify `tools/smoke_test_production.py` — Check `projects_route_cwd_remote`.

**mayring-claude-plugin:**
- Modify `hooks/_session_ctx.py` — `_git_remote`, `route_project`, `derive_task`.
- Modify `hooks/memory_inject.py` — `_search`/`_multi_lens_search` um `project_id`+`task_context`; `main()` Routing+Cache+Observability.
- Create `hooks/test_session_ctx_router.py` — stdlib-Assert-Script (kein pytest nötig).

---

## Task 1: Schema-Index `idx_projects_source` (MayringCoder)

**Files:**
- Modify: `core/mayring_core/memory/store.py` (`_migrate_schema`)
- Test: `tests/test_project_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_project_router.py
import sqlite3
from pathlib import Path
from mayring_core.memory.store import init_memory_db


def test_projects_source_index_exists(tmp_path: Path) -> None:
    p = tmp_path / "memory.db"
    init_memory_db(p).close()
    idx = {r[1] for r in sqlite3.connect(p).execute(
        "PRAGMA index_list('projects')").fetchall()}
    assert "idx_projects_source" in idx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nileneb/Desktop/MayringCoder && PYTHONPATH=. python -m pytest tests/test_project_router.py::test_projects_source_index_exists -v`
Expected: FAIL (index missing).

- [ ] **Step 3: Add the index in `_migrate_schema`**

Find the `_migrate_schema` body in `store.py` (where other `CREATE INDEX IF NOT EXISTS` migrations live) and add:

```python
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_projects_source "
            "ON projects(workspace_id, source_type, source_ref)"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_project_router.py::test_projects_source_index_exists -v`
Expected: PASS. Also run `init_memory_db` twice via existing `test_init_memory_db_idempotent` pattern is unaffected.

- [ ] **Step 5: Commit**

```bash
git add core/mayring_core/memory/store.py tests/test_project_router.py
git commit -m "feat(projects): index (workspace_id,source_type,source_ref) for remote-match"
```

---

## Task 2: Pure helper `_normalize_remote` (MayringCoder)

**Files:**
- Create: `src/api/routes/projects.py`
- Test: `tests/test_project_router.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_project_router.py
import pytest
from src.api.routes.projects import _normalize_remote


@pytest.mark.parametrize("url,expected", [
    ("git@github.com:Nileneb/MayringCoder.git", "nileneb/mayringcoder"),
    ("https://github.com/Nileneb/MayringCoder.git", "nileneb/mayringcoder"),
    ("https://github.com/Nileneb/MayringCoder", "nileneb/mayringcoder"),
    ("ssh://git@github.com/Nileneb/app.linn.games.git", "nileneb/app.linn.games"),
    ("", None),
    ("not-a-remote", None),
])
def test_normalize_remote(url, expected):
    assert _normalize_remote(url) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_project_router.py::test_normalize_remote -v`
Expected: FAIL (module/function missing).

- [ ] **Step 3: Create `src/api/routes/projects.py` with the helper**

```python
"""Project Router (Slice 1): POST /projects/route.

Attaches active_project_id from the strongest signal: cwd-git-remote (hard,
match-or-create) → semantic match against existing projects → null. See
docs/superpowers/specs/2026-05-24-project-router-design.md.
"""
from __future__ import annotations

import math
import re
import uuid
from datetime import datetime, timezone
from typing import Callable

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.auth import get_workspace
from src.api.dependencies import get_conn as _get_conn

router = APIRouter(tags=["projects"])

_REMOTE_RE = re.compile(
    r"github\.com[:/]+(?P<owner>[^/]+)/(?P<name>[^/]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)


def _normalize_remote(remote: str) -> str | None:
    """git@/https/ssh GitHub remote → 'owner/name' lowercased, else None."""
    if not remote:
        return None
    m = _REMOTE_RE.search(remote.strip())
    if not m:
        return None
    return f"{m.group('owner')}/{m.group('name')}".lower()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_project_router.py::test_normalize_remote -v`
Expected: PASS (6 cases).

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/projects.py tests/test_project_router.py
git commit -m "feat(projects): _normalize_remote helper + tests"
```

---

## Task 3: Pure helpers `_cosine` + `project_embed_text` (MayringCoder)

**Files:**
- Modify: `src/api/routes/projects.py`
- Test: `tests/test_project_router.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_project_router.py
from src.api.routes.projects import _cosine, project_embed_text


def test_cosine():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # zero vector → 0, no div0


def test_project_embed_text():
    assert project_embed_text("MayringCoder", "nileneb/mayringcoder", "github") == \
        "MayringCoder nileneb/mayringcoder github"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_project_router.py -k "cosine or embed_text" -v`
Expected: FAIL (functions missing).

- [ ] **Step 3: Implement in `src/api/routes/projects.py`**

```python
def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def project_embed_text(name: str, source_ref: str, source_type: str) -> str:
    return " ".join(p for p in (name, source_ref, source_type) if p).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_project_router.py -k "cosine or embed_text" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/projects.py tests/test_project_router.py
git commit -m "feat(projects): _cosine + project_embed_text helpers"
```

---

## Task 4: Core `route()` decision (MayringCoder)

`route()` is dependency-injected with `embed_fn` so tests run without Ollama/Chroma.

**Files:**
- Modify: `src/api/routes/projects.py`
- Test: `tests/test_project_router.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_project_router.py
import sqlite3
from src.api.routes.projects import route


class _FakeChroma:
    """get(include=['embeddings','metadatas']) → all project vectors."""
    def __init__(self, items):  # items: list[(id, embedding, metadata)]
        self._items = items

    def get(self, include=None):
        return {
            "ids": [i[0] for i in self._items],
            "embeddings": [i[1] for i in self._items],
            "metadatas": [i[2] for i in self._items],
        }

    def upsert(self, **kwargs):  # used by create path; record no-op
        for cid in kwargs.get("ids", []):
            self._items.append((cid, kwargs["embeddings"][kwargs["ids"].index(cid)],
                                {"project_id": cid}))


def _seed(db, rows):
    c = sqlite3.connect(db)
    now = "2026-05-24T00:00:00Z"
    for pid, st, ref, name in rows:
        c.execute("INSERT INTO projects(id,workspace_id,name,source_type,source_ref,"
                  "created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                  (pid, "ws1", name, st, ref, now, now))
    c.commit(); c.close()


def test_route_cwd_remote_match(tmp_path):
    db = tmp_path / "memory.db"; init_memory_db(db).close()
    _seed(db, [("p1", "github", "nileneb/mayringcoder", "MayringCoder")])
    conn = sqlite3.connect(db)
    out = route(conn, _FakeChroma([]), "ws1",
                cwd_remote="git@github.com:Nileneb/MayringCoder.git",
                prompt="fix the auth bug", embed_fn=lambda t: [0.0, 1.0])
    assert out["project_id"] == "p1"
    assert out["reason"] == "cwd-remote"
    assert out["mode"] == "coding"


def test_route_cwd_remote_create(tmp_path):
    db = tmp_path / "memory.db"; init_memory_db(db).close()
    conn = sqlite3.connect(db)
    out = route(conn, _FakeChroma([]), "ws1",
                cwd_remote="https://github.com/Nileneb/NewRepo.git",
                prompt="add feature", embed_fn=lambda t: [1.0, 0.0])
    assert out["project_id"]  # created
    row = conn.execute("SELECT source_ref FROM projects WHERE id=?",
                       (out["project_id"],)).fetchone()
    assert "newrepo" in (row[0] or "").lower()
    assert out["reason"] == "cwd-remote"


def test_route_semantic_match(tmp_path):
    db = tmp_path / "memory.db"; init_memory_db(db).close()
    _seed(db, [("p1", "github", "nileneb/mayringcoder", "MayringCoder")])
    conn = sqlite3.connect(db)
    chroma = _FakeChroma([("proj:p1", [0.0, 1.0], {"project_id": "p1"})])
    out = route(conn, chroma, "ws1", cwd_remote=None,
                prompt="memory retrieval pipeline", embed_fn=lambda t: [0.0, 1.0])
    assert out["project_id"] == "p1"
    assert out["reason"] == "semantic"


def test_route_null_when_uncertain(tmp_path):
    db = tmp_path / "memory.db"; init_memory_db(db).close()
    conn = sqlite3.connect(db)
    chroma = _FakeChroma([
        ("proj:p1", [1.0, 0.0], {"project_id": "p1"}),
        ("proj:p2", [0.99, 0.01], {"project_id": "p2"}),  # tiny margin
    ])
    out = route(conn, chroma, "ws1", cwd_remote=None,
                prompt="vague", embed_fn=lambda t: [1.0, 0.0])
    assert out["project_id"] is None
    assert out["reason"] == "no-match"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_project_router.py -k route -v`
Expected: FAIL (`route` missing).

- [ ] **Step 3: Implement `route()` + private helpers**

```python
_SEMANTIC_MIN = 0.55
_SEMANTIC_MARGIN = 0.08

_MODE_RE = {
    "coding": re.compile(r"\b(repo|migration|endpoint|deploy|CI|test|bug|refactor|"
                         r"commit|PR|merge|api|schema)\b", re.IGNORECASE),
    "research": re.compile(r"\b(paper|DOI|arxiv|pubmed|RQ|research question|"
                           r"systematic review|p[1-8]|hypoth)\b", re.IGNORECASE),
}


def _classify_mode(prompt: str) -> str:
    c = bool(_MODE_RE["coding"].search(prompt))
    r = bool(_MODE_RE["research"].search(prompt))
    if c and r:
        return "mixed"
    if c:
        return "coding"
    if r:
        return "research"
    return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upsert_embedding(chroma, project_id: str, text: str, embed_fn) -> None:
    if chroma is None:
        return
    emb = embed_fn(text)
    if not emb:
        return
    chroma.upsert(ids=[f"proj:{project_id}"], embeddings=[emb],
                  metadatas=[{"project_id": project_id}], documents=[text])


def _semantic_match(chroma, prompt_emb: list[float]) -> tuple[str | None, float, float]:
    if chroma is None or not prompt_emb:
        return None, 0.0, 0.0
    data = chroma.get(include=["embeddings", "metadatas"])
    ids = data.get("ids") or []
    embs = data.get("embeddings") or []
    metas = data.get("metadatas") or []
    scored = sorted(
        ((_cosine(prompt_emb, e), (m or {}).get("project_id"))
         for e, m in zip(embs, metas) if e),
        key=lambda x: x[0], reverse=True,
    )
    if not scored:
        return None, 0.0, 0.0
    top_score, top_pid = scored[0]
    margin = top_score - (scored[1][0] if len(scored) > 1 else 0.0)
    return top_pid, top_score, margin


def route(conn, chroma, workspace: str, *, cwd_remote: str | None,
          prompt: str, embed_fn: Callable[[str], list[float]]) -> dict:
    mode = _classify_mode(prompt)
    # 1) hard signal: cwd-remote → match-or-create
    owner_name = _normalize_remote(cwd_remote or "")
    if owner_name:
        row = conn.execute(
            "SELECT id, name FROM projects WHERE workspace_id=? AND "
            "source_type='github' AND lower(source_ref) LIKE ?",
            (workspace, f"%{owner_name}%"),
        ).fetchone()
        if row:
            return {"project_id": row[0], "name": row[1], "mode": "coding",
                    "confidence": 0.9, "reason": "cwd-remote"}
        pid = str(uuid.uuid4())
        name = owner_name.split("/")[-1]
        conn.execute(
            "INSERT INTO projects(id,workspace_id,name,source_type,source_ref,"
            "created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
            (pid, workspace, name, "github", owner_name, _now(), _now()))
        conn.commit()
        _upsert_embedding(chroma, pid, project_embed_text(name, owner_name, "github"),
                          embed_fn)
        return {"project_id": pid, "name": name, "mode": "coding",
                "confidence": 0.9, "reason": "cwd-remote"}
    # 2) semantic match (existing only, no create)
    pid, score, margin = _semantic_match(chroma, embed_fn(prompt))
    if pid and score >= _SEMANTIC_MIN and margin >= _SEMANTIC_MARGIN:
        row = conn.execute("SELECT name FROM projects WHERE id=?", (pid,)).fetchone()
        return {"project_id": pid, "name": row[0] if row else None, "mode": mode,
                "confidence": round(score, 3), "reason": "semantic"}
    # 3) null
    return {"project_id": None, "name": None, "mode": mode,
            "confidence": 0.0, "reason": "no-match"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_project_router.py -k route -v`
Expected: PASS (4 route tests).

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/projects.py tests/test_project_router.py
git commit -m "feat(projects): route() decision (cwd-remote → semantic → null) + tests"
```

---

## Task 5: Endpoint `POST /projects/route` + register (MayringCoder)

**Files:**
- Modify: `src/api/routes/projects.py`
- Modify: `src/api/server.py`

- [ ] **Step 1: Add the endpoint wrapper in `projects.py`**

```python
class RouteRequest(BaseModel):
    cwd_remote: str | None = None
    prompt: str = ""


def _embed_one(text: str) -> list[float]:
    from mayring_core.config import EMBEDDING_MODEL, OLLAMA_TIMEOUT
    from mayring_core.ollama_client import embed_single
    import os
    url = os.environ.get("OLLAMA_URL", "https://three.linn.games")
    try:
        return embed_single(url, EMBEDDING_MODEL, text, timeout=OLLAMA_TIMEOUT) or []
    except Exception:  # noqa: BLE001 — embed failure must not 500 the router
        return []


@router.post("/projects/route")
async def route_project(req: RouteRequest, ws: str = Depends(get_workspace)) -> dict:
    from mayring_core.memory.store import get_chroma_collection
    conn = _get_conn()
    chroma = get_chroma_collection("projects")
    return route(conn, chroma, ws, cwd_remote=req.cwd_remote,
                 prompt=req.prompt, embed_fn=_embed_one)
```

> NOTE: `_get_conn()` returns the DBAdapter used elsewhere in routes; it supports
> `.execute()`/`.commit()` like `codebooks.py` uses. If the adapter's cursor API
> differs, mirror exactly what `src/api/routes/codebooks.py` does.

- [ ] **Step 2: Register the router in `src/api/server.py`**

Find where `codebooks` router is included (added in Phase 1) and add directly after it:

```python
from src.api.routes import projects as projects_routes
app.include_router(projects_routes.router)
```

- [ ] **Step 3: Verify import + route registration**

Run: `PYTHONPATH=. python -c "from src.api.server import app; print([r.path for r in app.routes if 'projects' in r.path])"`
Expected: `['/projects/route']` present.

- [ ] **Step 4: Commit**

```bash
git add src/api/routes/projects.py src/api/server.py
git commit -m "feat(projects): POST /projects/route endpoint + register"
```

---

## Task 6: Backfill tool `tools/embed_projects.py` (MayringCoder)

**Files:**
- Create: `tools/embed_projects.py`

- [ ] **Step 1: Write the tool (mirrors tools/import_codebooks_to_db.py)**

```python
"""Embed all projects into the Chroma collection 'projects' (embedding_id=proj:<id>).

Idempotent upsert. Run with the chroma containers reachable. --dry-run default.
"""
from __future__ import annotations
import argparse, os, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(db_path: Path, apply: bool) -> dict:
    from mayring_core.ollama_client import embed_batch
    from mayring_core.memory.store import get_chroma_collection
    from src.api.routes.projects import project_embed_text
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, name, source_ref, source_type FROM projects").fetchall()
    rep = {"projects": len(rows), "embedded": 0}
    if not apply or not rows:
        return rep
    url = os.environ.get("OLLAMA_URL", "https://three.linn.games")
    model = os.environ.get("MAYRING_EMBED_MODEL", "nomic-embed-text")
    col = get_chroma_collection("projects")
    ids = [f"proj:{r[0]}" for r in rows]
    texts = [project_embed_text(r[1] or "", r[2] or "", r[3] or "") for r in rows]
    metas = [{"project_id": r[0]} for r in rows]
    for i in range(0, len(ids), 64):
        embs = embed_batch(url, model, texts[i:i + 64], timeout=120)
        if embs:
            col.upsert(ids=ids[i:i + 64], embeddings=embs,
                       documents=texts[i:i + 64], metadatas=metas[i:i + 64])
            rep["embedded"] += len(embs)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(ROOT / "cache" / "memory.db"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    rep = run(Path(args.db), args.apply)
    print(f"projects={rep['projects']} embedded={rep['embedded']} "
          f"({'APPLIED' if args.apply else 'DRY-RUN'})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run locally (no writes)**

Run: `PYTHONPATH=. python tools/embed_projects.py`
Expected: prints `projects=<n> embedded=0 (DRY-RUN)`.

- [ ] **Step 3: Commit (prod-apply happens at deploy, see Task 10)**

```bash
git add tools/embed_projects.py
git commit -m "feat(projects): embed_projects.py backfill → Chroma 'projects'"
```

---

## Task 7: Plugin helpers — `_git_remote`, `route_project`, `derive_task`

**Files:**
- Modify: `/home/nileneb/Desktop/mayring-claude-plugin/hooks/_session_ctx.py`
- Test: `/home/nileneb/Desktop/mayring-claude-plugin/hooks/test_session_ctx_router.py`

- [ ] **Step 1: Write the stdlib assert-test (no pytest infra in plugin repo)**

```python
# hooks/test_session_ctx_router.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _session_ctx as sc

# derive_task: regex-first, imperative + tail
t = sc.derive_task("implementiere die codebook API endpoints", "MayringCoder", "")
assert t and "codebook" in t.lower(), f"derive_task regex: {t!r}"
assert sc.derive_task("hi", "", "") == "", "too-short prompt → empty"

# _normalize done server-side; _git_remote returns str|None, never raises
r = sc._git_remote(cwd="/nonexistent-xyz")
assert r is None, f"_git_remote bad cwd → None, got {r!r}"

print("PASS test_session_ctx_router")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/nileneb/Desktop/mayring-claude-plugin && python3 hooks/test_session_ctx_router.py`
Expected: FAIL (`derive_task`/`_git_remote` missing → AttributeError).

- [ ] **Step 3: Implement in `_session_ctx.py`**

```python
import subprocess

_IMPERATIVES = re.compile(
    r"\b(implementier\w*|implement|fix|repariere?|debug\w*|analysier\w*|analyze|"
    r"refactor\w*|erstell\w*|create|add|füge|baue?|build|teste?|test|deploy\w*|"
    r"migrier\w*|migrate|untersuch\w*|investigate|optimier\w*|optimize|review|"
    r"prüf\w*|check|schreib\w*|write|entferne?|remove|delete|lösch\w*|"
    r"update|aktualisier\w*)\b", re.IGNORECASE)


def _git_remote(cwd: str | None = None) -> str | None:
    """`git -C <cwd> remote get-url origin`, fail-soft → None."""
    try:
        r = subprocess.run(
            ["git", "-C", cwd or os.getcwd(), "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=2)
        out = r.stdout.strip()
        return out or None
    except (OSError, subprocess.SubprocessError):
        return None


def route_project(token: str, cwd_remote: str | None, prompt: str) -> dict:
    """POST /projects/route, fail-soft → {project_id: None}."""
    api = _api()
    body = json.dumps({"cwd_remote": cwd_remote, "prompt": prompt[:600]}).encode()
    req = urllib.request.Request(
        f"{api}/projects/route", data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return {"project_id": None, "name": None, "mode": "unknown",
                "confidence": 0.0, "reason": "route-unreachable"}


def derive_task(prompt: str, project_name: str = "", goal: str = "") -> str:
    """Mayring-Selektionskriterium: Imperativ + Objekt, mit Projekt/Goal-Kontext.
    Regex-first; leerer String wenn nichts Sinnvolles (caller fällt zurück)."""
    p = (prompt or "").strip()
    if len(p) < 12:
        return ""
    m = _IMPERATIVES.search(p)
    seed = p[m.start():m.start() + 100].split("\n")[0].strip() if m else ""
    if not seed and goal:
        seed = goal[:80]
    if not seed:
        return ""
    prefix = f"{project_name}: " if project_name else ""
    return (prefix + seed)[:140]
```

> `re`, `os`, `json`, `urllib` sind in `_session_ctx.py` bereits importiert (Phase 2);
> nur `import subprocess` ergänzen.

- [ ] **Step 4: Run to verify it passes**

Run: `python3 hooks/test_session_ctx_router.py`
Expected: `PASS test_session_ctx_router`. Also `python3 -m py_compile hooks/_session_ctx.py`.

- [ ] **Step 5: Commit**

```bash
git add hooks/_session_ctx.py hooks/test_session_ctx_router.py
git commit -m "feat(hooks): _git_remote + route_project + derive_task"
```

---

## Task 8: Wire routing into `memory_inject.py` (plugin)

**Files:**
- Modify: `/home/nileneb/Desktop/mayring-claude-plugin/hooks/memory_inject.py`

- [ ] **Step 1: Extend `_search` to accept `project_id` + `task_context`**

In `_search(...)` add params and body fields. Change the signature line:

```python
def _search(
    query: str, token: str, *, top_k: int = TOP_K_PRIMARY,
    source_type: str | None = None, char_budget: int = CHAR_BUDGET,
    category_hint: list[str] | None = None,
    project_id: str | None = None, task_context: str = "",
) -> dict:
```

And inside, after `body_dict` is built (before `if source_type:`), add:

```python
    if project_id:
        body_dict["project"] = project_id
    if task_context:
        body_dict["task_context"] = task_context
```

- [ ] **Step 2: Thread through `_multi_lens_search`**

Change its signature + the `pool.submit` call:

```python
def _multi_lens_search(query: str, token: str, *,
                       category_hint: list[str] | None = None,
                       project_id: str | None = None,
                       task_context: str = "") -> dict[str, dict]:
```

In the lens dict, add `project_id`+`task_context` to the `primary` lens kwargs:

```python
        "primary": {"category_hint": category_hint, "project_id": project_id,
                     "task_context": task_context},
```

- [ ] **Step 3: Route + cache + observability in `main()`**

Add imports near the existing `_session_ctx` import:

```python
try:
    from _session_ctx import route_project, _git_remote, derive_task, read_session_ctx, write_session_ctx_field
except ImportError:
    def route_project(*_a, **_k): return {"project_id": None, "name": None, "mode": "unknown", "reason": "no-module"}
    def _git_remote(*_a, **_k): return None
    def derive_task(*_a, **_k): return ""
    def read_session_ctx(*_a, **_k): return None
    def write_session_ctx_field(*_a, **_k): return None
```

In `main()`, after `prompt_categories = _categorize_prompt(prompt, token)` and before `_multi_lens_search`, add:

```python
    ctx = read_session_ctx() or {}
    active = ctx.get("active_project")
    if active is None:
        active = route_project(token, _git_remote(), prompt)
        write_session_ctx_field("active_project", active)
    project_id = (active or {}).get("project_id")
    task = derive_task(prompt, (active or {}).get("name") or "")
```

Change the search call:

```python
    results = _multi_lens_search(prompt, token,
                                 category_hint=prompt_categories or None,
                                 project_id=project_id, task_context=task)
```

Build an observability line and prepend it to the injected block (add right after `pinned_prefix` is computed):

```python
    if active and project_id:
        proj_line = (f"📁 Projekt: {(active.get('name') or project_id)} "
                     f"({active.get('mode','?')}, conf={active.get('confidence',0)} "
                     f"· {active.get('reason','?')})"
                     + (f" · task={task}" if task else "") + "\n\n")
    else:
        proj_line = f"📁 Projekt: — (workspace-weit, {(active or {}).get('mode','?')})\n\n"
    pinned_prefix = proj_line + pinned_prefix
```

- [ ] **Step 4: Add `write_session_ctx_field` to `_session_ctx.py`**

```python
def write_session_ctx_field(key: str, value) -> None:
    """Merge a single field into session_ctx.json (best-effort)."""
    ctx = read_session_ctx(max_age=0) or {}
    ctx[key] = value
    try:
        os.makedirs(os.path.dirname(SESSION_CTX_PATH), exist_ok=True)
        with open(SESSION_CTX_PATH, "w", encoding="utf-8") as f:
            json.dump(ctx, f)
    except OSError:
        pass
```

> `read_session_ctx(max_age=0)` disables the TTL check so we don't drop the
> codebook block when only writing the project field.

- [ ] **Step 5: Verify compile + dry-run main with a fake prompt**

Run:
```bash
cd /home/nileneb/Desktop/mayring-claude-plugin
python3 -m py_compile hooks/memory_inject.py hooks/_session_ctx.py
echo '{"prompt":"fix the auth bug in MayringCoder","session_id":"t1"}' | python3 hooks/memory_inject.py | head -5
```
Expected: output starts with a `📁 Projekt:` line; no traceback.

- [ ] **Step 6: Commit**

```bash
git add hooks/memory_inject.py hooks/_session_ctx.py
git commit -m "feat(hooks): route active_project + thread project_id/task_context + observability"
```

---

## Task 9: Smoke check `projects_route_cwd_remote` (MayringCoder)

**Files:**
- Modify: `tools/smoke_test_production.py`

- [ ] **Step 1: Add the check function (near the other check_* defs)**

```python
def check_projects_route_cwd_remote(api: str, token: str) -> CheckResult:
    """POST /projects/route with a known remote → 200 + project_id set;
    with no signal + nonsense prompt → 200 + project_id null."""
    code, body, _ = _http("POST", f"{api}/projects/route", token,
        body={"cwd_remote": "git@github.com:Nileneb/MayringCoder.git",
              "prompt": "fix the retrieval pipeline"})
    if code != 200 or not isinstance(body, dict):
        return CheckResult("projects_route_cwd_remote", False, f"http={code}")
    if not body.get("project_id"):
        return CheckResult("projects_route_cwd_remote", False,
                           f"cwd-remote gave no project_id: {body}")
    code2, body2, _ = _http("POST", f"{api}/projects/route", token,
        body={"cwd_remote": None, "prompt": "zxqw nonsense %%%"})
    null_ok = code2 == 200 and (body2 or {}).get("project_id") is None
    return CheckResult("projects_route_cwd_remote",
                       bool(body.get("project_id")) and null_ok,
                       f"hard={body.get('reason')} null_branch_ok={null_ok}")
```

- [ ] **Step 2: Register it in the checks list**

Add to the registry tuple list (near `("stop_hook_e2e", ...)`):

```python
    ("projects_route_cwd_remote",     check_projects_route_cwd_remote),
```

- [ ] **Step 3: Add a coverage-map row** (so `coverage_map_complete` stays green if a tracking issue is opened) — only if an issue number is assigned; otherwise skip. Commit:

```bash
git add tools/smoke_test_production.py
git commit -m "feat(smoke): projects_route_cwd_remote check"
```

---

## Task 10: Deploy + prod verification

- [ ] **Step 1: Push MayringCoder (triggers Build & Push → deploy → smoke)**

```bash
cd /home/nileneb/Desktop/MayringCoder && git push origin master
```

- [ ] **Step 2: After deploy lands, run the project backfill on prod**

The backfill needs the prod DB + Chroma. Run inside the api container (mirrors codebook import):
```bash
ssh nileneb@u-server 'docker exec mayring-mayring-api-1 sh -c "cd /app && PYTHONPATH=. python3 tools/embed_projects.py --apply"'
```
Expected: `projects=<n> embedded=<n> (APPLIED)`.
> NOTE: SSH may require explicit user confirmation (prod write). If denied, ask the user to run it.

- [ ] **Step 3: Push the plugin + reload**

```bash
cd /home/nileneb/Desktop/mayring-claude-plugin && git push origin main
```
Then the user runs `/reload-plugins`.

- [ ] **Step 4: Verify end-to-end**

```bash
# route endpoint live
python3 - <<'PY'
import os,json,urllib.request
tok=open(os.path.expanduser("~/.config/mayring/hook.jwt")).read().strip()
body=json.dumps({"cwd_remote":"git@github.com:Nileneb/MayringCoder.git","prompt":"fix retrieval"}).encode()
req=urllib.request.Request("https://mcp.linn.games/projects/route",data=body,
  headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json"})
print(json.loads(urllib.request.urlopen(req,timeout=10).read()))
PY
```
Expected: `{'project_id': <id>, 'reason': 'cwd-remote', 'mode': 'coding', ...}`.

- [ ] **Step 5: Manually trigger smoke, confirm green incl. new check**

```bash
cd /home/nileneb/Desktop/MayringCoder
gh workflow run post-deploy-smoke.yml --ref master
# poll; expect 47/47 (or N+1) passed, projects_route_cwd_remote OK
```

---

## Self-Review

**Spec coverage:**
- cwd-remote hard match-or-create → Task 4 (`route`), Task 5 (endpoint). ✓
- Semantic match (existing only, margin) → Task 4 (`_semantic_match`). ✓
- Project embeddings in Chroma `projects`, `proj:<id>`, no SQLite column → Task 4 (`_upsert_embedding`), Task 6 (backfill). ✓
- `derive_task` + `task_context` → Task 7 (derive_task), Task 8 (threading). ✓
- session_ctx.json `active_project` cache → Task 8 (`write_session_ctx_field`). ✓
- Threading `project` into /memory/search (consumer exists) → Task 8 Step 1-2. ✓
- Observability line → Task 8 Step 3. ✓
- Index only DDL → Task 1. ✓
- Verification (route, semantic, cache, smoke) → Task 9, Task 10. ✓
- Fail-soft (git/route/embed) → Task 7 (`_git_remote`/`route_project` try/except), Task 5 (`_embed_one` try/except). ✓

**Placeholder scan:** Task 9 Step 3 is conditional ("only if issue number assigned") — acceptable (coverage-map only requires rows for *closed issues*; a new feature has no closed issue yet). No code placeholders.

**Type consistency:** `route(conn, chroma, workspace, *, cwd_remote, prompt, embed_fn)` used identically in Task 4 tests, Task 5 endpoint. `project_embed_text(name, source_ref, source_type)` consistent Task 3/4/6. `route_project`/`_git_remote`/`derive_task` signatures consistent Task 7/8. session_ctx field `active_project` consistent Task 8.
