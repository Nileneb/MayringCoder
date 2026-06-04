"""Project Router (Slice 1): POST /projects/route.

Attaches active_project_id from the strongest signal: cwd-git-remote (hard,
match-or-create) → semantic match against existing projects → null. See
docs/superpowers/specs/2026-05-24-project-router-design.md.
"""
from __future__ import annotations

import math
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from mayring_core.memory.store import PROJECT_GROUP_PALETTE
from src.api.auth import _is_privileged, get_token_info, get_workspace
from src.api.dependencies import get_conn as _get_conn
from src.api.jwt_auth import TokenInfo

router = APIRouter(tags=["projects"])

# WHY(py/polynomial-redos): single separator after host (git remotes have exactly
# one ':' or '/'); the prior `[:/]+` backtracked polynomially on '::' repetitions.
# WHY(2026-05-30, code-scanning py/polynomial-redos): the old pattern
# `(?P<name>[^/]+?)(?:\.git)?/?$` combined a lazy quantifier with an optional
# suffix + end-anchor under .search() → polynomial backtracking on crafted input.
# `name` is now a single greedy non-slash run (linear, bounded by '/'); the
# optional `.git` suffix + trailing slash are stripped in _normalize_remote.
_REMOTE_RE = re.compile(
    r"github\.com[:/](?P<owner>[^/]+)/(?P<name>[^/]+)",
    re.IGNORECASE,
)
_SEMANTIC_MIN = 0.55
_SEMANTIC_MARGIN = 0.08

_MODE_RE = {
    "coding": re.compile(r"\b(repo|migration|endpoint|deploy|CI|test|bug|refactor|"
                         r"commit|PR|merge|api|schema)\b", re.IGNORECASE),
    "research": re.compile(r"\b(paper|DOI|arxiv|pubmed|RQ|research question|"
                           r"systematic review|p[1-8]|hypoth)\b", re.IGNORECASE),
}


def _normalize_remote(remote: str) -> str | None:
    """git@/https/ssh GitHub remote → 'owner/name' lowercased, else None."""
    if not remote:
        return None
    m = _REMOTE_RE.search(remote.strip())
    if not m:
        return None
    name = m.group("name").rstrip("/").removesuffix(".git")
    return f"{m.group('owner')}/{name}".lower()


def canonical_repo_ref(ref: str) -> str:
    """DIE kanonische source_ref-Form für ein Repo, damit dasselbe Repo NIE als Dublette
    angelegt wird ('nileneb/x' via /projects/route vs 'https://github.com/Nileneb/x' via
    /repo-events). github → 'owner/name'-Slug, sonst canonicalize_url. EINE Funktion für
    beide Anlage-Stellen — uneinheitliche Keys spalten sonst die Daten (#4-Dublette 2026-05-29)."""
    from mayring_core.memory.schema import canonicalize_url
    return _normalize_remote(ref) or canonicalize_url(ref or "")


def _next_palette_color(conn, ws: str) -> str:
    used = {r[0] for r in conn.execute(
        "SELECT color FROM project_groups WHERE workspace_id=?", (ws,)).fetchall()}
    for c in PROJECT_GROUP_PALETTE:
        if c not in used:
            return c
    n = conn.execute("SELECT COUNT(*) FROM project_groups WHERE workspace_id=?",
                     (ws,)).fetchone()[0]
    return PROJECT_GROUP_PALETTE[n % len(PROJECT_GROUP_PALETTE)]


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def project_embed_text(name: str, source_ref: str, source_type: str) -> str:
    return " ".join(p for p in (name, source_ref, source_type) if p).strip()


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
    # ChromaDB returns embeddings as a numpy array → `x or []` raises
    # "truth value of an array is ambiguous". Use explicit None checks.
    embs = data.get("embeddings")
    embs = [] if embs is None else embs
    metas = data.get("metadatas")
    metas = [] if metas is None else metas
    scored = sorted(
        ((_cosine(prompt_emb, e), (m or {}).get("project_id"))
         for e, m in zip(embs, metas) if e is not None and len(e)),
        key=lambda x: x[0], reverse=True,
    )
    if not scored:
        return None, 0.0, 0.0
    top_score, top_pid = scored[0]
    margin = top_score - (scored[1][0] if len(scored) > 1 else 0.0)
    return top_pid, top_score, margin


def route(conn, chroma, workspace: str, *, cwd_remote: str | None,
          prompt: str, embed_fn: Callable[[str], list[float]]) -> dict:
    """Decide active project: cwd-remote (match-or-create) → semantic → null."""
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
        _upsert_embedding(chroma, pid,
                          project_embed_text(name, owner_name, "github"), embed_fn)
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


class RouteRequest(BaseModel):
    cwd_remote: str | None = None
    prompt: str = ""


class GroupCreate(BaseModel):
    name: str
    color: str | None = None


class GroupUpdate(BaseModel):
    name: str | None = None
    color: str | None = None


class GroupAssign(BaseModel):
    repo_slug: str
    group_id: str | None = None


def _embed_one(text: str) -> list[float]:
    from mayring_core.config import EMBEDDING_MODEL, OLLAMA_TIMEOUT
    from mayring_core.ollama_client import embed_single
    url = os.environ.get("OLLAMA_URL", "https://three.linn.games")
    try:
        return embed_single(url, EMBEDDING_MODEL, text, timeout=OLLAMA_TIMEOUT) or []
    except Exception:  # noqa: BLE001 — embed failure must never 500 the router
        return []


@router.post("/projects/route")
async def route_project(req: RouteRequest, ws: str = Depends(get_workspace)) -> dict:
    from mayring_core.memory.store import get_chroma_collection
    conn = _get_conn()
    chroma = get_chroma_collection("projects")
    return route(conn, chroma, ws, cwd_remote=req.cwd_remote,
                 prompt=req.prompt, embed_fn=_embed_one)


@router.get("/projects")
async def list_projects(ws: str = Depends(get_workspace)) -> dict:
    """Projekte des Workspace (id, name, repo, source_type) — die bisher fehlende Sicht auf
    die projects-Tabelle (Projekte→Repos). Read-only. Behebt die 'blackbox'-Lücke #4."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT p.id, p.name, p.source_type, p.source_ref, p.created_at, p.updated_at, "
        "       p.group_id, g.name, g.color "
        "FROM projects p LEFT JOIN project_groups g ON g.id = p.group_id "
        "WHERE p.workspace_id=? ORDER BY p.updated_at DESC",
        (ws,),
    ).fetchall()
    return {"workspace_id": ws, "count": len(rows), "projects": [
        {"id": r[0], "name": r[1], "source_type": r[2], "repo": r[3],
         "created_at": r[4], "updated_at": r[5],
         "group_id": r[6], "group_name": r[7], "group_color": r[8]} for r in rows]}


@router.post("/projects/claim-system")
async def claim_system_repos(
    info: TokenInfo = Depends(get_token_info),
    ws: str = Depends(get_workspace),
) -> dict:
    """Beansprucht alle 'system'-github-Projekte + ihre Repo-CI/Security-Events + repo_event-
    Chunks für DIESEN Workspace. Pre-launch single-user: alle Repos gehören dem einzigen User.
    Behebt die Blackbox (Repo-Events landeten in 'system' → user-gescoptes Dashboard zeigte 0).
    Idempotent + admin-gated."""
    if not _is_privileged(info):
        raise HTTPException(status_code=403, detail="admin/service token required")
    conn = _get_conn()
    # Smoke-Suite-Wegwerf-Repos (github.com/smoke/repo-*) NIE in den User-Workspace
    # (sonst pollutet jeder Smoke-Lauf die Projekt-Sicht).
    NOT_SMOKE = "lower(source_ref) NOT LIKE '%/smoke/repo-%'"
    repos = [r[0] for r in conn.execute(
        f"SELECT source_ref FROM projects WHERE workspace_id='system' AND source_type='github' "
        f"AND {NOT_SMOKE}").fetchall()]
    ev_n = conn.execute(
        "SELECT COUNT(*) FROM hook_events WHERE workspace_id='system' "
        "AND hook_type IN ('repo_ci','repo_security')").fetchone()[0]
    # repo_event-Chunks zuerst (über ihre 'system'-sources), DANN die sources umhängen.
    sids = [r[0] for r in conn.execute(
        "SELECT source_id FROM sources WHERE workspace_id='system' AND source_type='repo_event'"
    ).fetchall()]
    if sids:
        ph = ",".join("?" for _ in sids)
        conn.execute(f"UPDATE chunks SET workspace_id=? WHERE source_id IN ({ph})", (ws, *sids))
        conn.execute("UPDATE sources SET workspace_id=? WHERE workspace_id='system' "
                     "AND source_type='repo_event'", (ws,))
    conn.execute(f"UPDATE projects SET workspace_id=? WHERE workspace_id='system' "
                 f"AND source_type='github' AND {NOT_SMOKE}", (ws,))
    conn.execute("UPDATE hook_events SET workspace_id=? WHERE workspace_id='system' "
                 "AND hook_type IN ('repo_ci','repo_security')", (ws,))
    # Korrektur: bereits (über-)geclaimte Smoke-Projekte zurück nach 'system'.
    unsmoked = conn.execute(
        f"SELECT COUNT(*) FROM projects WHERE workspace_id=? AND source_type='github' "
        f"AND NOT ({NOT_SMOKE})", (ws,)).fetchone()[0]
    conn.execute(f"UPDATE projects SET workspace_id='system' WHERE workspace_id=? "
                 f"AND source_type='github' AND NOT ({NOT_SMOKE})", (ws,))
    conn.commit()
    return {"workspace_id": ws, "claimed_projects": len(repos), "claimed_events": ev_n,
            "repo_event_chunks": len(sids), "unsmoked_back_to_system": unsmoked, "repos": repos}


@router.post("/projects/dedup")
async def dedup_projects(
    info: TokenInfo = Depends(get_token_info),
    ws: str = Depends(get_workspace),
) -> dict:
    """Führt github-Projekt-Dubletten desselben Repos zusammen (nileneb/x vs
    https://github.com/Nileneb/x → eine kanonische source_ref). Behält das älteste,
    setzt dessen source_ref auf die kanonische Form, löscht die Dubletten. Idempotent +
    admin-gated. Behebt den Datenmüll, der sonst Routing/Anzeige spaltet (#4 2026-05-29)."""
    if not _is_privileged(info):
        raise HTTPException(status_code=403, detail="admin/service token required")
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, source_ref FROM projects WHERE workspace_id=? AND source_type='github' "
        "ORDER BY created_at", (ws,),
    ).fetchall()
    groups: dict[str, list[tuple[str, str]]] = {}
    for pid, ref in rows:
        groups.setdefault(canonical_repo_ref(ref), []).append((pid, ref))
    merged = 0
    for canon, items in groups.items():
        keep_id = items[0][0]
        # Kept-Projekt auf die kanonische source_ref normalisieren (auch bei Singles,
        # damit künftige /repo-events matchen statt eine neue Dublette anzulegen).
        conn.execute("UPDATE projects SET source_ref=? WHERE id=?", (canon, keep_id))
        for pid, _ref in items[1:]:
            conn.execute("DELETE FROM projects WHERE id=?", (pid,))
            merged += 1
    conn.commit()
    return {"workspace_id": ws, "merged": merged, "canonical_groups": len(groups)}


@router.get("/project-groups")
async def list_project_groups(ws: str = Depends(get_workspace)) -> dict:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT g.id, g.name, g.color, "
        "  (SELECT COUNT(*) FROM projects p WHERE p.group_id=g.id AND p.workspace_id=g.workspace_id) "
        "FROM project_groups g WHERE g.workspace_id=? ORDER BY g.name",
        (ws,),
    ).fetchall()
    return {"workspace_id": ws, "groups": [
        {"id": r[0], "name": r[1], "color": r[2], "member_count": r[3]} for r in rows]}


@router.post("/project-groups")
async def create_project_group(req: GroupCreate, ws: str = Depends(get_workspace)) -> dict:
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    conn = _get_conn()
    dupe = conn.execute(
        "SELECT 1 FROM project_groups WHERE workspace_id=? AND lower(name)=lower(?)",
        (ws, name)).fetchone()
    if dupe:
        raise HTTPException(status_code=409, detail=f"group '{name}' already exists")
    color = req.color or _next_palette_color(conn, ws)
    if color not in PROJECT_GROUP_PALETTE:
        raise HTTPException(status_code=422, detail="color not in palette")
    gid = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO project_groups(id,workspace_id,name,color,created_at,updated_at) "
        "VALUES (?,?,?,?,?,?)", (gid, ws, name, color, now, now))
    conn.commit()
    return {"id": gid, "name": name, "color": color, "member_count": 0}


@router.patch("/project-groups/{group_id}")
async def update_project_group(group_id: str, req: GroupUpdate,
                               ws: str = Depends(get_workspace)) -> dict:
    conn = _get_conn()
    row = conn.execute(
        "SELECT name, color FROM project_groups WHERE id=? AND workspace_id=?",
        (group_id, ws)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="group not found")
    name = row[0] if req.name is None else (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    if req.name is not None:
        dupe = conn.execute(
            "SELECT 1 FROM project_groups WHERE workspace_id=? AND lower(name)=lower(?) "
            "AND id<>?", (ws, name, group_id)).fetchone()
        if dupe:
            raise HTTPException(status_code=409, detail=f"group '{name}' already exists")
    color = row[1] if req.color is None else req.color
    if req.color is not None and color not in PROJECT_GROUP_PALETTE:
        raise HTTPException(status_code=422, detail="color not in palette")
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE project_groups SET name=?, color=?, updated_at=? WHERE id=? AND workspace_id=?",
        (name, color, now, group_id, ws))
    conn.commit()
    return {"id": group_id, "name": name, "color": color}


@router.delete("/project-groups/{group_id}")
async def delete_project_group(group_id: str, ws: str = Depends(get_workspace)) -> dict:
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM project_groups WHERE id=? AND workspace_id=?",
        (group_id, ws)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="group not found")
    conn.execute("UPDATE projects SET group_id=NULL WHERE group_id=? AND workspace_id=?",
                 (group_id, ws))
    conn.execute("DELETE FROM project_groups WHERE id=? AND workspace_id=?",
                 (group_id, ws))
    conn.commit()
    return {"ok": True, "id": group_id}


@router.post("/project-groups/assign")
async def assign_repo_group(req: GroupAssign, ws: str = Depends(get_workspace)) -> dict:
    """Ordnet ein Repo einer Gruppe zu. Legt das Projekt an, falls es noch keins gibt
    (canonical source_ref — gleiche Anlage-Logik wie /repo-events). group_id=None löst."""
    conn = _get_conn()
    if req.group_id is not None:
        grp = conn.execute(
            "SELECT 1 FROM project_groups WHERE id=? AND workspace_id=?",
            (req.group_id, ws)).fetchone()
        if not grp:
            raise HTTPException(status_code=404, detail="group not found")
    ref = canonical_repo_ref(req.repo_slug)
    now = datetime.now(timezone.utc).isoformat()
    row = conn.execute(
        "SELECT id FROM projects WHERE workspace_id=? AND source_type='github' "
        "AND lower(source_ref)=lower(?)", (ws, ref)).fetchone()
    if row:
        pid = row[0]
        conn.execute("UPDATE projects SET group_id=?, updated_at=? WHERE id=?",
                     (req.group_id, now, pid))
    else:
        pid = uuid.uuid4().hex
        name = ref.split("/")[-1] if "/" in ref else ref
        conn.execute(
            "INSERT INTO projects(id,workspace_id,name,source_type,source_ref,"
            "group_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (pid, ws, name, "github", ref, req.group_id, now, now))
    conn.commit()
    return {"ok": True, "project_id": pid, "repo": ref, "group_id": req.group_id}
