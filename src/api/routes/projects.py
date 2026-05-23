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

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.auth import get_workspace
from src.api.dependencies import get_conn as _get_conn

router = APIRouter(tags=["projects"])

_REMOTE_RE = re.compile(
    r"github\.com[:/]+(?P<owner>[^/]+)/(?P<name>[^/]+?)(?:\.git)?/?$",
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
    return f"{m.group('owner')}/{m.group('name')}".lower()


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
    embs = data.get("embeddings") or []
    metas = data.get("metadatas") or []
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
