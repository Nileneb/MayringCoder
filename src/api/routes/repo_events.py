"""POST /repo-events — the reusable GitHub Action posts repo push/CI/security
events here. Push re-ingests the newest version; CI/security are logged in
hook_events + a lightweight searchable repo_event chunk (recall + IGIO-Lens).

WHY(repo-watching C+D): closes the gap where only MayringCoder auto-ingested and
gives every watched repo's CI/security a memory presence."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import get_token_info, _is_privileged
from src.api.jwt_auth import TokenInfo
from src.api.routes.jobs import enqueue_populate
from src.api.routes.models import RepoEventRequest
from src.api.dependencies import get_conn as _get_conn

router = APIRouter()


def _resolve_workspace(conn, repo: str) -> str:
    """repo-url → projects.workspace_id; match-or-create under 'system' if unknown.

    WHY(repo-watching): mirrors src/api/routes/projects.py match-or-create so an
    unknown repo is never rejected — it gets a 'system' project on first sight.
    """
    row = conn.execute(
        "SELECT workspace_id FROM projects WHERE source_type='github' AND source_ref=?",
        (repo,),
    ).fetchone()
    if row is not None:
        return row[0]
    now = datetime.now(timezone.utc).isoformat()
    pid = str(uuid.uuid4())
    name = repo.rsplit("/", 1)[-1]
    conn.execute(
        "INSERT INTO projects (id, workspace_id, name, source_type, source_ref, created_at, updated_at) "
        "VALUES (?, 'system', ?, 'github', ?, ?, ?)",
        (pid, name, repo, now, now),
    )
    conn.commit()
    return "system"


_AXIS = {
    ("workflow_run", "failure"): "issue",
    ("workflow_run", "success"): "outcome",
    ("security", None): "issue",
}


def _record_repo_event(conn, workspace_id: str, hook_type: str, req: RepoEventRequest) -> None:
    """Log a CI/security event into hook_events (reuse payload JSON, NO migration).

    WHY(repo-watching): idempotent on the exact serialized payload so a
    GitHub re-delivery of the same event does not create a duplicate row.
    Exact-match (not LIKE) is used deliberately: it handles None sha/workflow
    correctly (json null), which a LIKE '%"workflow": "None"%' pattern would not."""
    payload = json.dumps({
        "repo": req.repo, "sha": req.sha, "ref": req.ref,
        "conclusion": req.conclusion, "workflow": req.workflow,
        "severity": req.severity, "summary": req.summary, "url": req.url,
    }, default=str)
    existing = conn.execute(
        "SELECT 1 FROM hook_events WHERE workspace_id=? AND hook_type=? AND payload=? LIMIT 1",
        (workspace_id, hook_type, payload),
    ).fetchone()
    if existing is not None:
        return
    conn.execute(
        "INSERT INTO hook_events (workspace_id, device_id, hook_type, fired_at, payload) "
        "VALUES (?, 'github-action', ?, ?, ?)",
        (workspace_id, hook_type, datetime.now(timezone.utc).isoformat(), payload),
    )
    conn.commit()


def _repo_event_chunk(conn, workspace_id: str, req: RepoEventRequest, axis: str) -> None:
    """Insert a lightweight, searchable source+chunk for a CI/security event,
    tagged with the deterministic igio axis (NO LLM in the hot path).

    WHY(repo-watching): gives every watched repo's CI/security a memory presence
    (recall + IGIO-Lens). igio_axis is written via update_chunk_igio_axis because
    insert_chunk does not persist igio columns."""
    from mayring_core.memory.store import upsert_source, insert_chunk, update_chunk_igio_axis
    from mayring_core.memory.schema import Source, Chunk
    now = datetime.now(timezone.utc).isoformat()
    if req.event_type == "workflow_run":
        text = f"CI {req.workflow or ''} {req.conclusion or ''} on {req.repo}@{(req.sha or '')[:8]}".strip()
    else:
        text = f"Security {req.severity or ''}: {req.summary or ''} in {req.repo}".strip()
    sid = f"repo_event:{req.repo}:{req.event_type}:{(req.sha or now)[:12]}"
    thash = Chunk.compute_text_hash(text)   # already 'sha256:...'
    src = Source(
        source_id=sid, source_type="repo_event", repo=req.repo,
        path=req.url or "", branch=req.ref or "main", commit=req.sha or "",
        content_hash=thash, captured_at=now,
    )
    upsert_source(conn, src, workspace_id=workspace_id)
    chunk = Chunk(
        chunk_id=Chunk.make_id(sid, 0, "event"), source_id=sid, chunk_level="event",
        ordinal=0, text=text, text_hash=thash, created_at=now, workspace_id=workspace_id,
    )
    insert_chunk(conn, chunk, workspace_id=workspace_id)
    if axis:
        update_chunk_igio_axis(conn, chunk.chunk_id, axis, confidence=0.9)


@router.post("/repo-events")
async def repo_events(req: RepoEventRequest, info: TokenInfo = Depends(get_token_info)) -> dict:
    if not _is_privileged(info):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="repo-events requires a service/admin token",
        )
    conn = _get_conn()
    workspace_id = _resolve_workspace(conn, req.repo)

    if req.event_type == "push":
        job_id = enqueue_populate(req.repo, workspace_id)
        return {"ok": True, "action": "populate", "job_id": job_id, "workspace_id": workspace_id}

    hook_type = "repo_ci" if req.event_type == "workflow_run" else "repo_security"
    _record_repo_event(conn, workspace_id, hook_type, req)
    axis = _AXIS.get((req.event_type, req.conclusion)) or _AXIS.get((req.event_type, None)) or ""
    _repo_event_chunk(conn, workspace_id, req, axis)
    return {"ok": True, "action": hook_type, "workspace_id": workspace_id, "igio_axis": axis}
