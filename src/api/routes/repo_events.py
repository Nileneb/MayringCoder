"""POST /repo-events — the reusable GitHub Action posts repo push/CI/security
events here. Push re-ingests the newest version; CI/security are logged in
hook_events + a lightweight searchable repo_event chunk (recall + IGIO-Lens).

WHY(repo-watching C+D): closes the gap where only MayringCoder auto-ingested and
gives every watched repo's CI/security a memory presence."""
from __future__ import annotations

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


def _record_repo_event(*a):  # filled in Task 3
    pass


def _repo_event_chunk(*a):   # filled in Task 4
    pass


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
