"""Server-managed repo-watch list — dashboard toggles active/inactive; the local
CI-warner hook fetches the active set from here (replacing the hardcoded default).

GET  /stats/watch-repos          → all watched repos for the workspace
POST /stats/watch-repos          → upsert one repo's active/alerts; on activate,
                                    kick off an ingest (enqueue_populate).
Workspace-scoped (a user manages their own workspace's repos).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api import watch_store
from src.api.auth import get_workspace

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats/watch-repos")
async def list_watch_repos(workspace_id: str = Depends(get_workspace)) -> dict:
    return {"repos": watch_store.get_watched(workspace_id)}


class WatchRepoRequest(BaseModel):
    repo_slug: str
    active: bool = True
    alerts: list[str] = ["ci", "code_scanning", "dependabot"]
    hook_id: int | None = None          # GitHub webhook id (for later deletion)
    secret: str | None = None           # HMAC secret — server-only, verifies incoming hooks
    source: str | None = None           # 'webhook' (new) vs legacy oidc/manual


@router.post("/stats/watch-repos")
async def set_watch_repo(
    req: WatchRepoRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    ingested_at = None
    if req.active:
        # Activating a repo ingests it so its content is searchable. Best-effort:
        # a failed ingest must not block the toggle (the watch state still flips).
        try:
            from src.api.routes.jobs import enqueue_populate
            job_id = enqueue_populate(req.repo_slug, workspace_id)
            ingested_at = datetime.now(timezone.utc).isoformat()
            logger.info("watch-repos: activated %s → ingest job %s", req.repo_slug, job_id)
        except Exception:
            logger.exception("watch-repos: ingest enqueue failed for %s", req.repo_slug)
    row = watch_store.set_watched(
        workspace_id, req.repo_slug, active=req.active,
        alerts=req.alerts, ingested_at=ingested_at,
        hook_id=req.hook_id, secret=req.secret, source=req.source,
    )
    return {"ok": True, "repo": row}
