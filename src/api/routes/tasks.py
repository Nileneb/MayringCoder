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
async def create_task(req: TaskCreateRequest, workspace_id: str = Depends(get_workspace),
                      info: TokenInfo = Depends(get_token_info)) -> dict:
    try:
        return _t.create_task(_get_conn(), workspace_id=workspace_id, title=req.title,
                              description=req.description, priority=req.priority,
                              due_date=req.due_date, tags=req.tags, created_by=info.sub,
                              linked_chunk_id=req.linked_chunk_id, scope_key=req.scope_key)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/tasks")
async def list_tasks(status: str | None = None, tag: str | None = None,
                     priority: str | None = None, include_existing: bool = True,
                     workspace_id: str = Depends(get_workspace)) -> dict:
    conn = _get_conn()
    tasks = _t.list_tasks(conn, workspace_id, status=status, tag=tag, priority=priority)
    # Surface the workspace's existing work (IGIO goals + research questions) as
    # read-only rows so an established workspace isn't shown an empty tracker.
    # Skipped when filtering by status/priority (those are real-task concepts).
    if include_existing and not status and not priority:
        tasks = tasks + _t.list_tracked_work(conn, workspace_id)
    return {"workspace_id": workspace_id, "tasks": tasks}


@router.patch("/tasks/{task_id}")
async def update_task(task_id: str, req: TaskUpdateRequest, workspace_id: str = Depends(get_workspace)) -> dict:
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        updated = _t.update_task(_get_conn(), workspace_id, task_id, **fields)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail="task not found")
    return updated


@router.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str, workspace_id: str = Depends(get_workspace)) -> dict:
    done = _t.complete_task(_get_conn(), workspace_id, task_id)
    if done is None:
        raise HTTPException(status_code=404, detail="task not found")
    return done


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, workspace_id: str = Depends(get_workspace)) -> dict:
    if not _t.delete_task(_get_conn(), workspace_id, task_id):
        raise HTTPException(status_code=404, detail="task not found")
    return {"deleted": task_id}
