"""REST endpoints for the task tracker — workspace-scoped via get_workspace."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from src.api.auth import get_workspace, get_token_info
from src.api.jwt_auth import TokenInfo
from src.api.dependencies import get_conn as _get_conn
from src.api.routes.models import TaskCreateRequest, TaskUpdateRequest
from mayring_core.memory import tasks as _t

router = APIRouter()


@router.post("/tasks")
async def create_task(req: TaskCreateRequest, workspace_id: str = Depends(get_workspace),
                      info: TokenInfo = Depends(get_token_info)) -> dict:
    try:
        return _t.create_task(_get_conn(), workspace_id=workspace_id, title=req.title,
                              description=req.description, priority=req.priority,
                              due_date=req.due_date, tags=req.tags, created_by=info.sub,
                              linked_chunk_id=req.linked_chunk_id, scope_key=req.scope_key,
                              external_id=req.external_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/tasks")
def list_tasks(status: str | None = None, tag: str | None = None,
                     priority: str | None = None,
                     workspace_id: str = Depends(get_workspace)) -> dict:
    return {"workspace_id": workspace_id,
            "tasks": _t.list_tasks(_get_conn(), workspace_id, status=status, tag=tag, priority=priority)}


@router.get("/tasks/goals")
def list_workspace_goals(workspace_id: str = Depends(get_workspace)) -> dict:
    return {"workspace_id": workspace_id,
            "goals": _t.list_workspace_goals(_get_conn(), workspace_id)}


@router.patch("/tasks/by-external/{external_id}")
def update_task_by_external(external_id: str, req: TaskUpdateRequest,
                                  workspace_id: str = Depends(get_workspace)) -> dict:
    row = _get_conn().execute(
        "SELECT task_id FROM tasks WHERE workspace_id=? AND external_id=?",
        (workspace_id, external_id),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="task not found")
    fields = {k: v for k, v in req.model_dump(exclude_none=True).items()}
    try:
        updated = _t.update_task(_get_conn(), workspace_id, row[0], **fields)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return updated


@router.patch("/tasks/{task_id}")
def update_task(task_id: str, req: TaskUpdateRequest, workspace_id: str = Depends(get_workspace)) -> dict:
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        updated = _t.update_task(_get_conn(), workspace_id, task_id, **fields)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail="task not found")
    return updated


@router.post("/tasks/{task_id}/complete")
def complete_task(task_id: str, workspace_id: str = Depends(get_workspace)) -> dict:
    done = _t.complete_task(_get_conn(), workspace_id, task_id)
    if done is None:
        raise HTTPException(status_code=404, detail="task not found")
    return done


@router.delete("/tasks/{task_id}")
def delete_task(task_id: str, workspace_id: str = Depends(get_workspace)) -> dict:
    if not _t.delete_task(_get_conn(), workspace_id, task_id):
        raise HTTPException(status_code=404, detail="task not found")
    return {"deleted": task_id}
