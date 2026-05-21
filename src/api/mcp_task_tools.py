"""MCP tools for the task tracker — agents & humans manage work items.

Mirrors the harness Task tools (create/list/update/complete). Workspace is
resolved exactly like the memory tools (active-workspace aware); created_by
is the JWT sub for humans, 'agent' when no human identity is present.
"""
from __future__ import annotations
from src.api.mcp_auth import _enforce_tenant, _effective_workspace_id, _effective_user_id
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
                "tasks": _t.list_tasks(_get_conn(), ws, status=status, tag=tag, priority=priority)}

    @mcp.tool()
    def task_update(task_id: str, title: str | None = None, description: str | None = None,
                    status: str | None = None, priority: str | None = None,
                    due_date: str | None = None, tags: str | None = None,
                    workspace_id: str | None = None) -> dict:
        """Update fields of a work item (status, priority, etc.)."""
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        fields = {k: v for k, v in dict(title=title, description=description, status=status,
                  priority=priority, due_date=due_date, tags=tags).items() if v is not None}
        updated = _t.update_task(_get_conn(), ws, task_id, **fields)
        return updated or {"error": "task not found", "task_id": task_id}

    @mcp.tool()
    def task_complete(task_id: str, workspace_id: str | None = None) -> dict:
        """Mark a work item as done."""
        ws = _enforce_tenant(workspace_id) or _effective_workspace_id()
        done = _t.complete_task(_get_conn(), ws, task_id)
        return done or {"error": "task not found", "task_id": task_id}
