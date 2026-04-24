from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_workspace

router = APIRouter(tags=["reports"])

_ROOT = Path(__file__).parent.parent.parent.parent


@router.get("/reports")
async def list_reports(
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """List analysis reports for this workspace."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", workspace_id):
        raise HTTPException(status_code=400, detail="Invalid workspace_id")
    safe_workspace_id = re.sub(r"[^A-Za-z0-9_-]", "", workspace_id)
    if not safe_workspace_id or safe_workspace_id != workspace_id:
        raise HTTPException(status_code=400, detail="Invalid workspace_id")
    base_reports_dir = (_ROOT / "reports").resolve()

    reports = []
    if base_reports_dir.exists():
        workspace_prefix = f"{safe_workspace_id}_"
        candidate_dirs = [base_reports_dir]
        candidate_dirs.extend(
            p
            for p in base_reports_dir.iterdir()
            if p.is_dir() and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", p.name)
        )
        for candidate_dir in candidate_dirs:
            markdown_files = [
                p
                for p in candidate_dir.iterdir()
                if p.is_file()
                and re.fullmatch(r"[A-Za-z0-9_.-]+\.md", p.name)
                and p.name.startswith(workspace_prefix)
            ]
            for f in sorted(markdown_files, key=lambda p: p.stat().st_mtime, reverse=True):
                reports.append({"name": f.name, "size": f.stat().st_size})
    return {"workspace_id": safe_workspace_id, "reports": reports, "count": len(reports)}
