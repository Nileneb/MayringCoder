"""Codebook-/goal-anchor stats for the admin dashboard.

Mounted under ``/stats/`` so the production nginx whitelist (``/stats/*``)
already covers it — no nginx config change required.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_token_info
from src.api.dependencies import get_conn as _conn
from src.api.jwt_auth import TokenInfo

router = APIRouter()


def _is_admin(info: TokenInfo) -> bool:
    return "*" in info.scopes or "admin" in info.scopes


@router.get("/stats/categories-overview")
def categories_overview(info: TokenInfo = Depends(get_token_info)) -> dict:
    """List all codebook categories grouped by Workspace → Goal, plus unlinked.

    Admin (scope '*') sees every workspace. Regular JWT sees only its own.
    A category can appear in multiple (workspace, goal) buckets when its chunks
    span several sources/goals.
    """
    conn = _conn()
    admin = _is_admin(info)
    params: list = []
    ws_clause = ""
    if not admin:
        ws_clause = "AND ch.workspace_id = ?"
        params.append(info.workspace_id)

    linked = conn.execute(
        "SELECT ch.workspace_id, COALESCE(sg.goal, '') AS goal, "
        "c.id, c.name, c.status, c.source, c.evidence_count, "
        "COUNT(DISTINCT cc.chunk_id) AS chunk_count "
        "FROM categories c "
        "JOIN chunk_categories cc ON cc.category_id = c.id "
        "JOIN chunks ch ON ch.chunk_id = cc.chunk_id AND ch.is_active = 1 "
        "LEFT JOIN source_goals sg ON sg.source_id = ch.source_id "
        f"WHERE 1=1 {ws_clause} "
        "GROUP BY ch.workspace_id, goal, c.id "
        "ORDER BY ch.workspace_id, goal, chunk_count DESC, c.name",
        params,
    ).fetchall()

    workspaces: dict = {}
    for r in linked:
        ws = workspaces.setdefault(r[0], {})
        ws.setdefault(r[1], []).append({
            "id": r[2], "name": r[3], "status": r[4],
            "source": r[5], "evidence_count": r[6],
            "chunk_count": r[7],
        })

    unlinked = [
        {"id": r[0], "name": r[1], "status": r[2],
         "source": r[3], "evidence_count": r[4], "chunk_count": 0}
        for r in conn.execute(
            "SELECT c.id, c.name, c.status, c.source, c.evidence_count "
            "FROM categories c "
            "LEFT JOIN chunk_categories cc ON cc.category_id = c.id "
            "WHERE cc.category_id IS NULL "
            "ORDER BY c.status, c.id"
        ).fetchall()
    ]

    total = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    return {
        "scope": "all" if admin else "workspace",
        "workspace_id": info.workspace_id,
        "total_categories": total,
        "workspaces": [
            {
                "workspace_id": w,
                "goals": [{"goal": g, "categories": cs} for g, cs in goals.items()],
            }
            for w, goals in workspaces.items()
        ],
        "unlinked": unlinked,
    }


@router.get("/stats/admin/goal-anchor-audit")
def goal_anchor_audit(info: TokenInfo = Depends(get_token_info)) -> dict:
    """Quantifiziert „wildes Codebook": wie viele Sources einen echten goal-Anker haben,
    wie viele auf den schwachen Fallback ('Inhalte aus …') liefen, und wie viele gar
    keinen Anker haben (vor v20 ingested → goal unbekannt). source_goals ist forward-only;
    Alt-Sources erscheinen als no_anchor_pre_v20 bis sie re-ingested werden."""
    if not _is_admin(info):
        raise HTTPException(status_code=403, detail="admin scope required")
    conn = _conn()
    total = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    with_goal = conn.execute("SELECT COUNT(*) FROM source_goals").fetchone()[0]
    fallback = conn.execute(
        "SELECT COUNT(*) FROM source_goals WHERE goal LIKE 'Inhalte aus %'").fetchone()[0]
    return {
        "total_sources": total,
        "with_goal_anchor": with_goal,
        "real_anchor": with_goal - fallback,
        "fallback_anchor": fallback,
        "no_anchor_pre_v20": total - with_goal,
    }
