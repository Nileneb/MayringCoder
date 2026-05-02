from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from src.api.auth import get_workspace
from src.api.job_queue import make_job as _make_job, _JOBS
from src.api.routes.models import (
    WikiConflictResolveRequest,
    WikiEdgeCreateRequest,
    WikiRebuildRequest,
)
from src.wiki_v2._path_utils import confined_path as _cp, safe_id

router = APIRouter(tags=["wiki"])


@router.get("/wiki/slugs")
async def wiki_slugs() -> dict:
    """List available wiki slugs (public — only exposes slug names, no content)."""
    from src.config import CACHE_DIR
    slugs = sorted({
        p.name.replace("_wiki_clusters.json", "").replace("_wiki_index.json", "")
        for p in CACHE_DIR.glob("*_wiki_*.json")
        if "_wiki_clusters.json" in p.name or "_wiki_index" in p.name
    })
    return {"slugs": slugs}


@router.get("/wiki/graph")
async def wiki_graph(slug: str = "", workspace_id: str = "", format: str = "json"):
    """Return wiki cluster graph + recent Pi-agent search activations for Brain visualization.

    Tries wiki_v2 graph.json first, falls back to legacy _wiki_clusters.json.
    No auth required — exposes only structural cluster metadata (no chunk content).
    """
    import json as _json
    import time as _time_g
    from src.config import CACHE_DIR, WIKI_DIR
    from src.api.memory_service import _RECENT_ACTIVATIONS

    if not slug and not workspace_id:
        return {"clusters": [], "edges": [], "activations": [], "error": "slug or workspace_id required"}

    wid = workspace_id or slug
    if not wid:
        return {"clusters": [], "edges": [], "activations": [], "error": "invalid workspace_id"}
    _wid_safe = safe_id(wid)
    _slug_safe = safe_id(slug) if slug else _wid_safe

    # --- Try wiki_v2 graph.json first ---
    graph_path = _cp(WIKI_DIR, _wid_safe, "graph.json")
    if graph_path.exists():
        data = _json.loads(graph_path.read_text())
        if format == "mermaid":
            from src.wiki_v2.renderer import to_mermaid
            return PlainTextResponse(to_mermaid(data), media_type="text/plain")
        now = _time_g.time()
        activations: list[dict] = []
        member_lookup: dict[str, str] = {}
        for c in data.get("clusters", []):
            for m in c.get("members", []):
                member_lookup[m] = c["cluster_id"]
        for ev in _RECENT_ACTIVATIONS:
            if ev["workspace_id"] != wid or now - ev["ts"] > 60:
                continue
            hit: set[str] = set()
            for sid in ev["source_ids"]:
                path = sid.split(":")[-1]
                for member, cid in member_lookup.items():
                    if path == member or path.endswith("/" + member) or member.endswith("/" + path):
                        hit.add(cid)
            activations.append({
                "query": ev["query"][:80],
                "clusters": list(hit),
                "age_s": round(now - ev["ts"], 1),
            })
        data["activations"] = activations
        return data

    # --- Fallback: legacy _wiki_clusters.json ---
    if not slug:
        return {"clusters": [], "edges": [], "activations": [],
                "error": f"No wiki found for workspace '{wid}'. Run POST /wiki/rebuild first."}

    _slug_safe = safe_id(slug)
    cluster_path = _cp(CACHE_DIR, f"{_slug_safe}_wiki_clusters.json")
    index_path = _cp(CACHE_DIR, f"{_slug_safe}_wiki_index.json")

    clusters: list[dict] = []
    raw: list[dict] = []
    if cluster_path.exists():
        raw = _json.loads(cluster_path.read_text())
        clusters = [
            {"name": c["name"], "files": c.get("files", []),
             "labels": c.get("labels", []), "size": max(1, len(c.get("files", [])))}
            for c in raw
        ]
    elif index_path.exists():
        idx = _json.loads(index_path.read_text())
        clusters = [{"name": k, "files": [], "labels": [], "size": 1} for k in idx]
    else:
        return {"clusters": [], "edges": [], "activations": [],
                "error": f"No wiki found for slug '{slug}'. Run POST /wiki/rebuild first."}

    edges: list[dict] = []
    for c in raw:
        for edge in c.get("edges", []):
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                edges.append({"source": c["name"], "target": edge[0],
                               "weight": edge[1], "rules": edge[2] if len(edge) > 2 else []})

    now = _time_g.time()
    activations = []
    for ev in _RECENT_ACTIVATIONS:
        if ev["workspace_id"] != wid or now - ev["ts"] > 60:
            continue
        hit: set[str] = set()
        for sid in ev["source_ids"]:
            path = sid.split(":")[-1]
            for c in clusters:
                if any(path == f or path.endswith("/" + f) or f.endswith("/" + path)
                       for f in c["files"]):
                    hit.add(c["name"])
        activations.append({"query": ev["query"][:80], "clusters": list(hit),
                              "age_s": round(now - ev["ts"], 1)})

    return {"clusters": clusters, "edges": edges, "activations": activations}


@router.post("/wiki/rebuild")
async def wiki_rebuild(
    request: WikiRebuildRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Rebuild wiki_v2 graph.json for a workspace: EdgeDetector + ClusterEngine + to_json()."""
    wid = request.workspace_id
    slug = request.repo_slug or wid
    try:
        _wid_safe = safe_id(wid)
        _slug_safe = safe_id(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid workspace_id or repo_slug") from exc
    job_id = _make_job(_wid_safe)

    async def _do_rebuild() -> None:
        try:
            _JOBS[job_id]["status"] = "running"
            import json as _j
            from src.config import CACHE_DIR
            from src.wiki_v2.graph import WikiGraph
            from src.wiki_v2.edge_detector import EdgeDetector
            from src.wiki_v2.clustering import ClusterEngine

            db = WikiGraph(_wid_safe, _slug_safe, CACHE_DIR / "wiki_v2.db")
            oc_path = _cp(CACHE_DIR, f"{_slug_safe}_overview_cache.json")
            oc = _j.loads(oc_path.read_text()) if oc_path.exists() else {}
            from src.memory.db_adapter import DBAdapter
            conn = DBAdapter.create(CACHE_DIR / "memory.db")
            detector = EdgeDetector()
            edges = detector.detect_from_overview(oc, conn, _wid_safe, _slug_safe)
            conn.close()
            from src.wiki_v2.models import WikiNode as _WikiNode
            node_ids = set(oc.keys()) | {e.source for e in edges} | {e.target for e in edges}
            for nid in sorted(node_ids):
                db.upsert_node(_WikiNode(id=nid, repo_slug=_slug_safe, workspace_id=_wid_safe))
            for e in edges:
                db.add_edge(e)
            ollama = request.ollama_url or "http://three.linn.games:11434"
            engine = ClusterEngine()
            engine.cluster(db, strategy=request.strategy, ollama_url=ollama, model=request.model)
            db.to_json()
            db.snapshot()
            db.close()
            n_edges = len(edges)
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["output"] = f"graph.json written: {len(oc)} nodes, {n_edges} edges"
        except Exception as exc:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["output"] = "[Fehler] Wiki-Rebuild fehlgeschlagen"

    asyncio.create_task(_do_rebuild())
    return {"job_id": job_id, "status": "started"}


@router.post("/wiki/edge")
async def wiki_edge_create(
    request: WikiEdgeCreateRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Manually create a wiki edge with user_id tracking via wiki_contributions."""
    if not request.source or not request.target:
        raise HTTPException(status_code=400, detail="source and target required")
    import os as _os_e
    _safe_ws = _os_e.path.basename(workspace_id.replace('/', '_').replace('\\', '_'))
    if not _safe_ws:
        raise HTTPException(status_code=400, detail="invalid workspace_id")
    try:
        from src.config import CACHE_DIR
        from src.wiki_v2.graph import WikiGraph
        from src.wiki_v2.models import WikiEdge
        from src.wiki_v2 import store as _ws
        db = WikiGraph(_safe_ws, _safe_ws, CACHE_DIR / "wiki_v2.db")
        edge = WikiEdge(
            source=request.source,
            target=request.target,
            repo_slug=_safe_ws,
            workspace_id=_safe_ws,
            type=request.type,
            weight=request.weight,
            context=request.context,
            validated=True,
        )
        db.add_edge(edge)
        _ws.log_contribution(db._conn, _safe_ws, request.user_id, "add_edge",
                             f"{request.source}→{request.target}")
        db.close()
        return {"status": "ok", "source": request.source, "target": request.target,
                "type": request.type}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/wiki/conflicts")
async def wiki_conflicts(
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Return edges where both existing and incoming are validated (manual conflict)."""
    try:
        import sqlite3 as _sq
        from src.config import CACHE_DIR
        conn = _sq.connect(str(CACHE_DIR / "wiki_v2.db"))
        conn.row_factory = _sq.Row
        rows = conn.execute(
            "SELECT source, target, type, weight, context FROM wiki_edges "
            "WHERE workspace_id=? AND validated=1 "
            "GROUP BY source, target HAVING COUNT(*)>1",
            (workspace_id,),
        ).fetchall()
        conn.close()
        return {"workspace_id": workspace_id, "conflicts": [dict(r) for r in rows]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/wiki/conflicts/resolve")
async def wiki_conflicts_resolve(
    request: WikiConflictResolveRequest,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Resolve a manual-edge conflict: keep the most recently validated edge."""
    try:
        import sqlite3 as _sq
        from src.config import CACHE_DIR
        conn = _sq.connect(str(CACHE_DIR / "wiki_v2.db"))
        conn.row_factory = _sq.Row
        rows = conn.execute(
            "SELECT rowid, * FROM wiki_edges WHERE workspace_id=? AND source=? AND target=? AND validated=1 "
            "ORDER BY updated_at DESC",
            (workspace_id, request.source, request.target),
        ).fetchall()
        if len(rows) <= 1:
            conn.close()
            return {"status": "no_conflict", "source": request.source, "target": request.target}
        keep_rowid = rows[0]["rowid"]
        conn.execute(
            "DELETE FROM wiki_edges WHERE workspace_id=? AND source=? AND target=? "
            "AND validated=1 AND rowid!=?",
            (workspace_id, request.source, request.target, keep_rowid),
        )
        from src.wiki_v2 import store as _ws
        _ws.log_contribution(conn, workspace_id, request.user_id, "resolve_conflict",
                             f"{request.source}→{request.target}")
        conn.commit()
        conn.close()
        return {"status": "resolved", "kept": dict(rows[0]), "removed": len(rows) - 1}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/wiki/history")
async def wiki_history(
    workspace_id: str = Depends(get_workspace),
    limit: int = 20,
) -> dict:
    """Return last `limit` graph snapshots for a workspace (newest first)."""
    try:
        import sqlite3 as _sq
        from src.config import CACHE_DIR
        from src.wiki_v2.history import WikiHistory
        conn = _sq.connect(str(CACHE_DIR / "wiki_v2.db"))
        conn.row_factory = _sq.Row
        hist = WikiHistory()
        snaps = hist.timeline(conn, workspace_id, limit=limit)
        conn.close()
        return {"snapshots": [
            {"snapshot_id": s.snapshot_id, "trigger": s.trigger,
             "node_count": s.node_count, "edge_count": s.edge_count,
             "cluster_count": s.cluster_count, "created_at": s.created_at}
            for s in snaps
        ]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/wiki/diff")
async def wiki_diff(
    from_date: str,
    to_date: str,
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Return diff between two ISO-8601 dates for a workspace."""
    try:
        import sqlite3 as _sq
        from src.config import CACHE_DIR
        from src.wiki_v2.history import WikiHistory
        conn = _sq.connect(str(CACHE_DIR / "wiki_v2.db"))
        conn.row_factory = _sq.Row
        hist = WikiHistory()
        diff = hist.diff(conn, workspace_id, from_date, to_date)
        conn.close()
        return {
            "nodes_added": diff.nodes_added,
            "nodes_removed": diff.nodes_removed,
            "edges_added": diff.edges_added,
            "edges_removed": diff.edges_removed,
            "clusters_added": diff.clusters_added,
            "clusters_removed": diff.clusters_removed,
            "contributors": diff.contributors,
            "summary": diff.summary(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/wiki/feedback-matrix")
async def wiki_feedback_matrix(
    limit: int = 50,
    query_filter: str | None = None,
    mode: str = "chunk",
    workspace_id: str = Depends(get_workspace),
) -> dict:
    """Return per-chunk feedback aggregation (positive/negative/neutral counts + net_score)."""
    try:
        from src.api.dependencies import get_conn
        from src.wiki_v2.store import get_feedback_matrix, get_task_feedback_matrix
        conn = get_conn()
        if mode == "task":
            data = get_task_feedback_matrix(conn, limit=limit, query_filter=query_filter)
            return {"workspace_id": workspace_id, "mode": "task", "tasks": data}
        data = get_feedback_matrix(conn, limit=limit)
        return {"workspace_id": workspace_id, "mode": "chunk", "chunks": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/wiki/team")
async def wiki_team_activity(
    workspace_id: str = Depends(get_workspace),
    days: int = 30,
) -> dict:
    """Return team contribution counts per user for the last `days` days."""
    try:
        import sqlite3 as _sq
        from src.config import CACHE_DIR
        from src.wiki_v2.history import team_activity
        conn = _sq.connect(str(CACHE_DIR / "wiki_v2.db"))
        conn.row_factory = _sq.Row
        activity = team_activity(conn, workspace_id, days=days)
        conn.close()
        return {"workspace_id": workspace_id, "days": days, "users": activity}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
