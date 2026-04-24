from __future__ import annotations
import re as _re
from pathlib import Path
from typing import Any

_FILE_PATH_RE = _re.compile(
    r'\b((?:src|tests?|lib|app|config)/[A-Za-z0-9_/.-]+\.(?:py|js|ts|java|go|rs|rb|php|cpp|c|h|yaml|yml|json))\b'
)


def _set_recluster_flag(workspace_id: str) -> None:
    try:
        from src.config import WIKI_DIR
        flag_path = WIKI_DIR / workspace_id / "recluster_needed"
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        flag_path.touch()
    except Exception:
        pass


def on_post_finding(
    workspace_id: str,
    repo_slug: str,
    analyzed_file: str,
    finding_text: str,
    db_path: Path | None = None,
) -> None:
    """Watcher hook: extracts file path references from finding_text,
    creates issue_mentions edges, sets recluster flag on >10% topology change.
    """
    try:
        from src.config import CACHE_DIR
        from src.wiki_v2.graph import WikiGraph
        from src.wiki_v2.models import WikiEdge

        db = WikiGraph(workspace_id, repo_slug, db_path or (CACHE_DIR / "wiki_v2.db"))
        count_before = db.edge_count()

        referenced = {m.group(1) for m in _FILE_PATH_RE.finditer(finding_text)}
        for ref in referenced:
            if ref != analyzed_file:
                db.add_edge(WikiEdge(
                    source=analyzed_file,
                    target=ref,
                    repo_slug=repo_slug,
                    workspace_id=workspace_id,
                    type="issue_mentions",
                    weight=1.0,
                    context="finding_ref",
                ))

        count_after = db.edge_count()
        if count_before > 0:
            if (count_after - count_before) / count_before > 0.10:
                _set_recluster_flag(workspace_id)
        elif count_after > 0:
            _set_recluster_flag(workspace_id)

        db.close()
    except Exception:
        pass


def on_post_analyze(
    workspace_id: str,
    repo_slug: str,
    analyzed_file: str,
    db_path: Path | None = None,
    overview_cache: dict | None = None,
) -> None:
    """Watcher hook after analysis run. Updates node + edges in wiki graph.
    Designed to be called non-blockingly; all exceptions are swallowed.
    """
    try:
        from src.config import CACHE_DIR
        from src.wiki_v2.graph import WikiGraph
        from src.wiki_v2.models import WikiNode
        from src.wiki_v2.edge_detector import EdgeDetector

        db = WikiGraph(workspace_id, repo_slug, db_path or (CACHE_DIR / "wiki_v2.db"))
        if analyzed_file:
            db.upsert_node(WikiNode(
                id=analyzed_file,
                repo_slug=repo_slug,
                workspace_id=workspace_id,
            ))
        if overview_cache:
            detector = EdgeDetector()
            edges = detector.detect_from_overview(overview_cache, None, workspace_id, repo_slug)
            for e in edges:
                db.add_edge(e)
        db.close()
    except Exception:
        pass


def on_post_ingest(
    workspace_id: str,
    repo_slug: str,
    source_id: str,
    db_path: Path | None = None,
) -> None:
    """Watcher hook after memory ingest. Adds node to wiki graph."""
    try:
        from src.config import CACHE_DIR
        from src.wiki_v2.graph import WikiGraph
        from src.wiki_v2.models import WikiNode

        path = source_id.split(":")[-1] if ":" in source_id else source_id
        db = WikiGraph(workspace_id, repo_slug, db_path or (CACHE_DIR / "wiki_v2.db"))
        db.upsert_node(WikiNode(id=path, repo_slug=repo_slug, workspace_id=workspace_id))
        db.close()
    except Exception:
        pass
