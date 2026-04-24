from __future__ import annotations
from pathlib import Path
from typing import Any


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
