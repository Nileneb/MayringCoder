"""Wiki-Haupteinstiegspunkt: generate_wiki, Keyword-Index, Cluster-Embeddings."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from src.memory.wiki_core import WikiCluster, build_connection_graph, cluster_themes, generate_wiki_markdown


def _build_keyword_index(clusters: list[WikiCluster]) -> dict[str, list[str]]:
    """Keyword → [cluster_name, ...] Mapping."""
    index: dict[str, list[str]] = {}
    for c in clusters:
        keywords = set()
        keywords.add(c.name.lower())
        for f in c.files:
            keywords.add(Path(f).stem.lower())
        for label in c.labels:
            keywords.update(label.lower().split())
        for kw in keywords:
            if len(kw) > 2:
                index.setdefault(kw, [])
                if c.name not in index[kw]:
                    index[kw].append(c.name)
    return index


def _build_cluster_embeddings(
    clusters: list[WikiCluster],
    ollama_url: str,
) -> dict[str, list[float]]:
    """cluster_name → embedding vector. Skips silently if ollama_url empty."""
    if not ollama_url or not clusters:
        return {}
    from src.analysis.context import _embed_texts
    texts = [f"{c.name} {' '.join(c.labels)}" for c in clusters]
    try:
        vecs = _embed_texts(texts, ollama_url)
        return {c.name: vec for c, vec in zip(clusters, vecs)}
    except Exception:
        return {}


def generate_wiki(
    conn: Any,
    chroma: Any,
    repo_url: str = "",
    ollama_url: str = "",
    model: str = "",
    workspace_id: str = "default",
    doc_type: str = "code",
) -> Path | None:
    """Orchestrate wiki generation.

    Workspace-mode (repo_url=""): zieht alle Chunks des workspace_id,
    merged Overview-Caches aller enthaltenen Repos.
    Repo-mode (repo_url gesetzt): bisheriges Verhalten für ein einzelnes Repo.
    """
    from src.analysis.context import load_overview_cache_raw
    from src.config import repo_slug as _repo_slug

    if not repo_url:
        rows = conn.execute(
            "SELECT source_id, category_labels, text FROM chunks WHERE is_active=1 AND workspace_id=?",
            (workspace_id,),
        ).fetchall()
        if not rows:
            print(f"[wiki] Keine Chunks für workspace={workspace_id}")
            return None
        chunks: list = [{"source_id": r[0], "category_labels": r[1], "text": r[2]} for r in rows]
        repo_urls: set[str] = set()
        for r in rows:
            sid = r[0]
            if sid.startswith("repo:"):
                parts = sid.split(":", 2)
                if len(parts) >= 2:
                    repo_urls.add(parts[1])
        overview_cache: dict[str, dict] = {}
        for url in repo_urls:
            overview_cache.update(load_overview_cache_raw(url) or {})
        slug = workspace_id
    else:
        slug = _repo_slug(repo_url)
        overview_cache = load_overview_cache_raw(repo_url) or {}
        if not overview_cache:
            print(f"[wiki] Kein Overview-Cache für {slug} — erst --mode overview ausführen")
            return None
        rows = conn.execute(
            "SELECT source_id, category_labels, text FROM chunks WHERE is_active=1"
        ).fetchall()
        chunks = [{"source_id": r[0], "category_labels": r[1], "text": r[2]} for r in rows]

    edges = build_connection_graph(doc_type, overview_cache, chunks, conn, chroma)
    clusters = cluster_themes(edges)

    out = Path("cache") / f"{slug}_wiki.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate_wiki_markdown(clusters, slug), encoding="utf-8")
    print(f"[wiki] {len(clusters)} Cluster → {out}")

    idx_path = Path("cache") / f"{slug}_wiki_index.json"
    idx_path.write_text(json.dumps(_build_keyword_index(clusters), ensure_ascii=False), encoding="utf-8")

    clusters_path = Path("cache") / f"{slug}_wiki_clusters.json"
    clusters_data = [
        {
            "name": c.name,
            "files": c.files,
            "labels": c.labels,
            "edges": [(e[0], e[1], e[2] if len(e) > 2 else []) for e in c.edges],
        }
        for c in clusters
    ]
    clusters_path.write_text(json.dumps(clusters_data, ensure_ascii=False), encoding="utf-8")

    emb_path = Path("cache") / f"{slug}_wiki_clusters_emb.json"
    emb = _build_cluster_embeddings(clusters, ollama_url)
    if emb:
        emb_path.write_text(json.dumps(emb, ensure_ascii=False), encoding="utf-8")

    print(f"[wiki] index → {idx_path} | clusters → {clusters_path}")
    return out
