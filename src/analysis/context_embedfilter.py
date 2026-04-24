"""Embedding-based file pre-filter."""

import json
import math
from pathlib import Path

from src.config import CACHE_DIR, EMBEDDING_MODEL, repo_slug as _repo_slug
from src.analysis.context_rag import _embed_texts


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _file_snippet(file: dict, max_chars: int = 800) -> str:
    cat = file.get("category", "uncategorized")
    fn = file.get("filename", "")
    content = (file.get("content") or "")[:max_chars]
    return f"[{cat}] {fn}\n{content}"


def _embed_index_path(repo_url: str, embedding_model: str) -> Path:
    slug = _repo_slug(repo_url)
    model_tag = embedding_model.replace(":", "_").replace("/", "_")
    return CACHE_DIR / f"{slug}_embed_{model_tag}.json"


_index_path = _embed_index_path


def build_file_index(
    files: list,
    ollama_url: str,
    embedding_model: str = "",
    repo_url: str = "",
    force: bool = False,
) -> list:
    _model = embedding_model or EMBEDDING_MODEL
    cache_path = _embed_index_path(repo_url, _model) if repo_url else None

    if cache_path and cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if {e["filename"] for e in cached} == {f["filename"] for f in files}:
                print(f"  Embedding-Index geladen aus Cache ({len(cached)} Einträge)")
                return cached
        except (ValueError, KeyError):
            pass

    snippets = []
    for f in files:
        cat = f.get("category", "uncategorized")
        fn = f.get("filename", "")
        content = (f.get("content") or "")[:800]
        snippets.append(f"[{cat}] {fn}\n{content}")

    print(f"  Embedding-Prefilter: {len(files)} Dateien einbetten ...")
    embeddings = _embed_texts(snippets, ollama_url)
    index = [{"filename": f["filename"], "embedding": emb} for f, emb in zip(files, embeddings)]

    if cache_path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    return index


def filter_by_embedding(
    files: list,
    query: str,
    ollama_url: str,
    top_k: int = 20,
    threshold: float | None = None,
    embedding_model: str = "",
    repo_url: str = "",
    force_reindex: bool = False,
) -> tuple:
    _model = embedding_model or EMBEDDING_MODEL
    if not files:
        return [], []
    index = build_file_index(files, ollama_url, _model, repo_url, force=force_reindex)
    query_emb = _embed_texts([query], ollama_url)[0]
    scored = [(_cosine_similarity(query_emb, e["embedding"]), e["filename"]) for e in index]
    if threshold is not None:
        scored = [(s, fn) for s, fn in scored if s >= threshold]
    scored.sort(key=lambda x: x[0], reverse=True)
    effective_k = len(scored) if top_k <= 0 else min(top_k, len(scored))
    selected = [fn for _, fn in scored[:effective_k]]
    selected_set = set(selected)
    filtered_out = sorted(f["filename"] for f in files if f["filename"] not in selected_set)
    return selected, filtered_out
