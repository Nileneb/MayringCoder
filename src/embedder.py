"""Embedding-based prefilter for file selection (Issue #11).

Optional pipeline step: embeds all file content snippets, then ranks them by
cosine similarity to a query (research question / categories). Only the top-K
most relevant files are passed to the LLM analyzer, reducing token usage for
large corpora.

Usage (from checker.py):
    selected, filtered_out = filter_by_embedding(
        files=files,
        query="code quality, bugs, security vulnerabilities",
        ollama_url=ollama_url,
        top_k=20,
        threshold=0.3,
        embedding_model="nomic-embed-text",
        repo_url=repo_url,
    )
"""

import json
import math
from pathlib import Path

from src.config import CACHE_DIR, EMBEDDING_MODEL, repo_slug as _repo_slug

# Snippet length when building the embedding document for a file.
# Long enough to capture intent; short enough to avoid OOM in the embedding model.
_SNIPPET_CHARS = 800


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity in [0, 1] between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _index_path(repo_url: str, embedding_model: str) -> Path:
    """Return the cache path for the embedding index of a repo + model pair."""
    slug = _repo_slug(repo_url)
    model_tag = embedding_model.replace(":", "_").replace("/", "_")
    return CACHE_DIR / f"{slug}_embed_{model_tag}.json"


def _file_snippet(file: dict, max_chars: int = _SNIPPET_CHARS) -> str:
    """Build a short text to embed for a file: category + filename + content prefix."""
    cat = file.get("category", "uncategorized")
    fn = file.get("filename", "")
    content = (file.get("content") or "")[:max_chars]
    return f"[{cat}] {fn}\n{content}"


def build_file_index(
    files: list[dict],
    ollama_url: str,
    embedding_model: str = EMBEDDING_MODEL,
    repo_url: str = "",
    force: bool = False,
) -> list[dict]:
    """Compute and cache embeddings for all files.

    Returns a list of ``{"filename": str, "embedding": list[float]}`` entries.
    If a valid cached index already exists (same set of filenames), it is
    returned immediately without calling Ollama.

    Parameters
    ----------
    files:
        List of file dicts (must contain at least ``filename`` and optionally
        ``content`` and ``category``).
    ollama_url:
        Ollama base URL (e.g. ``http://localhost:11434``).
    embedding_model:
        Ollama embedding model name (default: ``nomic-embed-text``).
    repo_url:
        Used to derive the cache file path. Empty string disables caching.
    force:
        If True, re-embed even when a valid cache exists.
    """
    from src.context import _embed_texts  # reuse existing batched embedding helper

    cache_path = _index_path(repo_url, embedding_model) if repo_url else None

    # --- Try loading from cache ---
    if cache_path and cache_path.exists() and not force:
        try:
            cached: list[dict] = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_names = {e["filename"] for e in cached}
            current_names = {f["filename"] for f in files}
            if cached_names == current_names:
                print(
                    f"  Embedding-Index geladen aus Cache"
                    f" ({len(cached)} Einträge, Modell: {embedding_model})"
                )
                return cached
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted cache → rebuild

    # --- Build new index ---
    snippets = [_file_snippet(f) for f in files]
    print(
        f"  Embedding-Prefilter: {len(files)} Dateien einbetten"
        f" (Modell: {embedding_model}) ..."
    )
    embeddings = _embed_texts(snippets, ollama_url)

    index = [
        {"filename": f["filename"], "embedding": emb}
        for f, emb in zip(files, embeddings)
    ]

    # --- Persist ---
    if cache_path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(index, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Embedding-Index gespeichert: {cache_path.name}")

    return index


def filter_by_embedding(
    files: list[dict],
    query: str,
    ollama_url: str,
    top_k: int = 20,
    threshold: float | None = None,
    embedding_model: str = EMBEDDING_MODEL,
    repo_url: str = "",
    force_reindex: bool = False,
) -> tuple[list[str], list[str]]:
    """Select the top-K most relevant files by cosine similarity to *query*.

    Parameters
    ----------
    files:
        Candidate files (already filtered by exclude patterns / categorized).
    query:
        Free-text research question or keyword list used as the retrieval query.
    ollama_url:
        Ollama base URL.
    top_k:
        Maximum number of files to keep (default: 20). If 0, all files above
        *threshold* are kept.
    threshold:
        Optional minimum cosine similarity. Files below this score are excluded
        even if they would make the top-K cut.
    embedding_model:
        Ollama embedding model to use.
    repo_url:
        Used to derive the embedding cache path. Pass empty string to disable.
    force_reindex:
        If True, re-embed all files even when a cache exists.

    Returns
    -------
    (selected, filtered_out)
        ``selected`` is an ordered list of filenames (most relevant first).
        ``filtered_out`` is a sorted list of filenames that were excluded.
    """
    from src.context import _embed_texts

    if not files:
        return [], []

    # 1. Build / load embedding index
    index = build_file_index(files, ollama_url, embedding_model, repo_url, force=force_reindex)

    # 2. Embed the query
    query_embedding = _embed_texts([query], ollama_url)[0]

    # 3. Score each file
    scored: list[tuple[float, str]] = [
        (_cosine_similarity(query_embedding, entry["embedding"]), entry["filename"])
        for entry in index
    ]

    # 4. Apply similarity threshold
    if threshold is not None:
        scored = [(s, fn) for s, fn in scored if s >= threshold]

    # 5. Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # 6. Apply top-K cap (0 means unlimited)
    effective_k = len(scored) if top_k <= 0 else min(top_k, len(scored))
    selected = [fn for _, fn in scored[:effective_k]]

    # 7. Compute filtered-out set
    selected_set = set(selected)
    filtered_out = sorted(f["filename"] for f in files if f["filename"] not in selected_set)

    return selected, filtered_out
