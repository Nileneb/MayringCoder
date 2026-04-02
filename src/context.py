"""Project context from Overview cache — Phase 1 + Phase 2 (RAG) of Issue #4.

Phase 1: Saves overview results to JSON in the cache dir.
         Loads them back as a compact context string for the analyze prompt.
Phase 2: Indexes overview entries into a local ChromaDB collection.
         Uses Ollama embeddings (nomic-embed-text) for similarity search.
         Falls back to Phase 1 if ChromaDB is unavailable.
"""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

from src.config import (
    CACHE_DIR,
    EMBEDDING_MODEL,
    MAX_CONTEXT_CHARS,
    OLLAMA_TIMEOUT,
    RAG_TOP_K,
)

_ACTIVE_MAX_CONTEXT_CHARS = MAX_CONTEXT_CHARS


def set_max_context_chars(limit: int) -> None:
    """Override project-context char budget at runtime."""
    global _ACTIVE_MAX_CONTEXT_CHARS
    _ACTIVE_MAX_CONTEXT_CHARS = max(1, int(limit))

try:
    import chromadb
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False


def _repo_slug(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    slug = parsed.path.strip("/").replace("/", "-").lower()
    return re.sub(r"[^a-z0-9\-]", "", slug) or "repo"


def _overview_path(repo_url: str) -> Path:
    return CACHE_DIR / f"{_repo_slug(repo_url)}_overview.json"


def save_overview_context(results: list[dict], repo_url: str) -> str:
    """Persist overview results as JSON. Returns the file path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for r in results:
        if "error" in r:
            continue
        entries.append({
            "filename": r["filename"],
            "category": r.get("category", "uncategorized"),
            "file_summary": r.get("file_summary", ""),
        })
    path = _overview_path(repo_url)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def load_overview_context(repo_url: str) -> str | None:
    """Load overview cache and build a compact context string.

    Returns None if no overview cache exists (no breaking change).
    The returned string is capped at the active context budget.
    """
    path = _overview_path(repo_url)
    if not path.exists():
        return None

    entries = json.loads(path.read_text(encoding="utf-8"))
    if not entries:
        return None

    lines = [
        "## Projektkontext",
        f"Das Projekt enthält {len(entries)} Dateien. Übersicht:",
        "",
    ]
    char_budget = _ACTIVE_MAX_CONTEXT_CHARS - sum(len(l) + 1 for l in lines)

    for e in entries:
        summary = e.get("file_summary", "").replace("\n", " ").strip()
        if not summary:
            summary = "(keine Zusammenfassung)"
        # Truncate long summaries to one line
        if len(summary) > 120:
            summary = summary[:117] + "..."
        line = f"- [{e.get('category', '?')}] {e['filename']} → {summary}"
        if len(line) + 1 > char_budget:
            lines.append(f"... und {len(entries) - len(lines) + 3} weitere Dateien")
            break
        lines.append(line)
        char_budget -= len(line) + 1

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 2: RAG — ChromaDB + Ollama embeddings
# ---------------------------------------------------------------------------

def _chroma_dir(repo_url: str) -> Path:
    return CACHE_DIR / f"{_repo_slug(repo_url)}_chroma"


def _embed_texts(texts: list[str], ollama_url: str) -> list[list[float]]:
    """Get embeddings from Ollama for a batch of texts."""
    embeddings: list[list[float]] = []
    for text in texts:
        resp = httpx.post(
            f"{ollama_url}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings.append(data["embedding"])
    return embeddings


def index_overview_to_vectordb(repo_url: str, ollama_url: str) -> int:
    """Index overview JSON entries into ChromaDB. Returns number of entries indexed."""
    if not _HAS_CHROMADB:
        return 0

    path = _overview_path(repo_url)
    if not path.exists():
        return 0

    entries = json.loads(path.read_text(encoding="utf-8"))
    if not entries:
        return 0

    chroma_path = _chroma_dir(repo_url)
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))
    # Delete existing collection to re-index
    try:
        client.delete_collection("overview")
    except Exception:
        pass
    collection = client.create_collection("overview")

    # Build documents + metadata
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for i, e in enumerate(entries):
        fn = e["filename"]
        cat = e.get("category", "uncategorized")
        summary = e.get("file_summary", "").replace("\n", " ").strip()
        doc = f"[{cat}] {fn}: {summary}"
        ids.append(str(i))
        documents.append(doc)
        metadatas.append({"filename": fn, "category": cat})

    # Get embeddings from Ollama
    embeddings = _embed_texts(documents, ollama_url)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(ids)


def query_similar_context(
    query_text: str,
    repo_url: str,
    ollama_url: str,
    top_k: int | None = None,
) -> str | None:
    """Find the most relevant overview entries for a given file via similarity search.

    Returns a formatted context string, or None if ChromaDB is unavailable or empty.
    Falls back to Phase 1 (load_overview_context) if the vector DB doesn't exist.
    """
    if not _HAS_CHROMADB:
        return None

    chroma_path = _chroma_dir(repo_url)
    if not chroma_path.exists():
        return None

    k = top_k or RAG_TOP_K

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection("overview")
    except Exception:
        return None

    if collection.count() == 0:
        return None

    # Embed the query
    query_embedding = _embed_texts([query_text], ollama_url)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
    )

    docs = results.get("documents", [[]])[0]
    if not docs:
        return None

    lines = [
        "## Projektkontext (relevante Dateien)",
        "",
    ]
    for doc in docs:
        lines.append(f"- {doc}")

    context = "\n".join(lines)
    # Cap to budget
    if len(context) > _ACTIVE_MAX_CONTEXT_CHARS:
        context = context[:_ACTIVE_MAX_CONTEXT_CHARS - 3] + "..."
    return context
