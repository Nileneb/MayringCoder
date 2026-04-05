"""Project context from Overview cache — Phase 1 + Phase 2 (RAG) of Issue #4.

Phase 1: Saves overview results to JSON in the cache dir.
         Loads them back as a compact context string for the analyze prompt.
Phase 2: Indexes overview entries into a local ChromaDB collection.
         Uses Ollama embeddings (nomic-embed-text) for similarity search.
         Falls back to Phase 1 if ChromaDB is unavailable.
Phase 3: Finding-reactive RAG queries (Issue #18).
         After primary analysis, each finding gets a semantically tailored
         RAG query. Context is stored in finding dict for validation steps.
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import httpx

from src.config import (
    BATCH_DELAY_SECONDS,
    BATCH_SIZE,
    CACHE_DIR,
    EMBEDDING_MODEL,
    MAX_CONTEXT_CHARS,
    OLLAMA_TIMEOUT,
    RAG_TOP_K,
    repo_slug as _repo_slug,
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


def _overview_path(repo_url: str) -> Path:
    return CACHE_DIR / f"{_repo_slug(repo_url)}_overview.json"


def save_overview_context(results: list[dict], repo_url: str) -> str:
    """Persist overview results as JSON. Returns the file path.

    Saves: filename, category, file_summary, file_type, key_responsibilities,
    dependencies, purpose_keywords, and _signatures (for redundancy checking).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for r in results:
        if "error" in r:
            continue
        entry = {
            "filename": r["filename"],
            "category": r.get("category", "uncategorized"),
            "file_summary": r.get("file_summary", ""),
        }
        # Phase 1 enrichment fields (from overview.md)
        for field in ("file_type", "key_responsibilities", "dependencies", "purpose_keywords",
                       "functions", "external_deps"):
            if field in r:
                entry[field] = r[field]
        # Signature extraction (for redundancy checking)
        if "_signatures" in r:
            entry["_signatures"] = r["_signatures"]
        entries.append(entry)
    path = _overview_path(repo_url)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def load_overview_cache_raw(repo_url: str) -> dict[str, dict] | None:
    """Load overview JSON as {filename: entry_dict} map.

    Used by the turbulence stage to reuse categories and function I/O
    from the overview stage (feed-forward pipeline, Issue #17).

    Returns None if no overview cache exists.
    """
    path = _overview_path(repo_url)
    if not path.exists():
        return None
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not entries:
        return None
    return {e["filename"]: e for e in entries if "filename" in e}


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


def build_inventory_context(repo_url: str) -> str | None:
    """Build a Phase-1 inventory context: all file types, responsibilities, and dependencies.

    This context is richer than load_overview_context — it includes key_responsibilities,
    file_type, dependencies, and purpose_keywords from the overview cache.
    It is used as the project_context for the analyze phase.

    Returns None if no overview cache exists.
    """
    path = _overview_path(repo_url)
    if not path.exists():
        return None
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not entries:
        return None

    lines = [
        "## Projekt-Inventar (alle Dateien)",
        "",
    ]
    char_budget = _ACTIVE_MAX_CONTEXT_CHARS - sum(len(l) + 1 for l in lines)

    for e in entries:
        fn = e.get("filename", "")
        cat = e.get("category", "?")
        ftype = e.get("file_type", "")
        responsibilities = e.get("key_responsibilities", [])
        deps = e.get("dependencies", [])
        keywords = e.get("purpose_keywords", [])

        summary = e.get("file_summary", "").replace("\n", " ").strip()
        if len(summary) > 100:
            summary = summary[:97] + "..."

        parts = [f"- [{cat}]"]
        if ftype:
            parts.append(f"type={ftype}")
        parts.append(fn)
        if summary:
            parts.append(f"→ {summary}")

        line = " ".join(parts)

        # Append responsibilities (capped)
        if responsibilities:
            resp_str = "  Verantwortlichkeiten: " + "; ".join(responsibilities[:3])
            if len(line) + len(resp_str) + 1 <= char_budget:
                line += resp_str
                char_budget -= len(resp_str)

        if len(line) + 1 > char_budget:
            lines.append(f"... und {len(entries) - len(lines) + 1} weitere Dateien")
            break

        lines.append(line)
        char_budget -= len(line) + 1

    return "\n".join(lines)


def build_dependency_context(
    repo_url: str, file: dict
) -> str | None:
    """Build context for a specific file: its overview entry + referenced files' entries.

    Phase 3 improvement: Instead of analyzing a file in isolation, this function
    injects the inventory entries of all files it imports or references.
    This reduces "Kontext fehlt" false positives.

    Returns None if no overview cache exists or the file has no entry.
    """
    path = _overview_path(repo_url)
    if not path.exists():
        return None
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    entry_map: dict[str, dict] = {e["filename"]: e for e in entries}
    current = entry_map.get(file.get("filename", ""))
    if current is None:
        return None

    lines = [
        "## Referenzierte Dateien",
        "",
    ]
    char_budget = _ACTIVE_MAX_CONTEXT_CHARS

    # Include self
    self_summary = current.get("file_summary", "").replace("\n", " ").strip()
    lines.append(f"### {file.get('filename')} (diese Datei)")
    if self_summary:
        lines.append(f"  {self_summary}")
    deps = current.get("dependencies", [])
    if deps:
        lines.append(f"  Abhängigkeiten: {', '.join(deps[:6])}")
    char_budget -= sum(len(l) + 1 for l in lines)

    # Include referenced files
    referenced: list[dict] = []
    for dep in deps:
        # dep can be a fully qualified name; try partial matches
        dep_key = dep.split("\\")[-1].split(".")[-1]  # last segment
        for fn, entry in entry_map.items():
            if dep_key.lower() in fn.lower() or dep_key.lower() in entry.get("file_summary", "").lower():
                referenced.append(entry)
                break

    # Deduplicate by filename
    seen: set[str] = {file.get("filename", "")}
    for ref in referenced:
        if ref["filename"] in seen:
            continue
        seen.add(ref["filename"])
        ref_summary = ref.get("file_summary", "").replace("\n", " ").strip()
        if len(ref_summary) > 100:
            ref_summary = ref_summary[:97] + "..."
        line = f"- {ref['filename']}: {ref_summary}"
        if len(line) + 1 > char_budget:
            break
        lines.append(line)
        char_budget -= len(line) + 1

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 2: RAG — ChromaDB + Ollama embeddings
# ---------------------------------------------------------------------------

def _chroma_dir(repo_url: str) -> Path:
    return CACHE_DIR / f"{_repo_slug(repo_url)}_chroma"


# ---------------------------------------------------------------------------
# Embedding cache + batch helpers
# ---------------------------------------------------------------------------

# In-memory cache keyed by sha256(text). Survives the entire process lifetime,
# avoids re-embedding the same query/document text on repeated calls.
_EMBEDDING_CACHE: dict[str, list[float]] = {}

# How many texts to send per /api/embed batch request.
_EMBED_BATCH_SIZE = 10


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _batch_embed_via_api_embed(
    texts: list[str], ollama_url: str
) -> list[list[float]] | None:
    """Try Ollama's /api/embed endpoint (supports batch input).

    Returns a list of embeddings in the same order as *texts*, or None if the
    endpoint is unavailable or returns an unexpected shape.
    """
    try:
        resp = httpx.post(
            f"{ollama_url}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": texts},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # /api/embed returns {"embeddings": [[...], ...]}
        batch_result = data.get("embeddings")
        if isinstance(batch_result, list) and len(batch_result) == len(texts):
            return batch_result
    except Exception:
        pass
    return None


def _single_embed_with_retry(text: str, ollama_url: str, label: str) -> list[float]:
    """Embed a single text via /api/embeddings with retry logic."""
    max_retries = 5
    retry_delays = (3, 6, 12, 20, 30)
    for attempt in range(max_retries):
        try:
            resp = httpx.post(
                f"{ollama_url}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
            if attempt < max_retries - 1:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                print(
                    f"    ⟳ Embedding-Retry {attempt + 1}/{max_retries} [{label}] (in {delay}s) …",
                    file=sys.stderr, flush=True,
                )
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("unreachable")


def _embed_texts(texts: list[str], ollama_url: str) -> list[list[float]]:
    """Get embeddings from Ollama with cache, batch requests, and GPU-friendly pausing.

    Strategy:
    1. Return cached embeddings immediately (no network call).
    2. For uncached texts, try /api/embed in batches of _EMBED_BATCH_SIZE.
    3. Fall back to individual /api/embeddings calls with retry if batch fails.
    """
    result: list[list[float] | None] = [None] * len(texts)

    # --- Pass 1: serve from cache ---
    uncached_indices: list[int] = []
    for i, text in enumerate(texts):
        cached = _EMBEDDING_CACHE.get(_cache_key(text))
        if cached is not None:
            result[i] = cached
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return result  # type: ignore[return-value]

    # --- Pass 2: batch-embed uncached texts ---
    total_uncached = len(uncached_indices)
    processed = 0

    for batch_start in range(0, total_uncached, _EMBED_BATCH_SIZE):
        batch_idx = uncached_indices[batch_start: batch_start + _EMBED_BATCH_SIZE]
        batch_texts = [texts[i] for i in batch_idx]

        batch_results = _batch_embed_via_api_embed(batch_texts, ollama_url)

        if batch_results is not None:
            # Batch succeeded → store in cache and result
            for i, embedding in zip(batch_idx, batch_results):
                _EMBEDDING_CACHE[_cache_key(texts[i])] = embedding
                result[i] = embedding
        else:
            # Fallback: individual calls with retry
            for i in batch_idx:
                label = f"{processed + 1}/{total_uncached}"
                embedding = _single_embed_with_retry(texts[i], ollama_url, label)
                _EMBEDDING_CACHE[_cache_key(texts[i])] = embedding
                result[i] = embedding
                processed += 1

        processed += len(batch_idx) if batch_results is not None else 0

        # GPU pause — same cadence as analyzer batch processing.
        items_done = min(batch_start + _EMBED_BATCH_SIZE, total_uncached)
        if BATCH_SIZE > 0 and items_done % BATCH_SIZE == 0 and items_done < total_uncached:
            print(
                f"  ⏸ Embedding-Pause ({BATCH_DELAY_SECONDS}s nach {items_done} Docs) …",
                flush=True,
            )
            time.sleep(BATCH_DELAY_SECONDS)

    return result  # type: ignore[return-value]


def index_overview_to_vectordb(repo_url: str, ollama_url: str, force: bool = False) -> int:
    """Index overview JSON entries into ChromaDB. Returns number of entries indexed.

    Skips re-indexing if the collection already has the same number of entries
    (embeddings use a fixed model and don't depend on the analysis model).
    Pass force=True to re-index regardless.
    """
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

    # Staleness check: skip if collection already has the right number of entries.
    if not force:
        try:
            existing = client.get_collection("overview")
            if existing.count() == len(entries):
                print(f"  Vektor-DB bereits aktuell ({len(entries)} Einträge) — übersprungen.", flush=True)
                return len(entries)
        except Exception:
            pass  # Collection doesn't exist yet — proceed with indexing.

    # (Re-)create collection.
    try:
        client.delete_collection("overview")
    except Exception:
        pass
    collection = client.create_collection("overview")

    # Build documents + metadata
    # Truncate summaries — embedding models work best with short texts and
    # very long inputs can cause Ollama 500 errors (GPU OOM / context overflow).
    _MAX_EMBED_CHARS = 500
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for i, e in enumerate(entries):
        fn = e["filename"]
        cat = e.get("category", "uncategorized")
        summary = (e.get("file_summary") or "").replace("\n", " ").strip()
        if len(summary) > _MAX_EMBED_CHARS:
            summary = summary[:_MAX_EMBED_CHARS - 3] + "..."
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
    query_type: str | None = None,
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
