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

    return result  # type: ignore[return-value]


def index_overview_to_vectordb(repo_url: str, ollama_url: str, force: bool = False) -> int:
    """Index overview JSON entries into ChromaDB. Returns number of documents indexed.

    Indexes two document types per file:
    - Summary document: prose summary (always, Typ 1)
    - Functions document: function signatures + external_deps (if functions[] non-empty, Typ 2)

    Skips re-indexing if the collection already has the expected number of documents.
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

    # Expected document count: 1 summary + 1 functions doc per file with functions[].
    n_fn_docs = sum(1 for e in entries if e.get("functions"))
    expected_count = len(entries) + n_fn_docs

    # Staleness check: skip if collection already has the right number of documents.
    if not force:
        try:
            existing = client.get_collection("overview")
            if existing.count() == expected_count:
                print(f"  Vektor-DB bereits aktuell ({expected_count} Dokumente) — übersprungen.", flush=True)
                return expected_count
        except Exception:
            pass  # Collection doesn't exist yet — proceed with indexing.

    # (Re-)create collection.
    try:
        client.delete_collection("overview")
    except Exception:
        pass
    collection = client.create_collection("overview")

    # Build documents + metadata
    # Truncate to _MAX_EMBED_CHARS — embedding models work best with short texts and
    # very long inputs can cause Ollama 500 errors (GPU OOM / context overflow).
    _MAX_EMBED_CHARS = 500
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for e in entries:
        fn = e["filename"]
        cat = e.get("category", "uncategorized")

        # Typ 1: Prosa-Summary (wie bisher)
        summary = (e.get("file_summary") or "").replace("\n", " ").strip()
        if len(summary) > _MAX_EMBED_CHARS:
            summary = summary[:_MAX_EMBED_CHARS - 3] + "..."
        doc = f"[{cat}] {fn}: {summary}"
        ids.append(f"{fn}::summary")
        documents.append(doc)
        metadatas.append({"filename": fn, "category": cat, "doc_type": "summary"})

        # Typ 2: Funktions-Signaturen-Dokument (Issue #21)
        functions = e.get("functions") or []
        if functions:
            fn_parts = []
            for f in functions[:10]:
                name = f.get("name", "")
                inputs = ", ".join(f.get("inputs", []))
                outputs = " → " + ", ".join(f.get("outputs", [])) if f.get("outputs") else ""
                calls = ", ".join(f.get("calls", []))
                part = f"{name}({inputs}){outputs}" + (f" [calls: {calls}]" if calls else "")
                fn_parts.append(part)
            ext_deps = e.get("external_deps") or []
            ext_str = f" | deps: {', '.join(ext_deps[:8])}" if ext_deps else ""
            fn_doc = f"[{cat}] {fn} functions: " + " | ".join(fn_parts) + ext_str
            fn_doc = fn_doc[:_MAX_EMBED_CHARS]
            ids.append(f"{fn}::functions")
            documents.append(fn_doc)
            metadatas.append({"filename": fn, "category": cat, "doc_type": "functions"})

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


# ---------------------------------------------------------------------------
# Phase 3: Finding-reactive RAG queries (Issue #18)
# ---------------------------------------------------------------------------

# Regex to extract function/method names from evidence excerpts.
_FN_EXTRACT_RE = re.compile(
    r"(?:def|function|public\s+function|private\s+function|"
    r"protected\s+function|func)\s+(\w+)",
    re.IGNORECASE,
)

# Type → query template. {fn_name}, {category}, {filename} are substituted.
_RAG_QUERY_TEMPLATES: dict[str, str] = {
    "redundanz": "Funktion {fn_name} ähnliche Implementierung",
    "sicherheit": "Auth Validation User-Input {category}",
    "zombie_code": "Aufruf {fn_name} Referenz",
    "inkonsistenz": "[{category}] Fehlerbehandlung Muster",
    "fehlerbehandlung": "Exception Try-Catch {category} Pattern",
    "overengineering": "[{category}] Abstraktion Vereinfachung",
    "unklar": "[{category}] {filename} Kontext Abhängigkeit",
}


def _build_rag_query(
    finding: dict, filename: str = "", category: str = ""
) -> str:
    """Map a finding to a semantically tailored RAG query string.

    Uses the finding type to select a query template, then substitutes
    a function name extracted from the evidence (if present).
    """
    ftype = (finding.get("type") or "").lower().strip()
    evidence = finding.get("evidence_excerpt") or ""

    # Try to extract a function name from the evidence excerpt.
    m = _FN_EXTRACT_RE.search(evidence)
    fn_name = m.group(1) if m else ""

    # Fallback: use the filename stem as pseudo function name.
    if not fn_name and filename:
        fn_name = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    template = _RAG_QUERY_TEMPLATES.get(ftype, "[{category}] {filename}")
    query = template.format(
        fn_name=fn_name,
        category=category or "uncategorized",
        filename=filename or "unknown",
    )
    return query.strip()


def enrich_findings_with_rag(
    results: list[dict],
    repo_url: str,
    ollama_url: str,
    top_k: int | None = None,
) -> list[dict]:
    """Enrich each finding with a finding-reactive RAG context (Issue #18).

    For every finding in *results*, builds a semantically tailored query,
    embeds it, queries ChromaDB, and stores the results as ``_rag_context``
    and ``_rag_query`` in the finding dict.

    Returns the (mutated) results list. If ChromaDB is unavailable or the
    vector DB does not exist, results are returned unchanged.
    """
    if not _HAS_CHROMADB:
        return results

    chroma_path = _chroma_dir(repo_url)
    if not chroma_path.exists():
        return results

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection("overview")
    except Exception:
        return results

    if collection.count() == 0:
        return results

    k = top_k or RAG_TOP_K

    # Collect all findings with their metadata.
    finding_refs: list[tuple[dict, str, str]] = []  # (finding_dict, filename, category)
    for r in results:
        if "error" in r:
            continue
        fn = r.get("filename", "")
        cat = r.get("category", "uncategorized")
        for smell in r.get("potential_smells", []):
            finding_refs.append((smell, fn, cat))
        for cod in r.get("codierungen", []):
            finding_refs.append((cod, fn, cat))

    if not finding_refs:
        return results

    # Build queries and batch-embed.
    queries: list[str] = []
    for finding, fn, cat in finding_refs:
        q = _build_rag_query(finding, fn, cat)
        queries.append(q)

    embeddings = _embed_texts(queries, ollama_url)

    # Query ChromaDB per finding and store context.
    n_results = min(k, collection.count())
    for i, (finding, fn, cat) in enumerate(finding_refs):
        finding["_rag_query"] = queries[i]
        try:
            results_db = collection.query(
                query_embeddings=[embeddings[i]],
                n_results=n_results,
            )
            docs = results_db.get("documents", [[]])[0]
            if docs:
                ctx_lines = ["## Projektkontext (ähnliche Dateien)", ""]
                for doc in docs:
                    ctx_lines.append(f"- {doc}")
                context = "\n".join(ctx_lines)
                if len(context) > _ACTIVE_MAX_CONTEXT_CHARS:
                    context = context[:_ACTIVE_MAX_CONTEXT_CHARS - 3] + "..."
                finding["_rag_context"] = context
            else:
                finding["_rag_context"] = ""
        except Exception:
            finding["_rag_context"] = ""

    return results
