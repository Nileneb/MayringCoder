"""Phase 2 + 3 RAG — ChromaDB, Ollama embeddings, finding-reactive queries."""

import hashlib
import json
import re
from pathlib import Path

from src.config import (
    CACHE_DIR,
    EMBEDDING_MODEL,
    OLLAMA_TIMEOUT,
    RAG_TOP_K,
    repo_slug as _repo_slug,
)

try:
    import chromadb
    _HAS_CHROMADB = True
except ImportError:
    chromadb = None  # type: ignore[assignment]
    _HAS_CHROMADB = False


def _chroma_dir(repo_url: str) -> Path:
    return CACHE_DIR / f"{_repo_slug(repo_url)}_chroma"


_EMBEDDING_CACHE: dict[str, list[float]] = {}
_EMBED_BATCH_SIZE = 10


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _batch_embed_via_api_embed(
    texts: list[str], ollama_url: str
) -> list[list[float]] | None:
    from src.ollama_client import embed_batch as _oc_embed_batch
    return _oc_embed_batch(ollama_url, EMBEDDING_MODEL, texts, timeout=OLLAMA_TIMEOUT)


def _single_embed_with_retry(text: str, ollama_url: str, label: str) -> list[float]:
    from src.ollama_client import embed_single as _oc_embed_single
    return _oc_embed_single(ollama_url, EMBEDDING_MODEL, text, timeout=OLLAMA_TIMEOUT, label=label)


def _embed_texts(texts: list[str], ollama_url: str) -> list[list[float]]:
    result: list[list[float] | None] = [None] * len(texts)

    uncached_indices: list[int] = []
    for i, text in enumerate(texts):
        cached = _EMBEDDING_CACHE.get(_cache_key(text))
        if cached is not None:
            result[i] = cached
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return result  # type: ignore[return-value]

    total_uncached = len(uncached_indices)
    processed = 0

    for batch_start in range(0, total_uncached, _EMBED_BATCH_SIZE):
        batch_idx = uncached_indices[batch_start: batch_start + _EMBED_BATCH_SIZE]
        batch_texts = [texts[i] for i in batch_idx]

        batch_results = _batch_embed_via_api_embed(batch_texts, ollama_url)

        if batch_results is not None:
            for i, embedding in zip(batch_idx, batch_results):
                _EMBEDDING_CACHE[_cache_key(texts[i])] = embedding
                result[i] = embedding
        else:
            for i in batch_idx:
                label = f"{processed + 1}/{total_uncached}"
                embedding = _single_embed_with_retry(texts[i], ollama_url, label)
                _EMBEDDING_CACHE[_cache_key(texts[i])] = embedding
                result[i] = embedding
                processed += 1

        processed += len(batch_idx) if batch_results is not None else 0

    return result  # type: ignore[return-value]


def index_overview_to_vectordb(repo_url: str, ollama_url: str, force: bool = False) -> int:
    if not _HAS_CHROMADB:
        return 0

    from src.analysis.context_cache import _overview_path
    path = _overview_path(repo_url)
    if not path.exists():
        return 0

    entries = json.loads(path.read_text(encoding="utf-8"))
    if not entries:
        return 0

    chroma_path = _chroma_dir(repo_url)
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))

    n_fn_docs = sum(1 for e in entries if e.get("functions"))
    expected_count = len(entries) + n_fn_docs

    if not force:
        try:
            existing = client.get_collection("overview")
            if existing.count() == expected_count:
                print(f"  Vektor-DB bereits aktuell ({expected_count} Dokumente) — übersprungen.", flush=True)
                return expected_count
        except Exception:
            pass

    try:
        client.delete_collection("overview")
    except Exception:
        pass
    collection = client.create_collection("overview")

    _MAX_EMBED_CHARS = 500
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for e in entries:
        fn = e["filename"]
        cat = e.get("category", "uncategorized")

        summary = (e.get("file_summary") or "").replace("\n", " ").strip()
        if len(summary) > _MAX_EMBED_CHARS:
            summary = summary[:_MAX_EMBED_CHARS - 3] + "..."
        doc = f"[{cat}] {fn}: {summary}"
        ids.append(f"{fn}::summary")
        documents.append(doc)
        metadatas.append({"filename": fn, "category": cat, "doc_type": "summary"})

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

    import src.analysis.context_cache as _cache_mod
    context = "\n".join(lines)
    if len(context) > _cache_mod._ACTIVE_MAX_CONTEXT_CHARS:
        context = context[:_cache_mod._ACTIVE_MAX_CONTEXT_CHARS - 3] + "..."
    return context


_FN_EXTRACT_RE = re.compile(
    r"(?:def|function|public\s+function|private\s+function|"
    r"protected\s+function|func)\s+(\w+)",
    re.IGNORECASE,
)

_RAG_QUERY_TEMPLATES: dict[str, str] = {
    "redundanz": "Funktion {fn_name} ähnliche Implementierung",
    "sicherheit": "Auth Validation User-Input {category}",
    "zombie_code": "Aufruf {fn_name} Referenz",
    "inkonsistenz": "[{category}] Fehlerbehandlung Muster",
    "fehlerbehandlung": "Exception Try-Catch {category} Pattern",
    "overengineering": "[{category}] Abstraktion Vereinfachung",
    "unklar": "[{category}] {filename} Kontext Abhängigkeit",
}


def _build_rag_query(finding: dict, filename: str = "", category: str = "") -> str:
    ftype = (finding.get("type") or "").lower().strip()
    evidence = finding.get("evidence_excerpt") or ""

    m = _FN_EXTRACT_RE.search(evidence)
    fn_name = m.group(1) if m else ""

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

    finding_refs: list[tuple[dict, str, str]] = []
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

    queries: list[str] = []
    for finding, fn, cat in finding_refs:
        q = _build_rag_query(finding, fn, cat)
        queries.append(q)

    embeddings = _embed_texts(queries, ollama_url)

    import src.analysis.context_cache as _cache_mod
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
                if len(context) > _cache_mod._ACTIVE_MAX_CONTEXT_CHARS:
                    context = context[:_cache_mod._ACTIVE_MAX_CONTEXT_CHARS - 3] + "..."
                finding["_rag_context"] = context
            else:
                finding["_rag_context"] = ""
        except Exception:
            finding["_rag_context"] = ""

    return results
