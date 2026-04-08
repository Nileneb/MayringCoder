"""Memory Ingestion Pipeline.

Stages:
    1. structural_chunk()    — split source content into Chunks
    2. mayring_categorize()  — optional LLM category labels (silently skips on error)
    3. resolve_dedup()       — exact dedup via text_hash
    4. ingest()              — orchestrate all stages + ChromaDB upsert + logging
"""

from __future__ import annotations

import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    import chromadb as _chromadb
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False

from src.config import CACHE_DIR, EMBEDDING_MODEL
from src.memory_schema import Chunk, Source
from src.memory_store import (
    find_by_text_hash,
    insert_chunk,
    kv_put,
    log_ingestion_event,
    upsert_source,
)

MEMORY_CHROMA_DIR: Path = CACHE_DIR / "memory_chroma"

# ---------------------------------------------------------------------------
# Optional JSONL logging (Phase 5 — opt-in)
# ---------------------------------------------------------------------------

_MEMORY_LOG_PATH: Path | None = None


def configure_memory_log(slug: str) -> None:
    """Enable JSONL logging to cache/<slug>_memory_log.jsonl."""
    global _MEMORY_LOG_PATH
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _MEMORY_LOG_PATH = CACHE_DIR / f"{slug}_memory_log.jsonl"


def _log_memory_event(event: dict) -> None:
    """Append one JSON line to the memory log. No-op if not configured."""
    if _MEMORY_LOG_PATH is None:
        return
    try:
        with _MEMORY_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Task 2.1 — Structural chunking
# ---------------------------------------------------------------------------

def _make_file_chunk(text: str, source_id: str, ordinal: int = 0) -> Chunk:
    """Single file-level fallback chunk."""
    text_hash = Chunk.compute_text_hash(text)
    return Chunk(
        chunk_id=Chunk.make_id(source_id, ordinal, "file"),
        source_id=source_id,
        chunk_level="file",
        ordinal=ordinal,
        start_offset=0,
        end_offset=len(text),
        text=text,
        text_hash=text_hash,
        dedup_key=text_hash,
        created_at=_now_iso(),
    )


def _chunk_python(text: str, source_id: str) -> list[Chunk]:
    """AST-based chunking for Python: top-level functions and classes."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines(keepends=True)
    # Build cumulative char offsets per line (0-indexed)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    chunks: list[Chunk] = []
    ordinal = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # Only top-level nodes (direct children of module)
        if node not in tree.body:  # type: ignore[attr-defined]
            continue

        level = "class" if isinstance(node, ast.ClassDef) else "function"
        start_line = node.lineno - 1  # 0-indexed
        end_line = getattr(node, "end_lineno", start_line) - 1  # 0-indexed
        start_off = line_offsets[start_line] if start_line < len(line_offsets) else 0
        end_off = (
            line_offsets[end_line] + len(lines[end_line])
            if end_line < len(lines)
            else len(text)
        )
        chunk_text = text[start_off:end_off]
        if not chunk_text.strip():
            continue

        text_hash = Chunk.compute_text_hash(chunk_text)
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(source_id, ordinal, level),
                source_id=source_id,
                chunk_level=level,
                ordinal=ordinal,
                start_offset=start_off,
                end_offset=end_off,
                text=chunk_text,
                text_hash=text_hash,
                dedup_key=text_hash,
                created_at=_now_iso(),
            )
        )
        ordinal += 1

    return chunks


def _chunk_js(text: str, source_id: str) -> list[Chunk]:
    """Regex + brace-depth chunking for JS/TS: functions and classes."""
    # Match: function X(, async function X(, class X {, const X = (async)? (, export variants
    pattern = re.compile(
        r"(?:^|\n)((?:export\s+default\s+|export\s+)?(?:async\s+)?function\s+\w+"
        r"|(?:export\s+)?class\s+\w+"
        r"|(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?\()",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    chunks: list[Chunk] = []
    ordinal = 0
    for i, m in enumerate(matches):
        start = m.start() if text[m.start()] != "\n" else m.start() + 1
        # Find end by counting braces from the match start
        brace_depth = 0
        end = len(text)
        found_open = False
        for j in range(start, len(text)):
            if text[j] == "{":
                brace_depth += 1
                found_open = True
            elif text[j] == "}":
                brace_depth -= 1
                if found_open and brace_depth == 0:
                    end = j + 1
                    break

        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue

        header = m.group(1)
        level = "class" if "class" in header else "function"
        text_hash = Chunk.compute_text_hash(chunk_text)
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(source_id, ordinal, level),
                source_id=source_id,
                chunk_level=level,
                ordinal=ordinal,
                start_offset=start,
                end_offset=end,
                text=chunk_text,
                text_hash=text_hash,
                dedup_key=text_hash,
                created_at=_now_iso(),
            )
        )
        ordinal += 1

    return chunks


def _chunk_markdown(text: str, source_id: str) -> list[Chunk]:
    """Split Markdown on headings (# / ## / ###)."""
    heading_re = re.compile(r"^#{1,3}\s+.+$", re.MULTILINE)
    matches = list(heading_re.finditer(text))
    if not matches:
        return []

    chunks: list[Chunk] = []
    ordinal = 0
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue
        text_hash = Chunk.compute_text_hash(chunk_text)
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(source_id, ordinal, "section"),
                source_id=source_id,
                chunk_level="section",
                ordinal=ordinal,
                start_offset=start,
                end_offset=end,
                text=chunk_text,
                text_hash=text_hash,
                dedup_key=text_hash,
                created_at=_now_iso(),
            )
        )
        ordinal += 1

    return chunks


def _chunk_yaml_json(text: str, source_id: str, filename: str) -> list[Chunk]:
    """Chunk YAML/JSON by top-level keys."""
    _MAX_CHUNK_CHARS = 2000
    try:
        if filename.endswith(".json"):
            data = json.loads(text)
        elif _HAS_YAML:
            data = _yaml.safe_load(text)
        else:
            return []
    except Exception:
        return []

    if not isinstance(data, dict) or not data:
        return []

    chunks: list[Chunk] = []
    ordinal = 0
    for key, value in data.items():
        chunk_text = json.dumps({key: value}, ensure_ascii=False)[:_MAX_CHUNK_CHARS]
        text_hash = Chunk.compute_text_hash(chunk_text)
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(source_id, ordinal, "block"),
                source_id=source_id,
                chunk_level="block",
                ordinal=ordinal,
                text=chunk_text,
                text_hash=text_hash,
                dedup_key=text_hash,
                created_at=_now_iso(),
            )
        )
        ordinal += 1

    return chunks


def structural_chunk(text: str, source_id: str, filename: str) -> list[Chunk]:
    """Dispatch to language-specific chunker based on file extension.

    Falls back to a single file-level chunk on parse failure or unknown extension.
    """
    if not text.strip():
        return [_make_file_chunk(text, source_id)]

    ext = Path(filename).suffix.lower()

    if ext == ".py":
        chunks = _chunk_python(text, source_id)
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        chunks = _chunk_js(text, source_id)
    elif ext in (".md", ".markdown"):
        chunks = _chunk_markdown(text, source_id)
    elif ext in (".yaml", ".yml", ".json"):
        chunks = _chunk_yaml_json(text, source_id, filename)
    else:
        chunks = []

    return chunks if chunks else [_make_file_chunk(text, source_id)]


# ---------------------------------------------------------------------------
# Task 2.2 — Optional Mayring categorization
# ---------------------------------------------------------------------------

_CATEGORIZE_SYSTEM_PROMPT = (
    "You are a code categorizer. Given a code or text chunk, "
    "respond with ONLY a comma-separated list of 3-5 short category labels "
    "(e.g. auth, error_handling, data_access, config, api, validation, test, utility). "
    "No explanation. No punctuation except commas."
)


def mayring_categorize(
    chunks: list[Chunk],
    ollama_url: str,
    model: str,
) -> list[Chunk]:
    """Assign Mayring category labels to each chunk via LLM (optional, best-effort)."""
    if not model or not ollama_url:
        return chunks

    # Import here to avoid circular imports at module level
    try:
        from src.analyzer import _ollama_generate
    except ImportError:
        return chunks

    for chunk in chunks:
        try:
            prompt = f"Chunk text (first 400 chars):\n\n{chunk.text[:400]}"
            response = _ollama_generate(
                prompt=prompt,
                ollama_url=ollama_url,
                model=model,
                label=f"categorize:{chunk.chunk_id[:8]}",
                system_prompt=_CATEGORIZE_SYSTEM_PROMPT,
            )
            labels = [
                lbl.strip().lower()
                for lbl in response.split(",")
                if lbl.strip()
            ]
            if labels:
                chunk.category_labels = labels[:5]
        except Exception:
            pass  # Categorization is strictly optional

    return chunks


# ---------------------------------------------------------------------------
# Task 2.3 — Dedup resolution
# ---------------------------------------------------------------------------

def resolve_dedup(
    conn: Any,  # sqlite3.Connection — avoid circular import with type hint
    chunk: Chunk,
) -> tuple[Chunk, bool]:
    """Exact dedup via text_hash.

    Returns (existing_chunk, True) if duplicate found.
    Returns (chunk, False) if no duplicate — caller should insert.
    """
    existing = find_by_text_hash(conn, chunk.text_hash)
    if existing is not None:
        return existing, True
    return chunk, False


# ---------------------------------------------------------------------------
# Task 2.4 — Ingestion orchestrator
# ---------------------------------------------------------------------------

def get_or_create_chroma_collection(chroma_dir: Path | None = None):
    """Get or create the 'memory_chunks' ChromaDB collection.

    Returns the collection object, or None if chromadb is unavailable.
    """
    if not _HAS_CHROMADB:
        return None
    path = chroma_dir or MEMORY_CHROMA_DIR
    path.mkdir(parents=True, exist_ok=True)
    client = _chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection("memory_chunks")


def ingest(
    source: Source,
    content: str,
    conn: Any,  # sqlite3.Connection
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    opts: dict | None = None,
) -> dict:
    """Orchestrate the full ingestion pipeline for one source.

    opts:
        categorize (bool, default False): run mayring_categorize()
        log (bool, default False): write JSONL event

    Returns:
        {source_id, chunk_ids, indexed, deduped, superseded}
    """
    opts = opts or {}
    do_categorize: bool = bool(opts.get("categorize", False))
    do_log: bool = bool(opts.get("log", False))

    # Import here to keep top-level imports clean
    from src.context import _embed_texts

    # Step 1: persist source
    upsert_source(conn, source)
    log_ingestion_event(conn, source.source_id, "ingest_start", {"path": source.path})

    # Step 2: structural chunking
    chunks = structural_chunk(content, source.source_id, source.path)

    # Step 3: optional categorization
    if do_categorize and model:
        chunks = mayring_categorize(chunks, ollama_url, model)

    # Step 4: dedup + embed + store
    new_chunk_ids: list[str] = []
    deduped_count = 0
    indexed = False

    for chunk in chunks:
        # 4a: exact dedup
        canonical, is_dup = resolve_dedup(conn, chunk)
        if is_dup:
            deduped_count += 1
            continue

        # 4b: insert into SQLite
        insert_chunk(conn, chunk)

        # 4c: embed
        try:
            emb = _embed_texts([chunk.text[:500]], ollama_url)[0]
        except Exception:
            emb = None

        # 4d: upsert to ChromaDB
        if chroma_collection is not None and emb is not None:
            try:
                chroma_collection.upsert(
                    ids=[chunk.chunk_id],
                    documents=[chunk.text[:500]],
                    embeddings=[emb],
                    metadatas=[{
                        "source_id": chunk.source_id,
                        "chunk_level": chunk.chunk_level,
                        "category_labels": ",".join(chunk.category_labels),
                        "is_active": 1,
                    }],
                )
                indexed = True
            except Exception:
                pass

        # 4e: KV cache
        kv_put(chunk.chunk_id, chunk.to_dict())
        new_chunk_ids.append(chunk.chunk_id)

    log_ingestion_event(
        conn,
        source.source_id,
        "ingest_done",
        {"chunks": len(new_chunk_ids), "deduped": deduped_count},
    )

    result = {
        "source_id": source.source_id,
        "chunk_ids": new_chunk_ids,
        "indexed": indexed,
        "deduped": deduped_count,
        "superseded": 0,
    }

    if do_log:
        _log_memory_event({"event": "ingest", "ts": _now_iso(), **result})

    return result
