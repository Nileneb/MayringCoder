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
import shutil
import subprocess
import sys
import tempfile
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
    add_source_ref,
    find_by_text_hash,
    get_source,
    init_memory_db,
    insert_chunk,
    kv_put,
    log_ingestion_event,
    upsert_source,
)

MEMORY_CHROMA_DIR: Path = CACHE_DIR / "memory_chroma"


def _coerce_str(val: object) -> str:
    """LLM-JSON-Felder können Liste, int oder None sein — immer zu str normalisieren."""
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v)
    return str(val)


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
    heading_re = re.compile(r"^#{1,3}\s+[^\n]+", re.MULTILINE)
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
# Image ingestion — Vision captioning via Ollama multimodal
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"})


def _is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in _IMAGE_EXTENSIONS


def ingest_image(
    source: "Source",
    image_path: "Path",
    conn: Any,
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    vision_model: str = "qwen2.5vl:3b",
    workspace_id: str = "default",
) -> dict:
    """Ingest a single image file via vision captioning.

    SVGs are ingested as raw text. Raster images are captioned via the Ollama
    multimodal model (vision_model) and stored as text chunks with embeddings.

    Returns:
        {source_id, chunk_ids, indexed, deduped, superseded}
    """
    from src.vision_captioner import caption_image, get_image_metadata
    from src.context import _embed_texts

    # Step 1: persist source + log start
    upsert_source(conn, source, workspace_id=workspace_id)
    log_ingestion_event(conn, source.source_id, "ingest_start", {"path": source.path})

    # Step 2: gather metadata
    metadata = get_image_metadata(image_path) or {}

    # Step 3: caption / read content
    caption = caption_image(image_path, ollama_url, vision_model)
    if not caption.strip():
        # Fallback: describe what we know from metadata
        fmt = metadata.get("format", "")
        w = metadata.get("width", 0)
        h = metadata.get("height", 0)
        size = metadata.get("file_size", 0)
        caption = (
            f"Image file: {image_path.name}"
            + (f" ({fmt}, {w}x{h} px, {size} bytes)" if fmt else f" ({size} bytes)")
        )

    # Step 4: determine category label
    is_svg = Path(source.path).suffix.lower() == ".svg"
    category_labels = ["diagram"] if is_svg else ["image"]

    # Step 5: create single image_caption chunk
    text_hash = Chunk.compute_text_hash(caption)
    chunk = Chunk(
        chunk_id=Chunk.make_id(source.source_id, 0, "image_caption"),
        source_id=source.source_id,
        chunk_level="image_caption",
        ordinal=0,
        start_offset=0,
        end_offset=len(caption),
        text=caption,
        text_hash=text_hash,
        dedup_key=text_hash,
        category_labels=category_labels,
        created_at=_now_iso(),
    )

    # Step 6: dedup check (workspace-scoped)
    canonical, is_dup = resolve_dedup(conn, chunk, workspace_id=workspace_id)
    new_chunk_ids: list[str] = []
    deduped_count = 0
    indexed = False

    if is_dup:
        deduped_count += 1
        add_source_ref(conn, canonical.chunk_id, source.source_id, workspace_id)
    else:
        # insert into SQLite
        insert_chunk(conn, chunk, workspace_id=workspace_id)
        add_source_ref(conn, chunk.chunk_id, source.source_id, workspace_id)

        # embed
        try:
            emb = _embed_texts([chunk.text[:500]], ollama_url)[0]
        except Exception:
            emb = None

        # upsert to ChromaDB
        if chroma_collection is not None and emb is not None:
            try:
                chroma_collection.upsert(
                    ids=[chunk.chunk_id],
                    documents=[chunk.text[:500]],
                    embeddings=[emb],
                    metadatas=[{
                        "workspace_id": workspace_id,
                        "source_id": chunk.source_id,
                        "chunk_level": chunk.chunk_level,
                        "category_labels": ",".join(chunk.category_labels),
                        "category_source": chunk.category_source,
                        "category_confidence": chunk.category_confidence,
                        "is_active": 1,
                    }],
                )
                indexed = True
            except Exception:
                pass

        # KV cache
        kv_put(chunk.chunk_id, chunk.to_dict())
        new_chunk_ids.append(chunk.chunk_id)

    # Step 7: log completion
    log_ingestion_event(
        conn,
        source.source_id,
        "ingest_done",
        {"chunks": len(new_chunk_ids), "deduped": deduped_count},
    )

    return {
        "source_id": source.source_id,
        "chunk_ids": new_chunk_ids,
        "indexed": indexed,
        "deduped": deduped_count,
        "superseded": 0,
    }


# ---------------------------------------------------------------------------
# Task 2.2 — Mayring categorization — Prompt-Template Ansatz
# ---------------------------------------------------------------------------

_PROMPTS_DIR: Path = Path(__file__).parent.parent / "prompts"

_ORIGINAL_MAYRING_CATEGORIES: list[str] = [
    "Zusammenfassung",
    "Explikation",
    "Strukturierung",
    "Paraphrase",
    "Reduktion",
    "Kategoriensystem",
    "Ankerbeispiel",
]

_MODE_TO_TEMPLATE: dict[str, str] = {
    "deductive": "mayring_deduktiv",
    "inductive": "mayring_induktiv",
    "hybrid": "mayring_hybrid",
}

# Ingest defaults per source_type — no more scattered opts in callers
_INGEST_DEFAULTS: dict[str, dict] = {
    "repo_file":            {"categorize": True,  "codebook": "auto", "mode": "hybrid", "multiview": False},
    "github_issue":         {"categorize": True,  "codebook": "social", "mode": "hybrid", "multiview": True},
    "conversation_summary": {"categorize": True,  "codebook": "social", "mode": "hybrid", "multiview": False},
    "note":                 {"categorize": True,  "codebook": "auto",   "mode": "hybrid", "multiview": False},
    "session_knowledge":    {"categorize": True,  "codebook": "social", "mode": "hybrid", "multiview": False},
    "session_note":         {"categorize": True,  "codebook": "social", "mode": "hybrid", "multiview": False},
    "image":                {"categorize": False, "codebook": "auto",   "mode": "hybrid", "multiview": False},
}
_INGEST_DEFAULT_FALLBACK: dict = {"categorize": True, "codebook": "auto", "mode": "hybrid", "multiview": False}

_CODEBOOK_DIR = Path(__file__).parent.parent / "codebooks"


def _resolve_codebook(codebook: str, source_type: str) -> list[str]:
    """Return category names for the given codebook/source_type.

    Resolution order:
      1. "auto" → maps source_type to "code" or "social"
      2. codebooks/<name>.yaml (code, social, or any custom)
      3. codebooks/profiles/<name>.yaml (generic, python, laravel, ...)
      4. Fallback → original Mayring categories
    """
    codebook = str(codebook).strip().lower()
    if codebook == "auto":
        _AUTO = {
            "repo_file": "code", "note": "code",
            "conversation": "social", "conversation_summary": "social",
            "session_knowledge": "social", "session_note": "social",
            "github_issue": "social",
        }
        codebook = _AUTO.get(source_type, "code")

    if codebook == "original":
        return list(_ORIGINAL_MAYRING_CATEGORIES)

    # Security: only alphanumeric + _ and - (blocks path traversal)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", codebook):
        return list(_ORIGINAL_MAYRING_CATEGORIES)

    for candidate in [
        _CODEBOOK_DIR / f"{codebook}.yaml",
        _CODEBOOK_DIR / "profiles" / f"{codebook}.yaml",
    ]:
        if candidate.exists() and _HAS_YAML:
            try:
                import yaml as _yaml_local
                data = _yaml_local.safe_load(candidate.read_text(encoding="utf-8"))
                cats = data.get("categories", []) if isinstance(data, dict) else data
                names: list[str] = []
                for c in cats:
                    if isinstance(c, str):
                        names.append(c)
                    elif isinstance(c, dict):
                        n = c.get("label") or c.get("name", "")
                        if n:
                            names.append(n)
                if names:
                    return names
            except Exception:
                continue

    return list(_ORIGINAL_MAYRING_CATEGORIES)


def _path_fallback_category(path: str) -> list[str]:
    """Regex-based category from file path when LLM categorization fails."""
    _RULES = [
        (r"test_|_test\.|/tests/|conftest", "tests"),
        (r"/api/|/routes/|/controllers/|router", "api"),
        (r"/models?/|/db/|/migration|repositor", "data_access"),
        (r"/auth|/security|/guards?/|/policies?", "auth"),
        (r"/service|/domain|/usecase|/business", "domain"),
        (r"/middleware|/pipeline", "middleware"),
        (r"config.*\.(py|yaml|yml|env)|settings\.|constants\.", "config"),
        (r"/utils?/|/helpers?/|/tools?/", "utils"),
        (r"/cache|redis|memcache", "caching"),
        (r"/log|monitor|metric|trace", "logging"),
    ]
    for pattern, cat in _RULES:
        if re.search(pattern, path, re.IGNORECASE):
            return [cat]
    return []


def _load_mayring_template(mode: str) -> str:
    """Load prompt template for the given mode. Falls back to inline default."""
    filename = _MODE_TO_TEMPLATE.get(mode, "mayring_hybrid") + ".md"
    template_path = _PROMPTS_DIR / filename
    try:
        return template_path.read_text(encoding="utf-8")
    except OSError:
        return (
            "Categorize this text chunk using these categories if applicable: {{categories}}. "
            "Respond with ONLY a comma-separated list of labels."
        )


def mayring_categorize(
    chunks: list[Chunk],
    ollama_url: str,
    model: str,
    mode: str = "hybrid",
    codebook: str = "auto",
    source_type: str = "repo_file",
    conn: Any = None,
) -> list[Chunk]:
    """Assign Mayring category labels to each chunk via LLM.

    Args:
        mode: "deductive" (closed category set), "inductive" (free derivation),
              "hybrid" (anchors + new categories marked with [neu])
        codebook: "auto" (detect from source_type), "code", "social", or profile name
        source_type: used for auto-detection of codebook
        conn: optional SQLite connection for error logging
    """
    if not model or not ollama_url:
        return chunks

    try:
        from src.analyzer import _ollama_generate
    except ImportError:
        return chunks

    categories = _resolve_codebook(codebook, source_type)
    valid_set = {c.lower() for c in categories if c}
    template = _load_mayring_template(mode)
    system_prompt = template.replace("{{categories}}", ", ".join(categories))

    for chunk in chunks:
        try:
            prompt = f"Text chunk:\n\n{chunk.text[:1200]}"
            response = _ollama_generate(
                prompt=prompt,
                ollama_url=ollama_url,
                model=model,
                label=f"mayring:{chunk.chunk_id[:8]}",
                system_prompt=system_prompt,
            )
            raw = [re.sub(r"^[-•*]\s*", "", p).strip()
                   for p in re.split(r"[,\n]", response)]

            validated: list[str] = []
            for lbl in raw:
                if not lbl or len(lbl) > 60 or "," in lbl or len(lbl.split()) > 4:
                    continue
                if mode == "inductive":
                    # Free derivation — accept any reasonable label
                    validated.append(lbl)
                elif mode == "hybrid" and lbl.lower().startswith("[neu]"):
                    validated.append(lbl)  # Keep [neu] prefix
                elif lbl.lower() in valid_set:
                    validated.append(lbl.lower())

            chunk.category_labels = validated[:5]
            chunk.category_source = mode
            chunk.category_confidence = 1.0 if validated else 0.0

        except Exception as exc:
            chunk.category_labels = _path_fallback_category(chunk.source_id or "")
            chunk.category_source = "fallback"
            chunk.category_confidence = 0.5 if chunk.category_labels else 0.0
            if conn is not None:
                try:
                    from src.memory_store import log_ingestion_event
                    log_ingestion_event(conn, chunk.chunk_id, "categorize_error",
                                        {"error": str(exc)[:200]})
                except Exception:
                    pass

    return chunks


# ---------------------------------------------------------------------------
# Task 2.3 — Dedup resolution
# ---------------------------------------------------------------------------

def resolve_dedup(
    conn: Any,  # sqlite3.Connection — avoid circular import with type hint
    chunk: Chunk,
    workspace_id: str = "default",
) -> tuple[Chunk, bool]:
    """Exact dedup via text_hash (workspace-scoped).

    Returns (existing_chunk, True) if duplicate found in same workspace.
    Returns (chunk, False) if no duplicate — caller should insert.
    """
    existing = find_by_text_hash(conn, chunk.text_hash, workspace_id=workspace_id)
    if existing is not None:
        return existing, True
    return chunk, False


# ---------------------------------------------------------------------------
# Task 2.4 — Ingestion orchestrator
# ---------------------------------------------------------------------------

def get_or_create_chroma_collection(chroma_dir: Path | None = None):
    """Get or create the 'memory_chunks' ChromaDB collection (process singleton)."""
    if not _HAS_CHROMADB:
        return None
    from src.memory_store import get_chroma_collection as get_collection
    return get_collection("memory_chunks", path=chroma_dir)


def ingest(
    source: Source,
    content: str,
    conn: Any,  # sqlite3.Connection
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    opts: dict | None = None,
    workspace_id: str = "default",
) -> dict:
    """Orchestrate the full ingestion pipeline for one source.

    opts:
        categorize (bool, default False): run mayring_categorize()
        log (bool, default False): write JSONL event

    Returns:
        {source_id, chunk_ids, indexed, deduped, superseded}
    """
    opts = opts or {}
    # Merge source_type defaults with caller overrides (caller wins)
    defaults = _INGEST_DEFAULTS.get(source.source_type, _INGEST_DEFAULT_FALLBACK)
    effective = {**defaults, **opts}

    do_categorize: bool = bool(effective.get("categorize", True)) and bool(model)
    do_log: bool = bool(effective.get("log", False))
    do_multiview: bool = bool(effective.get("multiview", False))
    mode: str = effective.get("mode", "hybrid")
    codebook_choice: str = effective.get("codebook", "auto")

    # Import here to keep top-level imports clean
    from src.context import _embed_texts

    # Step 0: skip if source content unchanged
    if source.content_hash:
        existing_src = get_source(conn, source.source_id)
        if existing_src and existing_src.content_hash == source.content_hash:
            return {
                "source_id": source.source_id,
                "chunk_ids": [], "indexed": False,
                "deduped": 0, "superseded": 0, "skipped": True,
            }

    # Step 1: persist source
    upsert_source(conn, source, workspace_id=workspace_id)
    log_ingestion_event(conn, source.source_id, "ingest_start", {"path": source.path})

    # Step 2: structural chunking (or multi-view for github_issue)
    if do_multiview and source.source_type == "github_issue" and model:
        chunks = generate_multiview_chunks(source.source_id, content, ollama_url, model)
    else:
        chunks = structural_chunk(content, source.source_id, source.path)

    # Step 3: optional categorization
    if do_categorize and model:
        chunks = mayring_categorize(
            chunks, ollama_url, model,
            mode=mode, codebook=codebook_choice,
            source_type=source.source_type,
            conn=conn,
        )

    # Step 4: dedup + embed + store
    new_chunk_ids: list[str] = []
    deduped_count = 0
    indexed = False

    for chunk in chunks:
        # 4a: exact dedup (workspace-scoped)
        canonical, is_dup = resolve_dedup(conn, chunk, workspace_id=workspace_id)
        if is_dup:
            deduped_count += 1
            add_source_ref(conn, canonical.chunk_id, source.source_id, workspace_id)
            continue

        # 4b: insert into SQLite + register source ref for new chunk
        insert_chunk(conn, chunk, workspace_id=workspace_id)
        add_source_ref(conn, chunk.chunk_id, source.source_id, workspace_id)

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
                        "workspace_id": workspace_id,
                        "source_id": chunk.source_id,
                        "chunk_level": chunk.chunk_level,
                        "category_labels": ",".join(chunk.category_labels),
                        "category_source": chunk.category_source,
                        "category_confidence": chunk.category_confidence,
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


# ---------------------------------------------------------------------------
# Multi-view Indexing for GitHub Issues
# ---------------------------------------------------------------------------

_MULTIVIEW_SYSTEM_PROMPT = """Du bist ein präziser Informationsextrahierer.
Antworte NUR mit dem angeforderten JSON-Objekt, ohne Erklärungen oder Markdown-Blöcke."""

_MULTIVIEW_PROMPT_TEMPLATE = """Analysiere dieses GitHub Issue und extrahiere vier strukturierte Sichten.

ISSUE-TEXT:
{content}

Antworte mit genau diesem JSON:
{{
  "fact_summary": "<2-4 Sätze: Wer meldet was, welcher konkrete Fehler/Feature-Request, betroffene Komponente>",
  "impl_summary": "<2-4 Sätze: Betroffene Module/Dateien/Services, vermutete Ursache, mögliche Fix-Strategien. Leer lassen wenn unklar>",
  "decision_summary": "<2-4 Sätze: Getroffene Entscheidungen, gewählte Ansätze, offene Fragen. Leer lassen wenn keine>",
  "entities_keywords": "<kommagetrennte Liste: Technologien, Fehlercodes, Dateinamen, Komponenten, Schlüsselbegriffe>"
}}"""


def generate_multiview_chunks(
    source_id: str,
    content: str,
    ollama_url: str,
    model: str,
) -> list["Chunk"]:
    """Generiert 5 semantische View-Chunks für ein GitHub Issue via LLM.

    Views:
      - view_fact: Fakten-Zusammenfassung (Wer, Was, Welcher Fehler)
      - view_impl: Betroffene Module, Ursache, Fix-Strategien
      - view_decision: Entscheidungen und offene Fragen
      - view_entities: Keywords und Entitäten
      - view_full: Originaltext als Fallback

    Falls der LLM-Call fehlschlägt, wird nur view_full zurückgegeben.
    """
    import json as _json_mod
    from src.analyzer import _ollama_generate

    view_full = Chunk(
        chunk_id=Chunk.make_id(source_id, 0, "view_full"),
        source_id=source_id,
        chunk_level="view_full",
        ordinal=0,
        text=content,
        text_hash=Chunk.compute_text_hash(content),
    )

    if not model or not ollama_url:
        return [view_full]

    try:
        prompt = _MULTIVIEW_PROMPT_TEMPLATE.format(content=content[:3000])
        raw = _ollama_generate(
            prompt, ollama_url, model,
            label="multiview",
            system_prompt=_MULTIVIEW_SYSTEM_PROMPT,
        )
        # Parse JSON — strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = _json_mod.loads(raw)
        if not isinstance(parsed, dict):
            return [view_full]
    except Exception:
        return [view_full]

    chunks: list[Chunk] = []
    for ordinal, (level, key) in enumerate([
        ("view_fact", "fact_summary"),
        ("view_impl", "impl_summary"),
        ("view_decision", "decision_summary"),
        ("view_entities", "entities_keywords"),
    ]):
        text = _coerce_str(parsed.get(key)).strip()
        if not text:
            continue
        chunks.append(Chunk(
            chunk_id=Chunk.make_id(source_id, ordinal, level),
            source_id=source_id,
            chunk_level=level,
            ordinal=ordinal,
            text=text,
            text_hash=Chunk.compute_text_hash(text),
        ))

    chunks.append(view_full)
    return chunks


# ---------------------------------------------------------------------------
# Task X — Conversation-Summary Ingestion
# ---------------------------------------------------------------------------

def ingest_conversation_summary(
    summary_text: str,
    conn: Any,
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    session_id: str | None = None,
    run_id: str | None = None,
    workspace_id: str = "default",
) -> dict:
    """Ingest a Claude /compact summary as a conversation_summary source.

    Args:
        summary_text: Raw Markdown text of the compaction summary.
        conn: SQLite connection (from init_memory_db()).
        chroma_collection: ChromaDB collection or None.
        ollama_url: Ollama base URL.
        model: Ollama model name (empty string = no embedding / categorization).
        session_id: Optional session identifier, stored in Source.branch.
        run_id: Optional run identifier, stored in Source.commit.

    Returns:
        {source_id, chunk_ids, indexed, deduped, superseded}
    """
    import hashlib

    content_hash = "sha256:" + hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
    path = f"summary/{session_id or 'unknown'}.md"
    source_id = Source.make_id("conversation", path)

    source = Source(
        source_id=source_id,
        source_type="conversation_summary",
        repo="conversation",
        path=path,
        branch=session_id or "",
        commit=run_id or "",
        content_hash=content_hash,
        captured_at=_now_iso(),
    )

    return ingest(
        source=source,
        content=summary_text,
        conn=conn,
        chroma_collection=chroma_collection,
        ollama_url=ollama_url,
        model=model,
        opts={
            "categorize": bool(model),
            "codebook": "social",
            "mode": "hybrid",
            "log": True,
        },
        workspace_id=workspace_id,
    )


# ---------------------------------------------------------------------------
# GitHub Issues ingestion (merged from ingest_github_issues.py)
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"})


def fetch_issues(repo: str, state: str = "open", limit: int = 100) -> list[dict]:
    """Fetch issues via gh CLI. Returns empty list on any error."""
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--repo", repo, "--state", state,
             "--limit", str(limit), "--json", "number,title,body,labels,state,url,createdAt"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def issues_to_sources(issues: list[dict], repo: str) -> list[tuple[Source, str]]:
    """Map gh issues to (Source, content) tuples for ingest()."""
    from src.memory_schema import source_fingerprint
    result: list[tuple[Source, str]] = []
    for issue in issues:
        if not isinstance(issue, dict) or "number" not in issue:
            continue
        title = issue.get("title") or ""
        body = issue.get("body") or ""
        content = f"# {title}\n\n{body}"
        content_hash = source_fingerprint(content)
        source = Source(
            source_id=f"github_issue:{repo}:issue/{issue['number']}:{content_hash[:16]}",
            source_type="github_issue",
            repo=repo,
            path=f"issue/{issue['number']}",
            content_hash=content_hash,
            branch=issue.get("state"),
            commit=str(issue["number"]),
        )
        result.append((source, content))
    return result


# ---------------------------------------------------------------------------
# Image ingestion (merged from image_ingest.py)
# ---------------------------------------------------------------------------

def discover_images(root: Path, max_file_bytes: int = 5 * 1024 * 1024, max_images: int = 50) -> list[Path]:
    """Walk root recursively, return image files within size and count limits."""
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if len(found) >= max_images:
            break
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                print(f"[ingest-images] Skip (too large): {path.relative_to(root)}")
                continue
        except OSError:
            continue
        found.append(path)
    return found


def run_image_ingest(
    repo_url: str,
    ollama_url: str,
    vision_model: str = "qwen2.5vl:3b",
    embed_model: str = "nomic-embed-text",
    max_images: int = 50,
    max_file_bytes: int = 5 * 1024 * 1024,
    force_reingest: bool = False,
    workspace_id: str = "default",
) -> dict:
    """Shallow-clone repo, caption all images, ingest into memory."""
    url = repo_url.rstrip("/").removesuffix(".git")
    parts = url.split("github.com/", 1)
    repo_owner_name = parts[1] if len(parts) == 2 else url
    repo_slug = repo_owner_name.replace("/", "-").lower()
    clone_dir = tempfile.mkdtemp(prefix=f"mayring-{repo_slug}-images-")
    try:
        return _run_image_ingest_from_clone(
            repo_url=url, clone_dir=Path(clone_dir),
            repo_owner_name=repo_owner_name, ollama_url=ollama_url,
            vision_model=vision_model, embed_model=embed_model,
            max_images=max_images, max_file_bytes=max_file_bytes,
            force_reingest=force_reingest, workspace_id=workspace_id,
        )
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)


def _run_image_ingest_from_clone(
    repo_url: str, clone_dir: Path, repo_owner_name: str, ollama_url: str,
    vision_model: str, embed_model: str, max_images: int, max_file_bytes: int,
    force_reingest: bool, workspace_id: str = "default",
) -> dict:
    print(f"[ingest-images] git clone --depth=1 {repo_url} ...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", repo_url, str(clone_dir)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")

    images = discover_images(clone_dir, max_file_bytes=max_file_bytes, max_images=max_images)
    print(f"[ingest-images] {len(images)} Bilder gefunden")
    if not images:
        return {"images_found": 0, "images_captioned": 0, "images_skipped": 0, "images_failed": 0, "repo": repo_owner_name}

    from src.memory_store import deactivate_chunks_by_source
    from src.memory_retrieval import invalidate_query_cache

    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()
    captioned = skipped = failed = 0

    if force_reingest:
        for img_path in images:
            rel = str(img_path.relative_to(clone_dir))
            deactivate_chunks_by_source(conn, Source.make_id(repo_owner_name, rel))
        invalidate_query_cache()

    try:
        for img_path in images:
            rel = str(img_path.relative_to(clone_dir))
            source = Source(
                source_id=Source.make_id(repo_owner_name, rel),
                source_type="repo_file", repo=repo_owner_name, path=rel,
            )
            try:
                r = ingest_image(
                    source=source, image_path=img_path, conn=conn,
                    chroma_collection=chroma, ollama_url=ollama_url,
                    model=embed_model, vision_model=vision_model, workspace_id=workspace_id,
                )
                if r.get("deduped", 0) > 0:
                    skipped += 1
                    print(f"[ingest-images] Dedup: {rel}")
                else:
                    captioned += 1
                    print(f"[ingest-images] OK: {rel}")
            except Exception as exc:
                failed += 1
                print(f"[ingest-images] FEHLER {rel}: {exc}")
    finally:
        conn.close()

    return {"images_found": len(images), "images_captioned": captioned,
            "images_skipped": skipped, "images_failed": failed, "repo": repo_owner_name}
