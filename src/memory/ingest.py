"""Memory Ingestion Pipeline.

Stages:
    1. structural_chunk()    — split source content into Chunks
    2. mayring_categorize()  — optional LLM category labels (silently skips on error)
    3. resolve_dedup()       — exact dedup via text_hash
    4. ingest()              — orchestrate all stages + ChromaDB upsert + logging
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace as _dc_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.model_router import ModelRouter

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **_kw):  # type: ignore[misc]
        return it

try:
    import chromadb as _chromadb
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False

from src.config import CACHE_DIR, EMBEDDING_MODEL
from src.memory.schema import Chunk, Source
from src.memory.store import (
    add_source_ref,
    find_by_text_hash,
    get_source,
    init_memory_db,
    insert_chunk,
    kv_put,
    log_ingestion_event,
    upsert_source,
)
from src.memory.chunker import (  # noqa: F401
    _make_file_chunk,
    _chunk_python,
    _chunk_js,
    _chunk_markdown,
    _chunk_yaml_json,
    structural_chunk,
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
    from src.agents.vision import caption_image, get_image_metadata
    from src.analysis.context import _embed_texts

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

_PROMPTS_DIR: Path = Path(__file__).parent.parent.parent / "prompts"

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
    "paper":                {"categorize": True,  "codebook": "social", "mode": "hybrid", "multiview": True},
}
_INGEST_DEFAULT_FALLBACK: dict = {"categorize": True, "codebook": "auto", "mode": "hybrid", "multiview": False}

_CODEBOOK_DIR = Path(__file__).parent.parent.parent / "codebooks"


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
    router: "ModelRouter | None" = None,
) -> list[Chunk]:
    """Assign Mayring category labels to each chunk via LLM.

    Args:
        mode: "deductive" (closed category set), "inductive" (free derivation),
              "hybrid" (anchors + new categories marked with [neu])
        codebook: "auto" (detect from source_type), "code", "social", or profile name
        source_type: used for auto-detection of codebook
        conn: optional SQLite connection for error logging
        router: optional ModelRouter for task-based model selection
    """
    if router is not None and not model:
        _task = "mayring_code" if source_type == "repo_file" else "mayring_social"
        if router.is_available(_task):
            model = router.resolve(_task)

    if not model or not ollama_url:
        return chunks

    try:
        from src.analysis.analyzer import _ollama_generate
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
                    from src.memory.store import log_ingestion_event
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
    from src.memory.store import get_chroma_collection as get_collection
    return get_collection("memory_chunks", path=chroma_dir)


def ingest(
    source: Source,
    content: str,
    conn: Any,  # sqlite3.Connection
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    opts: dict | None = None,
    router: "ModelRouter | None" = None,
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

    # ── Automatische Visual Pipeline ─────────────────────────────────────────
    # Delegate to ingest_image() for known image extensions when router provides
    # a vision model. This keeps a single image pipeline with consistent chunk
    # schema (chunk_level="image_caption") instead of a second ad-hoc path.
    if _is_image_file(source.path):
        if source.source_type not in ("image",):
            source = _dc_replace(source, source_type="image")
        if router is not None and router.is_available("vision"):
            try:
                return ingest_image(
                    source=source,
                    image_path=Path(source.path),
                    conn=conn,
                    chroma_collection=chroma_collection,
                    ollama_url=ollama_url,
                    model=model,
                    vision_model=router.resolve("vision"),
                    workspace_id=workspace_id,
                )
            except Exception:
                pass  # Vision failed — fall through to generic pipeline with source_type=image

    # Modell aus Router wenn kein explizites model übergeben
    if router is not None and not model:
        _task = "mayring_code" if source.source_type == "repo_file" else "mayring_social"
        if router.is_available(_task):
            model = router.resolve(_task)

    # Merge source_type defaults with caller overrides (caller wins)
    defaults = _INGEST_DEFAULTS.get(source.source_type, _INGEST_DEFAULT_FALLBACK)
    effective = {**defaults, **opts}

    do_categorize: bool = bool(effective.get("categorize", True)) and bool(model)
    do_log: bool = bool(effective.get("log", False))
    do_multiview: bool = bool(effective.get("multiview", False))
    mode: str = effective.get("mode", "hybrid")
    codebook_choice: str = effective.get("codebook", "auto")

    # Import here to keep top-level imports clean
    from src.analysis.context import _embed_texts

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
            router=router,
        )

    # Step 4: dedup + embed + store
    new_chunk_ids: list[str] = []
    deduped_count = 0
    indexed = False

    for chunk in _tqdm(chunks, desc="Chunks embedden", unit="chunk", leave=False):
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
    from src.analysis.analyzer import _ollama_generate

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
    from src.memory.schema import source_fingerprint
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

from src.memory.image_ingest import discover_images, run_image_ingest  # noqa: F401
