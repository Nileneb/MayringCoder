# Vision Image Ingest — Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Caption image files from GitHub repos (ERDs, architecture diagrams, screenshots) using `qwen2.5vl:3b` and ingest the captions into the MCP Memory (ChromaDB + SQLite), so Pi can retrieve visual context via `search_memory` before generating findings.

**Architecture:** New `src/image_ingest.py` orchestrates shallow clone → image discovery → captioning via existing `vision_captioner.py` → memory ingest via existing `memory_ingest.py`. A new `--ingest-images` flag in `checker.py` exposes the feature, analogous to the existing `--ingest-issues` flag.

**Tech Stack:** gitpython (or subprocess `git clone`), Pillow (already in vision_captioner.py), httpx, existing memory_ingest + memory_store modules, Ollama `/api/generate` multimodal endpoint.

---

## Problem Being Solved

Pi (qwen3.5:2b) has ~46% false positive rate on repos with rich domain structure because it lacks visual context. ERD diagrams, architecture PNGs, and schema screenshots encode relationships and conventions that are never present in the text-only gitingest output. When Pi calls `search_memory("user_id foreign key")` and finds an ERD caption confirming the relationship is intentional, it correctly suppresses the false finding.

---

## Flow

```
checker.py --ingest-images <repo-url>
    │
    ├─ git clone --depth=1 <repo> → /tmp/mayring-<slug>/
    │
    ├─ discover images: *.png *.jpg *.jpeg *.svg *.gif *.webp
    │       skip: > 5 MB, > 50 files total (configurable via --max-images)
    │
    ├─ per image:
    │       vision_captioner.caption_image(path, model=VISION_MODEL)
    │       → SVG: return raw text
    │       → raster: base64 → Ollama /api/generate
    │
    ├─ memory_ingest.ingest(
    │       source_type="repo_file",
    │       category="visual",
    │       filename=relative_path,
    │       content="[Caption] " + caption_text,
    │       repo=repo_slug,
    │   )
    │       → dedup via content hash (skip if unchanged)
    │       → ChromaDB collection: memory_chunks
    │       → SQLite: sources + chunks tables
    │
    └─ rm -rf /tmp/mayring-<slug>/   (always, even on error)
```

---

## Files

| File | Action | Responsibility |
|---|---|---|
| `src/image_ingest.py` | **NEW** | Clone, discover, caption, ingest, cleanup |
| `src/vision_captioner.py` | minimal | Already complete — add `timeout` param if missing |
| `checker.py` | +flag | `--ingest-images` flag, delegates to `image_ingest.run_image_ingest()` |
| `src/memory_ingest.py` | unchanged | `source_type="repo_file"` already supported |
| `src/memory_store.py` | unchanged | Hash-based dedup already implemented |

---

## Public Interface

### `src/image_ingest.py`

```python
def run_image_ingest(
    repo_url: str,
    ollama_url: str,
    vision_model: str = "qwen2.5vl:3b",
    max_images: int = 50,
    max_file_bytes: int = 5 * 1024 * 1024,  # 5 MB
    force_reingest: bool = False,
) -> dict:
    """
    Returns: {
        "images_found": int,
        "images_captioned": int,
        "images_skipped": int,   # too large, already ingested, or caption empty
        "images_failed": int,
        "repo_slug": str,
    }
    """
```

### `checker.py` CLI

```
--ingest-images     Caption and ingest repo images into Memory (requires qwen2.5vl:3b)
--vision-model STR  Vision model to use (default: qwen2.5vl:3b)
--max-images N      Max images per ingest run (default: 50)
--force-reingest    Re-caption even if image hash unchanged
```

---

## Memory Chunk Format

```json
{
  "source_type": "repo_file",
  "category": "visual",
  "filename": "docs/architecture.png",
  "content": "[Caption] The diagram shows a tasks table with columns id, title, status, user_id, project_id. tasks.user_id references users.id with onDelete cascade. tasks.project_id references projects.id.",
  "repo": "nileneb-applinngames",
  "content_hash": "<sha256>"
}
```

The `[Caption]` prefix lets Pi (and humans inspecting memory) immediately distinguish captioned images from source code chunks.

---

## Limits & Guards

- **Max file size:** 5 MB — larger files skipped with warning
- **Max images per run:** 50 (override with `--max-images`)
- **Dedup:** content hash via `memory_store` — unchanged images not re-captioned
- **Empty captions:** if `caption_image()` returns `""`, skip ingest, log warning
- **Cleanup:** `shutil.rmtree(clone_dir)` in `finally` block — always runs
- **VISION_MODEL env var:** overrides default `qwen2.5vl:3b`
- **Clone dir:** `/tmp/mayring-<repo_slug>-images/` to avoid collision with other pipeline temp dirs

---

## What Pi Gets

When Pi analyzes `app/Models/Task.php` and calls `search_memory("task user_id foreign key")`:

```
[Memory result]
Source: docs/erd.png (visual)
"[Caption] tasks table: id, title, status, priority (int 1-3), user_id → users.id (cascade delete), project_id → projects.id"
```

Pi now knows: `user_id` FK is correct, `priority` is integer not boolean — two of the five Task.php bugs correctly suppressed as framework conventions.

---

## Out of Scope

- Video files
- PDF diagrams (separate complexity)
- Private repos requiring token (supported via `GITHUB_TOKEN` env var passed to git clone, but not the focus)
- Automatic re-ingest on image changes (manual `--force-reingest` only)
