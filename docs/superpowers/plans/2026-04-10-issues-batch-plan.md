# Issues Batch Implementation Plan (#31, #32, #33, #34, #35, #37, #38)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all 7 open MayringCoder issues: smart chunking, visual RAG via Qwen2.5-VL, JWT workspace isolation, full-pipeline Docker, model docs, memory roadmap, and E2E web tests.

**Architecture:** Extends existing memory pipeline (memory_ingest → memory_store → memory_retrieval → mcp_server) with: (1) priority-based block selection in splitter.py, (2) vision captioning via Ollama multimodal API, (3) JWT auth middleware replacing static X-Auth-Token, (4) workspace-scoped Chroma collections and SQLite filtering.

**Tech Stack:** Python 3.11, SQLite, ChromaDB, Ollama (qwen2.5vl:3b for vision), FastMCP, PyJWT, Pillow, Docker Compose, pytest

---

## File Map

| File | Action | Issue | Purpose |
|---|---|---|---|
| `src/splitter.py` | Modify | #32 | Add `smart_split()` with AST blocks + priority scoring |
| `src/vision_captioner.py` | Create | #31 | Ollama multimodal captioning for images |
| `src/memory_ingest.py` | Modify | #31 | Image source_type support, call vision_captioner |
| `codebook.yaml` | Modify | #31 | Remove image extensions from exclude_patterns |
| `src/mcp_server.py` | Modify | #38 | JWT middleware replacing static X-Auth-Token |
| `src/memory_store.py` | Modify | #38 | workspace_id column + filtered queries |
| `src/memory_ingest.py` | Modify | #38 | Pass workspace_id to chroma collection |
| `src/memory_retrieval.py` | Modify | #38 | workspace_id scope filter |
| `tools/generate_mcp_token.py` | Create | #38 | JWT token generator CLI |
| `docker-compose.full.yml` | Create | #37 | Full pipeline: Ollama + analyzer + web-ui + MCP |
| `Dockerfile` | Modify | #37 | Add checker.py, prompts/, codebook.yaml |
| `README.md` | Modify | #35 | Model recommendation table with VRAM |
| `docs/memory_roadmap.md` | Create | #34 | Phase 2-4 roadmap |
| `tests/test_smart_split.py` | Create | #32 | Smart splitting tests |
| `tests/test_vision_captioner.py` | Create | #31 | Vision captioning tests |
| `tests/test_jwt_auth.py` | Create | #38 | JWT middleware tests |
| `tests/test_web_ui.py` | Modify | #33 | E2E flow tests |
| `requirements.txt` | Modify | #31, #38 | Add Pillow, PyJWT |

---

### Task 1: Smart Splitting — Priority-Based Block Selection (#32)

**Files:**
- Modify: `src/splitter.py`
- Create: `tests/test_smart_split.py`

- [ ] **Step 1: Write failing tests for smart_split**

Create `tests/test_smart_split.py`:

```python
"""Tests for smart_split() — priority-based block selection."""

import textwrap
import pytest
from src.splitter import smart_split


class TestSmartSplitPython:
    """Python files: AST-based blocks with priority scoring."""

    def test_extracts_functions_as_blocks(self):
        code = textwrap.dedent("""\
            def helper():
                return 1

            def main():
                return helper()
        """)
        result = smart_split(code, "app.py", max_chars=5000)
        assert result["blocks"]
        names = [b["name"] for b in result["blocks"]]
        assert "helper" in names
        assert "main" in names

    def test_security_functions_get_higher_priority(self):
        code = textwrap.dedent("""\
            def format_name(name):
                return name.strip()

            def authenticate_user(token):
                return validate(token)

            def delete_account(user_id):
                return remove(user_id)
        """)
        result = smart_split(code, "app.py", max_chars=5000)
        blocks = result["blocks"]
        auth_block = next(b for b in blocks if b["name"] == "authenticate_user")
        delete_block = next(b for b in blocks if b["name"] == "delete_account")
        format_block = next(b for b in blocks if b["name"] == "format_name")
        assert auth_block["priority"] > format_block["priority"]
        assert delete_block["priority"] > format_block["priority"]

    def test_respects_max_chars_limit(self):
        funcs = "\n".join(
            f"def func_{i}():\n    return {i}\n" for i in range(50)
        )
        result = smart_split(funcs, "big.py", max_chars=200)
        total = sum(len(b["text"]) for b in result["selected"])
        assert total <= 200
        assert result["skipped_summary"] != ""

    def test_skipped_summary_lists_omitted_functions(self):
        funcs = "\n".join(
            f"def func_{i}():\n    return {i}\n" for i in range(50)
        )
        result = smart_split(funcs, "big.py", max_chars=200)
        assert "func_" in result["skipped_summary"]

    def test_classes_are_extracted(self):
        code = textwrap.dedent("""\
            class UserService:
                def create(self):
                    pass
                def delete_user(self):
                    pass
        """)
        result = smart_split(code, "service.py", max_chars=5000)
        names = [b["name"] for b in result["blocks"]]
        assert "UserService" in names

    def test_syntax_error_falls_back_to_truncation(self):
        bad_code = "def foo(\n  broken syntax here"
        result = smart_split(bad_code, "broken.py", max_chars=100)
        assert result["selected"]
        assert result["selected"][0]["name"] == "__fallback__"


class TestSmartSplitJS:
    """JS/TS files: regex-based block extraction."""

    def test_extracts_js_functions(self):
        code = "function greet(name) {\n  return name;\n}\n\nfunction adminDelete(id) {\n  return id;\n}\n"
        result = smart_split(code, "app.js", max_chars=5000)
        names = [b["name"] for b in result["blocks"]]
        assert "greet" in names
        assert "adminDelete" in names

    def test_admin_function_higher_priority(self):
        code = "function greet(name) {\n  return name;\n}\n\nfunction adminDelete(id) {\n  return id;\n}\n"
        result = smart_split(code, "app.js", max_chars=5000)
        admin = next(b for b in result["blocks"] if b["name"] == "adminDelete")
        greet = next(b for b in result["blocks"] if b["name"] == "greet")
        assert admin["priority"] > greet["priority"]


class TestSmartSplitFallback:
    """Unknown extensions: simple truncation fallback."""

    def test_unknown_ext_returns_truncated(self):
        content = "x" * 500
        result = smart_split(content, "data.csv", max_chars=100)
        assert len(result["selected"][0]["text"]) <= 100

    def test_empty_content(self):
        result = smart_split("", "empty.py", max_chars=1000)
        assert result["selected"] == []
        assert result["blocks"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_smart_split.py -v`
Expected: ImportError — `smart_split` not found in `src.splitter`

- [ ] **Step 3: Implement smart_split in src/splitter.py**

Add to the end of `src/splitter.py` (after the existing `split_into_files` function):

```python
import ast as _ast

# ---------------------------------------------------------------------------
# Priority keywords for security/risk-relevant code blocks
# ---------------------------------------------------------------------------

_PRIORITY_KEYWORDS: dict[str, int] = {
    "auth": 3, "authenticate": 3, "login": 3, "password": 3, "secret": 3,
    "token": 3, "delete": 3, "admin": 3, "permission": 3, "csrf": 3,
    "export": 2, "__init__": 2, "create": 2, "update": 2, "save": 2,
    "except": 1, "error": 1, "catch": 1, "raise": 1, "fail": 1,
}


def _score_block(name: str, text: str) -> int:
    """Score a code block by keyword presence in name and body."""
    score = 0
    name_lower = name.lower()
    text_lower = text.lower()
    for kw, pts in _PRIORITY_KEYWORDS.items():
        if kw in name_lower:
            score += pts
        elif kw in text_lower:
            score += max(1, pts - 1)
    return score


def _extract_python_blocks(text: str) -> list[dict]:
    """AST-based extraction of top-level functions and classes."""
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    blocks: list[dict] = []
    for node in tree.body:
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            continue
        start_line = node.lineno - 1
        end_line = getattr(node, "end_lineno", start_line + 1) - 1
        start_off = line_offsets[start_line] if start_line < len(line_offsets) else 0
        end_off = (
            line_offsets[end_line] + len(lines[end_line])
            if end_line < len(lines)
            else len(text)
        )
        block_text = text[start_off:end_off]
        name = node.name
        priority = _score_block(name, block_text)
        blocks.append({
            "name": name,
            "text": block_text,
            "priority": priority,
            "start": start_off,
            "end": end_off,
        })

    return blocks


_JS_BLOCK_RE = re.compile(
    r"(?:^|\n)((?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)"
    r"|(?:export\s+)?class\s+(\w+))",
    re.MULTILINE,
)


def _extract_js_blocks(text: str) -> list[dict]:
    """Regex + brace-depth extraction for JS/TS functions and classes."""
    matches = list(_JS_BLOCK_RE.finditer(text))
    if not matches:
        return []

    blocks: list[dict] = []
    for m in matches:
        start = m.start() if text[m.start()] != "\n" else m.start() + 1
        name = m.group(2) or m.group(3) or "anonymous"

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

        block_text = text[start:end].strip()
        if not block_text:
            continue

        priority = _score_block(name, block_text)
        blocks.append({
            "name": name,
            "text": block_text,
            "priority": priority,
            "start": start,
            "end": end,
        })

    return blocks


def smart_split(content: str, filename: str, max_chars: int = 3000) -> dict:
    """Split a file into prioritized blocks, selecting the most important ones.

    Returns:
        {
            "blocks": list[dict] — all extracted blocks with name, text, priority
            "selected": list[dict] — blocks chosen within max_chars budget
            "skipped_summary": str — one-line summary of omitted blocks
        }
    """
    if not content.strip():
        return {"blocks": [], "selected": [], "skipped_summary": ""}

    ext = _Path(filename).suffix.lower()

    if ext == ".py":
        blocks = _extract_python_blocks(content)
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        blocks = _extract_js_blocks(content)
    else:
        blocks = []

    # Fallback: no blocks extracted → truncate
    if not blocks:
        truncated = content[:max_chars]
        fallback = {"name": "__fallback__", "text": truncated, "priority": 0, "start": 0, "end": len(truncated)}
        return {"blocks": [], "selected": [fallback], "skipped_summary": ""}

    # Sort by priority DESC, then by position ASC for stability
    blocks.sort(key=lambda b: (-b["priority"], b["start"]))

    selected: list[dict] = []
    used = 0
    skipped: list[str] = []

    for block in blocks:
        if used + len(block["text"]) <= max_chars:
            selected.append(block)
            used += len(block["text"])
        else:
            skipped.append(block["name"])

    # Re-sort selected by original position for readable output
    selected.sort(key=lambda b: b["start"])

    skipped_summary = ""
    if skipped:
        skipped_summary = f"{len(skipped)} weitere Blöcke übersprungen: {', '.join(skipped[:10])}"
        if len(skipped) > 10:
            skipped_summary += f" (+{len(skipped) - 10} weitere)"

    return {"blocks": blocks, "selected": selected, "skipped_summary": skipped_summary}
```

Also add the import at the top of `src/splitter.py`:
```python
from pathlib import Path as _Path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_smart_split.py -v`
Expected: All PASS

- [ ] **Step 5: Run existing splitter tests to check for regressions**

Run: `.venv/bin/python -m pytest tests/test_splitter.py -v`
Expected: All PASS (existing split_into_files unchanged)

- [ ] **Step 6: Commit**

```bash
git add src/splitter.py tests/test_smart_split.py
git commit -m "feat(#32): add smart_split() with AST-based priority block selection

Extracts functions/classes from Python (AST) and JS/TS (regex+brace),
scores blocks by security-relevance keywords, selects highest-priority
blocks within max_chars budget. Skipped blocks get a summary line."
```

---

### Task 2: Vision Captioner Module (#31)

**Files:**
- Create: `src/vision_captioner.py`
- Create: `tests/test_vision_captioner.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add Pillow to requirements.txt**

Append `Pillow` to `requirements.txt`:

```
gitingest
httpx
python-dotenv
pyyaml
chromadb
mcp
Pillow
```

- [ ] **Step 2: Install new dependency**

Run: `.venv/bin/pip install Pillow`

- [ ] **Step 3: Write failing tests for vision_captioner**

Create `tests/test_vision_captioner.py`:

```python
"""Tests for src/vision_captioner.py — Ollama multimodal image captioning."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.vision_captioner import caption_image, caption_images_batch, get_image_metadata


class TestGetImageMetadata:
    """Image metadata extraction without Ollama."""

    def test_png_metadata(self, tmp_path):
        # Create a minimal 1x1 red PNG
        from PIL import Image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 50), color="red")
        img.save(img_path)

        meta = get_image_metadata(img_path)
        assert meta["width"] == 100
        assert meta["height"] == 50
        assert meta["format"] == "PNG"
        assert meta["file_size"] > 0

    def test_nonexistent_file_returns_none(self, tmp_path):
        meta = get_image_metadata(tmp_path / "nope.png")
        assert meta is None

    def test_svg_returns_text_metadata(self, tmp_path):
        svg_path = tmp_path / "diagram.svg"
        svg_path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"></svg>')
        meta = get_image_metadata(svg_path)
        assert meta["format"] == "SVG"
        assert meta["is_text"] is True


class TestCaptionImage:
    """Ollama multimodal captioning."""

    def test_caption_returns_string(self, tmp_path):
        from PIL import Image
        img_path = tmp_path / "arch.png"
        Image.new("RGB", (10, 10)).save(img_path)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "A diagram showing system architecture."}

        with patch("httpx.post", return_value=mock_response):
            result = caption_image(img_path, "http://localhost:11434", "qwen2.5vl:3b")

        assert "architecture" in result.lower() or "diagram" in result.lower()

    def test_caption_with_ollama_error_returns_empty(self, tmp_path):
        from PIL import Image
        img_path = tmp_path / "err.png"
        Image.new("RGB", (10, 10)).save(img_path)

        with patch("httpx.post", side_effect=Exception("connection refused")):
            result = caption_image(img_path, "http://localhost:11434", "qwen2.5vl:3b")

        assert result == ""

    def test_svg_returns_text_content_not_caption(self, tmp_path):
        svg_path = tmp_path / "flow.svg"
        svg_content = '<svg><text>Login Flow</text></svg>'
        svg_path.write_text(svg_content)

        result = caption_image(svg_path, "http://localhost:11434", "qwen2.5vl:3b")
        assert result == svg_content


class TestCaptionBatch:
    """Batch captioning."""

    def test_batch_returns_list_of_dicts(self, tmp_path):
        from PIL import Image
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            Image.new("RGB", (10, 10)).save(p)
            paths.append(p)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "An image."}

        with patch("httpx.post", return_value=mock_response):
            results = caption_images_batch(paths, "http://localhost:11434", "qwen2.5vl:3b")

        assert len(results) == 3
        assert all("caption" in r for r in results)
        assert all("path" in r for r in results)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_vision_captioner.py -v`
Expected: ModuleNotFoundError — `src.vision_captioner`

- [ ] **Step 5: Implement src/vision_captioner.py**

Create `src/vision_captioner.py`:

```python
"""Vision captioning via Ollama multimodal models (e.g. qwen2.5vl:3b).

SVG files are returned as-is (already text). Raster images (PNG, JPG, etc.)
are sent to Ollama's /api/generate with base64-encoded image data.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx

_CAPTION_PROMPT = (
    "Describe this image in detail. Focus on architecture, data flow, "
    "technical content, labels, and any text visible in the image. "
    "If this is a diagram, describe the components and their relationships."
)

_SVG_EXTENSIONS = frozenset({".svg"})
_RASTER_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_ALL_IMAGE_EXTENSIONS = _SVG_EXTENSIONS | _RASTER_EXTENSIONS


def get_image_metadata(path: Path) -> dict[str, Any] | None:
    """Extract metadata from an image file without Ollama.

    Returns dict with keys: width, height, format, file_size, is_text.
    Returns None if file doesn't exist or can't be read.
    """
    if not path.exists():
        return None

    ext = path.suffix.lower()

    if ext in _SVG_EXTENSIONS:
        return {
            "width": 0,
            "height": 0,
            "format": "SVG",
            "file_size": path.stat().st_size,
            "is_text": True,
        }

    try:
        from PIL import Image
        with Image.open(path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format or ext.lstrip(".").upper(),
                "file_size": path.stat().st_size,
                "is_text": False,
            }
    except Exception:
        return None


def caption_image(
    path: Path,
    ollama_url: str,
    model: str = "qwen2.5vl:3b",
    timeout: float = 120.0,
) -> str:
    """Generate a text caption for an image via Ollama multimodal API.

    SVG files: returns the raw XML text (already searchable).
    Raster files: sends base64 to Ollama /api/generate with image parameter.

    Returns empty string on error.
    """
    ext = path.suffix.lower()

    # SVGs are already text — return raw content
    if ext in _SVG_EXTENSIONS:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    # Raster images — send to Ollama
    if ext not in _RASTER_EXTENSIONS:
        return ""

    try:
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return ""

    try:
        resp = httpx.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": _CAPTION_PROMPT,
                "images": [image_data],
                "stream": False,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return ""
    except Exception:
        return ""


def caption_images_batch(
    paths: list[Path],
    ollama_url: str,
    model: str = "qwen2.5vl:3b",
) -> list[dict[str, Any]]:
    """Caption multiple images. Returns list of {path, caption, metadata}."""
    results: list[dict[str, Any]] = []
    for p in paths:
        caption = caption_image(p, ollama_url, model)
        meta = get_image_metadata(p)
        results.append({
            "path": str(p),
            "caption": caption,
            "metadata": meta,
        })
    return results
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_vision_captioner.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/vision_captioner.py tests/test_vision_captioner.py requirements.txt
git commit -m "feat(#31): add vision_captioner module with Qwen2.5-VL support

SVGs returned as text, raster images captioned via Ollama /api/generate
with base64 image data. Batch support included. Pillow added for metadata."
```

---

### Task 3: Integrate Vision into Memory Pipeline (#31)

**Files:**
- Modify: `codebook.yaml:70-100` (exclude_patterns section 6)
- Modify: `src/memory_ingest.py` (add image ingestion path)

- [ ] **Step 1: Update codebook.yaml — remove image extensions from excludes**

In `codebook.yaml`, replace the image lines in section 6 (lines 70-78):

```yaml
  # 6. Binärdateien & Medien (Bilder separat behandelt via Vision-Pipeline)
  # Images are now processed by vision_captioner — only exclude non-visual binaries
  - "*.woff"
  - "*.woff2"
  - "*.ttf"
  - "*.eot"
  - "*.otf"
  - "*.pdf"
  - "*.zip"
  - "*.tar"
  - "*.gz"
  - "*.rar"
  - "*.exe"
  - "*.dll"
  - "*.so"
  - "*.dylib"
  - "*.pyc"
  - "*.pyo"
  - "*.class"
  - "*.o"
  - "*.a"
  - "*.wasm"
  - "*.sqlite"
  - "*.sqlite3"
  - "*.db"
```

The removed lines are: `*.png`, `*.jpg`, `*.jpeg`, `*.gif`, `*.ico`, `*.svg`, `*.webp`, `*.bmp`.

- [ ] **Step 2: Add image ingestion to memory_ingest.py**

Add after the `structural_chunk` function (after line 312) in `src/memory_ingest.py`:

```python
# ---------------------------------------------------------------------------
# Image ingestion — Vision captioning via Ollama multimodal
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"})


def _is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in _IMAGE_EXTENSIONS


def ingest_image(
    source: Source,
    image_path: Path,
    conn: Any,
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    vision_model: str = "qwen2.5vl:3b",
) -> dict:
    """Ingest an image file: caption it via Ollama, store as text chunk.

    For SVGs, the raw XML text is used directly.
    For raster images, a caption is generated via vision_model.
    """
    from src.context import _embed_texts
    from src.vision_captioner import caption_image, get_image_metadata

    upsert_source(conn, source)
    log_ingestion_event(conn, source.source_id, "ingest_start", {"path": source.path, "type": "image"})

    # Get metadata
    meta = get_image_metadata(image_path) or {}

    # Generate caption
    caption = caption_image(image_path, ollama_url, vision_model)
    if not caption:
        caption = f"Image: {source.path}"
        if meta:
            caption += f" ({meta.get('width', '?')}x{meta.get('height', '?')} {meta.get('format', '?')})"

    # Create chunk
    text_hash = Chunk.compute_text_hash(caption)
    chunk = Chunk(
        chunk_id=Chunk.make_id(source.source_id, 0, "image_caption"),
        source_id=source.source_id,
        chunk_level="image_caption",
        ordinal=0,
        text=caption,
        text_hash=text_hash,
        dedup_key=text_hash,
        category_labels=["diagram"] if meta.get("is_text") else ["image"],
        created_at=_now_iso(),
    )

    # Dedup
    canonical, is_dup = resolve_dedup(conn, chunk)
    if is_dup:
        log_ingestion_event(conn, source.source_id, "ingest_done", {"chunks": 0, "deduped": 1})
        return {"source_id": source.source_id, "chunk_ids": [], "indexed": False, "deduped": 1, "superseded": 0}

    # Store
    insert_chunk(conn, chunk)

    # Embed + ChromaDB
    indexed = False
    try:
        emb = _embed_texts([caption[:500]], ollama_url)[0]
        if chroma_collection is not None and emb is not None:
            chroma_collection.upsert(
                ids=[chunk.chunk_id],
                documents=[caption[:500]],
                embeddings=[emb],
                metadatas=[{
                    "source_id": chunk.source_id,
                    "chunk_level": "image_caption",
                    "category_labels": ",".join(chunk.category_labels),
                    "is_active": 1,
                }],
            )
            indexed = True
    except Exception:
        pass

    kv_put(chunk.chunk_id, chunk.to_dict())

    log_ingestion_event(conn, source.source_id, "ingest_done", {"chunks": 1, "deduped": 0, "type": "image"})
    return {"source_id": source.source_id, "chunk_ids": [chunk.chunk_id], "indexed": indexed, "deduped": 0, "superseded": 0}
```

- [ ] **Step 3: Run all memory_ingest tests for regressions**

Run: `.venv/bin/python -m pytest tests/test_memory_ingest.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add codebook.yaml src/memory_ingest.py
git commit -m "feat(#31): integrate vision captioning into memory pipeline

Remove image extensions from codebook excludes. Add ingest_image() to
memory_ingest.py: SVGs indexed as text, raster images captioned via
Qwen2.5-VL and stored as text chunks with embedding."
```

---

### Task 4: JWT Authentication + Workspace Isolation (#38)

**Files:**
- Modify: `requirements.txt`
- Create: `tests/test_jwt_auth.py`
- Modify: `src/mcp_server.py:63-93`
- Modify: `src/memory_store.py:64-129`
- Modify: `src/memory_ingest.py` (chroma collection per workspace)
- Create: `tools/generate_mcp_token.py`

- [ ] **Step 1: Add PyJWT to requirements.txt**

```
gitingest
httpx
python-dotenv
pyyaml
chromadb
mcp
Pillow
PyJWT
```

- [ ] **Step 2: Install dependency**

Run: `.venv/bin/pip install PyJWT`

- [ ] **Step 3: Write failing tests for JWT middleware**

Create `tests/test_jwt_auth.py`:

```python
"""Tests for JWT authentication middleware in mcp_server.py."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest


SECRET = "test-secret-key-for-jwt"
ALGORITHM = "HS256"


def _make_token(workspace_id: str = "default", scope: str = "repo", exp_offset: int = 3600) -> str:
    payload = {
        "workspace_id": workspace_id,
        "scope": scope,
        "exp": int(time.time()) + exp_offset,
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


class TestJWTMiddleware:
    """ASGI middleware JWT validation."""

    @pytest.fixture
    def middleware(self):
        """Create middleware instance with auth enabled."""
        # Import after patching env vars
        import importlib
        import os
        os.environ["MCP_AUTH_ENABLED"] = "true"
        os.environ["MCP_AUTH_SECRET"] = SECRET

        from src.mcp_server import _JWTAuthMiddleware
        inner_app = AsyncMock()
        return _JWTAuthMiddleware(inner_app), inner_app

    @pytest.mark.asyncio
    async def test_valid_token_passes_through(self, middleware):
        mw, inner = middleware
        token = _make_token(workspace_id="ws_123")
        scope = {
            "type": "http",
            "headers": [(b"x-auth-token", token.encode())],
        }
        receive = AsyncMock()
        send = AsyncMock()

        await mw(scope, receive, send)

        inner.assert_called_once()
        # workspace_id should be injected into scope
        assert scope.get("workspace_id") == "ws_123"

    @pytest.mark.asyncio
    async def test_bearer_header_also_accepted(self, middleware):
        mw, inner = middleware
        token = _make_token(workspace_id="ws_456")
        scope = {
            "type": "http",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        }
        receive = AsyncMock()
        send = AsyncMock()

        await mw(scope, receive, send)

        inner.assert_called_once()
        assert scope.get("workspace_id") == "ws_456"

    @pytest.mark.asyncio
    async def test_missing_token_returns_401(self, middleware):
        mw, inner = middleware
        scope = {"type": "http", "headers": []}
        receive = AsyncMock()
        send = AsyncMock()

        await mw(scope, receive, send)

        inner.assert_not_called()
        # Check 401 was sent
        calls = send.call_args_list
        assert any(c.args[0].get("status") == 401 for c in calls)

    @pytest.mark.asyncio
    async def test_expired_token_returns_401(self, middleware):
        mw, inner = middleware
        token = _make_token(exp_offset=-3600)  # expired 1h ago
        scope = {
            "type": "http",
            "headers": [(b"x-auth-token", token.encode())],
        }
        receive = AsyncMock()
        send = AsyncMock()

        await mw(scope, receive, send)

        inner.assert_not_called()
        calls = send.call_args_list
        assert any(c.args[0].get("status") == 401 for c in calls)

    @pytest.mark.asyncio
    async def test_invalid_signature_returns_401(self, middleware):
        mw, inner = middleware
        token = jwt.encode({"workspace_id": "ws"}, "wrong-secret", algorithm=ALGORITHM)
        scope = {
            "type": "http",
            "headers": [(b"x-auth-token", token.encode())],
        }
        receive = AsyncMock()
        send = AsyncMock()

        await mw(scope, receive, send)

        inner.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_http_scope_passes_through(self, middleware):
        mw, inner = middleware
        scope = {"type": "websocket", "headers": []}
        receive = AsyncMock()
        send = AsyncMock()

        await mw(scope, receive, send)

        inner.assert_called_once()


class TestJWTMiddlewareDisabled:
    """When MCP_AUTH_ENABLED=false, middleware is pass-through."""

    @pytest.mark.asyncio
    async def test_disabled_auth_passes_all_requests(self):
        import os
        os.environ["MCP_AUTH_ENABLED"] = "false"
        os.environ.pop("MCP_AUTH_SECRET", None)

        from src.mcp_server import _JWTAuthMiddleware
        inner = AsyncMock()
        mw = _JWTAuthMiddleware(inner)

        scope = {"type": "http", "headers": []}
        await mw(scope, AsyncMock(), AsyncMock())

        inner.assert_called_once()
        assert scope.get("workspace_id") == "default"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_jwt_auth.py -v`
Expected: ImportError — `_JWTAuthMiddleware` not found

- [ ] **Step 5: Implement JWT middleware in src/mcp_server.py**

Replace the auth-related code in `src/mcp_server.py` (lines 60-93) with:

```python
# ---------------------------------------------------------------------------
# HTTP transport + JWT authentication configuration
# ---------------------------------------------------------------------------

_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")       # stdio | http
_AUTH_ENABLED = os.getenv("MCP_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
_AUTH_SECRET = os.getenv("MCP_AUTH_SECRET", "")
_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "")           # legacy static token (backward compat)
_HTTP_PORT = int(os.getenv("MCP_HTTP_PORT", "8000"))
_HTTP_HOST = os.getenv("MCP_HTTP_HOST", "0.0.0.0")


class _JWTAuthMiddleware:
    """ASGI middleware: validates JWT from X-Auth-Token or Authorization: Bearer header.

    When MCP_AUTH_ENABLED=false: pass-through, workspace_id="default".
    When MCP_AUTH_ENABLED=true: validates JWT signature + expiry, extracts workspace_id.
    Falls back to legacy static X-Auth-Token comparison if MCP_AUTH_SECRET is empty.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if not _AUTH_ENABLED:
            scope["workspace_id"] = "default"
            await self._app(scope, receive, send)
            return

        # Extract token from headers
        headers = dict(scope.get("headers", []))
        token = ""

        # Try X-Auth-Token first
        raw = headers.get(b"x-auth-token", b"")
        if raw:
            token = raw.decode()

        # Fallback: Authorization: Bearer
        if not token:
            auth_header = headers.get(b"authorization", b"").decode()
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]

        if not token:
            await self._send_401(send, "Missing authentication token")
            return

        # JWT validation (if secret configured)
        if _AUTH_SECRET:
            try:
                import jwt as _jwt
                payload = _jwt.decode(token, _AUTH_SECRET, algorithms=["HS256"])
                scope["workspace_id"] = payload.get("workspace_id", "default")
            except _jwt.ExpiredSignatureError:
                await self._send_401(send, "Token expired")
                return
            except _jwt.InvalidTokenError:
                await self._send_401(send, "Invalid token")
                return
        elif _AUTH_TOKEN:
            # Legacy static token comparison
            if token != _AUTH_TOKEN:
                await self._send_401(send, "Invalid token")
                return
            scope["workspace_id"] = "default"
        else:
            await self._send_401(send, "No auth secret configured")
            return

        await self._app(scope, receive, send)

    @staticmethod
    async def _send_401(send: Any, message: str = "Unauthorized") -> None:
        body = message.encode()
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                [b"content-type", b"text/plain; charset=utf-8"],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})
```

Update the entry point (lines 453-466) to use the new middleware:

```python
if __name__ == "__main__":
    if _TRANSPORT == "http":
        import uvicorn

        _asgi_app = mcp.streamable_http_app()
        _wrapped = _JWTAuthMiddleware(_asgi_app)
        auth_mode = "JWT" if _AUTH_SECRET else ("static" if _AUTH_TOKEN else "disabled")
        print(
            f"[mcp-memory] HTTP mode on {_HTTP_HOST}:{_HTTP_PORT}"
            f" | auth={auth_mode}"
        )
        uvicorn.run(_wrapped, host=_HTTP_HOST, port=_HTTP_PORT)
    else:
        mcp.run()  # stdio transport (default for Claude Code)
```

- [ ] **Step 6: Run JWT tests**

Run: `.venv/bin/python -m pytest tests/test_jwt_auth.py -v`
Expected: All PASS

- [ ] **Step 7: Run existing MCP server tests for regressions**

Run: `.venv/bin/python -m pytest tests/test_mcp_server_http.py -v`
Expected: All PASS (legacy static token tests should still work)

- [ ] **Step 8: Commit JWT middleware**

```bash
git add src/mcp_server.py tests/test_jwt_auth.py requirements.txt
git commit -m "feat(#38): replace static X-Auth-Token with JWT auth middleware

JWT validation via PyJWT (HS256). Extracts workspace_id from token claims.
Backward-compatible: falls back to static MCP_AUTH_TOKEN if no JWT secret.
MCP_AUTH_ENABLED=false (default) keeps unauthenticated behavior."
```

- [ ] **Step 9: Add workspace_id to SQLite schema in memory_store.py**

In `src/memory_store.py`, modify `_init_schema()` to add workspace_id columns. After the existing `CREATE TABLE` statements (before the final `conn.commit()`), add migration:

```python
def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            source_id       TEXT PRIMARY KEY,
            source_type     TEXT NOT NULL DEFAULT 'repo_file',
            repo            TEXT NOT NULL DEFAULT '',
            path            TEXT NOT NULL DEFAULT '',
            branch          TEXT NOT NULL DEFAULT 'main',
            "commit"        TEXT NOT NULL DEFAULT '',
            content_hash    TEXT NOT NULL DEFAULT '',
            captured_at     TEXT NOT NULL,
            workspace_id    TEXT NOT NULL DEFAULT 'default'
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id            TEXT PRIMARY KEY,
            source_id           TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
            parent_chunk_id     TEXT,
            chunk_level         TEXT NOT NULL DEFAULT 'file',
            ordinal             INTEGER NOT NULL DEFAULT 0,
            start_offset        INTEGER NOT NULL DEFAULT 0,
            end_offset          INTEGER NOT NULL DEFAULT 0,
            text                TEXT NOT NULL DEFAULT '',
            text_hash           TEXT NOT NULL DEFAULT '',
            summary             TEXT NOT NULL DEFAULT '',
            category_labels     TEXT NOT NULL DEFAULT '',
            category_version    TEXT NOT NULL DEFAULT 'mayring-inductive-v1',
            embedding_model     TEXT NOT NULL DEFAULT 'nomic-embed-text',
            embedding_id        TEXT NOT NULL DEFAULT '',
            quality_score       REAL NOT NULL DEFAULT 0.0,
            dedup_key           TEXT NOT NULL DEFAULT '',
            created_at          TEXT NOT NULL,
            superseded_by       TEXT,
            is_active           INTEGER NOT NULL DEFAULT 1,
            workspace_id        TEXT NOT NULL DEFAULT 'default'
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_source_id
            ON chunks(source_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_text_hash
            ON chunks(text_hash);
        CREATE INDEX IF NOT EXISTS idx_chunks_dedup_key
            ON chunks(dedup_key);
        CREATE INDEX IF NOT EXISTS idx_chunks_is_active
            ON chunks(is_active);
        CREATE INDEX IF NOT EXISTS idx_chunks_workspace_id
            ON chunks(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_sources_workspace_id
            ON sources(workspace_id);

        CREATE TABLE IF NOT EXISTS chunk_feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id        TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            signal          TEXT NOT NULL,
            metadata        TEXT NOT NULL DEFAULT '{}',
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_chunk_id
            ON chunk_feedback(chunk_id);

        CREATE TABLE IF NOT EXISTS ingestion_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id       TEXT NOT NULL DEFAULT '',
            event_type      TEXT NOT NULL,
            payload         TEXT NOT NULL DEFAULT '{}',
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_ingestion_log_source_id
            ON ingestion_log(source_id);
    """)
    # Migration: add workspace_id to existing tables if missing
    _migrate_workspace_id(conn)
    conn.commit()


def _migrate_workspace_id(conn: sqlite3.Connection) -> None:
    """Add workspace_id column to sources/chunks if not present (migration)."""
    for table in ("sources", "chunks"):
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if "workspace_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'default'")
```

- [ ] **Step 10: Run memory_store tests**

Run: `.venv/bin/python -m pytest tests/test_memory_store.py -v`
Expected: All PASS

- [ ] **Step 11: Commit workspace schema**

```bash
git add src/memory_store.py
git commit -m "feat(#38): add workspace_id column to sources and chunks tables

Auto-migration adds column to existing DBs. Indexed for fast filtering.
Default value 'default' preserves backward compatibility."
```

- [ ] **Step 12: Create JWT token generator tool**

Create `tools/generate_mcp_token.py`:

```python
#!/usr/bin/env python3
"""Generate JWT tokens for MCP Memory Server authentication.

Usage:
    python tools/generate_mcp_token.py \
        --workspace myapp \
        --scope "repo:Nileneb/app.linn.games" \
        --secret $MCP_AUTH_SECRET \
        --expiry 30d
"""

from __future__ import annotations

import argparse
import re
import sys
import time

try:
    import jwt
except ImportError:
    print("Error: PyJWT not installed. Run: pip install PyJWT", file=sys.stderr)
    sys.exit(1)


def _parse_expiry(s: str) -> int:
    """Parse expiry string like '30d', '24h', '60m', '3600' to seconds."""
    m = re.match(r"^(\d+)([dhms]?)$", s.strip())
    if not m:
        raise ValueError(f"Invalid expiry format: {s!r}. Use: 30d, 24h, 60m, or seconds.")
    value = int(m.group(1))
    unit = m.group(2) or "s"
    multipliers = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return value * multipliers[unit]


def generate_token(
    workspace_id: str,
    scope: str,
    secret: str,
    expiry_seconds: int,
) -> str:
    payload = {
        "workspace_id": workspace_id,
        "scope": scope,
        "iat": int(time.time()),
        "exp": int(time.time()) + expiry_seconds,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate MCP JWT tokens")
    p.add_argument("--workspace", required=True, help="Workspace ID (e.g. 'myapp', 'default')")
    p.add_argument("--scope", default="repo", help="Scope string (e.g. 'repo:owner/name')")
    p.add_argument("--secret", required=True, help="HMAC secret (same as MCP_AUTH_SECRET)")
    p.add_argument("--expiry", default="30d", help="Token expiry (e.g. 30d, 24h, 3600)")
    args = p.parse_args()

    seconds = _parse_expiry(args.expiry)
    token = generate_token(args.workspace, args.scope, args.secret, seconds)
    print(token)


if __name__ == "__main__":
    main()
```

- [ ] **Step 13: Commit token generator**

```bash
git add tools/generate_mcp_token.py
git commit -m "feat(#38): add JWT token generator CLI tool

Usage: python tools/generate_mcp_token.py --workspace X --secret KEY --expiry 30d"
```

---

### Task 5: Docker Compose Full Pipeline (#37)

**Files:**
- Create: `docker-compose.full.yml`
- Modify: `Dockerfile`

- [ ] **Step 1: Extend Dockerfile for full pipeline**

Modify `Dockerfile` to include all files needed for checker.py:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn gradio

# Copy application code
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY codebook.yaml codebook_sozialforschung.yaml ./
COPY checker.py turbulence_run.py run.sh ./
COPY .env.example ./

# Create cache and reports directories
RUN mkdir -p /app/cache /app/reports

# Transport mode: stdio (default) or http
ENV MCP_TRANSPORT=stdio
ENV PYTHONUNBUFFERED=1

# Default command: stdio transport for Claude Code
CMD ["python", "-m", "src.mcp_server"]
```

- [ ] **Step 2: Create docker-compose.full.yml**

```yaml
# MayringCoder Full Pipeline — Ollama + Analyzer + Web-UI + MCP Memory
#
# Quick start:
#   docker compose -f docker-compose.full.yml up -d
#
# Run analysis:
#   docker compose -f docker-compose.full.yml run analyzer \
#     python checker.py --repo https://github.com/user/repo --no-limit
#
# GPU support (NVIDIA):
#   Uncomment the deploy section under ollama service

services:
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    # Uncomment for GPU support:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

  model-pull:
    image: ollama/ollama:latest
    depends_on:
      ollama:
        condition: service_healthy
    entrypoint: >
      sh -c "ollama pull $${OLLAMA_MODEL:-llama3.1:8b} && echo 'Model ready'"
    environment:
      OLLAMA_HOST: http://ollama:11434
    volumes:
      - ollama_models:/root/.ollama
    restart: "no"

  analyzer:
    build: .
    depends_on:
      ollama:
        condition: service_healthy
    volumes:
      - ./cache:/app/cache
      - ./reports:/app/reports
    env_file: .env
    environment:
      OLLAMA_URL: http://ollama:11434
    command: ["sleep", "infinity"]
    restart: unless-stopped

  web-ui:
    build: .
    depends_on:
      ollama:
        condition: service_healthy
    ports:
      - "${GRADIO_PORT:-7860}:7860"
    volumes:
      - ./cache:/app/cache
      - ./reports:/app/reports
    env_file: .env
    environment:
      OLLAMA_URL: http://ollama:11434
    command: ["python", "-m", "src.web_ui"]
    restart: unless-stopped

  mcp-memory:
    build: .
    ports:
      - "${MCP_HTTP_PORT:-8000}:8000"
    volumes:
      - ./cache:/app/cache
    env_file: .env
    environment:
      MCP_TRANSPORT: http
      MCP_HTTP_HOST: "0.0.0.0"
      MCP_HTTP_PORT: "8000"
      OLLAMA_URL: http://ollama:11434
    depends_on:
      ollama:
        condition: service_healthy
    restart: unless-stopped

volumes:
  ollama_models:
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile docker-compose.full.yml
git commit -m "feat(#37): add full-pipeline docker-compose with Ollama, analyzer, web-ui, MCP

docker-compose.full.yml: 4 services + auto model pull.
Dockerfile extended with checker.py, prompts, codebooks, gradio."
```

---

### Task 6: Model Recommendations in README (#35)

**Files:**
- Modify: `README.md:360-371`

- [ ] **Step 1: Replace the model table in README.md**

Replace the "Empfohlene Modelle" section (lines 360-371) with:

```markdown
## Empfohlene Modelle

| Modell | VRAM | Code-Review | Sozialforschung | Turbulenz | Vision | Hinweis |
|---|---|---|---|---|---|---|
| `llama3.1:8b` | ~5 GB | gut | gut | gut | — | Solide Allround-Wahl |
| `qwen2.5-coder:7b` | ~5 GB | sehr gut | — | gut | — | Spezialisiert auf Code |
| `qwen3.5:9b` | ~7 GB | exzellent | exzellent | gut | — | Bestes Allround-Modell |
| `deepseek-coder-v2:16b` | ~10 GB | exzellent | — | sehr gut | — | Für kritische Reviews, langsamer |
| `mistral:7b-instruct` | ~5 GB | gut | gut | empfohlen | — | Standard für Turbulenz (`TURB_MODEL`) |
| `qwen2.5vl:3b` | ~3 GB | — | — | — | empfohlen | Multimodal: Bilder → Text-Captions |

**VRAM-Hinweise:**
- Modelle teilen sich den VRAM mit Embedding (`nomic-embed-text`, ~270 MB)
- Bei gleichzeitigem Vision-Captioning: +3 GB für `qwen2.5vl:3b`
- 8 GB VRAM reicht für die meisten 7B-Modelle + Embedding
- 16 GB empfohlen für 16B-Modelle oder parallele Nutzung

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
ollama pull qwen3.5:9b
ollama pull qwen2.5vl:3b          # für Bild-Captioning
ollama pull nomic-embed-text       # für RAG/Embedding
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(#35): add model recommendations with VRAM requirements per use case"
```

---

### Task 7: Memory Roadmap Document (#34)

**Files:**
- Create: `docs/memory_roadmap.md`

- [ ] **Step 1: Create docs/memory_roadmap.md**

```markdown
# Memory-System Roadmap

Stand: 2026-04-10 — Referenz: `ARCHITECTURE.md` (Target-Architecture)

## Phase 1 — Grundfunktionen (abgeschlossen)

- [x] MCP Tool-Verträge: put, get, search_memory, invalidate, list_by_source, explain, reindex, feedback
- [x] SQLite Memory-DB (`cache/memory.db`) mit sources, chunks, chunk_feedback, ingestion_log
- [x] ChromaDB Embedding-Retrieval (`cache/memory_chroma/`)
- [x] Structural Chunking: Python (AST), JS/TS (Regex), Markdown (Headings), YAML/JSON (Top-Level-Keys)
- [x] Multi-View Indexing für GitHub Issues (fact/impl/decision/entities/full)
- [x] 4-Stufen Hybrid-Search: Scope Filter → Symbolic → Vector → Rerank
- [x] Exact Dedup via text_hash
- [x] KV-Cache (in-process)
- [x] Conversation-Summary Ingestion
- [x] HTTP/SSE Transport mit X-Auth-Token
- [x] Docker-Compose Profile (stdio/http)

## Phase 2 — Semantische Anreicherung (offen)

- [ ] Qwen-basierte induktive Kategorisierung (automatisch, nicht nur per `--memory-categorize`)
- [ ] Cross-Encoder Re-Ranker als 5. Stufe nach dem Hybrid-Reranking
- [ ] Codebook-Auto-Erkennung verfeinern: mehr source_types, bessere Heuristiken
- [ ] Near-Dedup: normalisierter Text-Hash für leicht abweichende Duplikate
- [ ] Chunk-Summaries via LLM (für `compress_for_prompt` bei langen Chunks)
- [ ] Vision-RAG: Bild-Captioning via Qwen2.5-VL → Text-Chunks (Issue #31)

## Phase 3 — Training & Feedback-Loop (offen)

- [ ] Feedback-gewichtetes Re-Ranking: positive/negative Signale beeinflussen score_final
- [ ] Training-Data-Export: Chunk-Paare (Query → relevanter Chunk) als Fine-Tuning-Datensatz
- [ ] Active Learning: Chunks mit niedrigem Quality-Score zur manuellen Review vorschlagen
- [ ] Embedding-Modell-Vergleich: Benchmark verschiedener Modelle auf realen Queries

## Phase 4 — Governance & Skalierung (offen)

- [ ] Retention Policies: automatisches Invalidieren alter Chunks (TTL-basiert)
- [ ] Workspace-Isolation: JWT-Auth + separate Chroma-Collections pro Workspace (Issue #38)
- [ ] Audit-Log: Wer hat wann welche Memory-Operation ausgeführt
- [ ] Rate-Limiting pro Workspace (429 bei Überschreitung)
- [ ] Schema-Versioning: explizite Migrationsskripte statt ALTER TABLE in init

## Architekturentscheidungen

| Entscheidung | Wahl | Begründung |
|---|---|---|
| Embedding-Modell | nomic-embed-text | Gute Balance aus Qualität und Geschwindigkeit, lokal via Ollama |
| Re-Ranker | Weighted Linear (0.45v+0.25s+0.15r+0.15a) | Einfach, erklärbar, keine Extra-Dependency |
| Dedup-Strategie | Exact (sha256) | Near-Dedup erst wenn Exact nicht reicht |
| Chroma-Isolation | Separate Collections pro Workspace | Kein Leaking möglich |
| Auth-Mechanismus | JWT (HS256) | Standard, leichtgewichtig, Workspace-ID als Claim |
```

- [ ] **Step 2: Commit**

```bash
git add docs/memory_roadmap.md
git commit -m "docs(#34): add memory system roadmap for phases 2-4"
```

---

### Task 8: E2E Tests for Web-UI (#33)

**Files:**
- Modify: `tests/test_web_ui.py`

- [ ] **Step 1: Add E2E test class to test_web_ui.py**

Append to `tests/test_web_ui.py`:

```python
# ---------------------------------------------------------------------------
# Test H: E2E flows — full user workflows
# ---------------------------------------------------------------------------

class TestE2EAnalysisFlow:
    """End-to-end: URL → analyze → report visible."""

    def test_ingest_then_search_returns_results(self):
        """Ingest text, then search for it — full roundtrip."""
        import src.web_ui as web_ui

        fake_conn = MagicMock(spec=sqlite3.Connection)
        ingest_result = {
            "source_id": "repo:test:e2e.py",
            "chunk_ids": ["chk_e2e001"],
            "indexed": True,
            "deduped": 0,
            "superseded": 0,
        }

        from src.memory_schema import RetrievalRecord
        search_result = RetrievalRecord(
            chunk_id="chk_e2e001",
            score_final=0.85,
            score_symbolic=0.6,
            source_id="repo:test:e2e.py",
            text="def authenticate(user): pass",
            category_labels=["auth"],
            reasons=["token_overlap", "embedding_similarity"],
        )

        with (
            patch.object(web_ui, "_MEMORY_READY", True),
            patch("src.web_ui._get_conn", return_value=fake_conn),
            patch("src.web_ui._get_chroma", return_value=None),
            patch("src.web_ui.ingest", return_value=ingest_result),
            patch("src.web_ui.Source") as MockSource,
            patch("src.web_ui.hashlib") as mock_hashlib,
            patch("src.web_ui.search", return_value=[search_result]),
        ):
            mock_hash = MagicMock()
            mock_hash.hexdigest.return_value = "b" * 64
            mock_hashlib.sha256.return_value = mock_hash
            MockSource.return_value = MagicMock()
            MockSource.make_id.return_value = "repo:test:e2e.py"

            # Step 1: Ingest
            raw = web_ui._do_ingest(
                text_input="def authenticate(user): pass",
                file_upload=None,
                source_path="e2e.py",
                repo="test",
                categorize=False,
                mode="hybrid",
                codebook="auto",
                model="",
                ollama_available=False,
            )
            result = json.loads(raw)
            assert result["source_id"] == "repo:test:e2e.py"

            # Step 2: Search
            status, rows = web_ui._do_search("authenticate", 5, False)
            assert len(rows) == 1
            assert "auth" in str(rows[0])

    def test_feedback_after_search(self):
        """Search → get chunk_id → submit feedback."""
        import src.web_ui as web_ui

        fake_conn = MagicMock(spec=sqlite3.Connection)

        with (
            patch.object(web_ui, "_MEMORY_READY", True),
            patch("src.web_ui._get_conn", return_value=fake_conn),
            patch("src.web_ui.add_feedback") as mock_fb,
        ):
            result = web_ui._do_feedback("chk_e2e001", "positive", "relevant")

        mock_fb.assert_called_once_with(fake_conn, "chk_e2e001", "positive", {"label": "relevant"})
        assert "gespeichert" in result.lower()


class TestE2EConversationFlow:
    """End-to-end: conversation summary ingestion + search."""

    def test_conversation_ingest_then_search(self):
        """Ingest a conversation summary, then find it via search."""
        import src.web_ui as web_ui

        fake_conn = MagicMock(spec=sqlite3.Connection)
        conv_result = {
            "source_id": "conv:sess-e2e",
            "chunk_ids": ["chk_conv_001"],
            "indexed": True,
            "deduped": 0,
            "superseded": 0,
        }

        from src.memory_schema import RetrievalRecord
        search_hit = RetrievalRecord(
            chunk_id="chk_conv_001",
            score_final=0.72,
            source_id="conv:sess-e2e",
            text="Wir haben HTTP-Transport implementiert.",
            category_labels=["Zusammenfassung"],
            reasons=["token_overlap"],
        )

        with (
            patch.object(web_ui, "_MEMORY_READY", True),
            patch("src.web_ui._get_conn", return_value=fake_conn),
            patch("src.web_ui._get_chroma", return_value=None),
            patch("src.web_ui.ingest_conversation_summary", return_value=conv_result),
            patch("src.web_ui.search", return_value=[search_hit]),
        ):
            # Step 1: Ingest conversation
            raw = web_ui._do_ingest_conversation(
                summary_text="## Summary\nWir haben HTTP-Transport implementiert.",
                session_id="sess-e2e",
                run_id="run-001",
                model="",
                ollama_available=False,
            )
            assert "conv:sess-e2e" in raw

            # Step 2: Search for it
            status, rows = web_ui._do_search("HTTP Transport", 5, False)
            assert len(rows) == 1


class TestE2EErrorCases:
    """E2E error handling."""

    def test_ingest_with_no_content_and_no_file(self):
        """Both text and file empty → error."""
        import src.web_ui as web_ui

        with patch.object(web_ui, "_MEMORY_READY", True):
            raw = web_ui._do_ingest("", None, "", "", False, "hybrid", "auto", "", False)

        result = json.loads(raw)
        assert "error" in result

    def test_search_with_memory_not_loaded(self):
        """Memory not initialized → helpful error message."""
        import src.web_ui as web_ui

        with patch.object(web_ui, "_MEMORY_READY", False):
            status, rows = web_ui._do_search("test", 5, True)

        assert rows == []
        assert "memory" in status.lower() or "nicht" in status.lower()

    def test_feedback_on_nonexistent_chunk(self):
        """Feedback on empty chunk_id → error."""
        import src.web_ui as web_ui

        with patch.object(web_ui, "_MEMORY_READY", True):
            result = web_ui._do_feedback("", "positive", "")

        assert "chunk" in result.lower() or "id" in result.lower()
```

- [ ] **Step 2: Run all web_ui tests**

Run: `.venv/bin/python -m pytest tests/test_web_ui.py -v`
Expected: All PASS (existing + new)

- [ ] **Step 3: Commit**

```bash
git add tests/test_web_ui.py
git commit -m "test(#33): add E2E tests for web-ui flows (ingest→search→feedback)"
```

---

### Task 9: Update .env.example + close issues

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Update .env.example with JWT vars**

Add to the MCP section in `.env.example`:

```
# MCP Auth (JWT) — only needed for HTTP transport
# MCP_AUTH_ENABLED=false
# MCP_AUTH_SECRET=your-hmac-secret-here
# Generate tokens: python tools/generate_mcp_token.py --workspace X --secret $MCP_AUTH_SECRET

# Docker Full Pipeline
# GRADIO_PORT=7860
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: update .env.example with JWT auth and Docker pipeline vars"
```

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: All PASS

- [ ] **Step 4: Close issues on GitHub**

```bash
gh issue close 31 --repo Nileneb/MayringCoder --comment "Stufe 1+2 implementiert: SVG als Text, Raster via Qwen2.5-VL Captioning. Stufe 3 (CLIP) bewusst aufgeschoben."
gh issue close 32 --repo Nileneb/MayringCoder --comment "smart_split() mit AST-basierter Priorisierung implementiert."
gh issue close 33 --repo Nileneb/MayringCoder --comment "E2E-Tests für Ingest→Search→Feedback Flows hinzugefügt."
gh issue close 34 --repo Nileneb/MayringCoder --comment "Roadmap-Dokument erstellt: docs/memory_roadmap.md"
gh issue close 35 --repo Nileneb/MayringCoder --comment "Modellempfehlungen mit VRAM-Anforderungen in README ergänzt."
gh issue close 37 --repo Nileneb/MayringCoder --comment "docker-compose.full.yml mit Ollama + Analyzer + Web-UI + MCP Memory."
gh issue close 38 --repo Nileneb/MayringCoder --comment "JWT-Auth (HS256) + workspace_id Schema. Token-Generator: tools/generate_mcp_token.py"
```
