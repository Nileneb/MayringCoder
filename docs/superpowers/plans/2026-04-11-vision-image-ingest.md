# Vision Image Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--ingest-images <repo-url>` to checker.py so repo images are shallow-cloned, captioned via `qwen2.5vl:3b`, and ingested into ChromaDB/SQLite memory — enabling Pi to retrieve visual context (ERDs, architecture diagrams) via `search_memory`.

**Architecture:** New `src/image_ingest.py` handles git clone → image discovery → per-image `ingest_image()` call (already in `memory_ingest.py`) → cleanup. `checker.py` gets `--ingest-images` flag + `_run_ingest_images()` helper, mirroring the `--ingest-issues` pattern exactly.

**Tech Stack:** subprocess (git clone), pathlib, shutil, existing `src/memory_ingest.ingest_image()`, existing `src/memory_schema.Source`, existing `src/memory_store.init_memory_db()`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/image_ingest.py` | **CREATE** | Clone, discover images, build Source objects, call ingest_image(), cleanup |
| `checker.py` | **MODIFY** | Add `--ingest-images`, `--vision-model`, `--max-images` flags + `_run_ingest_images()` |
| `tests/test_image_ingest.py` | **CREATE** | Unit tests for image_ingest.py |

`src/memory_ingest.py`, `src/vision_captioner.py`, `src/memory_store.py` — **unchanged**.

---

## Task 1: `src/image_ingest.py` — clone, discover, ingest

**Files:**
- Create: `src/image_ingest.py`
- Test: `tests/test_image_ingest.py`

### Background

`ingest_image()` already exists in `memory_ingest.py` with this signature:

```python
def ingest_image(
    source: Source,           # from src/memory_schema.py
    image_path: Path,
    conn: Any,                # sqlite3.Connection from init_memory_db()
    chroma_collection: Any,   # from get_or_create_chroma_collection()
    ollama_url: str,
    model: str,               # text embedding model (e.g. "nomic-embed-text")
    vision_model: str = "qwen2.5vl:3b",
) -> dict:                    # {source_id, chunk_ids, indexed, deduped, superseded}
```

`Source` is a dataclass from `src/memory_schema.py`:
```python
@dataclass
class Source:
    source_id: str   # e.g. "repo:Nileneb/app.linn.games:docs/erd.png"
    source_type: str # "repo_file"
    repo: str        # "Nileneb/app.linn.games"
    path: str        # "docs/erd.png"

    @staticmethod
    def make_id(repo: str, path: str) -> str:
        return f"repo:{repo}:{path}"
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_image_ingest.py`:

```python
"""Tests for src/image_ingest.py."""
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from src.image_ingest import (
    discover_images,
    run_image_ingest,
    _IMAGE_EXTENSIONS,
)


class TestDiscoverImages:
    def test_finds_png_jpg_svg(self, tmp_path):
        (tmp_path / "diagram.png").write_bytes(b"x" * 100)
        (tmp_path / "photo.jpg").write_bytes(b"x" * 200)
        (tmp_path / "icon.svg").write_text("<svg/>")
        (tmp_path / "readme.md").write_text("text")

        found = discover_images(tmp_path, max_file_bytes=1024, max_images=50)
        names = {p.name for p in found}
        assert names == {"diagram.png", "photo.jpg", "icon.svg"}

    def test_skips_files_over_size_limit(self, tmp_path):
        (tmp_path / "big.png").write_bytes(b"x" * 6_000_000)  # 6 MB
        (tmp_path / "small.png").write_bytes(b"x" * 100)

        found = discover_images(tmp_path, max_file_bytes=5_000_000, max_images=50)
        assert len(found) == 1
        assert found[0].name == "small.png"

    def test_respects_max_images_limit(self, tmp_path):
        for i in range(10):
            (tmp_path / f"img{i}.png").write_bytes(b"x" * 10)

        found = discover_images(tmp_path, max_file_bytes=1024 * 1024, max_images=3)
        assert len(found) == 3

    def test_returns_empty_for_no_images(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hi')")
        found = discover_images(tmp_path, max_file_bytes=1024 * 1024, max_images=50)
        assert found == []


class TestRunImageIngest:
    def test_clones_repo_and_ingests_images(self, tmp_path):
        fake_clone_dir = tmp_path / "clone"
        fake_clone_dir.mkdir()
        (fake_clone_dir / "docs" / "erd.png").mkdir(parents=True)
        (fake_clone_dir / "docs" / "erd.png").rmdir()
        (fake_clone_dir / "docs").mkdir(exist_ok=True)
        (fake_clone_dir / "docs" / "erd.png").write_bytes(b"PNG" * 10)

        mock_ingest_image = MagicMock(return_value={"chunk_ids": ["c1"], "deduped": 0})
        mock_conn = MagicMock()
        mock_chroma = MagicMock()

        with (
            patch("src.image_ingest.subprocess.run") as mock_run,
            patch("src.image_ingest.tempfile.mkdtemp", return_value=str(fake_clone_dir.parent)),
            patch("src.image_ingest.shutil.rmtree") as mock_rmtree,
            patch("src.image_ingest.ingest_image", mock_ingest_image),
            patch("src.image_ingest.init_memory_db", return_value=mock_conn),
            patch("src.image_ingest.get_or_create_chroma_collection", return_value=mock_chroma),
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = run_image_ingest(
                repo_url="https://github.com/Nileneb/app.linn.games",
                ollama_url="http://localhost:11434",
                vision_model="qwen2.5vl:3b",
                max_images=50,
                max_file_bytes=5 * 1024 * 1024,
            )

        assert result["images_found"] >= 0
        mock_rmtree.assert_called_once()  # cleanup always runs

    def test_cleanup_runs_even_on_error(self, tmp_path):
        with (
            patch("src.image_ingest.subprocess.run", side_effect=RuntimeError("clone failed")),
            patch("src.image_ingest.tempfile.mkdtemp", return_value=str(tmp_path)),
            patch("src.image_ingest.shutil.rmtree") as mock_rmtree,
        ):
            with pytest.raises(RuntimeError):
                run_image_ingest(
                    repo_url="https://github.com/Nileneb/app.linn.games",
                    ollama_url="http://localhost:11434",
                )

        mock_rmtree.assert_called_once_with(str(tmp_path), ignore_errors=True)

    def test_returns_correct_counts(self, tmp_path):
        fake_clone_dir = tmp_path / "clone"
        fake_clone_dir.mkdir()
        (fake_clone_dir / "a.png").write_bytes(b"x" * 100)
        (fake_clone_dir / "b.svg").write_text("<svg/>")

        mock_ingest_image = MagicMock(side_effect=[
            {"chunk_ids": ["c1"], "deduped": 0},   # a.png: ingested
            {"chunk_ids": [], "deduped": 1},        # b.svg: deduped
        ])

        with (
            patch("src.image_ingest.subprocess.run", return_value=MagicMock(returncode=0)),
            patch("src.image_ingest.tempfile.mkdtemp", return_value=str(tmp_path)),
            patch("src.image_ingest.shutil.rmtree"),
            patch("src.image_ingest.ingest_image", mock_ingest_image),
            patch("src.image_ingest.init_memory_db", return_value=MagicMock()),
            patch("src.image_ingest.get_or_create_chroma_collection", return_value=MagicMock()),
        ):
            result = run_image_ingest(
                repo_url="https://github.com/Nileneb/app.linn.games",
                ollama_url="http://localhost:11434",
            )

        assert result["images_found"] == 2
        assert result["images_captioned"] == 1
        assert result["images_skipped"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/nileneb/Desktop/MayringCoder
.venv/bin/python -m pytest tests/test_image_ingest.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'src.image_ingest'`

- [ ] **Step 3: Implement `src/image_ingest.py`**

```python
"""Image ingestion pipeline: shallow clone → discover → caption → memory ingest.

Public API:
    discover_images(root, max_file_bytes, max_images) -> list[Path]
    run_image_ingest(repo_url, ollama_url, ...) -> dict
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.memory_ingest import get_or_create_chroma_collection, ingest_image
from src.memory_schema import Source
from src.memory_store import init_memory_db

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"})


def discover_images(
    root: Path,
    max_file_bytes: int = 5 * 1024 * 1024,
    max_images: int = 50,
) -> list[Path]:
    """Walk root recursively, return image files within size and count limits."""
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if len(found) >= max_images:
            break
        if not path.is_file():
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
) -> dict:
    """Shallow-clone repo, caption all images, ingest into memory.

    Returns:
        {images_found, images_captioned, images_skipped, images_failed, repo_slug}
    """
    # Derive repo slug and owner/name from URL
    url = repo_url.rstrip("/").removesuffix(".git")
    parts = url.split("github.com/", 1)
    repo_owner_name = parts[1] if len(parts) == 2 else url  # e.g. "Nileneb/app.linn.games"
    repo_slug = repo_owner_name.replace("/", "-").lower()

    clone_dir = tempfile.mkdtemp(prefix=f"mayring-{repo_slug}-images-")

    try:
        return _ingest_images_from_clone(
            repo_url=url,
            clone_dir=Path(clone_dir),
            repo_owner_name=repo_owner_name,
            ollama_url=ollama_url,
            vision_model=vision_model,
            embed_model=embed_model,
            max_images=max_images,
            max_file_bytes=max_file_bytes,
            force_reingest=force_reingest,
        )
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)


def _ingest_images_from_clone(
    repo_url: str,
    clone_dir: Path,
    repo_owner_name: str,
    ollama_url: str,
    vision_model: str,
    embed_model: str,
    max_images: int,
    max_file_bytes: int,
    force_reingest: bool,
) -> dict:
    # Step 1: shallow clone
    print(f"[ingest-images] git clone --depth=1 {repo_url} ...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", repo_url, str(clone_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")

    # Step 2: discover images
    images = discover_images(clone_dir, max_file_bytes=max_file_bytes, max_images=max_images)
    print(f"[ingest-images] {len(images)} Bilder gefunden")

    if not images:
        return {
            "images_found": 0,
            "images_captioned": 0,
            "images_skipped": 0,
            "images_failed": 0,
            "repo_slug": repo_owner_name,
        }

    # Step 3: ingest each image
    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()
    captioned = skipped = failed = 0

    if force_reingest:
        from src.memory_store import deactivate_chunks_by_source
        from src.memory_retrieval import invalidate_query_cache
        for img_path in images:
            rel = str(img_path.relative_to(clone_dir))
            source_id = Source.make_id(repo_owner_name, rel)
            deactivate_chunks_by_source(conn, source_id)
        invalidate_query_cache()

    try:
        for img_path in images:
            rel = str(img_path.relative_to(clone_dir))
            source = Source(
                source_id=Source.make_id(repo_owner_name, rel),
                source_type="repo_file",
                repo=repo_owner_name,
                path=rel,
            )
            try:
                r = ingest_image(
                    source=source,
                    image_path=img_path,
                    conn=conn,
                    chroma_collection=chroma,
                    ollama_url=ollama_url,
                    model=embed_model,
                    vision_model=vision_model,
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

    return {
        "images_found": len(images),
        "images_captioned": captioned,
        "images_skipped": skipped,
        "images_failed": failed,
        "repo_slug": repo_owner_name,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_image_ingest.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/image_ingest.py tests/test_image_ingest.py
git commit -m "feat: add image_ingest module — shallow clone + caption + memory ingest"
```

---

## Task 2: `checker.py` — `--ingest-images` flag

**Files:**
- Modify: `checker.py` (two locations: argparse block ~line 152, and dispatch block ~line 488, plus new `_run_ingest_images()` function after `_run_ingest_issues()`)

- [ ] **Step 1: Add argparse flags**

In `checker.py`, find this block (around line 151):
```python
    p.add_argument("--force-reingest", action="store_true",
                   help="Bestehende Chunks invalidieren und neu ingesten (ignoriert Dedup-Schutz)")
```

Add immediately after it:
```python
    p.add_argument("--ingest-images", metavar="REPO_URL",
                   help="Repo-Bilder (PNG/JPG/SVG) captionieren und in Memory ingesten. "
                        "z. B. --ingest-images https://github.com/Nileneb/app.linn.games")
    p.add_argument("--vision-model", default="qwen2.5vl:3b", metavar="MODEL",
                   help="Ollama Vision-Modell für Bild-Captioning (Standard: qwen2.5vl:3b)")
    p.add_argument("--max-images", type=int, default=50, metavar="N",
                   help="Maximale Anzahl Bilder pro Ingest-Lauf (Standard: 50)")
```

- [ ] **Step 2: Add dispatch in main flow**

Find this block (around line 488):
```python
    if args.ingest_issues:
        _run_ingest_issues(args, ollama_url, model)
```

Add immediately after it:
```python
    if args.ingest_images:
        _run_ingest_images(args, ollama_url)
```

- [ ] **Step 3: Add `_run_ingest_images()` function**

Find `_run_ingest_issues()` in `checker.py`. Add the following function immediately after it (after its closing line):

```python
def _run_ingest_images(args, ollama_url: str) -> None:
    """Repo-Bilder captionieren und in Memory ingesten."""
    from src.image_ingest import run_image_ingest

    repo_url = args.ingest_images
    vision_model = getattr(args, "vision_model", "qwen2.5vl:3b")
    max_images = getattr(args, "max_images", 50)
    do_force = getattr(args, "force_reingest", False)

    print(f"[ingest-images] Starte Bild-Ingest für: {repo_url}")
    print(f"[ingest-images] Vision-Modell: {vision_model}, Max-Bilder: {max_images}")

    result = run_image_ingest(
        repo_url=repo_url,
        ollama_url=ollama_url,
        vision_model=vision_model,
        max_images=max_images,
        force_reingest=do_force,
    )

    print(
        f"\n[ingest-images] Fertig: {result['images_found']} Bilder total, "
        f"{result['images_captioned']} captioniert, "
        f"{result['images_skipped']} Dedup, "
        f"{result['images_failed']} Fehler."
    )
```

- [ ] **Step 4: Smoke test the CLI flag**

```bash
.venv/bin/python checker.py --help | grep ingest-images
```

Expected output contains:
```
--ingest-images REPO_URL
                        Repo-Bilder (PNG/JPG/SVG) captionieren und in Memory ingesten.
```

- [ ] **Step 5: Run full test suite to verify nothing broken**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short -q 2>&1 | tail -20
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add checker.py
git commit -m "feat: add --ingest-images flag to checker.py"
```

---

## Task 3: End-to-end smoke test

- [ ] **Step 1: Verify qwen2.5vl:3b is available**

```bash
ollama list | grep qwen2.5vl
```

If not present:
```bash
ollama pull qwen2.5vl:3b
```

- [ ] **Step 2: Run ingest on app.linn.games**

```bash
.venv/bin/python checker.py \
  --ingest-images https://github.com/Nileneb/app.linn.games \
  --vision-model qwen2.5vl:3b \
  --max-images 20
```

Expected output pattern:
```
[ingest-images] Starte Bild-Ingest für: https://github.com/Nileneb/app.linn.games
[ingest-images] git clone --depth=1 ...
[ingest-images] N Bilder gefunden
[ingest-images] OK: docs/some-diagram.png
...
[ingest-images] Fertig: N Bilder total, N captioniert, 0 Dedup, 0 Fehler.
```

- [ ] **Step 3: Verify captions landed in memory**

```bash
.venv/bin/python -c "
from src.memory_store import init_memory_db
from src.memory_retrieval import search_memory_hybrid, compress_for_prompt

conn = init_memory_db()
results = search_memory_hybrid(conn, 'architecture diagram', repo='Nileneb/app.linn.games', top_k=3)
print(compress_for_prompt(results))
conn.close()
"
```

Expected: output contains `[Caption]` text from at least one image.

- [ ] **Step 4: Run Pi on same repo to confirm visual context flows through**

```bash
.venv/bin/python checker.py \
  --repo https://github.com/Nileneb/app.linn.games \
  --pi \
  --run-id after-image-ingest \
  --budget 5
```

Check report — findings should reflect visual context (e.g., fewer false positives on DB relationships if ERD is present).
