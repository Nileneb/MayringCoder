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
    url = repo_url.rstrip("/").removesuffix(".git")
    parts = url.split("github.com/", 1)
    repo_owner_name = parts[1] if len(parts) == 2 else url
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
    print(f"[ingest-images] git clone --depth=1 {repo_url} ...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", repo_url, str(clone_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")

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
