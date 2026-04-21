"""Paper-PDF-Ingest Workflow.

Scans `papers_dir` (mounted from linn-papers-data) for PDF/TXT files produced
by mcp-paper-search, extracts text, and ingests them into the memory store.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def run_ingest_paper(
    papers_dir: str,
    ollama_url: str,
    model: str,
    repo_slug: str = "",
    force_reingest: bool = False,
    workspace_id: str = "default",
) -> dict:
    """Scan papers_dir for PDF/TXT files and ingest into memory.

    papers_dir is mounted from linn-papers-data volume at /data/papers.
    Files are named {paper_id}.pdf or {paper_id}.txt by the mcp-paper-search service.
    """
    from src.memory.chunker import chunk_paper, extract_pdf_text  # noqa: F401
    from src.memory.ingest import get_or_create_chroma_collection, ingest
    from src.memory.schema import Source
    from src.memory.store import init_memory_db

    conn = init_memory_db()
    chroma = get_or_create_chroma_collection()

    base = Path(papers_dir)
    if not base.exists():
        print(f"  [paper] Verzeichnis nicht gefunden: {papers_dir}")
        return {"ingested": 0, "skipped": 0, "failed": 0, "total": 0}

    candidates = list(base.glob("*.pdf")) + list(base.glob("*.txt"))
    ingested = skipped = failed = 0

    for fpath in sorted(candidates):
        paper_id = fpath.stem
        source_id = f"paper:{paper_id}"

        if fpath.suffix == ".pdf":
            text = extract_pdf_text(str(fpath))
            if not text:
                print(f"  [paper] Überspringe (kein Text extrahierbar): {fpath.name}")
                failed += 1
                continue
        else:
            text = fpath.read_text(encoding="utf-8", errors="replace")

        content_hash = "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]
        src = Source(
            source_id=source_id,
            source_type="paper",
            repo=repo_slug,
            path=str(fpath),
            branch="",
            commit="",
            content_hash=content_hash,
        )
        result = ingest(
            src, text, conn, chroma,
            ollama_url, model,
            opts={"categorize": True, "chunk_level": "paper"},
            workspace_id=workspace_id,
        )
        if result.get("skipped"):
            skipped += 1
        else:
            print(f"  [paper] Ingested: {paper_id} ({result.get('chunks', 0)} chunks)")
            ingested += 1

    return {"ingested": ingested, "skipped": skipped, "failed": failed, "total": len(candidates)}
