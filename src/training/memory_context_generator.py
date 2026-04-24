"""Pipeline 2 (Issue #87): Memory-Context-Injection mit hierarchischer Sortierung.

Erzeugt automatisch Feedback für Memory-Chunks anhand von Chunk-Eigenschaften
(category_labels, chunk_level, quality_score) und generiert Trainingsdaten-Paare
mit hierarchisch sortiertem <memory_context> Block.

CLI:
    python -m src.cli --generate-training-data memory --workspace-id <id>

Output:
    cache/finetuning/memory_context_pairs.jsonl
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config import CACHE_DIR
from src.training.base import get_training_conn, write_jsonl

LEVEL_ORDER: dict[str, int] = {
    "file": 0, "class": 1, "function": 2, "block": 3, "section": 4,
}
_LEVEL_ORDER_DEFAULT = 9

_FINETUNING_DIR = CACHE_DIR / "finetuning"
DEFAULT_OUTPUT = _FINETUNING_DIR / "memory_context_pairs.jsonl"


def _auto_rating(chunk: sqlite3.Row) -> str:
    """Derive 1–5 star rating from chunk properties — no LLM required.

    Token-efficient representation: signal stored as "1"–"5".

    5★ classified (Mayring labels) + specific level (function/class/block)
    4★ classified OR specific, decent quality
    3★ has some categorization or is a section/view chunk
    2★ file-level overview, no labels, low quality
    1★ empty text, no classification, no level info
    """
    has_labels = bool((chunk["category_labels"] or "").strip())
    is_specific = chunk["chunk_level"] in ("function", "class", "block")
    is_broad = chunk["chunk_level"] in ("file",)
    quality = float(chunk["quality_score"] or 0.0)
    has_text = bool((chunk.get("text", "") or "").strip()) if "text" in chunk.keys() else True

    if not has_text:
        return "1"
    if has_labels and is_specific:
        return "5"
    if has_labels or (is_specific and quality > 0):
        return "4"
    if has_labels or is_specific or chunk["chunk_level"] in ("section", "view_fact", "view_decision"):
        return "3"
    if is_broad and not has_labels:
        return "2"
    return "2"


def generate_auto_feedback(
    conn: sqlite3.Connection,
    workspace_id: str,
    batch: int = 2000,
) -> int:
    """Write feedback signals for active chunks that have no feedback yet.

    Returns the number of new feedback records created.
    """
    ws_filter = "AND c.workspace_id = ?" if workspace_id else ""
    params: list = [workspace_id] * bool(workspace_id) + [batch]

    rows = conn.execute(f"""
        SELECT c.chunk_id, c.chunk_level, c.category_labels,
               c.quality_score, c.source_id, c.created_at
        FROM chunks c
        LEFT JOIN chunk_feedback f ON c.chunk_id = f.chunk_id
        WHERE c.is_active = 1
          {ws_filter}
          AND f.id IS NULL
        LIMIT ?
    """, params).fetchall()

    if not rows:
        return 0

    now = datetime.now(tz=timezone.utc).isoformat()
    meta = '{"auto":true,"source":"memory_context_generator"}'
    conn.executemany(
        "INSERT INTO chunk_feedback (chunk_id, signal, metadata, created_at) VALUES (?,?,?,?)",
        [(r["chunk_id"], _auto_rating(r), meta, now) for r in rows],
    )
    conn.commit()
    return len(rows)


def _signal_to_score(signal: str) -> float:
    """Convert signal string to float score.

    Accepts both legacy strings (positive/negative/neutral) and
    1–5 star ratings (stored as "1"–"5").
    """
    if signal in ("1", "2", "3", "4", "5"):
        return (float(signal) - 3.0) / 2.0  # maps 1→-1, 3→0, 5→+1
    return {"positive": 1.0, "negative": -1.0, "neutral": 0.1}.get(signal, 0.0)


def _feedback_score_for_source(
    conn: sqlite3.Connection, source_id: str
) -> dict[str, float]:
    """Return {chunk_id: score} from chunk_feedback for this source."""
    rows = conn.execute("""
        SELECT f.chunk_id, f.signal
        FROM chunk_feedback f
        JOIN chunks c ON f.chunk_id = c.chunk_id
        WHERE c.source_id = ?
    """, (source_id,)).fetchall()
    scores: dict[str, float] = {}
    for r in rows:
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + _signal_to_score(r["signal"])
    return scores


def _build_context(chunks: list[sqlite3.Row], scores: dict[str, float], char_budget: int = 3000) -> str:
    """Build hierarchically sorted <memory_context> block."""
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (
            LEVEL_ORDER.get(c["chunk_level"], _LEVEL_ORDER_DEFAULT),
            -scores.get(c["chunk_id"], 0.0),
            c["created_at"] or "",
        ),
    )
    parts: list[str] = []
    used = 0
    for chunk in sorted_chunks:
        level = chunk["chunk_level"]
        cap = 500 if level == "file" else 300
        text = (chunk["text"] or "")[:cap]
        if not text:
            continue
        if used + len(text) > char_budget:
            break
        fname = (chunk["source_id"] or "").split("/")[-1]
        parts.append(f"### [{level}] {fname}\n{text}")
        used += len(text)
    return "\n\n".join(parts)


def _derive_task(source_id: str, labels: set[str]) -> str:
    label_hint = f" (Kategorien: {', '.join(sorted(labels))})" if labels else ""
    return f"Analysiere den Code in `{source_id}`{label_hint} und beschreibe die Hauptverantwortlichkeiten."


def generate_pairs(
    conn: sqlite3.Connection,
    workspace_id: str,
    limit: int = 500,
    char_budget: int = 3000,
) -> list[dict]:
    """Generate training pairs with hierarchical memory context."""
    ws_cond = "AND workspace_id = ?" if workspace_id else ""
    params: list = [workspace_id] * bool(workspace_id) + [limit]

    sources = conn.execute(f"""
        SELECT DISTINCT source_id FROM chunks
        WHERE is_active = 1 {ws_cond}
        LIMIT ?
    """, params).fetchall()

    pairs: list[dict] = []
    for (source_id,) in sources:
        chunks = conn.execute("""
            SELECT chunk_id, chunk_level, source_id, text, category_labels,
                   quality_score, created_at, ordinal
            FROM chunks
            WHERE source_id = ? AND is_active = 1
            ORDER BY ordinal
        """, (source_id,)).fetchall()
        if not chunks:
            continue

        scores = _feedback_score_for_source(conn, source_id)
        context = _build_context(chunks, scores, char_budget)
        if not context.strip():
            continue

        labels: set[str] = set()
        for c in chunks:
            for lbl in (c["category_labels"] or "").split(","):
                lbl = lbl.strip()
                if lbl:
                    labels.add(lbl)

        task = _derive_task(source_id, labels)
        completion = (
            f"Der Memory-Kontext für `{source_id}` enthält "
            f"{len(chunks)} Chunk(s) über {len(labels)} Kategorie(n)."
            + (f" Erkannte Labels: {', '.join(sorted(labels))}." if labels else "")
        )
        pairs.append({
            "source_id": source_id,
            "prompt": f"<memory_context>\n{context}\n</memory_context>\n\nAufgabe: {task}",
            "completion": completion,
            "chunk_count": len(chunks),
            "labels": sorted(labels),
            "avg_feedback_score": (
                sum(scores.values()) / len(scores) if scores else 0.0
            ),
        })
    return pairs


def run(
    workspace_id: str,
    output_path: Path = DEFAULT_OUTPUT,
    skip_feedback: bool = False,
    limit: int = 500,
) -> dict:
    """Main entry point: auto-feedback + training-pair generation.

    Args:
        workspace_id: Tenant workspace (e.g. 'mayringcoder' or 'bene-workspace')
        output_path:  JSONL output file (default: cache/finetuning/memory_context_pairs.jsonl)
        skip_feedback: If True, don't write auto-feedback (read-only run)
        limit:        Max source files to process
    """
    conn = get_training_conn()

    feedback_written = 0
    if not skip_feedback:
        feedback_written = generate_auto_feedback(conn, workspace_id)

    pairs = generate_pairs(conn, workspace_id, limit=limit)
    written = write_jsonl(output_path, pairs)

    return {
        "feedback_written": feedback_written,
        "pairs_written": written,
        "output": str(output_path),
        "workspace_id": workspace_id,
    }
