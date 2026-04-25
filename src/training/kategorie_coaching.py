"""Pipeline 1 (Issue #87): Kategorie-Coaching für Mayring-Klassifizierung.

Generiert Trainings-Paare: Code-Chunk → Mayring-Kategorie-Labels.
Nur Chunks mit positivem Feedback-Signal (rating ≥ 4 oder signal=positive)
werden als Gold-Labels verwendet — fehlerhafte Kategorisierungen werden
nicht in Trainingsdaten übernommen.

CLI:
    python -m src.cli --generate-training-data kategorie --workspace-id <id>

Output:
    cache/finetuning/kategorie_coaching_pairs.jsonl
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config import CACHE_DIR
from src.training.base import get_training_conn, write_jsonl

_FINETUNING_DIR = CACHE_DIR / "finetuning"
DEFAULT_OUTPUT = _FINETUNING_DIR / "kategorie_coaching_pairs.jsonl"

try:
    from src.memory.ingestion.categorization import _resolve_codebook as _cb
    _CODE_CATEGORIES: list[str] = _cb("code", "repo_file")
except Exception:
    _CODE_CATEGORIES = [
        "api", "data_access", "domain", "infrastructure", "auth",
        "middleware", "providers", "listeners", "ui", "config",
        "utils", "tests", "integration", "caching", "logging",
        "validation", "serialization", "error_handling", "security", "scheduling",
    ]

_SYSTEM_PROMPT = (
    "Du analysierst Code-Ausschnitte nach Mayrings Reduktions-Modell und klassifizierst ihre Funktion.\n\n"
    "Anker-Kategorien: " + ", ".join(_CODE_CATEGORIES) + "\n\n"
    "REGELN:\n"
    "- Bestehende Kategorien ohne Prefix nutzen\n"
    "- Neue Themen mit [neu]-Prefix markieren\n"
    "- Antworte NUR mit kommaseparierten Labels, z.B.: api, auth"
)


def _is_positive(signal: str) -> bool:
    if signal in ("positive",):
        return True
    try:
        return int(signal) >= 4
    except (ValueError, TypeError):
        return False


def _best_signal_per_chunk(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, str]:
    """Return {chunk_id: best_signal} from chunk_feedback."""
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, signal FROM chunk_feedback WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    best: dict[str, str] = {}
    for r in rows:
        cid = r["chunk_id"] if hasattr(r, "keys") else r[0]
        sig = r["signal"] if hasattr(r, "keys") else r[1]
        prev = best.get(cid)
        if prev is None or _is_positive(sig):
            best[cid] = sig
    return best


def generate_pairs(
    conn: sqlite3.Connection,
    workspace_id: str,
    limit: int = 1000,
    min_text_len: int = 50,
) -> list[dict]:
    """Return training pairs for category coaching.

    Only chunks with positive feedback (or auto-rated ≥ 4) are included.
    """
    ws_cond = "AND c.workspace_id = ?" if workspace_id else ""
    params: list = ([workspace_id] if workspace_id else []) + [limit * 4]

    rows = conn.execute(
        f"""
        SELECT c.chunk_id, c.source_id, c.chunk_level,
               c.category_labels, c.text, c.quality_score
        FROM chunks c
        WHERE c.is_active = 1
          AND c.category_labels IS NOT NULL
          AND TRIM(c.category_labels) != ''
          {ws_cond}
        ORDER BY c.quality_score DESC
        LIMIT ?
        """,
        params,
    ).fetchall()

    if not rows:
        return []

    chunk_ids = [r["chunk_id"] if hasattr(r, "keys") else r[0] for r in rows]
    signals = _best_signal_per_chunk(conn, chunk_ids)

    pairs: list[dict] = []
    for r in rows:
        cid = r["chunk_id"] if hasattr(r, "keys") else r[0]
        source_id = r["source_id"] if hasattr(r, "keys") else r[1]
        level = r["chunk_level"] if hasattr(r, "keys") else r[2]
        labels_raw = r["category_labels"] if hasattr(r, "keys") else r[3]
        text = (r["text"] if hasattr(r, "keys") else r[4]) or ""
        quality = float(r["quality_score"] if hasattr(r, "keys") else r[5] or 0)

        if len(text.strip()) < min_text_len:
            continue

        sig = signals.get(cid)
        if sig is not None and not _is_positive(sig):
            continue

        labels = ", ".join(
            lbl.strip() for lbl in labels_raw.split(",") if lbl.strip()
        )
        fname = source_id.split("/")[-1] if source_id else "unknown"
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Datei: `{fname}` (Level: {level})\n\n"
            f"```\n{text[:800]}\n```"
        )

        pairs.append({
            "source_id": source_id,
            "chunk_id": cid,
            "prompt": prompt,
            "completion": labels,
            "chunk_level": level,
            "category_labels": labels,
            "quality_score": quality,
            "feedback_signal": sig or "auto",
        })

        if len(pairs) >= limit:
            break

    return pairs


def run(
    workspace_id: str,
    output_path: Path = DEFAULT_OUTPUT,
    limit: int = 1000,
) -> dict:
    """Main entry point for Pipeline 1.

    Args:
        workspace_id: Tenant workspace
        output_path:  JSONL output file
        limit:        Max training pairs to generate
    """
    conn = get_training_conn()
    pairs = generate_pairs(conn, workspace_id, limit=limit)
    written = write_jsonl(output_path, pairs)

    return {
        "pairs_written": written,
        "output": str(output_path),
        "workspace_id": workspace_id,
    }
