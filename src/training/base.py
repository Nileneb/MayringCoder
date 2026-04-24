"""Shared DB access and JSONL export utilities for training pipelines."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.config import CACHE_DIR


def get_training_conn(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or (CACHE_DIR / "memory.db")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def write_jsonl(path: Path, records: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)
