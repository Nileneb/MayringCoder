"""Tests for training data pipelines (Issue #87)."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest


def _make_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL DEFAULT 'default',
            chunk_level TEXT NOT NULL DEFAULT 'file',
            category_labels TEXT DEFAULT '',
            text TEXT DEFAULT '',
            quality_score REAL DEFAULT 0.0,
            is_active INTEGER DEFAULT 1,
            ordinal INTEGER DEFAULT 0,
            created_at TEXT DEFAULT '',
            superseded_by TEXT DEFAULT NULL
        );
        CREATE TABLE chunk_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL,
            signal TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            created_at TEXT DEFAULT ''
        );
    """)
    return conn


# ─── Pipeline 1: Kategorie-Coaching ──────────────────────────────────────────

class TestKategorieCoaching:
    def test_basic_pair_generation(self):
        from src.training.kategorie_coaching import generate_pairs

        conn = _make_memory_db()
        conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("chk1", "src/auth.py", "default", "function",
             "Auth, Sicherheit", "def login(user, pw): return check(user, pw)  " + "a" * 20, 0.8, 1, 0, "", None),
        )
        conn.execute(
            "INSERT INTO chunk_feedback (chunk_id, signal, created_at) VALUES (?,?,?)",
            ("chk1", "positive", "2026-01-01"),
        )
        conn.commit()

        pairs = generate_pairs(conn, "default")
        assert len(pairs) == 1
        assert pairs[0]["completion"] == "Auth, Sicherheit"
        assert "auth.py" in pairs[0]["prompt"]

    def test_negative_feedback_excluded(self):
        from src.training.kategorie_coaching import generate_pairs

        conn = _make_memory_db()
        conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("chk2", "src/bad.py", "default", "function",
             "Logik", "x = 1 + 1  " + "a" * 60, 0.5, 1, 0, "", None),
        )
        conn.execute(
            "INSERT INTO chunk_feedback (chunk_id, signal, created_at) VALUES (?,?,?)",
            ("chk2", "negative", "2026-01-01"),
        )
        conn.commit()

        pairs = generate_pairs(conn, "default")
        assert len(pairs) == 0

    def test_no_feedback_chunk_included(self):
        """Chunks without any feedback (not yet rated) ARE included."""
        from src.training.kategorie_coaching import generate_pairs

        conn = _make_memory_db()
        conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("chk3", "src/svc.py", "default", "class",
             "Service", "class PaymentService: pass  " + "x" * 50, 0.7, 1, 0, "", None),
        )
        conn.commit()

        pairs = generate_pairs(conn, "default")
        assert len(pairs) == 1
        assert pairs[0]["feedback_signal"] == "auto"

    def test_star_rating_4_included(self):
        from src.training.kategorie_coaching import generate_pairs

        conn = _make_memory_db()
        conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("chk4", "src/api.py", "default", "function",
             "Routing", "def get_users(): return db.all()  " + "r" * 40, 0.6, 1, 0, "", None),
        )
        conn.execute(
            "INSERT INTO chunk_feedback (chunk_id, signal, created_at) VALUES (?,?,?)",
            ("chk4", "4", "2026-01-01"),
        )
        conn.commit()

        pairs = generate_pairs(conn, "default")
        assert len(pairs) == 1

    def test_star_rating_2_excluded(self):
        from src.training.kategorie_coaching import generate_pairs

        conn = _make_memory_db()
        conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("chk5", "src/junk.py", "default", "file",
             "Logik", "import os  " + "y" * 60, 0.1, 1, 0, "", None),
        )
        conn.execute(
            "INSERT INTO chunk_feedback (chunk_id, signal, created_at) VALUES (?,?,?)",
            ("chk5", "2", "2026-01-01"),
        )
        conn.commit()

        pairs = generate_pairs(conn, "default")
        assert len(pairs) == 0

    def test_short_text_excluded(self):
        from src.training.kategorie_coaching import generate_pairs

        conn = _make_memory_db()
        conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("chk6", "src/x.py", "default", "function", "KI", "short", 0.9, 1, 0, "", None),
        )
        conn.commit()

        pairs = generate_pairs(conn, "default")
        assert len(pairs) == 0  # text too short (<50 chars)

    def test_output_jsonl(self, tmp_path):
        from src.training.kategorie_coaching import run, generate_pairs
        from src.training.base import write_jsonl

        conn = _make_memory_db()
        conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("chkX", "src/z.py", "ws1", "function",
             "Config", "def cfg(): return {'k': 'v'}  " + "z" * 60, 0.8, 1, 0, "", None),
        )
        conn.commit()

        pairs = generate_pairs(conn, "ws1")
        out = tmp_path / "out.jsonl"
        written = write_jsonl(out, pairs)
        assert written == len(pairs)
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert all("prompt" in l and "completion" in l for l in lines)


# ─── Pipeline 2: Memory-Context-Injection ────────────────────────────────────

class TestMemoryContextGenerator:
    def test_generate_pairs_basic(self):
        from src.training.memory_context_generator import generate_pairs

        conn = _make_memory_db()
        conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("m1", "src/main.py", "default", "function",
             "Logik", "def main(): pass  " + "m" * 60, 0.7, 1, 0, "2026-01-01", None),
        )
        conn.commit()

        pairs = generate_pairs(conn, "default", limit=10)
        assert len(pairs) == 1
        assert "memory_context" in pairs[0]["prompt"]

    def test_auto_feedback_writes_signals(self):
        from src.training.memory_context_generator import generate_auto_feedback

        conn = _make_memory_db()
        conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("m2", "src/db.py", "default", "class",
             "Daten", "class DB: pass", 0.6, 1, 0, "", None),
        )
        conn.commit()

        written = generate_auto_feedback(conn, "default")
        assert written == 1
        row = conn.execute("SELECT signal FROM chunk_feedback WHERE chunk_id='m2'").fetchone()
        assert row is not None
