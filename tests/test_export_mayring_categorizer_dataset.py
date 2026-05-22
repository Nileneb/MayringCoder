"""Tests for tools/export_mayring_categorizer_dataset.py (#260)."""
import json
import sqlite3
from pathlib import Path

from export_mayring_categorizer_dataset import export


def _build_memory_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            text TEXT NOT NULL DEFAULT '',
            category_labels TEXT NOT NULL DEFAULT '',
            category_source TEXT NOT NULL DEFAULT '',
            is_active INTEGER NOT NULL DEFAULT 1,
            workspace_id TEXT NOT NULL DEFAULT 'default'
        );
        """
    )
    rows = [
        # hybrid → included
        ("chk_a", "auth handler text", "domain, config", "hybrid", 1, "ws1"),
        ("chk_b", "router text", "infrastructure", "hybrid", 1, "ws1"),
        # non-hybrid → excluded
        ("chk_c", "fallback text", "misc", "fallback", 1, "ws1"),
        # hybrid but inactive → excluded
        ("chk_d", "old text", "domain", "hybrid", 0, "ws1"),
        # hybrid but empty labels → excluded
        ("chk_e", "no labels", "", "hybrid", 1, "ws1"),
    ]
    conn.executemany(
        "INSERT INTO chunks VALUES (?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()


def _build_wiki_db(path: Path, with_evidence: bool) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE wiki_category_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            source_id TEXT NOT NULL DEFAULT '',
            chunk_id TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL,
            span_start INTEGER NOT NULL DEFAULT -1,
            span_end INTEGER NOT NULL DEFAULT -1,
            excerpt TEXT NOT NULL DEFAULT '',
            reasoning TEXT NOT NULL DEFAULT '',
            task TEXT NOT NULL DEFAULT ''
        );
        """
    )
    if with_evidence:
        conn.execute(
            "INSERT INTO wiki_category_evidence "
            "(workspace_id, chunk_id, category, span_start, span_end, excerpt, reasoning, task) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("ws1", "chk_a", "domain", 0, 4, "auth", "handles auth", "code review"),
        )
    conn.commit()
    conn.close()


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_only_hybrid_active_labeled_chunks_exported(tmp_path):
    db = tmp_path / "memory.db"
    wiki = tmp_path / "wiki_v2.db"
    _build_memory_db(db)
    _build_wiki_db(wiki, with_evidence=False)
    out = tmp_path / "ds.jsonl"

    n = export(db, wiki, out)

    assert n == 2
    rows = _read(out)
    ids = {r["chunk_id"] for r in rows}
    assert ids == {"chk_a", "chk_b"}
    for r in rows:
        assert r["category_source"] == "hybrid"


def test_labels_split_and_belege_joined(tmp_path):
    db = tmp_path / "memory.db"
    wiki = tmp_path / "wiki_v2.db"
    _build_memory_db(db)
    _build_wiki_db(wiki, with_evidence=True)
    out = tmp_path / "ds.jsonl"

    export(db, wiki, out)
    rows = {r["chunk_id"]: r for r in _read(out)}

    assert rows["chk_a"]["output"]["kategorien"] == ["domain", "config"]
    belege = rows["chk_a"]["output"]["belege"]
    assert len(belege) == 1
    assert belege[0]["kategorie"] == "domain"
    assert belege[0]["span"] == [0, 4]
    assert belege[0]["reasoning"] == "handles auth"
    assert rows["chk_a"]["input"]["task"] == "code review"
    # chunk without evidence → empty belege, empty task
    assert rows["chk_b"]["output"]["belege"] == []
    assert rows["chk_b"]["input"]["task"] == ""


def test_runs_without_wiki_db(tmp_path):
    db = tmp_path / "memory.db"
    _build_memory_db(db)
    out = tmp_path / "ds.jsonl"

    n = export(db, tmp_path / "missing_wiki.db", out)

    assert n == 2
    for r in _read(out):
        assert r["output"]["belege"] == []
