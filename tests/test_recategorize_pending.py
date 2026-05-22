"""Tests for the targeted MARK step in tools/recategorize_pending.py (#249)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mayring_core.memory.store import init_memory_db

_RECAT_PATH = Path(__file__).parent.parent / "tools" / "recategorize_pending.py"


def _load_recat():
    spec = importlib.util.spec_from_file_location("recat_tool", _RECAT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def conn(tmp_path: Path):
    c = init_memory_db(tmp_path / "memory.db")
    # two sources of different types
    now = "2026-05-12T00:00:00+00:00"
    c.execute("INSERT INTO sources (source_id, source_type, captured_at, workspace_id) VALUES (?,?,?,?)",
              ("paper:p1", "agent_result", now, "bene"))
    c.execute("INSERT INTO sources (source_id, source_type, captured_at, workspace_id) VALUES (?,?,?,?)",
              ("repo:r1", "repo_file", now, "bene"))
    # chunks: 2 paper chunks (one with code labels, one clean), 1 repo chunk with code labels
    rows = [
        ("ck_paper_bad",  "paper:p1", "argumentation,api,domain", "hybrid"),
        ("ck_paper_ok",   "paper:p1", "argumentation,methodik",   "hybrid"),
        ("ck_repo_code",  "repo:r1",  "api,data_access",          "hybrid"),
    ]
    for cid, sid, labels, src in rows:
        c.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, category_labels, category_source, is_active, workspace_id, created_at) "
            "VALUES (?,?,?,?,?,1,?,?)",
            (cid, sid, "some text", labels, src, "bene", now),
        )
    c.commit()
    yield c
    c.close()


def test_csv_set_parsing():
    recat = _load_recat()
    assert recat._csv_set("a, B ,c") == {"a", "b", "c"}
    assert recat._csv_set("") == set()
    assert recat._csv_set(None) == set()


def test_mark_by_source_type_and_label(conn):
    recat = _load_recat()
    n = recat._mark_subset_cleanup_pending(
        conn, source_types={"agent_result"}, label_tokens={"api", "domain"},
        workspace_id=None, dry_run=False,
    )
    assert n == 1  # only ck_paper_bad: agent_result + has api/domain
    marked = {r[0] for r in conn.execute("SELECT chunk_id FROM chunks WHERE category_source='cleanup-pending'")}
    assert marked == {"ck_paper_bad"}


def test_mark_dry_run_changes_nothing(conn):
    recat = _load_recat()
    n = recat._mark_subset_cleanup_pending(
        conn, source_types={"agent_result"}, label_tokens={"api"},
        workspace_id=None, dry_run=True,
    )
    assert n == 1
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE category_source='cleanup-pending'").fetchone()[0] == 0


def test_mark_label_token_is_exact_not_substring(conn):
    """`api` must not match a label like `capitalize` — exact comma-token match."""
    recat = _load_recat()
    conn.execute("UPDATE chunks SET category_labels='capitalize,methodik' WHERE chunk_id='ck_paper_ok'")
    conn.commit()
    n = recat._mark_subset_cleanup_pending(
        conn, source_types=set(), label_tokens={"api"}, workspace_id=None, dry_run=False,
    )
    # ck_paper_bad has `api`; ck_repo_code has `api`; ck_paper_ok has `capitalize` (NOT api)
    assert n == 2
    marked = {r[0] for r in conn.execute("SELECT chunk_id FROM chunks WHERE category_source='cleanup-pending'")}
    assert marked == {"ck_paper_bad", "ck_repo_code"}


def test_mark_source_type_only_no_label_filter(conn):
    recat = _load_recat()
    n = recat._mark_subset_cleanup_pending(
        conn, source_types={"repo_file"}, label_tokens=set(), workspace_id=None, dry_run=False,
    )
    assert n == 1  # only ck_repo_code
