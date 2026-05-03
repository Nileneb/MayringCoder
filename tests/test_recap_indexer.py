"""Hermetic tests for recap_indexer + recap_renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.store import init_memory_db
from src.wiki_v2.recap_indexer import (
    Recap,
    _plans_mentioning_issue,
    build_recap,
    discover_issue_ids,
)
from src.wiki_v2.recap_renderer import render_recap


@pytest.fixture
def populated_db(tmp_path: Path):
    db = tmp_path / "memory.db"
    conn = init_memory_db(db)
    conn.execute(
        "INSERT INTO sources (source_id, source_type, captured_at) VALUES (?, ?, ?)",
        ("repo:test:src/auth.py", "repo_file", "2026-01-01"),
    )
    conn.execute(
        "INSERT INTO sources (source_id, source_type, captured_at) VALUES (?, ?, ?)",
        ("github:nileneb/MayringCoder/issues/42", "github_issue", "2026-01-01"),
    )
    rows = [
        ("chk_iss_1", "github:nileneb/MayringCoder/issues/42",
         "Login flow drops session on refresh — see #42",
         "auth", "issue", 0.91),
        ("chk_int_1", "repo:test:src/auth.py",
         "Refactored session refresh to extend cookie lifetime — closes issue 42",
         "auth", "intervention", 0.88),
        ("chk_out_1", "repo:test:src/auth.py",
         "Tests grün after fix for #42 — 142 passed",
         "tests,auth", "outcome", 0.95),
        ("chk_unrelated", "repo:test:src/auth.py",
         "Some neutral change unrelated to any issue",
         "infra", "intervention", 0.7),
    ]
    for cid, sid, text, cats, axis, conf in rows:
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_id, text, category_labels, "
            "igio_axis, igio_confidence, igio_classified_at, created_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (cid, sid, text, cats, axis, conf, "2026-01-02", "2026-01-02"),
        )
    conn.commit()
    yield conn
    conn.close()


def test_build_recap_groups_chunks_by_axis(populated_db, tmp_path: Path):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    (plans_dir / "session-fix.md").write_text(
        "# Session-Fix\n\nFixing issue #42 — refresh flow.\n"
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    recap = build_recap(
        "42", conn=populated_db, plans_dir=plans_dir, repo_root=repo_root,
    )
    assert len(recap.issue_chunks) == 1
    assert recap.issue_chunks[0].chunk_id == "chk_iss_1"
    assert len(recap.intervention_chunks) == 1
    assert recap.intervention_chunks[0].chunk_id == "chk_int_1"
    assert len(recap.outcome_chunks) == 1
    assert recap.outcome_chunks[0].chunk_id == "chk_out_1"
    assert recap.goal_chunks == []
    assert len(recap.plans) == 1
    assert recap.plans[0].title == "Session-Fix"


def test_build_recap_strips_hash_prefix(populated_db, tmp_path: Path):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    a = build_recap("#42", conn=populated_db, plans_dir=plans_dir, repo_root=repo_root)
    b = build_recap("42", conn=populated_db, plans_dir=plans_dir, repo_root=repo_root)
    assert a.issue_id == b.issue_id == "42"


def test_build_recap_unknown_issue_returns_empty(populated_db, tmp_path: Path):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    recap = build_recap(
        "999", conn=populated_db, plans_dir=plans_dir, repo_root=repo_root,
    )
    assert recap.issue_chunks == []
    assert recap.intervention_chunks == []
    assert recap.outcome_chunks == []
    assert recap.plans == []


def test_plans_mentioning_issue_skips_unrelated(tmp_path: Path):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    (plans_dir / "match.md").write_text("# A\n\nfix for #42\n")
    (plans_dir / "miss.md").write_text("# B\n\nunrelated work\n")
    out = _plans_mentioning_issue(plans_dir, "42")
    assert len(out) == 1
    assert out[0].path.name == "match.md"


def test_discover_issue_ids(populated_db):
    ids = discover_issue_ids(populated_db)
    assert "42" in ids


def test_render_recap_shape(populated_db, tmp_path: Path):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    (plans_dir / "p.md").write_text("# Plan A\n\nissue #42\n")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    recap = build_recap(
        "42", conn=populated_db, plans_dir=plans_dir, repo_root=repo_root,
    )
    md = render_recap(recap)
    assert md.startswith("# Recap — Issue #42")
    assert "## Issue" in md
    assert "## Goal" in md
    assert "## Intervention — Plans" in md
    assert "## Intervention — Chunks" in md
    assert "## Outcome" in md
    assert "chk_iss_1" in md
    assert "(manuell ergänzen)" in md


def test_render_recap_empty():
    md = render_recap(Recap(issue_id="999", workspace_id="ws"))
    assert "Recap — Issue #999" in md
    assert "noch keine Daten" in md or "manuell ergänzen" in md
