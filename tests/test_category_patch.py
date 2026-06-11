"""Tests for PATCH /codebooks/{cb}/categories/{id} — rename + description update."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
from fastapi import HTTPException

from mayring_core.memory.store import init_memory_db
import src.api.dependencies as deps
import src.api.routes.codebooks as cb


def _run(c):
    return asyncio.run(c) if asyncio.iscoroutine(c) else c


def _seed(tmp_path: Path):
    conn = init_memory_db(tmp_path / "m.db")
    deps._conn = conn
    return conn


def test_patch_rename_and_desc(tmp_path):
    conn = _seed(tmp_path)
    conn.execute(
        "INSERT INTO categories(id,name,status,source,evidence_count) "
        "VALUES (50,'old_name','active','manual',1)")
    conn.commit()
    _run(cb.patch_category(1, 50, cb.CategoryPatchReq(name="new_name", description="desc"), _ws="ws"))
    row = conn.execute("SELECT name,description FROM categories WHERE id=50").fetchone()
    assert row[0] == "new_name" and row[1] == "desc"
    deps._conn = None


def test_patch_rename_collision(tmp_path):
    conn = _seed(tmp_path)
    conn.execute("INSERT INTO categories(id,name,status,source,evidence_count) VALUES (60,'taken','active','manual',1)")
    conn.execute("INSERT INTO categories(id,name,status,source,evidence_count) VALUES (61,'movable','active','manual',1)")
    conn.commit()
    with pytest.raises(HTTPException) as exc_info:
        _run(cb.patch_category(1, 61, cb.CategoryPatchReq(name="taken"), _ws="ws"))
    assert exc_info.value.status_code == 409
    deps._conn = None


def test_patch_404(tmp_path):
    _seed(tmp_path)
    with pytest.raises(HTTPException) as exc_info:
        _run(cb.patch_category(1, 9999, cb.CategoryPatchReq(name="x"), _ws="ws"))
    assert exc_info.value.status_code == 404
    deps._conn = None


def test_patch_description_only(tmp_path):
    conn = _seed(tmp_path)
    conn.execute(
        "INSERT INTO categories(id,name,status,source,evidence_count) "
        "VALUES (70,'stable_name','active','manual',1)")
    conn.commit()
    _run(cb.patch_category(1, 70, cb.CategoryPatchReq(description="new desc"), _ws="ws"))
    row = conn.execute("SELECT name,description FROM categories WHERE id=70").fetchone()
    assert row[0] == "stable_name"
    assert row[1] == "new desc"
    deps._conn = None


def test_patch_description_capped_at_500_chars(tmp_path):
    conn = _seed(tmp_path)
    conn.execute(
        "INSERT INTO categories(id,name,status,source,evidence_count) "
        "VALUES (80,'capped','active','manual',1)")
    conn.commit()
    long_desc = "x" * 600
    _run(cb.patch_category(1, 80, cb.CategoryPatchReq(description=long_desc), _ws="ws"))
    row = conn.execute("SELECT description FROM categories WHERE id=80").fetchone()
    assert len(row[0]) == 500
    deps._conn = None


def test_patch_noop_when_name_unchanged(tmp_path):
    conn = _seed(tmp_path)
    conn.execute(
        "INSERT INTO categories(id,name,status,source,evidence_count) "
        "VALUES (90,'same_name','active','manual',2)")
    conn.commit()
    result = _run(cb.patch_category(1, 90, cb.CategoryPatchReq(name="same_name"), _ws="ws"))
    assert result["status"] == "updated"
    row = conn.execute("SELECT name FROM categories WHERE id=90").fetchone()
    assert row[0] == "same_name"
    deps._conn = None
