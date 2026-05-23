import sqlite3
from pathlib import Path

import pytest

from mayring_core.memory.store import init_memory_db
from src.api.routes.projects import (
    _cosine,
    _normalize_remote,
    project_embed_text,
    route,
)


# ---- Task 1: schema index -------------------------------------------------

def test_projects_source_index_exists(tmp_path: Path) -> None:
    p = tmp_path / "memory.db"
    init_memory_db(p).close()
    idx = {r[1] for r in sqlite3.connect(p).execute(
        "PRAGMA index_list('projects')").fetchall()}
    assert "idx_projects_source" in idx


# ---- Task 2: _normalize_remote --------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("git@github.com:Nileneb/MayringCoder.git", "nileneb/mayringcoder"),
    ("https://github.com/Nileneb/MayringCoder.git", "nileneb/mayringcoder"),
    ("https://github.com/Nileneb/MayringCoder", "nileneb/mayringcoder"),
    ("ssh://git@github.com/Nileneb/app.linn.games.git", "nileneb/app.linn.games"),
    ("", None),
    ("not-a-remote", None),
])
def test_normalize_remote(url, expected):
    assert _normalize_remote(url) == expected


# ---- Task 3: _cosine + project_embed_text ---------------------------------

def test_cosine():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # zero vector → 0, no div0


def test_project_embed_text():
    assert project_embed_text("MayringCoder", "nileneb/mayringcoder", "github") == \
        "MayringCoder nileneb/mayringcoder github"


# ---- Task 4: route() decision ---------------------------------------------

class _FakeChroma:
    """get(include=['embeddings','metadatas']) → all project vectors."""

    def __init__(self, items):  # items: list[(id, embedding, metadata)]
        self._items = list(items)

    def get(self, include=None):
        return {
            "ids": [i[0] for i in self._items],
            "embeddings": [i[1] for i in self._items],
            "metadatas": [i[2] for i in self._items],
        }

    def upsert(self, ids=None, embeddings=None, metadatas=None, documents=None):
        for j, cid in enumerate(ids or []):
            self._items.append((cid, embeddings[j], (metadatas or [{}])[j]))


def _seed(db, rows):
    c = sqlite3.connect(db)
    now = "2026-05-24T00:00:00Z"
    for pid, st, ref, name in rows:
        c.execute("INSERT INTO projects(id,workspace_id,name,source_type,source_ref,"
                  "created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                  (pid, "ws1", name, st, ref, now, now))
    c.commit()
    c.close()


def test_route_cwd_remote_match(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    _seed(db, [("p1", "github", "nileneb/mayringcoder", "MayringCoder")])
    conn = sqlite3.connect(db)
    out = route(conn, _FakeChroma([]), "ws1",
                cwd_remote="git@github.com:Nileneb/MayringCoder.git",
                prompt="fix the auth bug", embed_fn=lambda t: [0.0, 1.0])
    assert out["project_id"] == "p1"
    assert out["reason"] == "cwd-remote"
    assert out["mode"] == "coding"


def test_route_cwd_remote_create(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    conn = sqlite3.connect(db)
    out = route(conn, _FakeChroma([]), "ws1",
                cwd_remote="https://github.com/Nileneb/NewRepo.git",
                prompt="add feature", embed_fn=lambda t: [1.0, 0.0])
    assert out["project_id"]  # created
    row = conn.execute("SELECT source_ref FROM projects WHERE id=?",
                       (out["project_id"],)).fetchone()
    assert "newrepo" in (row[0] or "").lower()
    assert out["reason"] == "cwd-remote"


def test_route_semantic_match(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    _seed(db, [("p1", "github", "nileneb/mayringcoder", "MayringCoder")])
    conn = sqlite3.connect(db)
    chroma = _FakeChroma([("proj:p1", [0.0, 1.0], {"project_id": "p1"})])
    out = route(conn, chroma, "ws1", cwd_remote=None,
                prompt="memory retrieval pipeline", embed_fn=lambda t: [0.0, 1.0])
    assert out["project_id"] == "p1"
    assert out["reason"] == "semantic"


def test_route_null_when_uncertain(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    conn = sqlite3.connect(db)
    chroma = _FakeChroma([
        ("proj:p1", [1.0, 0.0], {"project_id": "p1"}),
        ("proj:p2", [0.99, 0.01], {"project_id": "p2"}),  # tiny margin
    ])
    out = route(conn, chroma, "ws1", cwd_remote=None,
                prompt="vague", embed_fn=lambda t: [1.0, 0.0])
    assert out["project_id"] is None
    assert out["reason"] == "no-match"
