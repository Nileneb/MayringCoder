import sqlite3
from pathlib import Path

from mayring_core.memory.store import init_memory_db
from src.api.routes.reconcile_admin import (
    _desired_metadata,
    _repair_chroma_metadata,
)


class _FakeChroma:
    """Minimal Chroma stand-in: id → metadata. update() mutates metadata in place
    (no embeddings touched), mirroring chromadb's metadata-only update."""

    def __init__(self, store):  # store: dict[id] = metadata
        self._m = dict(store)

    def get(self, ids=None, include=None):
        ids = [i for i in (ids or []) if i in self._m]
        return {"ids": ids, "metadatas": [self._m[i] for i in ids]}

    def update(self, ids=None, metadatas=None):
        for j, cid in enumerate(ids or []):
            self._m[cid] = metadatas[j]


def _seed(db):
    c = sqlite3.connect(db)
    now = "2026-06-08T00:00:00Z"
    # one private repo_file source owned by user 1, plus a chunk under it
    c.execute("INSERT INTO sources(source_id,source_type,visibility,org_id,user_id,"
              "workspace_id,captured_at) VALUES (?,?,?,?,?,?,?)",
              ("repo:x:a.py", "repo_file", "private", "", "1", "ws1", now))
    c.execute("INSERT INTO chunks(chunk_id,source_id,workspace_id,chunk_level,"
              "category_labels,category_source,category_confidence,is_active,created_at) "
              "VALUES (?,?,?,?,?,?,?,?,?)",
              ("chk_a", "repo:x:a.py", "ws1", "function", "api", "deductive", 0.9, 1, now))
    c.commit()
    c.close()


def test_repair_fixes_stripped_tenancy_metadata(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    _seed(db)
    conn = sqlite3.connect(db)
    # chroma has the chunk but with the visibility axis stripped (the migration bug)
    chroma = _FakeChroma({"chk_a": {"source_id": "repo:x:a.py", "visibility": "",
                                    "user_id": "", "org_id": ""}})

    dry = _repair_chroma_metadata(conn, chroma, workspace_id="ws1",
                                  source_type="repo_file", dry_run=True)
    assert dry["scanned"] == 1 and dry["mismatched"] == 1 and dry["updated"] == 0
    assert chroma._m["chk_a"]["visibility"] == ""  # dry-run wrote nothing

    wet = _repair_chroma_metadata(conn, chroma, workspace_id="ws1",
                                  source_type="repo_file", dry_run=False)
    assert wet["updated"] == 1
    assert chroma._m["chk_a"]["visibility"] == "private"
    assert chroma._m["chk_a"]["user_id"] == "1"


def test_repair_is_idempotent(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    _seed(db)
    conn = sqlite3.connect(db)
    chroma = _FakeChroma({"chk_a": _desired_metadata({
        "workspace_id": "ws1", "source_id": "repo:x:a.py", "chunk_level": "function",
        "category_labels": "api", "category_source": "deductive",
        "category_confidence": 0.9, "visibility": "private", "org_id": "", "user_id": "1",
    })})
    # already-correct metadata → nothing to repair
    out = _repair_chroma_metadata(conn, chroma, workspace_id="ws1",
                                  source_type="repo_file", dry_run=False)
    assert out["scanned"] == 1 and out["mismatched"] == 0 and out["updated"] == 0


def test_repair_skips_chunks_absent_from_chroma(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    _seed(db)
    conn = sqlite3.connect(db)
    chroma = _FakeChroma({})  # vector missing entirely → repair can't help
    out = _repair_chroma_metadata(conn, chroma, workspace_id="ws1",
                                  source_type="repo_file", dry_run=False)
    assert out["scanned"] == 1 and out["in_chroma"] == 0
    assert out["missing_in_chroma"] == 1 and out["updated"] == 0
