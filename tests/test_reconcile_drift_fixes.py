"""Drift-fix regression tests (#drift 2026-06-12).

Covers three fixes in src/api/routes/reconcile_admin.py:
  1. reconcile orphan-diff must be workspace-scoped on the Chroma side too
     (else dry_run=False would delete FOREIGN-workspace vectors).
  2. process-reembed-queue worker drains reembed-pending → upsert + reembed-done.
  3. repair full-scope iterates per-workspace (no monolithic scan).
"""

import sqlite3

from mayring_core.memory.store import init_memory_db
from src.api.routes.reconcile_admin import (
    _process_reembed_queue,
    _reconcile_chroma_sqlite,
    _repair_all_workspaces,
)


class _FakeChroma:
    """Chroma stand-in supporting get(where=…), delete, upsert, update."""

    def __init__(self, store):  # store: dict[id] = metadata
        self._m = dict(store)

    def get(self, ids=None, include=None, where=None):
        if ids is not None:
            keys = [i for i in ids if i in self._m]
        elif where:
            keys = [i for i, md in self._m.items()
                    if all(md.get(k) == v for k, v in where.items())]
        else:
            keys = list(self._m)
        return {"ids": keys, "metadatas": [self._m[i] for i in keys]}

    def delete(self, ids=None):
        for i in ids or []:
            self._m.pop(i, None)

    def update(self, ids=None, metadatas=None):
        for j, cid in enumerate(ids or []):
            self._m[cid] = metadatas[j]

    def upsert(self, ids=None, documents=None, embeddings=None, metadatas=None):
        for j, cid in enumerate(ids or []):
            self._m[cid] = metadatas[j]


# --- Fix 1: workspace-scoped orphan-diff (data-loss bug) -------------------

def test_reconcile_does_not_list_foreign_workspace_as_orphan(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    c = sqlite3.connect(db)
    now = "2026-06-12T00:00:00Z"
    # one active chunk in ws A; ws B has its own chunk + vector
    c.execute("INSERT INTO chunks(chunk_id,source_id,workspace_id,is_active,created_at)"
              " VALUES (?,?,?,?,?)", ("chk_a", "s:a", "wsA", 1, now))
    c.commit()
    chroma = _FakeChroma({
        "chk_a": {"workspace_id": "wsA"},
        "chk_b1": {"workspace_id": "wsB"},
        "chk_b2": {"workspace_id": "wsB"},
    })

    out = _reconcile_chroma_sqlite(c, chroma, workspace_id="wsA")

    # foreign wsB ids MUST NOT appear as orphans (would be deleted on dry_run=False)
    assert "chk_b1" not in out["missing_in_sqlite"]
    assert "chk_b2" not in out["missing_in_sqlite"]
    assert out["missing_in_sqlite"] == []
    assert out["total_chroma"] == 1  # only wsA vectors counted


def test_reconcile_unscoped_still_sees_all(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    c = sqlite3.connect(db)
    now = "2026-06-12T00:00:00Z"
    c.execute("INSERT INTO chunks(chunk_id,source_id,workspace_id,is_active,created_at)"
              " VALUES (?,?,?,?,?)", ("chk_a", "s:a", "wsA", 1, now))
    c.commit()
    chroma = _FakeChroma({"chk_a": {"workspace_id": "wsA"},
                          "orphan": {"workspace_id": "wsB"}})

    out = _reconcile_chroma_sqlite(c, chroma, workspace_id=None)
    assert out["missing_in_sqlite"] == ["orphan"]  # truly orphan vector surfaces
    assert out["total_chroma"] == 2


# --- Fix 2: reembed-queue worker -------------------------------------------

def _seed_pending(db, chunk_id, text="def f(): pass", active=1):
    c = sqlite3.connect(db)
    now = "2026-06-12T00:00:00Z"
    c.execute("INSERT INTO sources(source_id,source_type,visibility,user_id,"
              "workspace_id,captured_at) VALUES (?,?,?,?,?,?)",
              ("s:" + chunk_id, "repo_file", "private", "1", "wsA", now))
    c.execute("INSERT INTO chunks(chunk_id,source_id,workspace_id,text,chunk_level,"
              "category_labels,category_source,category_confidence,is_active,created_at)"
              " VALUES (?,?,?,?,?,?,?,?,?,?)",
              (chunk_id, "s:" + chunk_id, "wsA", text, "function",
               "api", "deductive", 0.9, active, now))
    c.execute("INSERT INTO ingestion_log(source_id,event_type,payload,created_at)"
              " VALUES (?,?,?,?)", (chunk_id, "reembed-pending", "{}", now))
    c.commit()
    c.close()


def test_reembed_worker_processes_pending_and_marks_done(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    _seed_pending(db, "chk_a")
    c = sqlite3.connect(db)
    chroma = _FakeChroma({})
    calls = []

    def fake_embed(texts, ollama_url):
        calls.append((texts, ollama_url))
        return [[0.1, 0.2, 0.3]]

    out = _process_reembed_queue(c, chroma, embed_fn=fake_embed,
                                 ollama_url="http://x", dry_run=False, limit=100)
    assert out["queued"] == 1 and out["processed"] == 1 and out["failed"] == 0
    assert "chk_a" in chroma._m
    # canonical metadata carried over
    assert chroma._m["chk_a"]["workspace_id"] == "wsA"
    assert chroma._m["chk_a"]["visibility"] == "private"
    assert chroma._m["chk_a"]["user_id"] == "1"
    assert calls and calls[0][1] == "http://x"

    # idempotent: a second pass sees an empty queue (reembed-done now present)
    again = _process_reembed_queue(c, chroma, embed_fn=fake_embed,
                                   ollama_url="http://x", dry_run=False, limit=100)
    assert again["queued"] == 0 and again["processed"] == 0


def test_reembed_worker_dry_run_only_reports(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    _seed_pending(db, "chk_a")
    c = sqlite3.connect(db)
    chroma = _FakeChroma({})

    def fake_embed(texts, ollama_url):
        raise AssertionError("dry_run must not embed")

    out = _process_reembed_queue(c, chroma, embed_fn=fake_embed,
                                 ollama_url="http://x", dry_run=True, limit=100)
    assert out["queued"] == 1 and out["processed"] == 0
    assert "chk_a" not in chroma._m  # nothing written


def test_reembed_worker_skips_deleted_chunk(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    _seed_pending(db, "chk_a", active=0)  # deactivated chunk
    c = sqlite3.connect(db)
    chroma = _FakeChroma({})

    def fake_embed(texts, ollama_url):
        raise AssertionError("inactive chunk must not embed")

    out = _process_reembed_queue(c, chroma, embed_fn=fake_embed,
                                 ollama_url="http://x", dry_run=False, limit=100)
    assert out["queued"] == 1 and out["skipped_missing"] == 1 and out["processed"] == 0
    # marked done so it leaves the queue
    again = _process_reembed_queue(c, chroma, embed_fn=fake_embed,
                                   ollama_url="http://x", dry_run=False, limit=100)
    assert again["queued"] == 0


def test_reembed_worker_isolates_per_chunk_failure(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    _seed_pending(db, "chk_ok", text="good")
    _seed_pending(db, "chk_bad", text="boom")
    c = sqlite3.connect(db)
    chroma = _FakeChroma({})

    def fake_embed(texts, ollama_url):
        if texts[0] == "boom":
            raise RuntimeError("embed exploded")
        return [[0.0]]

    out = _process_reembed_queue(c, chroma, embed_fn=fake_embed,
                                 ollama_url="http://x", dry_run=False, limit=100)
    assert out["queued"] == 2 and out["processed"] == 1 and out["failed"] == 1
    assert "chk_ok" in chroma._m and "chk_bad" not in chroma._m


# --- Fix 3: repair full-scope iterates per workspace -----------------------

def test_repair_all_workspaces_aggregates_per_workspace(tmp_path):
    db = tmp_path / "memory.db"
    init_memory_db(db).close()
    c = sqlite3.connect(db)
    now = "2026-06-12T00:00:00Z"
    for ws, cid in (("wsA", "chk_a"), ("wsB", "chk_b")):
        c.execute("INSERT INTO sources(source_id,source_type,visibility,user_id,"
                  "workspace_id,captured_at) VALUES (?,?,?,?,?,?)",
                  ("s:" + cid, "repo_file", "private", "1", ws, now))
        c.execute("INSERT INTO chunks(chunk_id,source_id,workspace_id,chunk_level,"
                  "category_labels,category_source,category_confidence,is_active,created_at)"
                  " VALUES (?,?,?,?,?,?,?,?,?)",
                  (cid, "s:" + cid, ws, "function", "api", "deductive", 0.9, 1, now))
    c.commit()
    # both vectors present but with the visibility axis stripped (the migration bug)
    chroma = _FakeChroma({
        "chk_a": {"source_id": "s:chk_a", "visibility": "", "user_id": "", "org_id": ""},
        "chk_b": {"source_id": "s:chk_b", "visibility": "", "user_id": "", "org_id": ""},
    })

    out = _repair_all_workspaces(c, chroma, source_type="repo_file",
                                 dry_run=False, batch=200)
    assert out["scanned"] == 2 and out["mismatched"] == 2 and out["updated"] == 2
    assert out["workspaces_iterated"] == 2
    assert chroma._m["chk_a"]["visibility"] == "private"
    assert chroma._m["chk_b"]["user_id"] == "1"
