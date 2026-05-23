import sqlite3

import pytest

from mayring_core.memory.store import init_memory_db
from mayring_core.memory.ingestion.mayring_process import (
    ProcessResult,
    _cosine,
    mayring_process,
)


class _FakeChroma:
    """get(ids=, include=['embeddings']) → vectors for the requested ids."""

    def __init__(self, store: dict):
        self._store = dict(store)

    def get(self, ids=None, include=None):
        ids = ids if ids is not None else list(self._store)
        present = [i for i in ids if i in self._store]
        return {"ids": present, "embeddings": [self._store[i] for i in present]}

    def upsert(self, ids=None, embeddings=None, documents=None):
        for i, cid in enumerate(ids or []):
            self._store[cid] = embeddings[i]


def _make_embed(mapping, default=(0.0, 0.0, 1.0)):
    def _e(s: str):
        for key, vec in mapping.items():
            if key in s:
                return list(vec)
        return list(default)
    return _e


def _seed(db):
    c = sqlite3.connect(db)
    now = "2026-05-24T00:00:00Z"
    c.execute("INSERT INTO codebooks(slug, description, version, auto_promote_threshold, "
              "created_at, updated_at) VALUES ('t','test',1,3,?,?)", (now, now))
    cb_id = c.execute("SELECT id FROM codebooks WHERE slug='t'").fetchone()[0]
    for name, emb_id in [("api", "cb:t:api"), ("domain", "cb:t:domain")]:
        c.execute("INSERT INTO codebook_categories(codebook_id, name, description, status, "
                  "source, evidence_count, embedding_id) VALUES (?,?,?, 'active','imported',5,?)",
                  (cb_id, name, name, emb_id))
    c.commit()
    return c, cb_id


CHROMA = {"cb:t:api": [1.0, 0.0, 0.0], "cb:t:domain": [0.0, 1.0, 0.0]}


def _raises_llm(_):  # llm_fn that must never be called on the deductive path
    raise AssertionError("llm_fn must not be called")


# ---- fail-closed -----------------------------------------------------------

def test_failclosed_empty_text(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    with pytest.raises(ValueError):
        mayring_process("", "do a thing", cb, conn=conn, chroma_categories=_FakeChroma(CHROMA),
                        embed_fn=_make_embed({}), llm_fn=_raises_llm)


def test_failclosed_empty_task(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    with pytest.raises(ValueError):
        mayring_process("some text", "", cb, conn=conn, chroma_categories=_FakeChroma(CHROMA),
                        embed_fn=_make_embed({}), llm_fn=_raises_llm)


def test_failclosed_no_active_categories(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn = sqlite3.connect(tmp_path / "m.db")
    now = "2026-05-24T00:00:00Z"
    conn.execute("INSERT INTO codebooks(slug, description, version, auto_promote_threshold, "
                 "created_at, updated_at) VALUES ('empty','',1,3,?,?)", (now, now))
    cb = conn.execute("SELECT id FROM codebooks WHERE slug='empty'").fetchone()[0]
    conn.commit()
    with pytest.raises(ValueError):
        mayring_process("text", "task", cb, conn=conn, chroma_categories=_FakeChroma({}),
                        embed_fn=_make_embed({"text": [1, 0, 0]}), llm_fn=_raises_llm)


# ---- deductive (>=0.78) ----------------------------------------------------

def test_deductive_high_score_no_llm(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    embed = _make_embed({"api": [1.0, 0.0, 0.0]})
    out = mayring_process("implement the api endpoint", "build api", cb, conn=conn,
                          chroma_categories=_FakeChroma(CHROMA), embed_fn=embed, llm_fn=_raises_llm,
                          chunk_id="c1")
    assert out.decision == "deductive"
    assert out.category_name == "api"
    assert out.proposed is False
    assert out.confidence >= 0.78
    row = conn.execute("SELECT source, confidence FROM chunk_categories WHERE chunk_id='c1'").fetchone()
    assert row[0] == "deductive"


# ---- hybrid (0.55..0.78) ---------------------------------------------------

def test_hybrid_creates_proposal_and_links(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    # [1,1,0] → cosine 0.707 to api AND domain → hybrid band
    embed = _make_embed({"hybridish": [1.0, 1.0, 0.0]})
    before = conn.execute("SELECT evidence_count FROM codebook_categories WHERE name='api'").fetchone()[0]
    out = mayring_process("hybridish content here", "classify", cb, conn=conn,
                          chroma_categories=_FakeChroma(CHROMA),
                          embed_fn=embed, llm_fn=_raises_llm, chunk_id="c2")
    assert out.decision == "hybrid"
    assert out.proposed is True
    after = conn.execute("SELECT evidence_count FROM codebook_categories WHERE name='api'").fetchone()[0]
    assert after == before + 1
    assert conn.execute("SELECT count(*) FROM codebook_proposals").fetchone()[0] == 1
    assert conn.execute("SELECT source FROM chunk_categories WHERE chunk_id='c2'").fetchone()[0] == "hybrid-merge"


# ---- inductive (<0.55) -----------------------------------------------------

def test_inductive_new_category_with_parent_hint(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    embed = _make_embed({"fresh_topic": [0.0, 0.0, 1.0]})  # far from api/domain
    out = mayring_process("zzz unrelated rambling", "classify", cb, conn=conn,
                          chroma_categories=_FakeChroma(CHROMA),
                          embed_fn=embed, llm_fn=lambda p: "fresh_topic", chunk_id="c3")
    assert out.decision == "inductive"
    assert out.proposed is True
    row = conn.execute("SELECT name, status, parent_id, embedding_id FROM codebook_categories "
                       "WHERE name='fresh_topic'").fetchone()
    assert row[1] == "proposed"
    assert row[2] is not None  # parent_hint PFLICHT
    assert row[3].startswith("cb:proposed:")  # embedding hinterlegt für künftige Dedup
    assert conn.execute("SELECT source FROM chunk_categories WHERE chunk_id='c3'").fetchone()[0] == "inductive"


def test_inductive_dedup_evidence_not_new(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    # text far away → inductive; llm label embeds onto 'api' (cosine 1.0 > 0.92) → dedup
    embed = _make_embed({"api": [1.0, 0.0, 0.0], "zzz": [0.0, 0.0, 1.0]})
    n_before = conn.execute("SELECT count(*) FROM codebook_categories").fetchone()[0]
    ev_before = conn.execute("SELECT evidence_count FROM codebook_categories WHERE name='api'").fetchone()[0]
    out = mayring_process("zzz unrelated", "classify", cb, conn=conn,
                          chroma_categories=_FakeChroma(CHROMA),
                          embed_fn=embed, llm_fn=lambda p: "api_variant")
    assert out.decision == "inductive-dedup"
    assert out.category_name == "api"
    assert out.proposed is False
    n_after = conn.execute("SELECT count(*) FROM codebook_categories").fetchone()[0]
    assert n_after == n_before  # no new category
    ev_after = conn.execute("SELECT evidence_count FROM codebook_categories WHERE name='api'").fetchone()[0]
    assert ev_after == ev_before + 1


# ---- numpy regression ------------------------------------------------------

def test_numpy_embeddings_no_crash(tmp_path):
    import numpy as np

    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")

    class _NumpyChroma:
        def get(self, ids=None, include=None):
            present = [i for i in (ids or []) if i in CHROMA]
            return {"ids": present, "embeddings": np.array([CHROMA[i] for i in present])}

        def upsert(self, **kw):
            pass

    embed = _make_embed({"api": [1.0, 0.0, 0.0]})
    out = mayring_process("api work", "task", cb, conn=conn, chroma_categories=_NumpyChroma(),
                          embed_fn=embed, llm_fn=_raises_llm)
    assert out.decision == "deductive"
    assert out.category_name == "api"


def test_cosine_zero_safe():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
