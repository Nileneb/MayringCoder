import sqlite3

import pytest

from mayring_core.memory.store import init_memory_db
from mayring_core.memory.ingestion.mayring_process import (
    ProcessResult,
    _cosine,
    categorize_chunks,
    link_chunks_deductive,
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
    """Embeds by substring match — keyed on the CANDIDATE LABEL the llm_fn returns
    (the canonical flow embeds the reduced category, not the raw text)."""
    def _e(s: str):
        for key, vec in mapping.items():
            if key in s:
                return list(vec)
        return list(default)
    return _e


def _seed(db, threshold=3):
    c = sqlite3.connect(db)
    now = "2026-05-24T00:00:00Z"
    c.execute("INSERT INTO codebooks(slug, description, version, auto_promote_threshold, "
              "created_at, updated_at) VALUES ('t','test',1,?,?,?)", (threshold, now, now))
    cb_id = c.execute("SELECT id FROM codebooks WHERE slug='t'").fetchone()[0]
    for name, emb_id in [("api", "cb:t:api"), ("domain", "cb:t:domain")]:
        c.execute("INSERT INTO codebook_categories(codebook_id, name, description, status, "
                  "source, evidence_count, embedding_id) VALUES (?,?,?, 'active','imported',5,?)",
                  (cb_id, name, name, emb_id))
    c.commit()
    return c, cb_id


CHROMA = {"cb:t:api": [1.0, 0.0, 0.0], "cb:t:domain": [0.0, 1.0, 0.0]}


def _raises_llm(_):  # must never be called when text/task validation fails first
    raise AssertionError("llm_fn must not be called")


# ---- fail-closed -----------------------------------------------------------

def test_failclosed_empty_text(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    with pytest.raises(ValueError):
        mayring_process("", "do a thing", cb, conn=conn, chroma_categories=_FakeChroma(CHROMA),
                        embed_fn=_make_embed({}), llm_fn=_raises_llm)


def test_failclosed_empty_task(tmp_path):
    """Zielbezug ist obligatorisch — leeres task fail-closed (Kern der Methode)."""
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    with pytest.raises(ValueError):
        mayring_process("some text", "", cb, conn=conn, chroma_categories=_FakeChroma(CHROMA),
                        embed_fn=_make_embed({}), llm_fn=_raises_llm)


def test_failclosed_empty_reduction_label(tmp_path):
    """Reduktion liefert kein Label → fail-closed, kein stilles 'uncategorized'."""
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    with pytest.raises(ValueError):
        mayring_process("text", "task", cb, conn=conn, chroma_categories=_FakeChroma(CHROMA),
                        embed_fn=_make_embed({"x": [1, 0, 0]}), llm_fn=lambda p: "   ")


# ---- reduce-FIRST + deductive (>=0.70) -------------------------------------

def test_reduce_first_then_deductive(tmp_path):
    """Kanonisch: ZIEL→Reduktion(LLM) ZUERST → die abgeleitete Kategorie cosine-matcht
    >= 0.70 eine vorhandene → deduktiv zuordnen. Der LLM-Reduktionsschritt läuft IMMER
    (nicht erst als induktiver Notnagel)."""
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    called = {"n": 0}

    def _llm(prompt):
        called["n"] += 1
        assert "Aufgabe" in prompt and "build api" in prompt  # goal flows into the prompt
        return "api_endpoint"

    embed = _make_embed({"api": [1.0, 0.0, 0.0]})  # candidate 'api_endpoint' → [1,0,0]
    out = mayring_process("implement the endpoint", "build api", cb, conn=conn,
                          chroma_categories=_FakeChroma(CHROMA), embed_fn=embed,
                          llm_fn=_llm, chunk_id="c1")
    assert called["n"] == 1                 # reduction ran first, always
    assert out.decision == "deductive"
    assert out.category_name == "api"
    assert out.proposed is False
    assert out.confidence >= 0.70


def test_deductive_links_chunk(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    embed = _make_embed({"api": [1.0, 0.0, 0.0]})
    mayring_process("implement endpoint", "build api", cb, conn=conn,
                    chroma_categories=_FakeChroma(CHROMA), embed_fn=embed,
                    llm_fn=lambda p: "api_call", chunk_id="c1")
    row = conn.execute("SELECT source FROM chunk_categories WHERE chunk_id='c1'").fetchone()
    assert row[0] == "deductive"


# ---- inductive (<0.70): create new ----------------------------------------

def test_inductive_new_category(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    embed = _make_embed({"fresh": [0.0, 0.0, 1.0]})  # orthogonal to api/domain
    out = mayring_process("zzz unrelated rambling", "classify", cb, conn=conn,
                          chroma_categories=_FakeChroma(CHROMA),
                          embed_fn=embed, llm_fn=lambda p: "fresh_topic", chunk_id="c3")
    assert out.decision == "inductive"
    assert out.proposed is True
    row = conn.execute("SELECT name, status, parent_id, embedding_id FROM codebook_categories "
                       "WHERE name='fresh_topic'").fetchone()
    assert row[1] == "proposed"
    assert row[2] is not None                     # parent_hint = nearest active category
    assert row[3].startswith("cb:proposed:")      # embedding hinterlegt → ab Promotion matchbar
    assert conn.execute("SELECT source FROM chunk_categories WHERE chunk_id='c3'").fetchone()[0] == "inductive"


# ---- inductive dedup onto a PROPOSED category (accumulation) ---------------

def test_inductive_dedup_onto_proposed(tmp_path):
    """Kandidat matcht keine ACTIVE >= 0.70, aber >0.92 eine PROPOSED → Evidenz auf die
    proposed (Akkumulation Richtung Promotion) statt ein paralleles Fragment."""
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    conn.execute("INSERT INTO codebook_categories(codebook_id, name, description, status, "
                 "source, evidence_count, embedding_id) VALUES (?,?,?, 'proposed','induced',1,?)",
                 (cb, "emerging", "x", "cb:proposed:emerging"))
    conn.commit()
    chroma = _FakeChroma({**CHROMA, "cb:proposed:emerging": [0.0, 0.0, 1.0]})
    embed = _make_embed({"emerging": [0.0, 0.0, 1.0]})
    n_before = conn.execute("SELECT count(*) FROM codebook_categories").fetchone()[0]
    out = mayring_process("zzz", "classify", cb, conn=conn, chroma_categories=chroma,
                          embed_fn=embed, llm_fn=lambda p: "emerging_variant", chunk_id="c4")
    assert out.decision == "inductive-dedup"
    assert out.category_name == "emerging"
    n_after = conn.execute("SELECT count(*) FROM codebook_categories").fetchone()[0]
    assert n_after == n_before                    # no new fragment created
    assert conn.execute("SELECT evidence_count FROM codebook_categories WHERE name='emerging'"
                        ).fetchone()[0] == 2


# ---- auto-promotion proposed→active ----------------------------------------

def test_auto_promotion_at_threshold(tmp_path):
    """evidence_count erreicht auto_promote_threshold → status='active' + promoted_at,
    damit der Reranker (cat_match, active-only) die induktive Kategorie sieht."""
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db", threshold=3)
    conn.execute("INSERT INTO codebook_categories(codebook_id, name, description, status, "
                 "source, evidence_count, embedding_id) VALUES (?,?,?, 'proposed','induced',2,?)",
                 (cb, "emerging", "x", "cb:proposed:emerging"))
    conn.commit()
    chroma = _FakeChroma({**CHROMA, "cb:proposed:emerging": [0.0, 0.0, 1.0]})
    embed = _make_embed({"emerging": [0.0, 0.0, 1.0]})
    out = mayring_process("zzz", "classify", cb, conn=conn, chroma_categories=chroma,
                          embed_fn=embed, llm_fn=lambda p: "emerging_again", chunk_id="c5")
    assert out.decision == "inductive-dedup"
    assert out.proposed is False                  # promoted in this very call
    row = conn.execute("SELECT status, promoted_at FROM codebook_categories WHERE name='emerging'"
                       ).fetchone()
    assert row[0] == "active"
    assert row[1] is not None


# ---- bootstrap: empty codebook → pure inductive (no fail-closed) -----------

def test_empty_codebook_bootstraps_inductive(tmp_path):
    """Kein ACTIVE vorhanden → kein fail-closed: die erste Kategorie wird induktiv gebildet
    (der kanonische Ablauf erlaubt 'kein Treffer → neu', auch ab Null)."""
    init_memory_db(tmp_path / "m.db").close()
    conn = sqlite3.connect(tmp_path / "m.db")
    now = "2026-05-24T00:00:00Z"
    conn.execute("INSERT INTO codebooks(slug, description, version, auto_promote_threshold, "
                 "created_at, updated_at) VALUES ('empty','',1,3,?,?)", (now, now))
    cb = conn.execute("SELECT id FROM codebooks WHERE slug='empty'").fetchone()[0]
    conn.commit()
    out = mayring_process("first ever text", "bootstrap", cb, conn=conn,
                          chroma_categories=_FakeChroma({}),
                          embed_fn=_make_embed({"first": [1, 0, 0]}),
                          llm_fn=lambda p: "first_category")
    assert out.decision == "inductive"
    assert conn.execute("SELECT count(*) FROM codebook_categories WHERE codebook_id=?",
                        (cb,)).fetchone()[0] == 1


# ---- batched bulk path shares the same decision logic ----------------------

def test_categorize_chunks_batched(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    embed = _make_embed({"api": [1.0, 0.0, 0.0], "fresh": [0.0, 0.0, 1.0]})
    items = [("c1", "implement the api endpoint"), ("c2", "totally unrelated rambling")]

    res = categorize_chunks(
        items, "build the service", cb, conn=conn,
        chroma_categories=_FakeChroma(CHROMA),
        batch_embed_fn=lambda labels: [embed(x) for x in labels],
        batch_reduce_fn=lambda pairs: ["api_handler", "fresh_concept"],
        match_codebook_id=None)
    assert [r.decision for r in res] == ["deductive", "inductive"]
    assert res[0].category_name == "api"
    assert conn.execute("SELECT source FROM chunk_categories WHERE chunk_id='c1'").fetchone()[0] == "deductive"
    assert conn.execute("SELECT source FROM chunk_categories WHERE chunk_id='c2'").fetchone()[0] == "inductive"


def test_categorize_chunks_intra_batch_dedup(tmp_path):
    """Zwei Chunks im selben Batch, deren Reduktion auf dieselbe NEUE Kategorie zeigt:
    der zweite dedupt in-place auf die vom ersten erzeugte proposed (kein Fragment)."""
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    embed = _make_embed({"novel": [0.0, 0.0, 1.0]})
    items = [("c1", "x"), ("c2", "y")]
    res = categorize_chunks(
        items, "ziel", cb, conn=conn, chroma_categories=_FakeChroma(CHROMA),
        batch_embed_fn=lambda labels: [embed(x) for x in labels],
        batch_reduce_fn=lambda pairs: ["novel_thing", "novel_thing_two"],
        match_codebook_id=None)
    assert res[0].decision == "inductive"
    assert res[1].decision == "inductive-dedup"          # second deduped onto the first
    assert conn.execute("SELECT count(*) FROM codebook_categories WHERE source='induced'"
                        ).fetchone()[0] == 1               # only ONE new category


def test_categorize_chunks_requires_task(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    with pytest.raises(ValueError):
        categorize_chunks([("c1", "text")], "", cb, conn=conn,
                          chroma_categories=_FakeChroma(CHROMA),
                          batch_embed_fn=lambda x: [], batch_reduce_fn=lambda x: [])


def test_categorize_chunks_count_mismatch_raises(tmp_path):
    """Reduktion liefert falsche Label-Zahl → laut scheitern, nicht stumm danebenlabeln."""
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    with pytest.raises(ValueError):
        categorize_chunks([("c1", "a"), ("c2", "b")], "ziel", cb, conn=conn,
                          chroma_categories=_FakeChroma(CHROMA),
                          batch_embed_fn=lambda x: [[1, 0, 0]],
                          batch_reduce_fn=lambda pairs: ["only_one_label"])


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
                          embed_fn=embed, llm_fn=lambda p: "api_thing")
    assert out.decision == "deductive"
    assert out.category_name == "api"


def test_cosine_zero_safe():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_clean_label_recovers_or_rejects_json():
    """Wenn das LLM JSON statt eines bloßen Labels leakt: Value bergen, sonst verwerfen —
    NIE einen JSON-Blob wie {"x":""} als Kategoriename speichern (Prod-Junk-Root-Cause)."""
    from mayring_core.memory.ingestion.mayring_process import _clean_label
    assert _clean_label("api_endpoint") == "api_endpoint"
    assert _clean_label('{"category_0": "org_revocation"}') == "org_revocation"
    assert _clean_label('{"multi_org_management": ""}') == "multi_org_management"
    assert _clean_label('["first_label", "second"]') == "first_label"
    assert _clean_label('user_authentication", "oauth_flow", "jwt_auth"') == "user_authentication"
    for junk in ('{}', '[]', '{"x":{"y":1}}', '{"a": ""}'):
        assert not any(ch in _clean_label(junk) for ch in '{}[]"'), f"JSON-Blob durchgelassen: {junk}"


# ---- batched reduction: determinism opts + parse + per-item fallback --------

def test_batch_reduce_labels_happy_json(monkeypatch):
    import mayring_core.providers as providers
    from mayring_core.memory.ingestion.core import _batch_reduce_labels

    seen = {}

    def _gen(**kw):
        seen.update(kw)
        return '["api_call", "session_token"]'

    monkeypatch.setattr(providers, "generate_text", _gen)
    out = _batch_reduce_labels([("text a", "goal"), ("text b", "goal")], "url", "mistral")
    assert out == ["api_call", "session_token"]
    # determinism (temp 0 + fixer seed) + JSON mode MUST be passed through
    assert seen["options"] == {"temperature": 0.0, "seed": 7}
    assert seen["response_format"] == "json"


def test_batch_reduce_labels_per_item_fallback(monkeypatch):
    """Batch-JSON unparsbar → per-item fallback liefert trotzdem N Labels (Quelle nicht gedroppt)."""
    import mayring_core.providers as providers
    from mayring_core.memory.ingestion.core import _batch_reduce_labels

    def _gen(**kw):
        if kw.get("label") == "mayring_reduce_batch":
            return "totally not json garbage"
        return f"label_for::{kw['prompt'][:20]}"  # per-item

    monkeypatch.setattr(providers, "generate_text", _gen)
    out = _batch_reduce_labels([("alpha", "g"), ("beta", "g"), ("gamma", "g")], "url", "m")
    assert len(out) == 3                       # never drops the whole source
    assert all(o.startswith("label_for::") for o in out)


# ---- link_chunks_deductive (unchanged cosine plumbing, query-side mirror) --

def test_link_chunks_deductive_links_best_match(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, _cb = _seed(tmp_path / "m.db")
    n = link_chunks_deductive(conn, _FakeChroma(CHROMA),
                              [("c1", [1.0, 0.0, 0.0]), ("c2", [0.0, 0.0, 1.0])],
                              min_score=0.55)
    assert n == 1  # c1→api (cos 1.0); c2 unrelated (cos 0) → no link
    row = conn.execute("SELECT source FROM chunk_categories WHERE chunk_id='c1'").fetchone()
    assert row[0] == "deductive"
    assert conn.execute("SELECT count(*) FROM chunk_categories WHERE chunk_id='c2'").fetchone()[0] == 0


def test_derive_query_category_ids_maps_query_to_codebook(tmp_path, monkeypatch):
    from mayring_core.memory.ingestion import mayring_process as mp
    init_memory_db(tmp_path / "m.db").close()
    conn, _cb = _seed(tmp_path / "m.db")
    api_id = conn.execute("SELECT id FROM codebook_categories WHERE name='api'").fetchone()[0]
    dom_id = conn.execute("SELECT id FROM codebook_categories WHERE name='domain'").fetchone()[0]
    chroma = _FakeChroma(CHROMA)
    monkeypatch.setattr(mp, "_CAT_EMB_CACHE", {})
    assert mp.derive_query_category_ids(conn, chroma, [0.95, 0.05, 0.0]) == {api_id}
    assert mp.derive_query_category_ids(conn, chroma, [0.0, 0.98, 0.1]) == {dom_id}
    assert mp.derive_query_category_ids(conn, chroma, [0.0, 0.0, 1.0]) == set()
    assert mp.derive_query_category_ids(conn, chroma, []) == set()


def test_link_scopes_by_project(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    conn.execute("INSERT INTO codebook_categories(codebook_id, name, description, status, "
                 "source, evidence_count, embedding_id, project_id) "
                 "VALUES (?,?,?, 'active','induced',1,?,?)",
                 (cb, "proj_only", "x", "cb:t:projonly", "proj-A"))
    conn.commit()
    chroma = _FakeChroma({**CHROMA, "cb:t:projonly": [0.0, 0.0, 1.0]})
    assert link_chunks_deductive(conn, chroma, [("c1", [0.0, 0.0, 1.0])],
                                 project_id=None, min_score=0.55) == 0
    assert link_chunks_deductive(conn, chroma, [("c2", [0.0, 0.0, 1.0])],
                                 project_id="proj-A", min_score=0.55) == 1


def test_inductive_tags_active_project(tmp_path):
    init_memory_db(tmp_path / "m.db").close()
    conn, cb = _seed(tmp_path / "m.db")
    embed = _make_embed({"fresh": [0.0, 0.0, 1.0]})
    out = mayring_process("zzz unrelated rambling", "classify", cb, conn=conn,
                          chroma_categories=_FakeChroma(CHROMA), embed_fn=embed,
                          llm_fn=lambda p: "fresh_proj_cat", active_project_id="proj-Z")
    assert out.decision == "inductive"
    row = conn.execute("SELECT project_id FROM codebook_categories WHERE name='fresh_proj_cat'").fetchone()
    assert row[0] == "proj-Z"
