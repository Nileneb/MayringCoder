"""Category-consolidation step 1 (additive): multi-link + label derivation.

SoT direction: chunk_categories (cosine FK) becomes the single source; the
free-string category_labels become a derived view (so the ~10 consumers that
read chunks.category_labels keep working after the LLM mayring_categorize pass
is retired). These helpers are additive — link_chunks_deductive default top_n=1
keeps the prior single-link behavior unchanged.
"""
from __future__ import annotations

import sqlite3

from mayring_core.memory.ingestion import mayring_process as mp


def test_best_matches_topn_and_threshold():
    pairs = [
        ({"id": 1, "name": "auth"}, [1.0, 0.0]),
        ({"id": 2, "name": "api"}, [0.95, 0.05]),
        ({"id": 3, "name": "db"}, [0.0, 1.0]),
    ]
    out = mp._best_matches([1.0, 0.0], pairs, top_n=2, min_score=0.5)
    assert [c["id"] for c, _ in out] == [1, 2]  # top-2 ≥ 0.5; db (cosine 0) excluded


def _codebook_conn() -> sqlite3.Connection:
    # WHY(v19-drop-codebook): categories statt codebook_categories (kein codebook_id)
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT, "
        "igio_axis TEXT, parent_id INTEGER, embedding_id TEXT, status TEXT, project_id TEXT)"
    )
    conn.execute(
        "CREATE TABLE chunk_categories (chunk_id TEXT, category_id INTEGER, "
        "codebook_version INTEGER, confidence REAL, source TEXT, PRIMARY KEY(chunk_id, category_id))"
    )
    for i, name in enumerate(["auth", "api", "db"], 1):
        conn.execute(
            "INSERT INTO categories VALUES (?,?,?,?,?,?,?)",
            (i, name, "", None, f"emb_{i}", "active", None),
        )
    conn.commit()
    return conn


def test_link_chunks_deductive_multi_link(monkeypatch):
    conn = _codebook_conn()
    embs = {1: [1.0, 0.0], 2: [0.95, 0.05], 3: [0.0, 1.0]}
    monkeypatch.setattr(mp, "_category_embeddings", lambda chroma, cats: [
        (c, embs[c["id"]]) for c in cats
    ])
    n = mp.link_chunks_deductive(conn, object(), [("chk_1", [1.0, 0.0])],
                                 min_score=0.5, top_n=3)
    linked = {r[0] for r in conn.execute(
        "SELECT category_id FROM chunk_categories WHERE chunk_id='chk_1'").fetchall()}
    assert linked == {1, 2}  # auth + api ≥ 0.5; db excluded
    assert n == 2


def test_link_chunks_deductive_default_topn1_single_link(monkeypatch):
    conn = _codebook_conn()
    embs = {1: [1.0, 0.0], 2: [0.95, 0.05], 3: [0.0, 1.0]}
    monkeypatch.setattr(mp, "_category_embeddings", lambda chroma, cats: [
        (c, embs[c["id"]]) for c in cats
    ])
    n = mp.link_chunks_deductive(conn, object(), [("chk_1", [1.0, 0.0])], min_score=0.5)
    assert n == 1  # default top_n=1 = unchanged single-link (backward-compat)


def test_derive_labels_from_categories():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE chunk_categories (chunk_id TEXT, category_id INTEGER, confidence REAL)")
    conn.executemany("INSERT INTO categories VALUES (?,?)", [(1, "auth"), (2, "api")])
    conn.executemany("INSERT INTO chunk_categories VALUES (?,?,?)",
                     [("chk_1", 1, 0.9), ("chk_1", 2, 0.7)])
    conn.commit()
    assert mp.derive_labels_from_categories(conn, ["chk_1"]) == {"chk_1": ["auth", "api"]}
    assert mp.derive_labels_from_categories(conn, []) == {}
