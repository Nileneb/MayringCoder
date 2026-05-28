"""Regression: the query-category embedding cache must NOT negative-cache an
empty result.

The chroma codebook_categories collection is empty right after a deploy cutover
and only repopulated by the post-deploy reembed. The old code cached pairs=[]
for the full 300s TTL → derive_query_category_ids returned empty → reranker-v3
cat_match stayed dead for up to 300s AFTER a reembed (recurring red
reranker_cat_match_fires smoke; a reembed had no immediate effect).
"""
from __future__ import annotations

from mayring_core.memory.ingestion import mayring_process as mp


class _FakeConn:
    def execute(self, *a):
        class _Cur:
            def fetchall(self_inner):
                return [(1, "auth", "I", None, "emb_1")]
        return _Cur()


def test_active_category_pairs_does_not_negative_cache_empty(monkeypatch):
    mp._CAT_EMB_CACHE.clear()
    conn = _FakeConn()

    # 1) collection empty (post-deploy, pre-reembed) → [] must NOT be cached.
    monkeypatch.setattr(mp, "_category_embeddings", lambda chroma, cats: [])
    assert mp._active_category_pairs(conn, object(), None) == []
    assert "__base__" not in mp._CAT_EMB_CACHE

    # 2) reembed populates the collection → next call re-queries (no stale empty
    #    cache), returns non-empty, and THAT gets cached.
    monkeypatch.setattr(mp, "_category_embeddings",
                        lambda chroma, cats: [({"id": 1}, [0.1, 0.2])])
    out = mp._active_category_pairs(conn, object(), None)
    assert out == [({"id": 1}, [0.1, 0.2])]
    assert "__base__" in mp._CAT_EMB_CACHE


def test_active_category_pairs_caches_nonempty(monkeypatch):
    """Non-empty results are still cached (the optimization is preserved)."""
    mp._CAT_EMB_CACHE.clear()
    conn = _FakeConn()
    calls = []

    def _emb(chroma, cats):
        calls.append(1)
        return [({"id": 1}, [0.1, 0.2])]

    monkeypatch.setattr(mp, "_category_embeddings", _emb)
    mp._active_category_pairs(conn, object(), None)
    mp._active_category_pairs(conn, object(), None)  # served from cache
    assert len(calls) == 1
