"""Unit tests for src.embedder (Issue #11 — Embedding-based prefilter)."""

import json
import math
from unittest.mock import patch

import pytest

from src.context import (
    _cosine_similarity,
    _file_snippet,
    _index_path,
    build_file_index,
    filter_by_embedding,
)


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        # cosine of opposite vectors is -1
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_partial_similarity(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = _cosine_similarity(a, b)
        expected = 1.0 / math.sqrt(2)
        assert sim == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# _file_snippet
# ---------------------------------------------------------------------------

class TestFileSnippet:
    def test_includes_category_and_filename(self):
        f = {"filename": "src/foo.py", "category": "domain", "content": "x = 1"}
        snippet = _file_snippet(f)
        assert "[domain]" in snippet
        assert "src/foo.py" in snippet

    def test_truncates_content(self):
        f = {"filename": "f.py", "category": "api", "content": "A" * 2000}
        snippet = _file_snippet(f, max_chars=100)
        assert len(snippet) < 200  # category+filename+100 chars ≈ well under limit

    def test_missing_content_field(self):
        f = {"filename": "f.py", "category": "api"}
        # Should not raise; content defaults to ""
        snippet = _file_snippet(f)
        assert "f.py" in snippet

    def test_missing_category_uses_uncategorized(self):
        f = {"filename": "x.py", "content": ""}
        snippet = _file_snippet(f)
        assert "uncategorized" in snippet


# ---------------------------------------------------------------------------
# _index_path
# ---------------------------------------------------------------------------

class TestIndexPath:
    def test_path_contains_slug_and_model(self, tmp_path):
        from src import config as cfg
        cfg.CACHE_DIR = tmp_path
        path = _index_path("https://github.com/owner/repo.git", "nomic-embed-text")
        assert "owner-repo" in path.name
        assert "nomic-embed-text" in path.name

    def test_colon_in_model_name_replaced(self, tmp_path):
        from src import config as cfg
        cfg.CACHE_DIR = tmp_path
        path = _index_path("https://github.com/a/b", "mxbai:latest")
        assert ":" not in path.name


# ---------------------------------------------------------------------------
# build_file_index
# ---------------------------------------------------------------------------

def _make_file(name: str, content: str = "", cat: str = "domain") -> dict:
    return {"filename": name, "content": content, "category": cat}


def _fake_embed(texts, ollama_url):
    """Return a deterministic fake embedding: [len(t), 0.0] for each text."""
    return [[float(len(t)), 0.0] for t in texts]


class TestBuildFileIndex:
    def test_returns_one_entry_per_file(self, tmp_path):
        from src import config as cfg, context as embedder
        cfg.CACHE_DIR = tmp_path
        embedder.CACHE_DIR = tmp_path

        files = [_make_file("a.py"), _make_file("b.py")]
        with patch("src.context._embed_texts", side_effect=_fake_embed):
            index = build_file_index(files, "http://localhost:11434", repo_url="")
        assert len(index) == 2
        assert {e["filename"] for e in index} == {"a.py", "b.py"}

    def test_each_entry_has_embedding(self, tmp_path):
        from src import config as cfg, context as embedder
        cfg.CACHE_DIR = tmp_path
        embedder.CACHE_DIR = tmp_path

        files = [_make_file("x.py", "hello")]
        with patch("src.context._embed_texts", side_effect=_fake_embed):
            index = build_file_index(files, "http://localhost:11434", repo_url="")
        assert isinstance(index[0]["embedding"], list)
        assert len(index[0]["embedding"]) == 2

    def test_cache_is_written_and_reused(self, tmp_path):
        from src import config as cfg, context as embedder
        cfg.CACHE_DIR = tmp_path
        embedder.CACHE_DIR = tmp_path

        files = [_make_file("z.py", "code")]
        call_count = {"n": 0}

        def counting_embed(texts, url):
            call_count["n"] += 1
            return _fake_embed(texts, url)

        repo = "https://github.com/test/repo"
        with patch("src.context._embed_texts", side_effect=counting_embed):
            build_file_index(files, "http://localhost:11434", repo_url=repo)
            # Second call — should hit cache, not call embed again
            build_file_index(files, "http://localhost:11434", repo_url=repo)

        assert call_count["n"] == 1, "Embed should only be called once; second call uses cache"

    def test_force_reindex_bypasses_cache(self, tmp_path):
        from src import config as cfg, context as embedder
        cfg.CACHE_DIR = tmp_path
        embedder.CACHE_DIR = tmp_path

        files = [_make_file("f.py")]
        call_count = {"n": 0}

        def counting_embed(texts, url):
            call_count["n"] += 1
            return _fake_embed(texts, url)

        repo = "https://github.com/test/repo2"
        with patch("src.context._embed_texts", side_effect=counting_embed):
            build_file_index(files, "http://localhost:11434", repo_url=repo)
            build_file_index(files, "http://localhost:11434", repo_url=repo, force=True)

        assert call_count["n"] == 2

    def test_cache_invalidated_on_file_set_change(self, tmp_path):
        from src import config as cfg, context as embedder
        cfg.CACHE_DIR = tmp_path
        embedder.CACHE_DIR = tmp_path

        files_v1 = [_make_file("a.py")]
        files_v2 = [_make_file("a.py"), _make_file("b.py")]
        call_count = {"n": 0}

        def counting_embed(texts, url):
            call_count["n"] += 1
            return _fake_embed(texts, url)

        repo = "https://github.com/test/repo3"
        with patch("src.context._embed_texts", side_effect=counting_embed):
            build_file_index(files_v1, "http://localhost:11434", repo_url=repo)
            build_file_index(files_v2, "http://localhost:11434", repo_url=repo)

        assert call_count["n"] == 2, "Different file set must trigger re-embedding"


# ---------------------------------------------------------------------------
# filter_by_embedding
# ---------------------------------------------------------------------------

class TestFilterByEmbedding:
    """Tests for the main prefilter function."""

    def _run_filter(self, files, query, top_k=10, threshold=None, tmp_path=None):
        from src import config as cfg, context as embedder
        if tmp_path is not None:
            cfg.CACHE_DIR = tmp_path
            embedder.CACHE_DIR = tmp_path

        with patch("src.context._embed_texts", side_effect=_fake_embed):
            return filter_by_embedding(
                files=files,
                query=query,
                ollama_url="http://localhost:11434",
                top_k=top_k,
                threshold=threshold,
                repo_url="",
            )

    def test_empty_files_returns_empty(self, tmp_path):
        selected, filtered = self._run_filter([], "anything", tmp_path=tmp_path)
        assert selected == []
        assert filtered == []

    def test_top_k_limits_selection(self, tmp_path):
        files = [_make_file(f"f{i}.py", "x" * i) for i in range(10)]
        selected, filtered = self._run_filter(files, "query", top_k=3, tmp_path=tmp_path)
        assert len(selected) == 3
        assert len(filtered) == 7

    def test_selected_and_filtered_are_disjoint_and_cover_all(self, tmp_path):
        files = [_make_file(f"x{i}.py") for i in range(5)]
        selected, filtered = self._run_filter(files, "q", top_k=3, tmp_path=tmp_path)
        all_names = {f["filename"] for f in files}
        assert set(selected) | set(filtered) == all_names
        assert set(selected) & set(filtered) == set()

    def test_top_k_zero_keeps_all_files(self, tmp_path):
        files = [_make_file(f"g{i}.py") for i in range(8)]
        selected, filtered = self._run_filter(files, "q", top_k=0, tmp_path=tmp_path)
        assert len(selected) == 8
        assert filtered == []

    def test_threshold_filters_low_similarity(self, tmp_path):
        """Files with zero-length content produce short snippets → small embedding value.
        Files with longer content produce longer snippets → higher embedding value.
        Our fake embed uses [len(text), 0], so longer snippets score higher vs a long query.
        Use a high threshold to force some files to be excluded."""
        from src import config as cfg, context as embedder
        cfg.CACHE_DIR = tmp_path
        embedder.CACHE_DIR = tmp_path

        # Use a high threshold — only files whose cosine sim with query >= threshold pass.
        # With _fake_embed: sim = len(snippet) / (|snippet| * |query|) = 1 always for nonzero.
        # So we need a threshold > 1 to exclude everything.
        selected, filtered = self._run_filter(
            [_make_file("a.py", "code")],
            "q",
            threshold=2.0,  # impossible cosine value → all excluded
            tmp_path=tmp_path,
        )
        assert selected == []
        assert filtered == ["a.py"]

    def test_filtered_out_is_sorted(self, tmp_path):
        files = [_make_file("z.py"), _make_file("a.py"), _make_file("m.py")]
        _, filtered = self._run_filter(files, "q", top_k=1, tmp_path=tmp_path)
        assert filtered == sorted(filtered)
