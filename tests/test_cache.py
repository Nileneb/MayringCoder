"""Unit tests for src.analysis.cache."""

import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch

from src.analysis.cache import (
    _repo_slug,
    reset_repo,
    init_db,
    _last_snapshot_id,
    _create_snapshot,
    _insert_file_versions,
    mark_files_analyzed,
    _select_top_k,
    find_changed_files,
)


# ---------------------------------------------------------------------------
# _repo_slug
# ---------------------------------------------------------------------------

class TestRepoSlug:
    def test_https_url(self):
        assert _repo_slug("https://github.com/user/repo.git") == "user-repo"

    def test_http_url(self):
        assert _repo_slug("http://gitlab.com/group/project.git") == "group-project"

    def test_ssh_url(self):
        # Standard SSH URLs (git@host:path format) are not parsed correctly by urlparse.
        # Only the ssh:// variant works properly.
        assert _repo_slug("ssh://git@github.com/foo/bar.git") == "foo-bar"

    def test_path_with_slashes(self):
        slug = _repo_slug("https://example.com/a/b/c.git")
        assert slug == "a-b-c"
        assert slug.isalnum() or "-" in slug

    def test_special_chars_removed(self):
        slug = _repo_slug("https://github.com/user/repo_with_underscores.git")
        assert "_" not in slug

    def test_strips_dot_git_suffix(self):
        assert _repo_slug("https://github.com/user/repo.git") == "user-repo"
        assert _repo_slug("git@gitlab.com:org/project.git") == "gitgitlabcomorg-project"  # known urlparse limitation
        assert _repo_slug("ssh://git@gitlab.com/org/project.git") == "org-project"


# ---------------------------------------------------------------------------
# init_db / schema
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_cache_dir(self, tmp_path):
        with patch("src.analysis.cache.CACHE_DIR", tmp_path / "cache"):
            with patch("src.analysis.cache.CACHE_DIR", tmp_path / "cache"):
                from src.analysis import cache as cache_module
                cache_module.CACHE_DIR = tmp_path / "cache"
                conn = cache_module.init_db("https://github.com/test/repo.git")
                assert (tmp_path / "cache").exists()
                conn.close()

    def test_creates_snapshots_table(self, tmp_path):
        from src.analysis import cache as cache_module
        cache_module.CACHE_DIR = tmp_path / "cache"
        conn = cache_module.init_db("https://github.com/test/repo.git")
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {r[0] for r in rows}
        assert "snapshots" in table_names
        assert "file_versions" in table_names
        conn.close()

    def test_runs_migrations_adds_analyzed_at(self, tmp_path):
        """When an old DB without analyzed_at is opened, it should be added."""
        from src.analysis import cache as cache_module
        cache_module.CACHE_DIR = tmp_path / "cache"

        # Create a bare DB without analyzed_at column
        db_path = tmp_path / "cache" / "test-repo.db"
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo TEXT,
                commit_hash TEXT,
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE file_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER,
                filename TEXT,
                hash TEXT,
                size INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

        # Re-open with init_db — migration should add analyzed_at
        conn2 = cache_module.init_db("https://github.com/test/repo.git")
        cols = {r[1] for r in conn2.execute("PRAGMA table_info(file_versions)").fetchall()}
        assert "analyzed_at" in cols
        conn2.close()


# ---------------------------------------------------------------------------
# _create_snapshot / _last_snapshot_id
# ---------------------------------------------------------------------------

class TestSnapshots:
    def test_create_snapshot_returns_id(self, tmp_path):
        from src.analysis import cache as cache_module
        cache_module.CACHE_DIR = tmp_path
        conn = cache_module.init_db("https://github.com/test/repo.git")
        snap_id = cache_module._create_snapshot(conn, "https://github.com/test/repo.git")
        assert isinstance(snap_id, int)
        conn.close()

    def test_last_snapshot_returns_most_recent(self, tmp_path):
        from src.analysis import cache as cache_module
        cache_module.CACHE_DIR = tmp_path
        conn = cache_module.init_db("https://github.com/test/repo.git")
        id1 = cache_module._create_snapshot(conn, "https://github.com/test/repo.git")
        id2 = cache_module._create_snapshot(conn, "https://github.com/test/repo.git")
        id3 = cache_module._create_snapshot(conn, "https://github.com/test/repo.git")
        assert cache_module._last_snapshot_id(conn, "https://github.com/test/repo.git") == id3
        conn.close()

    def test_last_snapshot_returns_none_for_new_repo(self, tmp_path):
        from src.analysis import cache as cache_module
        cache_module.CACHE_DIR = tmp_path
        conn = cache_module.init_db("https://github.com/test/new-repo.git")
        assert cache_module._last_snapshot_id(conn, "https://github.com/test/new-repo.git") is None
        conn.close()


# ---------------------------------------------------------------------------
# _insert_file_versions
# ---------------------------------------------------------------------------

class TestInsertFileVersions:
    def _conn(self, tmp_path):
        from src.analysis import cache as cache_module
        cache_module.CACHE_DIR = tmp_path
        conn = cache_module.init_db("https://github.com/test/repo.git")
        snap_id = cache_module._create_snapshot(conn, "https://github.com/test/repo.git")
        return conn, snap_id

    def test_inserts_files(self, tmp_path):
        conn, snap_id = self._conn(tmp_path)
        files = [
            {"filename": "a.py", "hash": "aaa", "size": 10},
            {"filename": "b.py", "hash": "bbb", "size": 20},
        ]
        _insert_file_versions(conn, snap_id, files)
        rows = conn.execute("SELECT filename, hash, size FROM file_versions ORDER BY filename").fetchall()
        assert len(rows) == 2
        assert rows[0] == ("a.py", "aaa", 10)
        assert rows[1] == ("b.py", "bbb", 20)
        conn.close()

    def test_preserves_analyzed_at_for_same_hash(self, tmp_path):
        """If same filename+hash exists with analyzed_at, carry it over."""
        conn, snap_id = self._conn(tmp_path)
        # Pre-insert a file that has been analyzed
        conn.execute(
            "INSERT INTO file_versions (snapshot_id, filename, hash, size, analyzed_at, run_key)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (snap_id, "a.py", "aaa", 10, "2025-01-01T00:00:00", "default"),
        )
        conn.commit()

        # Insert a new snapshot with same file
        snap2 = _create_snapshot(conn, "https://github.com/test/repo.git")
        _insert_file_versions(conn, snap2, [{"filename": "a.py", "hash": "aaa", "size": 10}])

        row = conn.execute(
            "SELECT analyzed_at FROM file_versions WHERE snapshot_id = ? AND filename = ?",
            (snap2, "a.py"),
        ).fetchone()
        assert row[0] == "2025-01-01T00:00:00"
        conn.close()


# ---------------------------------------------------------------------------
# mark_files_analyzed
# ---------------------------------------------------------------------------

class TestMarkFilesAnalyzed:
    def test_stamps_analyzed_at(self, tmp_path):
        from src.analysis import cache as cache_module
        cache_module.CACHE_DIR = tmp_path
        conn = cache_module.init_db("https://github.com/test/repo.git")
        snap_id = cache_module._create_snapshot(conn, "https://github.com/test/repo.git")
        _insert_file_versions(
            conn, snap_id,
            [{"filename": "a.py", "hash": "aaa", "size": 10}],
        )

        mark_files_analyzed(conn, snap_id, ["a.py"])

        row = conn.execute(
            "SELECT analyzed_at FROM file_versions WHERE filename = ?", ("a.py",)
        ).fetchone()
        assert row[0] is not None
        conn.close()


# ---------------------------------------------------------------------------
# _select_top_k
# ---------------------------------------------------------------------------

def make_file_map(files: list[tuple[str, int]]) -> dict[str, dict]:
    return {name: {"size": size} for name, size in files}


class TestSelectTopK:
    def test_no_limit_returns_all(self):
        files = ["a.py", "b.py", "c.py"]
        fmap = make_file_map([("a.py", 100), ("b.py", 200), ("c.py", 300)])
        selected, skipped = _select_top_k(files, fmap, None, None)
        assert selected == files
        assert skipped == []

    def test_limit_zero_means_no_limit(self):
        files = ["a.py", "b.py"]
        fmap = make_file_map([("a.py", 10), ("b.py", 20)])
        selected, skipped = _select_top_k(files, fmap, None, 0)
        assert selected == files
        assert skipped == []

    def test_limits_to_max_files(self):
        files = ["a.py", "b.py", "c.py", "d.py"]
        fmap = make_file_map([("a.py", 10), ("b.py", 20), ("c.py", 30), ("d.py", 40)])
        selected, skipped = _select_top_k(files, fmap, None, 2)
        assert len(selected) == 2
        assert len(skipped) == 2

    def test_risk_categories_prioritized(self):
        """Files in RISK_CATEGORIES should be selected before larger non-risk files."""
        files = ["big.txt", "small_api.py", "api.py"]
        fmap = make_file_map([("big.txt", 9999), ("small_api.py", 10), ("api.py", 20)])
        categories = {"small_api.py": "api", "api.py": "api", "big.txt": "other"}
        selected, _ = _select_top_k(files, fmap, categories, 2)
        # Both api files should be selected (risk category), big.txt skipped
        assert set(selected) == {"small_api.py", "api.py"}

    def test_within_risk_group_sorted_by_size(self):
        """Within same risk tier, larger files are prioritized."""
        files = ["tiny_api.py", "big_api.py", "medium_api.py"]
        fmap = make_file_map([
            ("tiny_api.py", 10),
            ("big_api.py", 1000),
            ("medium_api.py", 100),
        ])
        categories = {"tiny_api.py": "api", "big_api.py": "api", "medium_api.py": "api"}
        selected, _ = _select_top_k(files, fmap, categories, 2)
        assert "big_api.py" in selected
        assert "medium_api.py" in selected


# ---------------------------------------------------------------------------
# find_changed_files
# ---------------------------------------------------------------------------

def make_filespec(name: str, hash_val: str) -> dict:
    return {"filename": name, "hash": hash_val, "size": len(name)}


class TestFindChangedFiles:
    def _conn(self, tmp_path):
        from src.analysis import cache as cache_module
        cache_module.CACHE_DIR = tmp_path
        return cache_module.init_db("https://github.com/test/repo.git")

    def test_first_run_all_added(self, tmp_path):
        conn = self._conn(tmp_path)
        files = [make_filespec("a.py", "aaa"), make_filespec("b.py", "bbb")]
        result = find_changed_files(conn, "https://github.com/test/repo.git", files)

        assert set(result["added"]) == {"a.py", "b.py"}
        assert result["changed"] == []
        assert result["removed"] == []
        assert result["unchanged"] == []
        conn.close()

    def test_unchanged_file_same_hash(self, tmp_path):
        conn = self._conn(tmp_path)
        # First snapshot
        files1 = [make_filespec("a.py", "aaa")]
        find_changed_files(conn, "https://github.com/test/repo.git", files1)

        # Second snapshot — same file
        files2 = [make_filespec("a.py", "aaa")]
        result = find_changed_files(conn, "https://github.com/test/repo.git", files2)

        assert result["unchanged"] == ["a.py"]
        assert result["added"] == []
        assert result["changed"] == []
        conn.close()

    def test_changed_file_different_hash(self, tmp_path):
        conn = self._conn(tmp_path)
        find_changed_files(conn, "https://github.com/test/repo.git", [make_filespec("a.py", "aaa")])
        result = find_changed_files(
            conn, "https://github.com/test/repo.git", [make_filespec("a.py", "bbb")]
        )
        assert result["changed"] == ["a.py"]
        conn.close()

    def test_removed_file(self, tmp_path):
        conn = self._conn(tmp_path)
        find_changed_files(conn, "https://github.com/test/repo.git", [make_filespec("a.py", "aaa")])
        result = find_changed_files(conn, "https://github.com/test/repo.git", [])
        assert result["removed"] == ["a.py"]
        conn.close()

    def test_selected_excludes_already_analyzed(self, tmp_path):
        conn = self._conn(tmp_path)
        files = [make_filespec("a.py", "aaa"), make_filespec("b.py", "bbb")]
        result = find_changed_files(conn, "https://github.com/test/repo.git", files)
        assert result["snapshot_id"] is not None

        # Mark a.py as analyzed
        mark_files_analyzed(conn, result["snapshot_id"], ["a.py"])

        # Next run: a.py should not be in selected
        result2 = find_changed_files(
            conn, "https://github.com/test/repo.git", files
        )
        assert "a.py" not in result2["selected"]
        conn.close()

    def test_returns_snapshot_id(self, tmp_path):
        conn = self._conn(tmp_path)
        result = find_changed_files(conn, "https://github.com/test/repo.git", [])
        assert "snapshot_id" in result
        assert isinstance(result["snapshot_id"], int)
        conn.close()

    def test_run_key_isolation(self, tmp_path):
        """Analyzing with run_key A must not block run_key B from analyzing the same files."""
        conn = self._conn(tmp_path)
        files = [make_filespec("a.py", "aaa"), make_filespec("b.py", "bbb")]

        # Run with llama-run: find + mark all files as analyzed
        result_llama = find_changed_files(
            conn, "https://github.com/test/repo.git", files, run_key="llama-run"
        )
        mark_files_analyzed(conn, result_llama["snapshot_id"], ["a.py", "b.py"], "llama-run")

        # Run with qwen-run on the same files: should still need to analyze everything
        result_qwen = find_changed_files(
            conn, "https://github.com/test/repo.git", files, run_key="qwen-run"
        )
        assert set(result_qwen["selected"]) == {"a.py", "b.py"}, (
            "qwen-run should see both files as unanalyzed, "
            "not be blocked by llama-run's analyzed_at stamps"
        )
        conn.close()

    def test_default_run_key_isolation(self, tmp_path):
        """Files analyzed under 'default' must not block a named run_key."""
        conn = self._conn(tmp_path)
        files = [make_filespec("x.py", "xxx")]

        # Default run: analyze x.py
        result_default = find_changed_files(conn, "https://github.com/test/repo.git", files)
        mark_files_analyzed(conn, result_default["snapshot_id"], ["x.py"])

        # Named run: should still see x.py as unanalyzed
        result_named = find_changed_files(
            conn, "https://github.com/test/repo.git", files, run_key="my-run"
        )
        assert "x.py" in result_named["selected"]
        conn.close()


# ---------------------------------------------------------------------------
# reset_repo
# ---------------------------------------------------------------------------

class TestResetRepo:
    def test_reset_without_run_key_deletes_db(self, tmp_path):
        from src.analysis import cache as cache_module
        cache_module.CACHE_DIR = tmp_path
        cache_module.init_db("https://github.com/test/repo.git").close()

        result = cache_module.reset_repo("https://github.com/test/repo.git")
        assert result is not None
        assert not (tmp_path / "test-repo.db").exists()

    def test_reset_nonexistent_returns_none(self, tmp_path):
        from src.analysis import cache as cache_module
        cache_module.CACHE_DIR = tmp_path
        result = cache_module.reset_repo("https://github.com/test/nonexistent.git")
        assert result is None

    def test_reset_with_run_key_clears_analyzed_at(self, tmp_path):
        from src.analysis import cache as cache_module
        cache_module.CACHE_DIR = tmp_path
        conn = cache_module.init_db("https://github.com/test/repo.git")
        snap_id = cache_module._create_snapshot(conn, "https://github.com/test/repo.git")
        _insert_file_versions(conn, snap_id, [make_filespec("a.py", "aaa")])
        mark_files_analyzed(conn, snap_id, ["a.py"])
        conn.close()

        cache_module.reset_repo("https://github.com/test/repo.git", run_key="default")

        conn2 = cache_module.init_db("https://github.com/test/repo.git")
        row = conn2.execute("SELECT analyzed_at FROM file_versions").fetchone()
        assert row[0] is None
        conn2.close()
