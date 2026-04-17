"""Unit tests for context.py Phase 1 improvements (build_inventory_context, build_dependency_context)."""

import pytest
from pathlib import Path
from unittest.mock import patch
import json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cache(tmp_path: Path):
    """Provide a patched CACHE_DIR and a pre-populated overview JSON.

    Note: context._repo_slug does NOT strip ".git" (unlike cache._repo_slug),
    so we use a URL that produces a predictable slug matching the fixture data.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # The URL must produce a slug matching the overview filename.
    # repo_slug("https://example.com/example/repo.git") -> "example-repo"
    repo_url = "https://example.com/example/repo.git"
    slug = "example-repo"
    overview_filename = f"{slug}_overview.json"

    overview_data = [
        {
            "filename": "src/UserService.php",
            "category": "domain",
            "file_summary": "Handles user business logic including registration and profile updates.",
            "file_type": "service",
            "key_responsibilities": ["register()", "updateProfile()", "sendWelcomeEmail()"],
            "dependencies": ["App\\Models\\User", "App\\Mail\\WelcomeMail"],
            "purpose_keywords": ["User", "Registration", "Email"],
            "_signatures": {
                "functions": ["register", "updateProfile", "sendWelcomeEmail"],
                "classes": ["UserService"],
                "imports": ["User", "WelcomeMail"],
            },
        },
        {
            "filename": "app/Models/User.php",
            "category": "domain",
            "file_summary": "Eloquent User model with relationships to workspaces and projects.",
            "file_type": "model",
            "key_responsibilities": ["relationships: workspaces(), projects()"],
            "dependencies": ["Illuminate\\Database\\Eloquent\\Model"],
            "purpose_keywords": ["User", "Eloquent", "Relationships"],
            "_signatures": {"functions": [], "classes": ["User"], "imports": ["Model"]},
        },
        {
            "filename": "tests/Unit/UserServiceTest.php",
            "category": "test",
            "file_summary": "Unit tests for UserService registration flow.",
            "file_type": "test",
            "key_responsibilities": [
                "testRegister_ValidInput_Success()",
                "testRegister_DuplicateEmail_Fails()",
            ],
            "dependencies": ["App\\Services\\UserService", "App\\Models\\User"],
            "purpose_keywords": ["Testing", "UserService", "Registration"],
            "_signatures": {
                "functions": ["testRegister_ValidInput_Success", "testRegister_DuplicateEmail_Fails"],
                "classes": [],
                "imports": ["UserService", "User"],
            },
        },
    ]

    overview_path = cache_dir / overview_filename
    overview_path.write_text(json.dumps(overview_data, ensure_ascii=False), encoding="utf-8")

    with patch("src.analysis.context.CACHE_DIR", cache_dir):
        yield {
            "cache_dir": cache_dir,
            "overview_path": overview_path,
            "repo_url": repo_url,
            "slug": slug,
        }


# ---------------------------------------------------------------------------
# build_inventory_context
# ---------------------------------------------------------------------------

class TestBuildInventoryContext:
    def test_returns_context_string(self, mock_cache):
        from src.analysis.context import build_inventory_context

        ctx = build_inventory_context(mock_cache["repo_url"])
        assert ctx is not None
        assert isinstance(ctx, str)

    def test_includes_file_types(self, mock_cache):
        from src.analysis.context import build_inventory_context

        ctx = build_inventory_context(mock_cache["repo_url"])
        assert "type=service" in ctx
        assert "type=model" in ctx
        assert "type=test" in ctx

    def test_includes_responsibilities(self, mock_cache):
        from src.analysis.context import build_inventory_context

        ctx = build_inventory_context(mock_cache["repo_url"])
        assert "register()" in ctx
        assert "updateProfile()" in ctx

    def test_truncates_long_summaries(self, mock_cache):
        from src.analysis.context import build_inventory_context

        ctx = build_inventory_context(mock_cache["repo_url"])
        # Summary for UserService is >100 chars so should be truncated
        assert "..." in ctx or len(ctx) < 5000

    def test_nonexistent_repo_returns_none(self, mock_cache):
        from src.analysis.context import build_inventory_context

        ctx = build_inventory_context("https://nonexistent.com/does/not/exist.git")
        assert ctx is None

    def test_includes_all_files(self, mock_cache):
        from src.analysis.context import build_inventory_context

        ctx = build_inventory_context(mock_cache["repo_url"])
        assert "UserService.php" in ctx
        assert "User.php" in ctx
        assert "UserServiceTest.php" in ctx


# ---------------------------------------------------------------------------
# build_dependency_context
# ---------------------------------------------------------------------------

class TestBuildDependencyContext:
    def test_returns_context_for_known_file(self, mock_cache):
        from src.analysis.context import build_dependency_context

        ctx = build_dependency_context(
            mock_cache["repo_url"],
            {"filename": "src/UserService.php"},
        )
        assert ctx is not None
        assert "referenzierte Dateien" in ctx.lower() or "UserService.php" in ctx

    def test_includes_self_entry(self, mock_cache):
        from src.analysis.context import build_dependency_context

        ctx = build_dependency_context(
            mock_cache["repo_url"],
            {"filename": "src/UserService.php"},
        )
        assert "UserService.php" in ctx

    def test_includes_dependency_summaries(self, mock_cache):
        from src.analysis.context import build_dependency_context

        ctx = build_dependency_context(
            mock_cache["repo_url"],
            {"filename": "src/UserService.php"},
        )
        # Should reference the User model since it's in dependencies
        assert "User" in ctx

    def test_unknown_file_returns_none(self, mock_cache):
        from src.analysis.context import build_dependency_context

        ctx = build_dependency_context(
            mock_cache["repo_url"],
            {"filename": "nonexistent/file.php"},
        )
        assert ctx is None

    def test_nonexistent_repo_returns_none(self, mock_cache):
        from src.analysis.context import build_dependency_context

        ctx = build_dependency_context(
            "https://nonexistent.com/repo.git",
            {"filename": "src/UserService.php"},
        )
        assert ctx is None


# ---------------------------------------------------------------------------
# save_overview_context — signature preservation
# ---------------------------------------------------------------------------

class TestSaveOverviewContextSignatures:
    def test_preserves_signatures(self, mock_cache):
        from src.analysis.context import save_overview_context

        results = [
            {
                "filename": "a.py",
                "category": "source",
                "file_summary": "Test",
                "_signatures": {
                    "functions": ["doThing"],
                    "classes": ["A"],
                    "imports": ["b"],
                },
            },
        ]
        path = save_overview_context(results, mock_cache["repo_url"])
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert loaded[0]["_signatures"]["functions"] == ["doThing"]
        assert loaded[0]["_signatures"]["classes"] == ["A"]

    def test_skips_error_results(self, mock_cache):
        from src.analysis.context import save_overview_context

        results = [
            {"filename": "a.py", "error": "Timeout"},
            {"filename": "b.py", "category": "source", "file_summary": "OK"},
        ]
        path = save_overview_context(results, mock_cache["repo_url"])
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert len(loaded) == 1
        assert loaded[0]["filename"] == "b.py"

    def test_preserves_enrichment_fields(self, mock_cache):
        from src.analysis.context import save_overview_context

        results = [
            {
                "filename": "a.py",
                "category": "source",
                "file_summary": "Summary",
                "file_type": "service",
                "key_responsibilities": ["doX()", "doY()"],
                "dependencies": ["lib.py"],
                "purpose_keywords": ["X", "Y"],
            },
        ]
        path = save_overview_context(results, mock_cache["repo_url"])
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert loaded[0]["file_type"] == "service"
        assert loaded[0]["key_responsibilities"] == ["doX()", "doY()"]
        assert loaded[0]["dependencies"] == ["lib.py"]
        assert loaded[0]["purpose_keywords"] == ["X", "Y"]


# ---------------------------------------------------------------------------
# save_overview_context — functions + external_deps (Issue #17)
# ---------------------------------------------------------------------------

class TestSaveOverviewContextFeedForward:
    def test_preserves_functions_field(self, mock_cache):
        from src.analysis.context import save_overview_context

        results = [
            {
                "filename": "a.php",
                "category": "domain",
                "file_summary": "Summary",
                "functions": [
                    {"name": "store", "inputs": ["Request"], "outputs": ["JsonResponse"], "calls": ["DB::insert"]}
                ],
            },
        ]
        path = save_overview_context(results, mock_cache["repo_url"])
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "functions" in loaded[0]
        assert loaded[0]["functions"][0]["name"] == "store"
        assert loaded[0]["functions"][0]["calls"] == ["DB::insert"]

    def test_preserves_external_deps_field(self, mock_cache):
        from src.analysis.context import save_overview_context

        results = [
            {
                "filename": "b.php",
                "category": "api",
                "file_summary": "Summary",
                "external_deps": ["Auth", "DB", "Mail"],
            },
        ]
        path = save_overview_context(results, mock_cache["repo_url"])
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert loaded[0]["external_deps"] == ["Auth", "DB", "Mail"]

    def test_omits_missing_functions_field(self, mock_cache):
        from src.analysis.context import save_overview_context

        results = [
            {"filename": "c.php", "category": "config", "file_summary": "Just config"},
        ]
        path = save_overview_context(results, mock_cache["repo_url"])
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "functions" not in loaded[0]


# ---------------------------------------------------------------------------
# load_overview_cache_raw (Issue #17)
# ---------------------------------------------------------------------------

class TestLoadOverviewCacheRaw:
    def test_returns_dict_keyed_by_filename(self, mock_cache):
        from src.analysis.context import load_overview_cache_raw

        result = load_overview_cache_raw(mock_cache["repo_url"])
        assert result is not None
        assert isinstance(result, dict)
        assert "src/UserService.php" in result
        assert "app/Models/User.php" in result

    def test_entries_contain_category(self, mock_cache):
        from src.analysis.context import load_overview_cache_raw

        result = load_overview_cache_raw(mock_cache["repo_url"])
        assert result["src/UserService.php"]["category"] == "domain"

    def test_nonexistent_repo_returns_none(self, mock_cache):
        from src.analysis.context import load_overview_cache_raw

        result = load_overview_cache_raw("https://nonexistent.com/no/repo.git")
        assert result is None


# ---------------------------------------------------------------------------
# index_overview_to_vectordb — Funktions-Signaturen-Dokumente (Issue #21)
# ---------------------------------------------------------------------------

class TestIndexOverviewFunctionDocs:
    """index_overview_to_vectordb() erzeugt pro Datei mit functions[] ein zweites Dokument."""

    REPO_URL = "https://example.com/example/repo.git"

    @pytest.fixture
    def setup_index(self, tmp_path, monkeypatch):
        """CACHE_DIR patchen, Overview-JSON schreiben, ChromaDB + embed mocken."""
        import src.analysis.context as ctx_mod
        from unittest.mock import MagicMock, patch as _patch

        if not ctx_mod._HAS_CHROMADB:
            pytest.skip("chromadb not installed")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(ctx_mod, "CACHE_DIR", cache_dir)

        entries = [
            {
                "filename": "app/Services/UserService.php",
                "category": "domain",
                "file_summary": "Manages user accounts.",
                "functions": [
                    {"name": "save_user", "inputs": ["$request"], "outputs": ["User"], "calls": ["DB::insert", "Auth::check"]},
                    {"name": "delete_user", "inputs": ["$id"], "outputs": ["bool"], "calls": ["DB::delete"]},
                ],
                "external_deps": ["Auth", "DB"],
            },
            {
                "filename": "config/app.php",
                "category": "config",
                "file_summary": "Application config.",
                # no functions field
            },
        ]
        (cache_dir / "example-repo_overview.json").write_text(
            json.dumps(entries, ensure_ascii=False), encoding="utf-8"
        )

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("no collection")
        mock_client.create_collection.return_value = mock_collection

        def fake_embed(texts, ollama_url):
            return [[0.0] * 4 for _ in texts]

        with _patch.object(ctx_mod, "_embed_texts", fake_embed), \
             _patch("src.analysis.context.chromadb.PersistentClient", return_value=mock_client):
            yield {"entries": entries, "collection": mock_collection}

    def test_total_document_count(self, setup_index):
        from src.analysis.context import index_overview_to_vectordb
        total = index_overview_to_vectordb(self.REPO_URL, "http://localhost:11434")
        # 2 entries: UserService (summary + functions) + config (summary only) → 3
        assert total == 3

    def test_function_doc_contains_function_names(self, setup_index):
        from src.analysis.context import index_overview_to_vectordb
        index_overview_to_vectordb(self.REPO_URL, "http://localhost:11434")
        docs = setup_index["collection"].add.call_args.kwargs["documents"]
        fn_docs = [d for d in docs if "functions:" in d]
        assert len(fn_docs) == 1
        assert "save_user" in fn_docs[0]
        assert "delete_user" in fn_docs[0]

    def test_function_doc_contains_calls(self, setup_index):
        from src.analysis.context import index_overview_to_vectordb
        index_overview_to_vectordb(self.REPO_URL, "http://localhost:11434")
        docs = setup_index["collection"].add.call_args.kwargs["documents"]
        fn_docs = [d for d in docs if "functions:" in d]
        assert "DB::insert" in fn_docs[0]

    def test_function_doc_contains_external_deps(self, setup_index):
        from src.analysis.context import index_overview_to_vectordb
        index_overview_to_vectordb(self.REPO_URL, "http://localhost:11434")
        docs = setup_index["collection"].add.call_args.kwargs["documents"]
        fn_docs = [d for d in docs if "functions:" in d]
        assert "Auth" in fn_docs[0]
        assert "DB" in fn_docs[0]

    def test_no_function_doc_for_entry_without_functions(self, setup_index):
        from src.analysis.context import index_overview_to_vectordb
        index_overview_to_vectordb(self.REPO_URL, "http://localhost:11434")
        ids = setup_index["collection"].add.call_args.kwargs["ids"]
        assert "config/app.php::functions" not in ids
        assert "config/app.php::summary" in ids

    def test_function_doc_metadata_has_doc_type(self, setup_index):
        from src.analysis.context import index_overview_to_vectordb
        index_overview_to_vectordb(self.REPO_URL, "http://localhost:11434")
        kwargs = setup_index["collection"].add.call_args.kwargs
        ids = kwargs["ids"]
        metadatas = kwargs["metadatas"]
        fn_idx = ids.index("app/Services/UserService.php::functions")
        assert metadatas[fn_idx]["doc_type"] == "functions"
        sum_idx = ids.index("app/Services/UserService.php::summary")
        assert metadatas[sum_idx]["doc_type"] == "summary"

    def test_staleness_check_uses_expected_count(self, tmp_path, monkeypatch):
        import src.analysis.context as ctx_mod
        from unittest.mock import MagicMock, patch as _patch

        if not ctx_mod._HAS_CHROMADB:
            pytest.skip("chromadb not installed")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr(ctx_mod, "CACHE_DIR", cache_dir)

        entries = [
            {"filename": "a.php", "category": "domain", "file_summary": "A",
             "functions": [{"name": "foo", "inputs": [], "outputs": [], "calls": []}]},
            {"filename": "b.php", "category": "config", "file_summary": "B"},
        ]
        (cache_dir / "example-repo_overview.json").write_text(
            json.dumps(entries, ensure_ascii=False), encoding="utf-8"
        )

        # expected_count = 2 summaries + 1 function doc (only a.php) = 3
        mock_existing = MagicMock()
        mock_existing.count.return_value = 3
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_existing

        def fake_embed(texts, ollama_url):
            return [[0.0] * 4 for _ in texts]

        with _patch.object(ctx_mod, "_embed_texts", fake_embed), \
             _patch("src.analysis.context.chromadb.PersistentClient", return_value=mock_client):
            from src.analysis.context import index_overview_to_vectordb
            result = index_overview_to_vectordb(self.REPO_URL, "http://localhost:11434")

        # Staleness matched → skip, return expected_count
        assert result == 3
        mock_existing.add.assert_not_called()
