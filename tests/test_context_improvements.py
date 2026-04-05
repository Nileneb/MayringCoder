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

    with patch("src.context.CACHE_DIR", cache_dir):
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
        from src.context import build_inventory_context

        ctx = build_inventory_context(mock_cache["repo_url"])
        assert ctx is not None
        assert isinstance(ctx, str)

    def test_includes_file_types(self, mock_cache):
        from src.context import build_inventory_context

        ctx = build_inventory_context(mock_cache["repo_url"])
        assert "type=service" in ctx
        assert "type=model" in ctx
        assert "type=test" in ctx

    def test_includes_responsibilities(self, mock_cache):
        from src.context import build_inventory_context

        ctx = build_inventory_context(mock_cache["repo_url"])
        assert "register()" in ctx
        assert "updateProfile()" in ctx

    def test_truncates_long_summaries(self, mock_cache):
        from src.context import build_inventory_context

        ctx = build_inventory_context(mock_cache["repo_url"])
        # Summary for UserService is >100 chars so should be truncated
        assert "..." in ctx or len(ctx) < 5000

    def test_nonexistent_repo_returns_none(self, mock_cache):
        from src.context import build_inventory_context

        ctx = build_inventory_context("https://nonexistent.com/does/not/exist.git")
        assert ctx is None

    def test_includes_all_files(self, mock_cache):
        from src.context import build_inventory_context

        ctx = build_inventory_context(mock_cache["repo_url"])
        assert "UserService.php" in ctx
        assert "User.php" in ctx
        assert "UserServiceTest.php" in ctx


# ---------------------------------------------------------------------------
# build_dependency_context
# ---------------------------------------------------------------------------

class TestBuildDependencyContext:
    def test_returns_context_for_known_file(self, mock_cache):
        from src.context import build_dependency_context

        ctx = build_dependency_context(
            mock_cache["repo_url"],
            {"filename": "src/UserService.php"},
        )
        assert ctx is not None
        assert "referenzierte Dateien" in ctx.lower() or "UserService.php" in ctx

    def test_includes_self_entry(self, mock_cache):
        from src.context import build_dependency_context

        ctx = build_dependency_context(
            mock_cache["repo_url"],
            {"filename": "src/UserService.php"},
        )
        assert "UserService.php" in ctx

    def test_includes_dependency_summaries(self, mock_cache):
        from src.context import build_dependency_context

        ctx = build_dependency_context(
            mock_cache["repo_url"],
            {"filename": "src/UserService.php"},
        )
        # Should reference the User model since it's in dependencies
        assert "User" in ctx

    def test_unknown_file_returns_none(self, mock_cache):
        from src.context import build_dependency_context

        ctx = build_dependency_context(
            mock_cache["repo_url"],
            {"filename": "nonexistent/file.php"},
        )
        assert ctx is None

    def test_nonexistent_repo_returns_none(self, mock_cache):
        from src.context import build_dependency_context

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
        from src.context import save_overview_context

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
        from src.context import save_overview_context

        results = [
            {"filename": "a.py", "error": "Timeout"},
            {"filename": "b.py", "category": "source", "file_summary": "OK"},
        ]
        path = save_overview_context(results, mock_cache["repo_url"])
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert len(loaded) == 1
        assert loaded[0]["filename"] == "b.py"

    def test_preserves_enrichment_fields(self, mock_cache):
        from src.context import save_overview_context

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
        from src.context import save_overview_context

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
        from src.context import save_overview_context

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
        from src.context import save_overview_context

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
        from src.context import load_overview_cache_raw

        result = load_overview_cache_raw(mock_cache["repo_url"])
        assert result is not None
        assert isinstance(result, dict)
        assert "src/UserService.php" in result
        assert "app/Models/User.php" in result

    def test_entries_contain_category(self, mock_cache):
        from src.context import load_overview_cache_raw

        result = load_overview_cache_raw(mock_cache["repo_url"])
        assert result["src/UserService.php"]["category"] == "domain"

    def test_nonexistent_repo_returns_none(self, mock_cache):
        from src.context import load_overview_cache_raw

        result = load_overview_cache_raw("https://nonexistent.com/no/repo.git")
        assert result is None
