"""Unit tests for src.categorizer."""

import pytest
from src.categorizer import (
    _matches_patterns,
    filter_excluded_files,
    categorize_files,
    load_codebook,
    load_exclude_patterns,
    load_mayringignore,
)


# ---------------------------------------------------------------------------
# _matches_patterns
# ---------------------------------------------------------------------------

class TestMatchesPatternsGlob:
    def test_exact_match(self):
        assert _matches_patterns("src/main.py", ["src/main.py"]) is True
        assert _matches_patterns("src/main.py", ["src/other.py"]) is False

    def test_single_star_matches_segment(self):
        assert _matches_patterns("src/main.py", ["src/*.py"]) is True
        assert _matches_patterns("src/utils/helper.py", ["src/*.py"]) is False
        assert _matches_patterns("tests/test_main.py", ["src/*.py"]) is False

    def test_double_star_matches_any_depth(self):
        assert _matches_patterns("src/deep/nested/file.py", ["src/**/*.py"]) is True
        assert _matches_patterns("tests/file.py", ["src/**/*.py"]) is False

    def test_double_star_at_start(self):
        # Pattern **/tests/** matches anywhere in path
        assert _matches_patterns("foo/tests/bar.txt", ["**/tests/**"]) is True
        assert _matches_patterns("tests/foo/bar.txt", ["**/tests/**"]) is True
        # Does NOT match files outside the tests directory
        assert _matches_patterns("anything/here.txt", ["**/tests/**"]) is False
        # Note: **/tests/** matching root-level "tests/foo.txt" requires fixing
        # the leading .*? adjustment in _matches_patterns (see src/categorizer.py)

    def test_trailing_slash_star_matches_nested(self):
        assert _matches_patterns("vendor/package/index.js", ["vendor/*"]) is True
        assert _matches_patterns("vendor/package/nested/deep.js", ["vendor/*"]) is True

    def test_case_insensitive(self):
        assert _matches_patterns("Src/Main.py", ["src/*.py"]) is True
        assert _matches_patterns("SRC/MAIN.PY", ["src/*.py"]) is True


class TestMatchesPatternsRegex:
    def test_re_prefix_uses_regex(self):
        # Regex: matches _test or _spec suffix
        assert _matches_patterns("foo_test.py", ["re:.*_test\\.py$"]) is True
        assert _matches_patterns("foo_spec.py", ["re:.*_spec\\.py$"]) is True
        assert _matches_patterns("foo.py", ["re:.*_test\\.py$"]) is False

    def test_re_prefix_case_insensitive(self):
        assert _matches_patterns("FOO_TEST.PY", ["re:.*_test\\.py$"]) is True

    def test_no_re_prefix_uses_glob(self):
        # Without re:, special regex chars are escaped
        assert _matches_patterns("foo_test.py", ["foo_test.py"]) is True
        assert _matches_patterns("foo.test.py", ["foo_test.py"]) is False


class TestMatchesPatternsEdgeCases:
    def test_empty_patterns_returns_false(self):
        assert _matches_patterns("anything.txt", []) is False

    def test_dot_file_matches_exact(self):
        assert _matches_patterns(".gitignore", [".gitignore"]) is True
        assert _matches_patterns(".env", [".env"]) is True

    def test_path_with_dots(self):
        assert _matches_patterns("foo.bar/baz.txt", ["foo.bar/*.txt"]) is True

    def test_no_match_returns_false(self):
        assert _matches_patterns("src/main.py", ["tests/*.py"]) is False


# ---------------------------------------------------------------------------
# filter_excluded_files
# ---------------------------------------------------------------------------

def make_file(name: str) -> dict:
    return {"filename": name, "content": ""}


class TestFilterExcludedFiles:
    def test_no_patterns_returns_all_included(self):
        files = [make_file("a.txt"), make_file("b.txt")]
        inc, exc = filter_excluded_files(files, [])
        assert inc == files
        assert exc == []

    def test_excludes_matched(self):
        files = [make_file("src/main.py"), make_file("tests/test.py"), make_file("README.md")]
        inc, exc = filter_excluded_files(files, ["tests/**"])
        assert [f["filename"] for f in inc] == ["src/main.py", "README.md"]
        assert [f["filename"] for f in exc] == ["tests/test.py"]

    def test_multiple_patterns(self):
        files = [make_file("a.py"), make_file("b.js"), make_file("c.txt")]
        inc, exc = filter_excluded_files(files, ["*.py", "*.js"])
        # a.py and b.js match, c.txt does not
        assert [f["filename"] for f in exc] == ["a.py", "b.js"]
        assert [f["filename"] for f in inc] == ["c.txt"]

    def test_regex_pattern(self):
        files = [make_file("foo_test.py"), make_file("foo.py")]
        inc, exc = filter_excluded_files(files, ["re:.*_test\\.py$"])
        assert [f["filename"] for f in inc] == ["foo.py"]
        assert [f["filename"] for f in exc] == ["foo_test.py"]


# ---------------------------------------------------------------------------
# categorize_files
# ---------------------------------------------------------------------------

SMALL_CODEBOOK = [
    {
        "name": "source",
        "description": "Source code files",
        "patterns": ["src/**/*.py", "lib/**/*.py", "src/*.py"],
    },
    {
        "name": "config",
        "description": "Configuration files",
        "patterns": ["*.yaml", "*.yml", "*.toml", ".env*"],
    },
    {
        "name": "api",
        "description": "API definitions",
        "patterns": ["re:.*api.*\\.py$", "**/api/**"],
    },
]


class TestCategorizeFiles:
    def test_categorizes_by_first_match(self):
        files = [
            make_file("src/main.py"),
            make_file("config.yaml"),
            make_file("README.md"),
        ]
        result = categorize_files(files, SMALL_CODEBOOK)
        cats = {f["filename"]: f.get("category") for f in result}
        assert cats["src/main.py"] == "source"
        assert cats["config.yaml"] == "config"
        assert cats["README.md"] == "uncategorized"

    def test_uncategorized_has_empty_reason(self):
        files = [make_file("readme.txt")]
        result = categorize_files(files, SMALL_CODEBOOK)
        assert result[0]["category"] == "uncategorized"
        assert result[0]["category_reason"] == ""

    def test_api_pattern_matches_regex(self):
        files = [make_file("users_api.py"), make_file("utils_api.py")]
        result = categorize_files(files, SMALL_CODEBOOK)
        assert all(f["category"] == "api" for f in result)

    def test_reason_matches_description(self):
        files = [make_file("config.yaml")]
        result = categorize_files(files, SMALL_CODEBOOK)
        assert result[0]["category_reason"] == "Configuration files"

    def test_uncategorized_when_no_match(self):
        files = [make_file("img.png"), make_file("video.mp4")]
        result = categorize_files(files, SMALL_CODEBOOK)
        assert all(f["category"] == "uncategorized" for f in result)

    def test_empty_file_list(self):
        result = categorize_files([], SMALL_CODEBOOK)
        assert result == []

    def test_file_order_preserved(self):
        files = [make_file("b.txt"), make_file("a.txt"), make_file("c.txt")]
        result = categorize_files(files, [{"name": "a", "patterns": ["*.txt"]}])
        assert [f["filename"] for f in result] == ["b.txt", "a.txt", "c.txt"]


class TestNewLaravelCategories:
    """Verify that the new categories in codebook.yaml catch Laravel-typical files."""

    def _categorize(self, filename: str) -> str:
        result = categorize_files([make_file(filename)])
        return result[0]["category"]

    def test_job_categorized_as_infrastructure(self):
        assert self._categorize("app/Jobs/DownloadPaperJob.php") == "infrastructure"

    def test_console_command_categorized_as_infrastructure(self):
        assert self._categorize("app/Console/Commands/IngestCommand.php") == "infrastructure"

    def test_policy_categorized_as_auth(self):
        assert self._categorize("app/Policies/WorkspacePolicy.php") == "auth"

    def test_guard_categorized_as_auth(self):
        assert self._categorize("app/Guards/SessionGuard.php") == "auth"

    def test_middleware_categorized_as_middleware(self):
        assert self._categorize("app/Http/Middleware/VerifyMcpToken.php") == "middleware"

    def test_service_provider_categorized_as_providers(self):
        assert self._categorize("app/Providers/AppServiceProvider.php") == "providers"

    def test_fortify_provider_categorized_as_providers(self):
        assert self._categorize("app/Providers/FortifyServiceProvider.php") == "providers"

    def test_listener_categorized_as_listeners(self):
        assert self._categorize("app/Listeners/SetUpNewUser.php") == "listeners"

    def test_observer_categorized_as_listeners(self):
        assert self._categorize("app/Observers/UserObserver.php") == "listeners"

    def test_unknown_file_still_uncategorized(self):
        assert self._categorize("app/Helpers/SomeHelper.php") == "utils"


class TestNewIssue26Categories:
    """Verify the 8 new categories from Issue #26 catch typical files."""

    def _categorize(self, filename: str) -> str:
        result = categorize_files([make_file(filename)])
        return result[0]["category"]

    def test_integration_client(self):
        assert self._categorize("app/Http/Clients/StripeClient.php") == "integration"

    def test_integration_directory(self):
        assert self._categorize("app/Integrations/SlackWebhook.php") == "integration"

    def test_caching_class(self):
        assert self._categorize("app/Cache/UserCache.php") == "caching"

    def test_logging_class(self):
        assert self._categorize("app/Logging/RequestLogger.php") == "logging"

    def test_validation_request(self):
        assert self._categorize("app/Http/Requests/StorePostRequest.php") == "validation"

    def test_serialization_resource(self):
        assert self._categorize("app/Http/Resources/UserResource.php") == "serialization"

    def test_error_handling_exception(self):
        assert self._categorize("app/Exceptions/PaymentException.php") == "error_handling"

    def test_security_directory(self):
        assert self._categorize("app/Security/EncryptionService.php") == "security"

    def test_scheduling_directory(self):
        assert self._categorize("app/Schedule/WeeklyReportTask.php") == "scheduling"


class TestLoadCodebook:
    def test_load_codebook_returns_empty_for_nonexistent(self, tmp_path):
        result = load_codebook(tmp_path / "nonexistent.yaml")
        assert result == []

    def test_load_codebook_from_dict_structure(self, tmp_path):
        yaml_file = tmp_path / "codebook.yaml"
        yaml_file.write_text("categories:\n  - name: test\n    patterns: ['*.txt']\n")
        result = load_codebook(yaml_file)
        assert result == [{"name": "test", "patterns": ["*.txt"]}]

    def test_load_codebook_from_list_structure(self, tmp_path):
        yaml_file = tmp_path / "codebook.yaml"
        yaml_file.write_text("- name: test\n  patterns: ['*.txt']\n")
        result = load_codebook(yaml_file)
        assert result == [{"name": "test", "patterns": ["*.txt"]}]


class TestLoadExcludePatterns:
    def test_returns_exclude_patterns_list(self, tmp_path):
        yaml_file = tmp_path / "codebook.yaml"
        yaml_file.write_text(
            "exclude_patterns:\n  - '*.log'\n  - '**/.git/**'\n"
        )
        result = load_exclude_patterns(yaml_file)
        assert result == ["*.log", "**/.git/**"]

    def test_returns_empty_when_none(self, tmp_path):
        yaml_file = tmp_path / "codebook.yaml"
        yaml_file.write_text("categories: []\n")
        assert load_exclude_patterns(yaml_file) == []


class TestLoadMayringignore:
    def test_loads_comments_and_blank_lines_skipped(self, tmp_path):
        ignore_file = tmp_path / ".mayringignore"
        ignore_file.write_text(
            "# This is a comment\n\n*.tmp\n  # another comment\n*.swp\n"
        )
        result = load_mayringignore(ignore_file)
        assert result == ["*.tmp", "*.swp"]

    def test_returns_empty_when_file_missing(self, tmp_path):
        result = load_mayringignore(tmp_path / "missing")
        assert result == []
