"""Tests for src/ingest_github_issues.py and --ingest-issues in checker.py."""
from unittest.mock import MagicMock, patch
import json
import pytest

from src.memory_ingest import fetch_issues, issues_to_sources


# ---------------------------------------------------------------------------
# fetch_issues()
# ---------------------------------------------------------------------------

class TestFetchIssues:
    def test_returns_parsed_json_on_success(self):
        mock_issues = [
            {"number": 1, "title": "Bug", "body": "Details", "labels": [], "state": "open",
             "url": "https://github.com/x/y/issues/1", "createdAt": "2026-04-01T00:00:00Z"},
        ]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_issues)

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = fetch_issues("Nileneb/MayringCoder", state="open", limit=50)

        assert result == mock_issues
        # Verify gh CLI was called with correct flags
        call_args = mock_run.call_args[0][0]
        assert "gh" in call_args
        assert "--repo" in call_args
        assert "Nileneb/MayringCoder" in call_args
        assert "--state" in call_args
        assert "open" in call_args

    def test_returns_empty_list_on_nonzero_returncode(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = fetch_issues("Nileneb/MayringCoder")

        assert result == []

    def test_returns_empty_list_on_exception(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("gh not found")):
            result = fetch_issues("Nileneb/MayringCoder")

        assert result == []


# ---------------------------------------------------------------------------
# issues_to_sources()
# ---------------------------------------------------------------------------

class TestIssuesToSources:
    def _sample_issues(self):
        return [
            {"number": 1, "title": "First Issue", "body": "Body of first issue",
             "state": "open", "url": "https://github.com/x/y/issues/1", "createdAt": "2026-04-01"},
            {"number": 2, "title": "Second Issue", "body": None,
             "state": "closed", "url": "https://github.com/x/y/issues/2", "createdAt": "2026-04-02"},
        ]

    def test_source_type_is_github_issue(self):
        issues = self._sample_issues()
        sources = issues_to_sources(issues, "Nileneb/MayringCoder")
        assert all(s.source_type == "github_issue" for s, _ in sources)

    def test_path_format(self):
        issues = self._sample_issues()
        sources = issues_to_sources(issues, "Nileneb/MayringCoder")
        paths = [s.path for s, _ in sources]
        assert paths == ["issue/1", "issue/2"]

    def test_repo_field(self):
        issues = self._sample_issues()
        sources = issues_to_sources(issues, "Nileneb/MayringCoder")
        assert all(s.repo == "Nileneb/MayringCoder" for s, _ in sources)

    def test_content_contains_title_and_body(self):
        issues = self._sample_issues()
        sources = issues_to_sources(issues, "Nileneb/MayringCoder")
        _, content1 = sources[0]
        assert "First Issue" in content1
        assert "Body of first issue" in content1

    def test_none_body_handled(self):
        """Issue with body=None should not raise."""
        issues = self._sample_issues()
        sources = issues_to_sources(issues, "Nileneb/MayringCoder")
        _, content2 = sources[1]
        assert "Second Issue" in content2

    def test_content_hash_populated(self):
        issues = self._sample_issues()
        sources = issues_to_sources(issues, "Nileneb/MayringCoder")
        assert all(s.content_hash for s, _ in sources)

    def test_returns_correct_count(self):
        issues = self._sample_issues()
        sources = issues_to_sources(issues, "Nileneb/MayringCoder")
        assert len(sources) == 2

    def test_empty_input(self):
        sources = issues_to_sources([], "Nileneb/MayringCoder")
        assert sources == []

    def test_non_dict_entries_are_skipped(self):
        """If issues list contains non-dict entries, they are skipped silently."""
        issues = ["not a dict", None, 42, {"number": 5, "title": "Valid", "body": "ok", "state": "open"}]
        sources = issues_to_sources(issues, "Nileneb/MayringCoder")
        assert len(sources) == 1
        assert sources[0][0].path == "issue/5"

    def test_missing_number_field_is_skipped(self):
        """Issues without 'number' field are skipped."""
        issues = [{"title": "No number", "body": "body", "state": "open"}]
        sources = issues_to_sources(issues, "Nileneb/MayringCoder")
        assert sources == []


class TestFetchIssuesGuards:
    def test_non_list_json_response_returns_empty(self):
        """If gh CLI returns a JSON object instead of array, return empty list."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"error": "unexpected"})

        with patch("subprocess.run", return_value=mock_result):
            result = fetch_issues("Nileneb/MayringCoder")

        assert result == []

    def test_json_string_response_returns_empty(self):
        """If gh CLI returns a JSON string, return empty list."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps("some string")

        with patch("subprocess.run", return_value=mock_result):
            result = fetch_issues("Nileneb/MayringCoder")

        assert result == []


# ---------------------------------------------------------------------------
# _run_ingest_issues() in checker.py
# ---------------------------------------------------------------------------

class TestRunIngestIssues:
    def _make_args(self, repo="Nileneb/MayringCoder", state="open", limit=10):
        args = MagicMock()
        args.ingest_issues = repo
        args.issues_state = state
        args.issues_limit = limit
        return args

    def test_ingest_called_per_issue(self):
        from checker import _run_ingest_issues

        mock_issues = [
            {"number": 1, "title": "T1", "body": "B1", "state": "open",
             "url": "", "createdAt": "2026-04-01"},
            {"number": 2, "title": "T2", "body": "B2", "state": "open",
             "url": "", "createdAt": "2026-04-02"},
        ]

        with (
            patch("src.memory_ingest.subprocess.run",
                  return_value=MagicMock(returncode=0, stdout=json.dumps(mock_issues))),
            patch("src.memory_store.init_memory_db", return_value=MagicMock()),
            patch("src.memory_ingest.get_or_create_chroma_collection", return_value=MagicMock()),
            patch("src.memory_ingest.ingest", return_value={"chunk_ids": ["c1"], "deduped": 0}) as mock_ingest,
        ):
            _run_ingest_issues(self._make_args(), "http://localhost:11434", "mistral")

        assert mock_ingest.call_count == 2

    def test_source_type_in_ingest_call(self):
        from checker import _run_ingest_issues

        mock_issues = [
            {"number": 5, "title": "Issue", "body": "Text", "state": "open",
             "url": "", "createdAt": "2026-04-01"},
        ]
        captured = []

        def capture(source, content, conn, chroma, ollama_url, model, opts=None, workspace_id="default"):
            captured.append(source)
            return {"chunk_ids": [], "deduped": 0}

        with (
            patch("src.memory_ingest.subprocess.run",
                  return_value=MagicMock(returncode=0, stdout=json.dumps(mock_issues))),
            patch("src.memory_store.init_memory_db", return_value=MagicMock()),
            patch("src.memory_ingest.get_or_create_chroma_collection", return_value=MagicMock()),
            patch("src.memory_ingest.ingest", side_effect=capture),
        ):
            _run_ingest_issues(self._make_args(), "http://localhost:11434", "mistral")

        assert captured[0].source_type == "github_issue"
        assert captured[0].path == "issue/5"
