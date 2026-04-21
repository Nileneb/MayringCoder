"""Tests for --populate-memory mode (run_populate_memory in src/pipeline.py)."""
from unittest.mock import MagicMock, patch
import pytest

from src.pipeline import run_populate_memory


def _make_args(memory_categorize=False, codebook=None):
    args = MagicMock()
    args.memory_categorize = memory_categorize
    args.codebook = codebook
    return args


def test_populate_memory_calls_ingest_per_file(tmp_path):
    """ingest() is called once per non-excluded file."""
    files = [
        {"filename": "src/main.py", "content": "def main(): pass"},
        {"filename": "README.md", "content": "# Readme"},
    ]
    args = _make_args()
    with (
        patch("src.workflows.memory_ingest.fetch_repo", return_value=("slug", "commit", "content")),
        patch("src.workflows.memory_ingest.split_into_files", return_value=files),
        patch("src.workflows.memory_ingest.filter_excluded_files", return_value=(files, [])),
        patch("src.workflows.memory_ingest.load_codebook", return_value={}),
        patch("src.workflows.memory_ingest.load_exclude_patterns", return_value=[]),
        patch("src.workflows.memory_ingest.load_mayringignore", return_value=[]),
        patch("src.memory.store.init_memory_db", return_value=MagicMock()),
        patch("src.memory.ingest.get_or_create_chroma_collection", return_value=MagicMock()),
        patch("src.memory.ingest.ingest", return_value={"chunk_ids": ["c1"], "deduped": 0}) as mock_ingest,
    ):
        run_populate_memory(args, "https://github.com/test/repo", "http://localhost:11434", "mistral")
        assert mock_ingest.call_count == 2


def test_populate_memory_source_metadata():
    """Source object has correct source_type and repo."""
    files = [{"filename": "foo.py", "content": "x = 1"}]
    args = _make_args()
    captured_sources = []

    def capture_ingest(source, content, conn, chroma, ollama_url, model, opts=None, workspace_id="default"):
        captured_sources.append(source)
        return {"chunk_ids": ["c1"], "deduped": 0}

    with (
        patch("src.workflows.memory_ingest.fetch_repo", return_value=("slug", "commit", "content")),
        patch("src.workflows.memory_ingest.split_into_files", return_value=files),
        patch("src.workflows.memory_ingest.filter_excluded_files", return_value=(files, [])),
        patch("src.workflows.memory_ingest.load_codebook", return_value={}),
        patch("src.workflows.memory_ingest.load_exclude_patterns", return_value=[]),
        patch("src.workflows.memory_ingest.load_mayringignore", return_value=[]),
        patch("src.memory.store.init_memory_db", return_value=MagicMock()),
        patch("src.memory.ingest.get_or_create_chroma_collection", return_value=MagicMock()),
        patch("src.memory.ingest.ingest", side_effect=capture_ingest),
    ):
        run_populate_memory(args, "https://github.com/test/repo", "http://localhost:11434", "mistral")

    assert len(captured_sources) == 1
    assert captured_sources[0].source_type == "repo_file"
    assert captured_sources[0].repo == "https://github.com/test/repo"
    assert captured_sources[0].path == "foo.py"


def test_populate_memory_error_resilience():
    """A single ingest() exception does not abort the entire loop."""
    files = [
        {"filename": "a.py", "content": "x"},
        {"filename": "b.py", "content": "y"},
    ]
    args = _make_args()
    call_count = 0

    def flaky_ingest(source, content, conn, chroma, ollama_url, model, opts=None, workspace_id="default"):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated failure")
        return {"chunk_ids": ["c1"], "deduped": 0}

    with (
        patch("src.workflows.memory_ingest.fetch_repo", return_value=("slug", "commit", "content")),
        patch("src.workflows.memory_ingest.split_into_files", return_value=files),
        patch("src.workflows.memory_ingest.filter_excluded_files", return_value=(files, [])),
        patch("src.workflows.memory_ingest.load_codebook", return_value={}),
        patch("src.workflows.memory_ingest.load_exclude_patterns", return_value=[]),
        patch("src.workflows.memory_ingest.load_mayringignore", return_value=[]),
        patch("src.memory.store.init_memory_db", return_value=MagicMock()),
        patch("src.memory.ingest.get_or_create_chroma_collection", return_value=MagicMock()),
        patch("src.memory.ingest.ingest", side_effect=flaky_ingest),
    ):
        # Must not raise
        run_populate_memory(args, "https://github.com/test/repo", "http://localhost:11434", "mistral")
    assert call_count == 2  # Both files attempted
