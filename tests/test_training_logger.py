"""Tests for the training-data logger in src.analysis.analyzer."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_logger():
    """Reset module-level logger state between tests."""
    import src.analysis.analyzer as mod
    mod._training_log_path = None
    mod._training_run_id = "default"


def _read_log(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# configure_training_log
# ---------------------------------------------------------------------------

class TestConfigureTrainingLog:
    def test_sets_path_and_run_id(self, tmp_path):
        import src.analysis.analyzer as mod
        _reset_logger()
        from src.analysis.analyzer import configure_training_log
        configure_training_log(tmp_path / "log.jsonl", run_id="bench_test")
        assert mod._training_log_path == tmp_path / "log.jsonl"
        assert mod._training_run_id == "bench_test"
        _reset_logger()

    def test_creates_parent_dir(self, tmp_path):
        from src.analysis.analyzer import configure_training_log
        _reset_logger()
        deep = tmp_path / "a" / "b" / "log.jsonl"
        configure_training_log(deep)
        assert deep.parent.exists()
        _reset_logger()


# ---------------------------------------------------------------------------
# _log_training_entry
# ---------------------------------------------------------------------------

class TestLogTrainingEntry:
    def test_no_op_when_not_configured(self, tmp_path):
        _reset_logger()
        from src.analysis.analyzer import _log_training_entry
        # Should not raise and should not create any file
        _log_training_entry("m", "f.py", "prompt", "response", True, 3)

    def test_appends_jsonl_entry(self, tmp_path):
        from src.analysis.analyzer import configure_training_log, _log_training_entry
        _reset_logger()
        log = tmp_path / "log.jsonl"
        configure_training_log(log, run_id="r1")
        _log_training_entry("qwen:7b", "app/Foo.php", "prompt text", "response text", True, 2)
        entries = _read_log(log)
        assert len(entries) == 1
        e = entries[0]
        assert e["model"] == "qwen:7b"
        assert e["label"] == "app/Foo.php"
        assert e["run_id"] == "r1"
        assert e["parsed_ok"] is True
        assert e["findings_count"] == 2
        assert e["prompt"] == "prompt text"
        assert e["raw_response"] == "response text"
        assert "timestamp" in e
        assert "prompt_hash" in e
        _reset_logger()

    def test_call_type_stored(self, tmp_path):
        from src.analysis.analyzer import configure_training_log, _log_training_entry
        _reset_logger()
        log = tmp_path / "log.jsonl"
        configure_training_log(log)
        _log_training_entry("m", "f.py", "p", "r", True, 0, call_type="overview")
        entries = _read_log(log)
        assert entries[0]["call_type"] == "overview"
        _reset_logger()

    def test_multiple_entries_appended(self, tmp_path):
        from src.analysis.analyzer import configure_training_log, _log_training_entry
        _reset_logger()
        log = tmp_path / "log.jsonl"
        configure_training_log(log)
        for i in range(5):
            _log_training_entry("m", f"f{i}.py", "p", "r", True, i)
        entries = _read_log(log)
        assert len(entries) == 5
        assert [e["findings_count"] for e in entries] == [0, 1, 2, 3, 4]
        _reset_logger()

    def test_prompt_hash_is_16_chars(self, tmp_path):
        from src.analysis.analyzer import configure_training_log, _log_training_entry
        _reset_logger()
        log = tmp_path / "log.jsonl"
        configure_training_log(log)
        _log_training_entry("m", "f.py", "some prompt", "resp", False, 0)
        e = _read_log(log)[0]
        assert len(e["prompt_hash"]) == 16
        _reset_logger()


# ---------------------------------------------------------------------------
# Integration: analyze_file() writes to log
# ---------------------------------------------------------------------------

class TestAnalyzeFileLogging:
    def _make_prompt(self, tmp_path):
        p = tmp_path / "prompt.md"
        p.write_text("Analyze this.", encoding="utf-8")
        return p

    def test_analyze_file_logs_entry(self, tmp_path):
        from src.analysis.analyzer import configure_training_log, analyze_file, _load_prompt
        _reset_logger()
        log = tmp_path / "log.jsonl"
        configure_training_log(log, run_id="test_run")

        prompt_template = "Analyze this."
        file = {"filename": "app/Foo.php", "content": "<?php echo 'hi';", "category": "domain"}

        with patch("src.analysis.analyzer._ollama_generate",
                   return_value='{"file_summary": "ok", "potential_smells": []}'):
            analyze_file(file, prompt_template, "http://localhost", "model")

        entries = _read_log(log)
        assert len(entries) == 1
        assert entries[0]["label"] == "app/Foo.php"
        assert entries[0]["call_type"] == "analyze"
        _reset_logger()

    def test_parsed_ok_false_when_no_json(self, tmp_path):
        from src.analysis.analyzer import configure_training_log, analyze_file
        _reset_logger()
        log = tmp_path / "log.jsonl"
        configure_training_log(log)

        file = {"filename": "f.php", "content": "x", "category": "domain"}

        with patch("src.analysis.analyzer._ollama_generate", return_value="plain text no json"):
            analyze_file(file, "Analyze.", "http://localhost", "model")

        entries = [e for e in _read_log(log) if e["call_type"] == "analyze"]
        assert entries[-1]["parsed_ok"] is False
        _reset_logger()

    def test_no_log_when_not_configured(self, tmp_path):
        from src.analysis.analyzer import analyze_file
        _reset_logger()

        file = {"filename": "f.php", "content": "x", "category": "domain"}
        with patch("src.analysis.analyzer._ollama_generate",
                   return_value='{"potential_smells": []}'):
            analyze_file(file, "Analyze.", "http://localhost", "model")
        # No crash, no file created
        assert not any(tmp_path.glob("*.jsonl"))
        _reset_logger()
