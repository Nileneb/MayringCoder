"""Tests for Phase 2 Prompt Hardening changes in src/analyzer.py:
- _parse_llm_json: delimiter parsing + backward compat
- _ollama_generate: system_prompt parameter
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# _parse_llm_json — delimiter parsing
# ---------------------------------------------------------------------------

class TestParseLlmJsonDelimiters:
    def _parse(self, text):
        from src.analysis.analyzer import _parse_llm_json
        return _parse_llm_json(text)

    def test_delimiter_primary_strategy(self):
        raw = (
            "Some preamble text that should be ignored.\n"
            "---BEGIN_JSON---\n"
            '{"file_summary": "ok", "potential_smells": []}\n'
            "---END_JSON---\n"
            "Some trailing text."
        )
        result = self._parse(raw)
        assert result is not None
        assert result["file_summary"] == "ok"
        assert result["potential_smells"] == []

    def test_delimiter_with_whitespace_around_json(self):
        raw = "---BEGIN_JSON---\n\n  {\"x\": 1}  \n\n---END_JSON---"
        result = self._parse(raw)
        assert result == {"x": 1}

    def test_delimiter_invalid_json_falls_through_to_fence(self):
        """If delimiter block contains invalid JSON, fall through to next strategy."""
        raw = (
            "---BEGIN_JSON---\nnot valid json\n---END_JSON---\n"
            '```json\n{"file_summary": "from_fence"}\n```'
        )
        result = self._parse(raw)
        assert result is not None
        assert result["file_summary"] == "from_fence"

    def test_markdown_fence_backward_compat(self):
        """Old-style responses without delimiters still parse correctly."""
        raw = '```json\n{"file_summary": "legacy", "potential_smells": []}\n```'
        result = self._parse(raw)
        assert result is not None
        assert result["file_summary"] == "legacy"

    def test_bare_json_block_backward_compat(self):
        """Bare JSON object (no fences, no delimiters) still parses correctly."""
        raw = '{"codierungen": [], "file_summary": "bare"}'
        result = self._parse(raw)
        assert result is not None
        assert result["file_summary"] == "bare"

    def test_no_json_returns_none(self):
        result = self._parse("This is just plain text with no JSON.")
        assert result is None

    def test_delimiter_takes_priority_over_fence(self):
        """When both delimiters AND fences are present, delimiters win."""
        raw = (
            "---BEGIN_JSON---\n"
            '{"source": "delimiter"}\n'
            "---END_JSON---\n"
            '```json\n{"source": "fence"}\n```'
        )
        result = self._parse(raw)
        assert result["source"] == "delimiter"


# ---------------------------------------------------------------------------
# _ollama_generate — system_prompt parameter
# ---------------------------------------------------------------------------

def _make_mock_response(text="ok"):
    line = json.dumps({"response": text, "done": True})
    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = iter([line])
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _capturing_stream(captured: dict):
    def fake_stream(method, url, json=None, timeout=None, verify=None, **kwargs):
        captured["json"] = json
        return _make_mock_response()
    return fake_stream


class TestOllamaGenerateSystemPrompt:
    def test_system_prompt_included_in_request(self):
        from src.analysis.analyzer import _ollama_generate

        captured = {}
        with patch("src.analysis.analyzer.httpx.stream", side_effect=_capturing_stream(captured)):
            _ollama_generate("user content", "http://localhost", "model", "test",
                             system_prompt="system instructions")

        assert "system" in captured["json"]
        assert captured["json"]["system"] == "system instructions"
        assert captured["json"]["prompt"] == "user content"

    def test_no_system_prompt_omits_key(self):
        from src.analysis.analyzer import _ollama_generate

        captured = {}
        with patch("src.analysis.analyzer.httpx.stream", side_effect=_capturing_stream(captured)):
            _ollama_generate("user content", "http://localhost", "model", "test")

        assert "system" not in captured["json"]

    def test_none_system_prompt_omits_key(self):
        from src.analysis.analyzer import _ollama_generate

        captured = {}
        with patch("src.analysis.analyzer.httpx.stream", side_effect=_capturing_stream(captured)):
            _ollama_generate("user content", "http://localhost", "model", "test",
                             system_prompt=None)

        assert "system" not in captured["json"]

    def test_analyze_file_sends_system_prompt(self):
        """analyze_file() passes prompt_template as system_prompt to _ollama_generate."""
        from src.analysis.analyzer import analyze_file

        captured = {}

        def fake_generate(prompt, url, model, label, *, system_prompt=None):
            captured["system_prompt"] = system_prompt
            captured["prompt"] = prompt
            return '{"file_summary": "ok", "potential_smells": []}'

        with patch("src.analysis.analyzer._ollama_generate", side_effect=fake_generate):
            file = {"filename": "app/Foo.php", "content": "<?php echo 1;", "category": "domain"}
            analyze_file(file, "SYSTEM INSTRUCTIONS", "http://localhost", "model")

        assert captured["system_prompt"] == "SYSTEM INSTRUCTIONS"
        assert "SYSTEM INSTRUCTIONS" not in captured["prompt"]
        assert "app/Foo.php" in captured["prompt"]

    def test_overview_file_sends_system_prompt(self):
        """overview_file() passes prompt_template as system_prompt."""
        from src.analysis.analyzer import overview_file

        captured = {}

        def fake_generate(prompt, url, model, label, *, system_prompt=None):
            captured["system_prompt"] = system_prompt
            captured["prompt"] = prompt
            return '{"file_summary": "summary", "file_type": "service"}'

        with patch("src.analysis.analyzer._ollama_generate", side_effect=fake_generate):
            file = {"filename": "app/Foo.php", "content": "<?php", "category": "domain"}
            overview_file(file, "OVERVIEW TEMPLATE", "http://localhost", "model")

        assert captured["system_prompt"] == "OVERVIEW TEMPLATE"
        assert "OVERVIEW TEMPLATE" not in captured["prompt"]
