"""Tests für Pi-Agent (src/pi_agent.py) mit gemockten HTTP und Memory-Calls."""
import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.pi import analyze_with_memory, run_task_with_memory, _execute_search_memory


class TestAnalyzeWithMemory:
    def _mock_ollama_final(self, content: str):
        """Mock Response ohne tool_calls (final answer)."""
        return MagicMock(
            status_code=200,
            json=lambda: {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": None,
                }
            },
        )

    def _mock_ollama_tool_call(self, query: str, call_id: str = "call_1"):
        """Mock Response mit einem search_memory tool_call."""
        return MagicMock(
            status_code=200,
            json=lambda: {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {
                                "name": "search_memory",
                                "arguments": {"query": query, "top_k": 5},
                            },
                        }
                    ],
                }
            },
        )

    @patch("src.agents.pi.init_memory_db")
    @patch("src.agents.pi.get_or_create_chroma_collection")
    @patch("httpx.post")
    def test_direct_json_response(self, mock_post, mock_chroma, mock_db):
        """Model antwortet direkt ohne Tool-Call."""
        mock_db.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        expected = {"file_summary": "Standard Laravel entry point.", "potential_smells": []}
        mock_post.return_value = self._mock_ollama_final(json.dumps(expected))

        result = analyze_with_memory(
            {"filename": "artisan", "content": "#!/usr/bin/env php", "category": "entrypoint"},
            "http://localhost:11434",
            "qwen3.5:2b",
        )

        assert result["file_summary"] == "Standard Laravel entry point."
        assert result["potential_smells"] == []
        assert result["_pi_tool_calls"] == 0

    @patch("src.agents.pi._execute_search_memory")
    @patch("src.agents.pi.init_memory_db")
    @patch("src.agents.pi.get_or_create_chroma_collection")
    @patch("httpx.post")
    def test_tool_call_then_final(self, mock_post, mock_chroma, mock_db, mock_search):
        """Model ruft search_memory auf, dann final response."""
        mock_db.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        mock_search.return_value = "## Memory Context\n- Laravel artisan is the CLI entry point."

        final_content = json.dumps({"file_summary": "Entry point.", "potential_smells": []})
        mock_post.side_effect = [
            self._mock_ollama_tool_call("artisan Laravel convention"),
            self._mock_ollama_final(final_content),
        ]

        result = analyze_with_memory(
            {"filename": "artisan", "content": "#!/usr/bin/env php", "category": "entrypoint"},
            "http://localhost:11434",
            "qwen3.5:2b",
        )

        assert mock_search.called
        assert result["_pi_tool_calls"] == 1
        assert result["potential_smells"] == []

    @patch("src.agents.pi._execute_search_memory")
    @patch("src.agents.pi.init_memory_db")
    @patch("src.agents.pi.get_or_create_chroma_collection")
    @patch("httpx.post")
    def test_max_tool_calls_limit(self, mock_post, mock_chroma, mock_db, mock_search):
        """Nach max_tool_calls wird kein weiteres Tool aufgerufen."""
        mock_db.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        mock_search.return_value = "some context"

        final_json = json.dumps({"file_summary": "Done.", "potential_smells": []})
        # 2 tool calls, dann final
        mock_post.side_effect = [
            self._mock_ollama_tool_call("query 1", "c1"),
            self._mock_ollama_tool_call("query 2", "c2"),
            self._mock_ollama_final(final_json),
        ]

        result = analyze_with_memory(
            {"filename": "test.php", "content": "<?php echo 'hi';", "category": "api"},
            "http://localhost:11434",
            "qwen3.5:2b",
            max_tool_calls=2,
        )

        assert mock_search.call_count == 2
        assert result["_pi_tool_calls"] == 2

    @patch("src.agents.pi.init_memory_db")
    @patch("src.agents.pi.get_or_create_chroma_collection")
    @patch("httpx.post")
    def test_parse_error_fallback(self, mock_post, mock_chroma, mock_db):
        """Kein valides JSON → _parse_error=True, leere findings."""
        mock_db.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        mock_post.return_value = self._mock_ollama_final("Das ist kein JSON.")

        result = analyze_with_memory(
            {"filename": "foo.py", "content": "x = 1", "category": "utils"},
            "http://localhost:11434",
            "qwen3.5:2b",
        )

        assert result["_parse_error"] is True
        assert result["potential_smells"] == []

    @patch("src.agents.pi.init_memory_db")
    @patch("src.agents.pi.get_or_create_chroma_collection")
    @patch("httpx.post")
    def test_http_error_returns_error_dict(self, mock_post, mock_chroma, mock_db):
        """HTTP-Fehler → error-Dict zurückgeben, kein Crash."""
        mock_db.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        mock_post.side_effect = Exception("Connection refused")

        result = analyze_with_memory(
            {"filename": "foo.py", "content": "x = 1", "category": "utils"},
            "http://localhost:11434",
            "qwen3.5:2b",
        )

        assert "error" in result
        assert result["_parse_error"] is True


class TestRunTaskWithMemory:
    def _mock_final(self, content: str):
        return MagicMock(
            status_code=200,
            json=lambda: {"message": {"role": "assistant", "content": content, "tool_calls": None}},
        )

    def _mock_tool_call(self, query: str):
        return MagicMock(
            status_code=200,
            json=lambda: {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "function": {"name": "search_memory", "arguments": {"query": query}}}],
                }
            },
        )

    @patch("src.agents.pi.init_memory_db")
    @patch("src.agents.pi.get_or_create_chroma_collection")
    @patch("httpx.post")
    def test_returns_plain_text(self, mock_post, mock_chroma, mock_db):
        """run_task_with_memory gibt Freitext zurück (kein JSON-Zwang)."""
        mock_db.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        mock_post.return_value = self._mock_final("## PICO-Suchterms\n- Population: Erwachsene\n- Intervention: Yoga")

        result = run_task_with_memory(
            "Entwickle PICO-Suchterms für Yoga-Studien",
            "http://localhost:11434",
            "qwen3.5:2b",
        )

        assert isinstance(result, str)
        assert "PICO" in result or "Yoga" in result

    @patch("src.agents.pi._execute_search_memory")
    @patch("src.agents.pi.init_memory_db")
    @patch("src.agents.pi.get_or_create_chroma_collection")
    @patch("httpx.post")
    def test_uses_memory_tool(self, mock_post, mock_chroma, mock_db, mock_search):
        """Pi ruft search_memory auf und injiziert Ergebnis."""
        mock_db.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        mock_search.return_value = "## Memory\nProjekt nutzt Laravel-Konventionen."

        mock_post.side_effect = [
            self._mock_tool_call("Laravel Konventionen"),
            self._mock_final("Suchterm: Laravel artisan pattern"),
        ]

        result = run_task_with_memory(
            "Welche Frameworks nutzt das Projekt?",
            "http://localhost:11434",
            "qwen3.5:2b",
        )

        assert mock_search.called
        assert isinstance(result, str)

    @patch("src.agents.pi.init_memory_db")
    @patch("src.agents.pi.get_or_create_chroma_collection")
    @patch("httpx.post")
    def test_http_error_returns_error_string(self, mock_post, mock_chroma, mock_db):
        """HTTP-Fehler → Fehler-String, kein Crash."""
        mock_db.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        mock_post.side_effect = Exception("Connection refused")

        result = run_task_with_memory(
            "Irgendein Auftrag",
            "http://localhost:11434",
            "qwen3.5:2b",
        )

        assert isinstance(result, str)
        assert "Fehler" in result or "Error" in result or "error" in result.lower()


class TestExecuteSearchMemory:
    def test_returns_string(self):
        """_execute_search_memory gibt immer einen String zurück."""
        mock_conn = MagicMock()
        mock_chroma = MagicMock()

        with patch("src.agents.pi.search") as mock_search, \
             patch("src.agents.pi.compress_for_prompt") as mock_compress:
            mock_search.return_value = []
            mock_compress.return_value = ""

            result = _execute_search_memory("test query", 5, mock_conn, mock_chroma, "http://localhost:11434", None)

        assert isinstance(result, str)
