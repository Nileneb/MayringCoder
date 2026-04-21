"""Pi-Agent: optionaler endpoint-Parameter überschreibt ollama_url + model."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.pi import _resolve_ollama_compatible, run_task_with_memory
from src.llm.endpoint import LLMEndpoint


def test_resolve_ollama_compatible_unpacks_ollama_endpoint():
    endpoint = LLMEndpoint(provider="ollama", base_url="http://three.linn.games:11434", model="qwen3:30b")
    assert _resolve_ollama_compatible(endpoint) == ("http://three.linn.games:11434", "qwen3:30b")


def test_resolve_ollama_compatible_accepts_platform():
    endpoint = LLMEndpoint(provider="platform", base_url="http://localhost:11434", model="llama3.2")
    assert _resolve_ollama_compatible(endpoint) == ("http://localhost:11434", "llama3.2")


def test_resolve_ollama_compatible_accepts_openai():
    endpoint = LLMEndpoint(provider="openai", base_url="https://openrouter.ai/api", model="gpt-4o-mini")
    assert _resolve_ollama_compatible(endpoint) == ("https://openrouter.ai/api", "gpt-4o-mini")


def test_resolve_ollama_compatible_rejects_anthropic():
    endpoint = LLMEndpoint(
        provider="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-6",
        api_key="sk-ant-test",
    )
    with pytest.raises(NotImplementedError, match="anthropic"):
        _resolve_ollama_compatible(endpoint)


def test_run_task_passes_endpoint_url_and_model_to_agent_loop():
    """The endpoint wins over explicit ollama_url+model parameters."""
    endpoint = LLMEndpoint(provider="ollama", base_url="http://user-ollama:11434", model="custom-model")

    with patch("src.agents.pi._agent_loop") as mock_loop, \
         patch("src.agents.pi.init_memory_db"), \
         patch("src.agents.pi.get_or_create_chroma_collection"):
        mock_loop.return_value = ("some result text", 0)

        run_task_with_memory(
            task="Test task",
            ollama_url="http://default:11434",
            model="default-model",
            repo_slug="repo-x",
            endpoint=endpoint,
        )

    assert mock_loop.called
    kwargs = mock_loop.call_args.kwargs
    assert kwargs["ollama_url"] == "http://user-ollama:11434"
    assert kwargs["model"] == "custom-model"


def test_run_task_without_endpoint_uses_default_args():
    """Backward compat: callers not passing endpoint still use ollama_url+model."""
    with patch("src.agents.pi._agent_loop") as mock_loop, \
         patch("src.agents.pi.init_memory_db"), \
         patch("src.agents.pi.get_or_create_chroma_collection"):
        mock_loop.return_value = ("ok", 0)

        run_task_with_memory(
            task="Test",
            ollama_url="http://platform:11434",
            model="platform-model",
            repo_slug="repo-x",
        )

    kwargs = mock_loop.call_args.kwargs
    assert kwargs["ollama_url"] == "http://platform:11434"
    assert kwargs["model"] == "platform-model"
