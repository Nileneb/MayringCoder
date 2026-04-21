"""Tests for src/model_router.py."""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.model_router import ModelRouter, RouteConfig


class TestModelRouterDefaults:
    def test_all_tasks_have_routes(self):
        router = ModelRouter.__new__(ModelRouter)
        router._routes = {}
        router._availability_cache = {}
        router._cache_ttl = 30.0
        router._ollama_url = "http://localhost:11434"
        # Load defaults via __init__ without YAML
        with patch("src.model_router._CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = False
            router2 = ModelRouter("http://localhost:11434")
        for task in ModelRouter.TASKS:
            assert task in router2._routes

    def test_resolve_vision_returns_configured_model(self):
        with patch("src.model_router._CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = False
            router = ModelRouter("http://localhost:11434")
        assert router.resolve("vision") == "qwen2.5vl:3b"

    def test_resolve_empty_model_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "my-env-model:latest")
        with patch("src.model_router._CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = False
            router = ModelRouter("http://localhost:11434")
        # analysis model is "" by default → should return env var
        result = router.resolve("analysis")
        assert result == "my-env-model:latest"

    def test_resolve_unknown_task_returns_env_model(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "fallback-model")
        with patch("src.model_router._CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = False
            router = ModelRouter("http://localhost:11434")
        assert router.resolve("nonexistent_task") == "fallback-model"


class TestModelRouterAvailability:
    def _make_router(self) -> ModelRouter:
        with patch("src.model_router._CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = False
            return ModelRouter("http://localhost:11434")

    def test_is_available_true_when_model_in_ollama(self):
        router = self._make_router()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "qwen2.5vl:3b"}, {"name": "nomic-embed-text:latest"}]
        }
        with patch("httpx.get", return_value=mock_resp):
            assert router.is_available("vision") is True

    def test_is_available_false_when_model_not_in_ollama(self):
        router = self._make_router()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
        with patch("httpx.get", return_value=mock_resp):
            assert router.is_available("vision") is False

    def test_is_available_false_on_network_error(self):
        router = self._make_router()
        with patch("httpx.get", side_effect=Exception("connection refused")):
            assert router.is_available("vision") is False

    def test_availability_is_cached(self):
        router = self._make_router()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "qwen2.5vl:3b"}]}
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            router.is_available("vision")
            router.is_available("vision")  # second call — should use cache
        # httpx.get should only be called once
        assert mock_get.call_count == 1

    def test_is_available_false_for_empty_model(self, monkeypatch):
        # Test-Isolation: verhindert dass eine lokale .env (via load_dotenv in
        # src.api.server) OLLAMA_MODEL pollutet — is_available würde sonst das
        # echte lokale Ollama abfragen und True zurückgeben.
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        with patch("src.model_router._CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = False
            router = ModelRouter("http://localhost:11434")
        router._routes["analysis"].model = ""
        assert router.is_available("analysis") is False


class TestModelRouterConfig:
    def test_load_config_from_yaml(self, tmp_path):
        yaml_content = """
mayring_code:
  model: "custom-model:v2"
  fallback: "llama3.1:8b"
  timeout: 300
"""
        cfg_file = tmp_path / "routes.yaml"
        cfg_file.write_text(yaml_content)

        with patch("src.model_router._CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = False
            router = ModelRouter("http://localhost:11434")

        router.load_config(cfg_file)
        assert router.resolve("mayring_code") == "custom-model:v2"
        assert router._routes["mayring_code"].timeout == 300

    def test_to_dict_contains_all_tasks(self):
        with patch("src.model_router._CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = False
            router = ModelRouter("http://localhost:11434")
        d = router.to_dict()
        for task in ModelRouter.TASKS:
            assert task in d
            assert "model" in d[task]
            assert "fallback" in d[task]

    def test_set_route_updates_model(self):
        with patch("src.model_router._CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = False
            router = ModelRouter("http://localhost:11434")
        router.set_route("vision", "llava:7b", fallback="")
        assert router.resolve("vision") == "llava:7b"
