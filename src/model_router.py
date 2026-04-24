"""Centralized model routing for MayringCoder tasks.

Maps task names to Ollama model names with fallback and availability caching.
Config loaded from config/model_routes.yaml (optional — silent defaults if missing).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import httpx as _httpx
except ImportError:
    _httpx = None  # type: ignore

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config" / "model_routes.yaml"

_DEFAULTS: dict[str, dict] = {
    "mayring_code":   {"model": "mayringqwen:latest", "fallback": "llama3.1:8b",       "timeout": 240},
    "mayring_social": {"model": "llama3.1:8b",        "fallback": "mistral:7b-instruct","timeout": 240},
    "mayring_hybrid": {"model": "llama3.1:8b",        "fallback": "mistral:7b-instruct","timeout": 240},
    "vision":         {"model": "qwen2.5vl:3b",       "fallback": "",                  "timeout": 120},
    "analysis":       {"model": "",                   "fallback": "llama3.1:8b",       "timeout": 240},
    "embedding":      {"model": "nomic-embed-text",   "fallback": "nomic-embed-text",  "timeout": 60},
}


@dataclass
class RouteConfig:
    model: str
    fallback: str = ""
    timeout: int = 240


class ModelRouter:
    """Routes MayringCoder tasks to specific Ollama models.

    Usage:
        router = ModelRouter(ollama_url="http://localhost:11434")
        model = router.resolve("mayring_code")      # "mayringqwen:latest" if available
        if router.is_available("vision"):
            caption = caption_image(path, router.resolve("vision"))
    """

    TASKS: list[str] = [
        "mayring_code",
        "mayring_social",
        "mayring_hybrid",
        "vision",
        "analysis",
        "embedding",
    ]

    def __init__(self, ollama_url: str = "http://localhost:11434") -> None:
        self._ollama_url = ollama_url.rstrip("/")
        self._routes: dict[str, RouteConfig] = {
            task: RouteConfig(**cfg) for task, cfg in _DEFAULTS.items()
        }
        self._availability_cache: dict[str, tuple[bool, float]] = {}  # model → (available, timestamp)
        self._cache_ttl = 30.0  # seconds

        if _CONFIG_PATH.exists():
            self.load_config(_CONFIG_PATH)

    def load_config(self, path: Path) -> None:
        """Load routes from YAML file. Unknown tasks are ignored."""
        if not _HAS_YAML:
            return
        try:
            with path.open(encoding="utf-8") as f:
                data = _yaml.safe_load(f) or {}
            for task, cfg in data.items():
                if not isinstance(cfg, dict):
                    continue
                existing = self._routes.get(task)
                if existing:
                    if "model" in cfg:
                        existing.model = str(cfg["model"])
                    if "fallback" in cfg:
                        existing.fallback = str(cfg["fallback"])
                    if "timeout" in cfg:
                        existing.timeout = int(cfg["timeout"])
                else:
                    self._routes[task] = RouteConfig(
                        model=str(cfg.get("model", "")),
                        fallback=str(cfg.get("fallback", "")),
                        timeout=int(cfg.get("timeout", 240)),
                    )
        except Exception as e:
            import logging
            logging.warning("model_routes.yaml konnte nicht geladen werden: %s — Defaults aktiv", e)

    def save_config(self, path: Path | None = None) -> None:
        """Persist current routes to YAML."""
        target = path or _CONFIG_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        if not _HAS_YAML:
            return
        data = {
            task: {"model": cfg.model, "fallback": cfg.fallback, "timeout": cfg.timeout}
            for task, cfg in self._routes.items()
        }
        with target.open("w", encoding="utf-8") as f:
            _yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def resolve(self, task: str) -> str:
        """Return model name for task.

        Priority: configured model (if available) → fallback → OLLAMA_MODEL env → "".
        Does NOT check Ollama availability — use is_available() separately if needed.
        """
        cfg = self._routes.get(task)
        if cfg is None:
            return os.getenv("OLLAMA_MODEL", "")

        # Empty model → use env var
        model = cfg.model or os.getenv("OLLAMA_MODEL", "")
        return model

    def resolve_with_fallback(self, task: str) -> str:
        """Return model name, falling back to fallback if primary not available."""
        cfg = self._routes.get(task)
        if cfg is None:
            return os.getenv("OLLAMA_MODEL", "")

        primary = cfg.model or os.getenv("OLLAMA_MODEL", "")
        if primary and self._check_model_available(primary):
            return primary
        return cfg.fallback or os.getenv("OLLAMA_MODEL", "")

    def is_available(self, task: str) -> bool:
        """True if the model for this task is available in Ollama (cached 30s TTL).

        Returns False if:
        - task has no model configured
        - model name is empty string
        - httpx not installed
        - Ollama not reachable
        - model not in Ollama model list
        """
        cfg = self._routes.get(task)
        if cfg is None:
            return False
        model = cfg.model or os.getenv("OLLAMA_MODEL", "")
        if not model:
            return False
        return self._check_model_available(model)

    def _check_model_available(self, model: str) -> bool:
        """Check if model is in Ollama, with 30s cache."""
        now = time.monotonic()
        cached = self._availability_cache.get(model)
        if cached is not None:
            available, ts = cached
            if now - ts < self._cache_ttl:
                return available

        available = self._query_ollama_for_model(model)
        self._availability_cache[model] = (available, now)
        return available

    def _query_ollama_for_model(self, model: str) -> bool:
        """GET /api/tags and check if model is present."""
        if _httpx is None:
            return False
        try:
            resp = _httpx.get(f"{self._ollama_url}/api/tags", timeout=2.0)
            if resp.status_code != 200:
                return False
            tags = resp.json()
            names = {m["name"] for m in tags.get("models", [])}
            # Match exact name or name without tag
            model_base = model.split(":")[0]
            return model in names or any(
                n == model or n.split(":")[0] == model_base for n in names
            )
        except Exception:
            return False

    def set_route(self, task: str, model: str, fallback: str = "", timeout: int = 240) -> None:
        """Update a route at runtime (and invalidate availability cache)."""
        self._routes[task] = RouteConfig(model=model, fallback=fallback, timeout=timeout)
        # Invalidate cache entries for old+new model
        self._availability_cache.clear()

    def to_dict(self) -> dict[str, dict]:
        """Serialize routes for WebUI or JSON responses."""
        return {
            task: {
                "model": cfg.model,
                "fallback": cfg.fallback,
                "timeout": cfg.timeout,
                "available": self._availability_cache.get(cfg.model, (None, 0))[0],
            }
            for task, cfg in self._routes.items()
        }

    def __repr__(self) -> str:
        routes_str = ", ".join(f"{t}={cfg.model!r}" for t, cfg in self._routes.items())
        return f"ModelRouter({routes_str})"
