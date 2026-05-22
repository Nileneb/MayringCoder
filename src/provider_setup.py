"""Wire host-side implementations into ``mayring_core.providers`` (#267).

Core defines a provider registry so it never imports ``src.analysis`` /
``src.agents`` at runtime. The host registers the rich implementations here:
the cached/batched embedder, the Ollama-streaming generator and the Pillow
vision captioner.

The wrappers resolve the target function *inside the call* rather than capturing
it at registration time. This keeps the canonical functions
(``src.analysis.context_rag._embed_texts`` etc.) as the single patch point, so
existing ``unittest.mock.patch("src.analysis...")`` test seams keep working.

Call ``setup_providers()`` once per process — done from ``src/main.py``,
``src/api/server.py`` and the test conftest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def _embed(*args, **kwargs) -> list[list[float]]:
    from src.analysis.context_rag import _embed_texts
    return _embed_texts(*args, **kwargs)


def _generate(*args, **kwargs) -> str:
    from src.analysis.analyzer import _ollama_generate
    return _ollama_generate(*args, **kwargs)


def _vision_caption(*args, **kwargs) -> str:
    from src.agents.vision import caption_image
    return caption_image(*args, **kwargs)


def _vision_metadata(path: Path) -> Optional[dict]:
    from src.agents.vision import get_image_metadata
    return get_image_metadata(path)


def setup_providers() -> None:
    from mayring_core import providers

    providers.register_embedder(_embed)
    providers.register_generator(_generate)
    providers.register_vision(_vision_caption, _vision_metadata)
