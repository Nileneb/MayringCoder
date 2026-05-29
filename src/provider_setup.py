"""Wire host-side implementations into ``mayring_core.providers`` (#267).

Core defines a provider registry so it never imports ``src.analysis`` /
``mayring_pi_agent`` at runtime. The host registers the rich implementations here:
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

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Positional order of the canonical generator signature (mirrors
# mayring_core.providers._default_generate). Used to bind *args for queue-routing.
_GEN_PARAM_ORDER = ("prompt", "ollama_url", "model", "label")


def _embed(*args, **kwargs) -> list[list[float]]:
    # NOTE: embeddings NEVER route through the PiQueue — they stay on three.linn.games
    # (ollama.com has no embedding models). The queue is generate/chat-only.
    from src.analysis.context_rag import _embed_texts
    return _embed_texts(*args, **kwargs)


def _generate_via_queue(args: tuple, kwargs: dict) -> str | None:
    """Route a generate job through the central PiQueue (POST /pi/run) instead of
    hitting Ollama directly. Returns the content on success, or ``None`` to signal
    fail-soft fallback to the direct path (already logged).

    WHY(#1): the ingest subprocess is the dominant generate load (categorize/igio);
    funnelling it through /pi/run gives ONE bounded, observable, worker-delegatable
    entry. Determinism is preserved by forwarding the caller's ``options`` verbatim;
    ``num_predict`` is merged INTO options so it overrides the handler's 1024 cap
    (ollama_client merges {'num_predict': …} then options.update()) — no truncation.
    """
    import httpx

    bound = dict(kwargs)
    for name, val in zip(_GEN_PARAM_ORDER, args):
        bound[name] = val

    prompt = bound.get("prompt", "") or ""
    model = bound.get("model", "") or ""
    options = dict(bound.get("options") or {})
    options.setdefault("num_predict", bound.get("num_predict", 4096))
    response_format = bound.get("response_format") or ""

    try:
        from mayring_core.config import OLLAMA_TIMEOUT as _job_timeout
    except Exception:
        _job_timeout = 240.0

    api = os.getenv("MAYRING_API_URL", "http://localhost:8090").rstrip("/")
    token = os.getenv("MCP_SERVICE_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {
        "prompt": prompt,
        "model": model,
        "kind": "categorize",       # non-'pi-task' → plain generate, no memory aug
        "job_class": "background",   # ingest must not starve hook/interactive lanes
        "timeout": float(_job_timeout),
        "response_format": response_format,
        "options": options,
    }

    last_exc: Exception | None = None
    for _ in range(2):  # initial try + 1 retry
        try:
            resp = httpx.post(
                f"{api}/pi/run", json=payload, headers=headers,
                timeout=float(_job_timeout) + 30.0,
            )
            resp.raise_for_status()
            return resp.json().get("content") or ""
        except Exception as exc:  # noqa: BLE001 — fail-soft is intentional + logged below
            last_exc = exc

    # NOT silent: surface the queue breakage, then degrade to a correct direct call.
    logger.warning(
        "MAYRING_GENERATE_VIA_QUEUE=1 but POST %s/pi/run failed (%s) — fail-soft to "
        "direct generate (output stays correct, central routing degraded)",
        api, last_exc,
    )
    return None


def _generate(*args, **kwargs) -> str:
    if os.getenv("MAYRING_GENERATE_VIA_QUEUE") == "1":
        routed = _generate_via_queue(args, kwargs)
        if routed is not None:
            return routed
        # fail-soft → fall through to the direct path below
    from src.analysis.analyzer import _ollama_generate
    return _ollama_generate(*args, **kwargs)


def _vision_caption(*args, **kwargs) -> str:
    from mayring_pi_agent.vision import caption_image
    return caption_image(*args, **kwargs)


def _vision_metadata(path: Path) -> Optional[dict]:
    from mayring_pi_agent.vision import get_image_metadata
    return get_image_metadata(path)


def setup_providers() -> None:
    from mayring_core import providers

    providers.register_embedder(_embed)
    providers.register_generator(_generate)
    providers.register_vision(_vision_caption, _vision_metadata)
