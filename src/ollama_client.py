"""Unified Ollama HTTP client + availability check.

Public API:
    generate(url, model, prompt, *, system, images, stream, timeout, ...) -> str
    embed_batch(url, model, texts, *, timeout) -> list[list[float]] | None
    embed_single(url, model, text, *, timeout, label, ...) -> list[float]
    chat(url, model, messages, *, system, tools, options, timeout) -> dict
    check_ollama(url) -> tuple[bool, list[str]]
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any

import httpx

_GENERATE_MAX_RETRIES = 3
_GENERATE_RETRY_DELAYS = (2, 5, 10)

_EMBED_MAX_RETRIES = 5
_EMBED_RETRY_DELAYS = (3, 6, 12, 20, 30)


def generate(
    url: str,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    images: list[str] | None = None,
    stream: bool = True,
    timeout: float = 240.0,
    max_retries: int = _GENERATE_MAX_RETRIES,
    retry_delays: tuple[int, ...] = _GENERATE_RETRY_DELAYS,
    label: str = "",
    think: bool | None = None,
    num_predict: int = 4096,
    options: dict | None = None,
) -> str:
    """POST to /api/generate and return the complete response text.

    Uses stream=True by default to prevent read-timeout hangs on large models.
    Pass stream=False (e.g. for image captioning) to get a single-shot response.

    ``num_predict`` defaults to 4096 (not Ollama's 128). The first prod
    smoke showed thinking-capable models (qwen3, deepseek-r1) never closed
    ``</think>`` at 128 tokens, so ``response`` stayed empty. 4096 gives
    Mayring categorization room to reason AND emit the label line. Pass a
    lower value for cheap single-shot calls if you know the budget.

    ``think`` is left to the model's own default (``None`` means: don't send
    the field, let Ollama decide). Callers can force it (True/False) per call
    — e.g. image captioning with stream=False might opt out explicitly.
    """
    base = url.rstrip("/")
    body: dict[str, Any] = {"model": model, "prompt": prompt, "stream": stream}
    if think is not None:
        body["think"] = think
    if system:
        body["system"] = system
    if images:
        body["images"] = images

    merged_options: dict[str, Any] = {"num_predict": num_predict}
    if options:
        merged_options.update(options)
    body["options"] = merged_options

    for attempt in range(max_retries):
        try:
            if stream:
                chunks: list[str] = []
                with httpx.stream("POST", f"{base}/api/generate", json=body, timeout=timeout) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunks.append(data.get("response", ""))
                        if data.get("done"):
                            break
                return "".join(chunks)
            else:
                resp = httpx.post(f"{base}/api/generate", json=body, timeout=timeout)
                resp.raise_for_status()
                return resp.json().get("response", "")
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            if attempt < max_retries - 1:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                _log_retry("generate", label, attempt + 1, max_retries, delay)
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("unreachable")


def embed_batch(
    url: str,
    model: str,
    texts: list[str],
    *,
    timeout: float = 240.0,
) -> list[list[float]] | None:
    """POST to /api/embed (batch endpoint).

    Returns a list of float vectors in the same order as *texts*, or None if
    the endpoint is unavailable or returns an unexpected shape.
    """
    try:
        resp = httpx.post(
            f"{url.rstrip('/')}/api/embed",
            json={"model": model, "input": texts},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("embeddings")
        if isinstance(result, list) and len(result) == len(texts):
            return result
    except Exception:
        pass
    return None


def embed_single(
    url: str,
    model: str,
    text: str,
    *,
    timeout: float = 240.0,
    label: str = "",
    max_retries: int = _EMBED_MAX_RETRIES,
    retry_delays: tuple[int, ...] = _EMBED_RETRY_DELAYS,
) -> list[float]:
    """POST to /api/embeddings (legacy single-text endpoint) with retry."""
    base = url.rstrip("/")
    for attempt in range(max_retries):
        try:
            resp = httpx.post(
                f"{base}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
            if attempt < max_retries - 1:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                _log_retry("embed", label, attempt + 1, max_retries, delay)
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("unreachable")


def chat(
    url: str,
    model: str,
    messages: list[dict],
    *,
    system: str | None = None,
    tools: list[dict] | None = None,
    options: dict | None = None,
    stream: bool = False,
    timeout: float = 120.0,
) -> dict:
    """POST to /api/chat and return the parsed JSON response dict.

    Raises httpx exceptions on failure — callers handle retries/errors.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "think": False,
    }
    if system is not None:
        body["system"] = system
    if tools is not None:
        body["tools"] = tools
    if options is not None:
        body["options"] = options

    resp = httpx.post(f"{url.rstrip('/')}/api/chat", json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def check_ollama(url: str) -> tuple[bool, list[str]]:
    """Return (reachable, model_list). Never raises. Tries subprocess then HTTP."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            models = [l.split()[0] for l in lines[1:] if l.split()]
            return True, models
    except Exception:
        pass
    try:
        resp = httpx.get(url.rstrip("/") + "/api/tags", timeout=3.0)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
            return True, models
    except Exception:
        pass
    return False, []


def _log_retry(kind: str, label: str, attempt: int, max_retries: int, delay: int) -> None:
    tag = f" [{label}]" if label else ""
    print(f"    ⟳ {kind}-Retry {attempt}/{max_retries}{tag} (in {delay}s) …", flush=True)
