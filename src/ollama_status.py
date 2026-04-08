"""Ollama availability check — never raises, always returns (bool, list[str])."""

from __future__ import annotations

import subprocess


def check_ollama(ollama_url: str) -> tuple[bool, list[str]]:
    """Return (reachable, model_list). Never raises an exception.

    Strategy:
      1. Try ``ollama list`` via subprocess (fast, no network round-trip needed).
      2. Fall back to GET {ollama_url}/api/tags via httpx.
      3. On any error: return (False, []).
    """
    # --- Attempt 1: subprocess ---
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            models = _parse_ollama_list(result.stdout)
            return True, models
    except Exception:
        pass

    # --- Attempt 2: httpx HTTP fallback ---
    try:
        import httpx  # already in requirements.txt

        resp = httpx.get(ollama_url.rstrip("/") + "/api/tags", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            return True, models
    except Exception:
        pass

    return False, []


def _parse_ollama_list(stdout: str) -> list[str]:
    """Parse model names from ``ollama list`` output.

    The first line is a header (NAME  ID  SIZE  MODIFIED).
    Subsequent lines start with the model name as first whitespace-separated token.
    """
    models: list[str] = []
    lines = stdout.strip().splitlines()
    for line in lines[1:]:  # skip header
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models
