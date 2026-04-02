"""Ollama model discovery and interactive model selection."""

from __future__ import annotations

import urllib.request
import urllib.error
import json
import sys


_FALLBACK_MODEL = "llama3.1:8b"


def fetch_ollama_models(ollama_url: str, timeout: int = 2) -> list[str] | None:
    """Query /api/tags and return a sorted list of model names.

    Returns None if Ollama is unreachable or the response is malformed.
    """
    url = ollama_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        # Exclude embedding-only models — they can't be used for text generation.
        _EMBED_KEYWORDS = ("embed", "embedding")
        models = [
            m["name"] for m in data.get("models", [])
            if "name" in m and not any(kw in m["name"].lower() for kw in _EMBED_KEYWORDS)
        ]
        return sorted(models) if models else None
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def prompt_user_for_model(available_models: list[str]) -> str:
    """Display a numbered menu of models and return the user's selection.

    All display output goes to stderr so that stdout stays clean for shell
    command substitution (e.g. OLLAMA_MODEL=$(checker.py --resolve-model-only)).
    Falls back to the first model in the list on invalid input after 3 attempts.
    """
    print("\nKein Modell konfiguriert. Verfügbare Ollama-Modelle:", file=sys.stderr)
    for i, name in enumerate(available_models, 1):
        print(f"  {i}. {name}", file=sys.stderr)

    names_lower = {m.lower(): m for m in available_models}

    attempts = 0
    while attempts < 3:
        try:
            print(f"\nModell auswählen [1–{len(available_models)} oder Name]: ", end="", flush=True, file=sys.stderr)
            raw = input("").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return available_models[0]

        # Accept numeric index.
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(available_models):
                return available_models[idx - 1]

        # Accept exact or case-insensitive name match.
        if raw.lower() in names_lower:
            return names_lower[raw.lower()]

        attempts += 1
        remaining = 3 - attempts
        if remaining > 0:
            print(f"Ungültige Eingabe (Nummer oder Name). Noch {remaining} Versuch(e).", file=sys.stderr)

    print(f"Zu viele ungültige Eingaben — verwende '{available_models[0]}'.", file=sys.stderr)
    return available_models[0]


def resolve_model(ollama_url: str, cli_model: str | None, env_model: str | None) -> str:
    """Resolve the model to use, prompting interactively if nothing is configured.

    Priority: CLI flag → env var → interactive Ollama query → hardcoded fallback.
    """
    if cli_model:
        return cli_model
    if env_model:
        return env_model

    available = fetch_ollama_models(ollama_url)
    if available:
        selected = prompt_user_for_model(available)
        print(f"Modell ausgewählt: {selected}", file=sys.stderr)
        return selected

    print(f"Ollama nicht erreichbar — verwende Standardmodell '{_FALLBACK_MODEL}'.", file=sys.stderr)
    return _FALLBACK_MODEL
