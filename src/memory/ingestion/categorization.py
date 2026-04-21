"""Mayring-based chunk categorization — codebook resolution + LLM labelling."""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.memory.schema import Chunk
    from src.model_router import ModelRouter

try:
    import yaml as _yaml  # noqa: F401
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


_PROMPTS_DIR: Path = Path(__file__).parent.parent.parent.parent / "prompts"
_CODEBOOK_DIR: Path = Path(__file__).parent.parent.parent.parent / "codebooks"

_ORIGINAL_MAYRING_CATEGORIES: list[str] = [
    "Zusammenfassung",
    "Explikation",
    "Strukturierung",
    "Paraphrase",
    "Reduktion",
    "Kategoriensystem",
    "Ankerbeispiel",
]

_MODE_TO_TEMPLATE: dict[str, str] = {
    "deductive": "mayring_deduktiv",
    "inductive": "mayring_induktiv",
    "hybrid":    "mayring_hybrid",
}

# Ingest defaults per source_type — no more scattered opts in callers
_INGEST_DEFAULTS: dict[str, dict] = {
    "repo_file":            {"categorize": True,  "codebook": "auto",   "mode": "hybrid", "multiview": False},
    "github_issue":         {"categorize": True,  "codebook": "social", "mode": "hybrid", "multiview": True},
    "conversation_summary": {"categorize": True,  "codebook": "social", "mode": "hybrid", "multiview": False},
    "note":                 {"categorize": True,  "codebook": "auto",   "mode": "hybrid", "multiview": False},
    "session_knowledge":    {"categorize": True,  "codebook": "social", "mode": "hybrid", "multiview": False},
    "session_note":         {"categorize": True,  "codebook": "social", "mode": "hybrid", "multiview": False},
    "image":                {"categorize": False, "codebook": "auto",   "mode": "hybrid", "multiview": False},
    "paper":                {"categorize": True,  "codebook": "social", "mode": "hybrid", "multiview": True},
}
_INGEST_DEFAULT_FALLBACK: dict = {
    "categorize": True, "codebook": "auto", "mode": "hybrid", "multiview": False,
}


def _resolve_codebook(codebook: str, source_type: str) -> list[str]:
    """Return category names for the given codebook/source_type.

    Resolution order:
      1. "auto" → maps source_type to "code" or "social"
      2. codebooks/<name>.yaml (code, social, or any custom)
      3. codebooks/profiles/<name>.yaml (generic, python, laravel, ...)
      4. Fallback → original Mayring categories
    """
    codebook = str(codebook).strip().lower()
    if codebook == "auto":
        _AUTO = {
            "repo_file": "code", "note": "code",
            "conversation": "social", "conversation_summary": "social",
            "session_knowledge": "social", "session_note": "social",
            "github_issue": "social",
        }
        codebook = _AUTO.get(source_type, "code")

    if codebook == "original":
        return list(_ORIGINAL_MAYRING_CATEGORIES)

    # Security: only alphanumeric + _ and - (blocks path traversal)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", codebook):
        return list(_ORIGINAL_MAYRING_CATEGORIES)

    for candidate in [
        _CODEBOOK_DIR / f"{codebook}.yaml",
        _CODEBOOK_DIR / "profiles" / f"{codebook}.yaml",
    ]:
        if candidate.exists() and _HAS_YAML:
            try:
                import yaml as _yaml_local
                data = _yaml_local.safe_load(candidate.read_text(encoding="utf-8"))
                cats = data.get("categories", []) if isinstance(data, dict) else data
                names: list[str] = []
                for c in cats:
                    if isinstance(c, str):
                        names.append(c)
                    elif isinstance(c, dict):
                        n = c.get("label") or c.get("name", "")
                        if n:
                            names.append(n)
                if names:
                    return names
            except Exception:
                continue

    return list(_ORIGINAL_MAYRING_CATEGORIES)


def _path_fallback_category(path: str) -> list[str]:
    """Regex-based category from file path when LLM categorization fails."""
    _RULES = [
        (r"test_|_test\.|/tests/|conftest", "tests"),
        (r"/api/|/routes/|/controllers/|router", "api"),
        (r"/models?/|/db/|/migration|repositor", "data_access"),
        (r"/auth|/security|/guards?/|/policies?", "auth"),
        (r"/service|/domain|/usecase|/business", "domain"),
        (r"/middleware|/pipeline", "middleware"),
        (r"config.*\.(py|yaml|yml|env)|settings\.|constants\.", "config"),
        (r"/utils?/|/helpers?/|/tools?/", "utils"),
        (r"/cache|redis|memcache", "caching"),
        (r"/log|monitor|metric|trace", "logging"),
    ]
    for pattern, cat in _RULES:
        if re.search(pattern, path, re.IGNORECASE):
            return [cat]
    return []


def _load_mayring_template(mode: str) -> str:
    """Load prompt template for the given mode. Falls back to inline default."""
    filename = _MODE_TO_TEMPLATE.get(mode, "mayring_hybrid") + ".md"
    template_path = _PROMPTS_DIR / filename
    try:
        return template_path.read_text(encoding="utf-8")
    except OSError:
        return (
            "Categorize this text chunk using these categories if applicable: {{categories}}. "
            "Respond with ONLY a comma-separated list of labels."
        )


def mayring_categorize(
    chunks: "list[Chunk]",
    ollama_url: str,
    model: str,
    mode: str = "hybrid",
    codebook: str = "auto",
    source_type: str = "repo_file",
    conn: Any = None,
    router: "ModelRouter | None" = None,
) -> "list[Chunk]":
    """Assign Mayring category labels to each chunk via LLM.

    Args:
        mode: "deductive" (closed category set), "inductive" (free derivation),
              "hybrid" (anchors + new categories marked with [neu])
        codebook: "auto" (detect from source_type), "code", "social", or profile name
        source_type: used for auto-detection of codebook
        conn: optional SQLite connection for error logging
        router: optional ModelRouter for task-based model selection
    """
    if router is not None and not model:
        _task = "mayring_code" if source_type == "repo_file" else "mayring_social"
        if router.is_available(_task):
            model = router.resolve(_task)

    if not model or not ollama_url:
        return chunks

    try:
        from src.analysis.analyzer import _ollama_generate
    except ImportError:
        return chunks

    categories = _resolve_codebook(codebook, source_type)
    valid_set = {c.lower() for c in categories if c}
    template = _load_mayring_template(mode)
    system_prompt = template.replace("{{categories}}", ", ".join(categories))

    for chunk in chunks:
        try:
            prompt = f"Text chunk:\n\n{chunk.text[:1200]}"
            response = _ollama_generate(
                prompt=prompt,
                ollama_url=ollama_url,
                model=model,
                label=f"mayring:{chunk.chunk_id[:8]}",
                system_prompt=system_prompt,
            )
            raw = [re.sub(r"^[-•*]\s*", "", p).strip()
                   for p in re.split(r"[,\n]", response)]

            validated: list[str] = []
            for lbl in raw:
                if not lbl or len(lbl) > 60 or "," in lbl or len(lbl.split()) > 4:
                    continue
                if mode == "inductive":
                    validated.append(lbl)
                elif mode == "hybrid" and lbl.lower().startswith("[neu]"):
                    validated.append(lbl)
                elif lbl.lower() in valid_set:
                    validated.append(lbl.lower())

            chunk.category_labels = validated[:5]
            chunk.category_source = mode
            chunk.category_confidence = 1.0 if validated else 0.0

        except Exception as exc:
            chunk.category_labels = _path_fallback_category(chunk.source_id or "")
            chunk.category_source = "fallback"
            chunk.category_confidence = 0.5 if chunk.category_labels else 0.0
            if conn is not None:
                try:
                    from src.memory.store import log_ingestion_event
                    log_ingestion_event(conn, chunk.chunk_id, "categorize_error",
                                        {"error": str(exc)[:200]})
                except Exception:
                    pass

    return chunks
