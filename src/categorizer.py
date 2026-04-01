"""Categorize files using a YAML codebook.

Each entry in the codebook has:
    name:        str
    description: str
    patterns:    list[str]  # glob-style or 're:...' for regex

Adds 'category' and 'category_reason' to each file dict in-place.
"""

import re
from pathlib import Path

from src.config import CODEBOOK_PATH

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def load_codebook(path: Path | None = None) -> list[dict]:
    target = path or CODEBOOK_PATH
    if not _HAS_YAML or not target.exists():
        return []
    with target.open("r", encoding="utf-8") as fh:
        data = _yaml.safe_load(fh)
    if isinstance(data, dict):
        return data.get("categories", [])
    return []


def _matches_patterns(filename: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if pat.startswith("re:"):
            if re.search(pat[3:], filename, re.IGNORECASE):
                return True
        else:
            # Convert glob wildcards to regex
            # "**" matches any path segment, "*" matches within a segment
            regex = re.escape(pat).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
            if re.search(regex + r"$", filename, re.IGNORECASE):
                return True
    return False


def categorize_files(
    files: list[dict], codebook: list[dict] | None = None
) -> list[dict]:
    if codebook is None:
        codebook = load_codebook()
    for file in files:
        fn = file["filename"]
        matched = False
        for entry in codebook:
            if _matches_patterns(fn, entry.get("patterns", [])):
                file["category"] = entry["name"]
                file["category_reason"] = entry.get("description", "")
                matched = True
                break
        if not matched:
            file["category"] = "uncategorized"
            file["category_reason"] = ""
    return files
