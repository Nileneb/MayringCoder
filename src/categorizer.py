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


def _load_yaml(path: Path) -> dict | list | None:
    if not _HAS_YAML or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return _yaml.safe_load(fh)


def load_codebook(path: Path | None = None) -> list[dict]:
    data = _load_yaml(path or CODEBOOK_PATH)
    if isinstance(data, dict):
        return data.get("categories", [])
    if isinstance(data, list):
        return data
    return []


def load_exclude_patterns(path: Path | None = None) -> list[str]:
    """Load exclude_patterns from codebook YAML."""
    data = _load_yaml(path or CODEBOOK_PATH)
    if isinstance(data, dict):
        return data.get("exclude_patterns", [])
    return []


def load_mayringignore(path: Path | None = None) -> list[str]:
    """Load extra exclude patterns from a .mayringignore file (optional)."""
    target = path or Path(".mayringignore")
    if not target.exists():
        return []
    patterns: list[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped)
    return patterns


def filter_excluded_files(
    files: list[dict], patterns: list[str]
) -> tuple[list[dict], list[dict]]:
    """Split *files* into (included, excluded) based on exclude patterns."""
    if not patterns:
        return files, []
    included, excluded = [], []
    for f in files:
        if _matches_patterns(f["filename"], patterns):
            excluded.append(f)
        else:
            included.append(f)
    return included, excluded


def _matches_patterns(filename: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if pat.startswith("re:"):
            if re.search(pat[3:], filename, re.IGNORECASE):
                return True
        else:
            # Convert glob wildcards to regex
            # "**" matches any path segment, "*" matches within a segment
            regex = re.escape(pat).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
            # Allow */dir/* to also match at root level (dir/*)
            if regex.startswith("[^/]*/"):
                regex = "(?:[^/]*/)?" + regex[len("[^/]*/"):]
            # Trailing /* should match any depth inside a directory
            if regex.endswith("/[^/]*"):
                regex = regex[: -len("[^/]*")] + ".*"
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
