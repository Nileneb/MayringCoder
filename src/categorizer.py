"""Categorize files using a YAML codebook.

Each entry in the codebook has:
    name:        str
    description: str
    patterns:    list[str]  # glob-style or 're:...' for regex

Adds 'category' and 'category_reason' to each file dict in-place.
"""

import re
from pathlib import Path

from src.config import CODEBOOK_PATH, CODEBOOKS_DIR

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
            # "**" matches any number of path segments (including zero)
            # "*" matches any non-slash characters (zero or more)
            regex = re.escape(pat).replace(r"\*\*", "SPLIT_MARKER")
            regex = regex.replace(r"\*", "[^/]*").replace("SPLIT_MARKER", ".*?")
            # Allow */dir/* to also match at root level (dir/*)
            if regex.startswith("[^/]*/"):
                regex = "(?:[^/]*/)?" + regex[len("[^/]*/"):]
            # Trailing /* should match any depth inside a directory
            if regex.endswith("/[^/]*"):
                regex = regex[: -len("/[^/]*")] + ".*"
            # **/foo/** — allow the leading .*? to also match empty (root level)
            if regex.startswith(".*?/"):
                regex = "(?:.*?/)?" + regex[len(".*?/"):]
            if re.search(regex + r"$", filename, re.IGNORECASE):
                return True
    return False


def load_codebook_modular(profile: str = "generic") -> tuple[list[str], list[dict]]:
    """Load exclude patterns and categories from a codebook profile.

    Reads codebooks/profiles/{profile}.yaml, then loads each referenced
    exclude and category submodule.

    Returns:
        (exclude_patterns: list[str], categories: list[dict])
    Fallback: if codebooks/ doesn't exist or the profile is not found,
    delegates to load_codebook() + load_exclude_patterns().
    """
    profile_path = CODEBOOKS_DIR / "profiles" / f"{profile}.yaml"
    if not CODEBOOKS_DIR.exists() or not profile_path.exists():
        return load_exclude_patterns(), load_codebook()

    profile_data = _load_yaml(profile_path)
    if not isinstance(profile_data, dict):
        return load_exclude_patterns(), load_codebook()

    # Collect exclude patterns from all referenced exclude submodules
    all_exclude_patterns: list[str] = []
    for name in profile_data.get("excludes", []):
        exclude_file = CODEBOOKS_DIR / "excludes" / f"{name}.yaml"
        data = _load_yaml(exclude_file)
        if isinstance(data, dict):
            all_exclude_patterns.extend(data.get("patterns", []))

    # Collect categories from all referenced category submodules
    all_categories: list[dict] = []
    for name in profile_data.get("categories", []):
        cat_file = CODEBOOKS_DIR / "categories" / f"{name}.yaml"
        data = _load_yaml(cat_file)
        if isinstance(data, dict):
            cat = {
                "name": data.get("name", name),
                "description": data.get("description", ""),
                "patterns": data.get("patterns", []),
                "risk_level": data.get("risk_level", "medium"),
            }
            all_categories.append(cat)

    return all_exclude_patterns, all_categories


def detect_profile(files: list[dict]) -> str:
    """Auto-detect codebook profile from file list.

    Heuristic:
    - If any file matches 'artisan' or '*.blade.php' or 'app/Http/*' → 'laravel'
    - If any file matches '*.py' and ('setup.py' or 'pyproject.toml') → 'python'
    - Otherwise → 'generic'
    """
    filenames = [f.get("filename", "") for f in files]

    # Check for Laravel markers
    for fn in filenames:
        if fn == "artisan" or fn.endswith(".blade.php") or fn.startswith("app/Http/"):
            return "laravel"

    # Check for Python markers
    has_py = any(fn.endswith(".py") for fn in filenames)
    has_py_marker = any(fn in ("setup.py", "pyproject.toml") for fn in filenames)
    if has_py and has_py_marker:
        return "python"

    # Check pyproject.toml alone (could be Python project without .py listed)
    if has_py_marker:
        return "python"

    return "generic"


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
