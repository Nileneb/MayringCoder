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
    _yaml = None  # type: ignore[assignment]
    _HAS_YAML = False


def _load_yaml(path: Path) -> dict | list | None:
    if not _HAS_YAML or _yaml is None or not path.exists():
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


def _is_safe_profile_name(profile: str) -> bool:
    """Allow only simple profile identifiers (no path chars / traversal)."""
    if not isinstance(profile, str) or not profile:
        return False
    return re.fullmatch(r"[A-Za-z0-9_-]+", profile) is not None


def _is_within_dir(path: Path, base_dir: Path) -> bool:
    try:
        path.resolve().relative_to(base_dir.resolve())
        return True
    except Exception:
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
    if not _is_safe_profile_name(profile):
        return load_exclude_patterns(), load_codebook()

    profiles_dir = (CODEBOOKS_DIR / "profiles").resolve()
    profile_path = (profiles_dir / f"{profile}.yaml").resolve()
    try:
        profile_path.relative_to(profiles_dir)
    except ValueError:
        return load_exclude_patterns(), load_codebook()

    if not CODEBOOKS_DIR.exists() or not profile_path.exists():
        return load_exclude_patterns(), load_codebook()

    profile_data = _load_yaml(profile_path)
    if not isinstance(profile_data, dict):
        return load_exclude_patterns(), load_codebook()

    # Collect exclude patterns from all referenced exclude submodules
    all_exclude_patterns: list[str] = []
    excludes_dir = CODEBOOKS_DIR / "excludes"
    for name in profile_data.get("excludes", []):
        if not isinstance(name, str) or not _is_safe_profile_name(name):
            continue
        exclude_file = excludes_dir / f"{name}.yaml"
        if not _is_within_dir(exclude_file, excludes_dir):
            continue
        data = _load_yaml(exclude_file)
        if isinstance(data, dict):
            all_exclude_patterns.extend(data.get("patterns", []))

    # Collect categories from all referenced category submodules
    all_categories: list[dict] = []
    categories_dir = CODEBOOKS_DIR / "categories"
    for name in profile_data.get("categories", []):
        if not isinstance(name, str) or not _is_safe_profile_name(name):
            continue
        cat_file = categories_dir / f"{name}.yaml"
        if not _is_within_dir(cat_file, categories_dir):
            continue
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


def parse_tree(tree: str) -> list[str]:
    """Parse a gitingest directory tree string into a flat list of file paths.

    Input example:
        Directory structure:
        └── repo-name/
            ├── artisan
            ├── app/
            │   └── Http/
            │       └── Controller.php

    Returns: ["artisan", "app/Http/Controller.php"]
    """
    paths: list[str] = []
    prefix_stack: list[str] = []

    for line in tree.splitlines():
        # Skip header / empty lines
        stripped = line.strip()
        if not stripped or stripped.startswith("Directory"):
            continue

        # Remove tree drawing chars: └── ├── │   ─
        cleaned = re.sub(r"[│├└─┌┐┘┤┬┴┼]", "", line)
        cleaned = cleaned.replace("  ", "\t")  # normalize indent

        # Extract the name (last non-whitespace segment)
        name = cleaned.strip()
        if not name:
            continue

        # Calculate depth by leading whitespace in original cleaned line
        indent = len(cleaned) - len(cleaned.lstrip())

        # Trim prefix stack to current depth
        depth = indent // 1  # rough approximation
        while len(prefix_stack) > depth:
            prefix_stack.pop()

        if name.endswith("/"):
            # Directory — push to stack
            prefix_stack.append(name)
        else:
            # File — build full path
            path = "/".join(prefix_stack) + name
            # Clean up double slashes and leading slashes
            path = re.sub(r"/+", "/", path).strip("/")
            if path:
                paths.append(path)

    return paths


def detect_profile_from_tree(tree: str) -> str:
    """Auto-detect codebook profile from a gitingest directory tree string.

    Faster than detect_profile(files) — no need to parse file contents,
    works directly on the tree string with simple keyword checks.
    """
    # Flatten tree chars to make substring matching reliable
    # (tree has line breaks between directory levels)
    flat = re.sub(r"[│├└─┌┐┘┤┬┴┼\s]+", " ", tree).lower()

    # Laravel markers
    has_artisan = "artisan" in flat
    has_laravel_dirs = any(marker in flat for marker in (
        "app/ http/", "app/ livewire/", "app/ filament/",
        "app/http/", "app/livewire/", "app/filament/",
    ))
    has_blade = ".blade.php" in flat
    has_composer = "composer.json" in flat

    if (has_artisan and has_composer) or has_blade or (has_artisan and has_laravel_dirs):
        return "laravel"

    # Python markers (setup.py/pyproject.toml OR requirements.txt + .py files)
    has_py = ".py" in flat
    has_py_marker = "setup.py" in flat or "pyproject.toml" in flat or "requirements.txt" in flat
    if has_py and has_py_marker:
        return "python"

    return "universal"


# ---------------------------------------------------------------------------
# Language detection via Pygments (no new dependency — already installed)
# ---------------------------------------------------------------------------

def detect_languages(tree: str) -> dict[str, int]:
    """Count programming languages in a repo based on file extensions.

    Uses Pygments lexer database (~500 languages). Returns {language: file_count}
    sorted by count descending.
    """
    try:
        from pygments.lexers import get_lexer_for_filename
    except ImportError:
        return {}

    counts: dict[str, int] = {}
    for line in tree.splitlines():
        # Extract filename from tree line
        cleaned = re.sub(r"[│├└─┌┐┘┤┬┴┼]", "", line).strip()
        if not cleaned or cleaned.endswith("/") or cleaned.startswith("Directory"):
            continue
        try:
            lex = get_lexer_for_filename(cleaned)
            lang = lex.name
            counts[lang] = counts.get(lang, 0) + 1
        except Exception:
            pass

    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def detect_profile(files: list[dict]) -> str:
    """Auto-detect codebook profile from file list.

    Heuristic:
    - If any file matches 'artisan' or '*.blade.php' or 'app/Http/*' → 'laravel'
    - If any file matches '*.py' and ('setup.py' or 'pyproject.toml') → 'python'
    - Otherwise → 'universal'
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

    return "universal"


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
