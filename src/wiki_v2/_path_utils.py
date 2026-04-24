from __future__ import annotations
import re
from pathlib import Path

_SAFE_WID_RE = re.compile(r'[^A-Za-z0-9_\-/]')
_SAFE_NAME_RE = re.compile(r'[^A-Za-z0-9_\-]')


def safe_workspace_id(wid: str) -> str:
    """Sanitize workspace ID for path use: replaces unsafe chars, strips leading slashes."""
    return _SAFE_WID_RE.sub('_', wid).lstrip('/')


def safe_filename_part(name: str) -> str:
    """Sanitize a filename component: replaces any non-alphanum/-/_ char (including slashes)."""
    return _SAFE_NAME_RE.sub('_', name)


def confined_path(root: Path, *parts: str) -> Path:
    """Build a path under root. Raises ValueError if the result escapes root."""
    root_r = Path(root).resolve()
    p = root_r
    for part in parts:
        p = (p / part).resolve()
    if not p.is_relative_to(root_r):
        raise ValueError(f"Path traversal: {parts!r} escapes {root!r}")
    return p
