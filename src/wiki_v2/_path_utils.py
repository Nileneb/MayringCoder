from __future__ import annotations
import os
import re
from pathlib import Path

_SAFE_NAME_RE = re.compile(r'[^A-Za-z0-9_\-]')


def safe_workspace_id(wid: str) -> str:
    return os.path.basename(wid.replace('/', '_').replace('\\', '_'))


def safe_filename_part(name: str) -> str:
    return _SAFE_NAME_RE.sub('_', name)


def confined_path(root: Path, *parts: str) -> Path:
    root_r = Path(root).resolve()
    p = root_r
    for part in parts:
        _s = safe_filename_part(str(part))
        if not _s or _s in {".", ".."}:
            raise ValueError(f"Invalid path segment: {part!r}")
        p = (p / _s).resolve()
    if not p.is_relative_to(root_r):
        raise ValueError(f"Path traversal: {parts!r} escapes {root!r}")
    return p
