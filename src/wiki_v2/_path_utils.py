from __future__ import annotations
import os
import re
from pathlib import Path

_SAFE_NAME_RE = re.compile(r'[^A-Za-z0-9_\-]')


def safe_workspace_id(wid: str | None) -> str:
    s = os.path.basename(str(wid or "").replace('/', '_').replace('\\', '_'))
    s = _SAFE_NAME_RE.sub('_', s).strip("._-")
    if not s or s in {".", ".."}:
        raise ValueError(f"Invalid workspace id: {wid!r}")
    return s


def safe_filename_part(name: str) -> str:
    return _SAFE_NAME_RE.sub('_', name)


def confined_path(root: Path, *parts: str) -> Path:
    root_r = Path(root).resolve()
    p = root_r
    for part in parts:
        _raw = os.path.basename(str(part).replace('/', '_').replace('\\', '_'))
        _s = _SAFE_NAME_RE.sub('_', _raw).strip("._-")
        if not _s or _s in {".", ".."}:
            raise ValueError(f"Invalid path segment: {part!r}")
        p = (p / _s).resolve()
    if not p.is_relative_to(root_r):
        raise ValueError(f"Path traversal: {parts!r} escapes {root!r}")
    return p
