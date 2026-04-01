"""Parse gitingest output into individual file dicts.

Each returned dict has the shape:
    {
        "filename":      str,   # path as reported by gitingest
        "content":       str,   # EOL-normalized content
        "hash":          str,   # full SHA256 hex digest (for DB storage)
        "hash_short":    str,   # first 16 chars (for display only)
        "size":          int,   # byte size of content (UTF-8)
        "line_estimate": int,   # approximate line count
    }
"""

import hashlib
import re

from src.config import INGEST_SEPARATOR

_SKIP_MARKERS = frozenset({"[Binary file]", "[Empty file]"})

_FILE_BLOCK_RE = re.compile(
    rf"{re.escape(INGEST_SEPARATOR)}\nFILE: (.+?)\n{re.escape(INGEST_SEPARATOR)}\n(.*?)(?=\n{re.escape(INGEST_SEPARATOR)}\n|\Z)",
    re.DOTALL,
)


def _normalize_eol(text: str) -> str:
    """Normalize line endings to LF so hashes are OS-independent."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_into_files(content: str) -> list[dict]:
    content = _normalize_eol(content)
    files = []
    for match in _FILE_BLOCK_RE.finditer(content):
        filename = match.group(1).strip()
        raw_content = match.group(2)
        normalized = _normalize_eol(raw_content).rstrip("\n")
        if normalized.strip() in _SKIP_MARKERS:
            continue
        encoded = normalized.encode("utf-8")
        full_hash = hashlib.sha256(encoded).hexdigest()
        files.append({
            "filename": filename,
            "content": normalized,
            "hash": full_hash,
            "hash_short": full_hash[:16],
            "size": len(encoded),
            "line_estimate": normalized.count("\n") + 1 if normalized else 0,
        })
    return files
