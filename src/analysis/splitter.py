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

import ast as _ast
import hashlib
import re
from pathlib import Path as _Path

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


# ---------------------------------------------------------------------------
# smart_split() — Priority-based block selection for the analysis pipeline
# ---------------------------------------------------------------------------

# Security-relevance keyword scores.
# Keys are matched case-insensitively against function/class names and bodies.
_PRIORITY_KEYWORDS: dict[str, int] = {
    "auth": 3,
    "authenticate": 3,
    "login": 3,
    "password": 3,
    "secret": 3,
    "token": 3,
    "delete": 3,
    "admin": 3,
    "permission": 3,
    "csrf": 3,
    "export": 2,
    "__init__": 2,
    "create": 2,
    "update": 2,
    "save": 2,
    "except": 1,
    "error": 1,
    "catch": 1,
    "raise": 1,
    "fail": 1,
}


def _score_block(name: str, text: str) -> int:
    """Score a block by keyword presence in its name and body.

    Name matches earn full points; body matches earn (pts - 1), minimum 1.
    """
    score = 0
    name_lower = name.lower()
    body_lower = text.lower()
    for keyword, pts in _PRIORITY_KEYWORDS.items():
        if keyword in name_lower:
            score += pts
        elif keyword in body_lower:
            score += max(pts - 1, 1)
    return score


def _extract_python_blocks(text: str) -> list[dict]:
    """AST-based extraction of top-level functions and classes.

    Returns list of dicts: {name, text, priority, start, end}.
    Returns empty list on SyntaxError (caller falls back to truncation).
    """
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    blocks: list[dict] = []
    for node in tree.body:
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            continue
        start_line = node.lineno - 1  # 0-indexed
        end_line = getattr(node, "end_lineno", start_line) - 1  # 0-indexed
        start_off = line_offsets[start_line] if start_line < len(line_offsets) else 0
        end_off = (
            line_offsets[end_line] + len(lines[end_line])
            if end_line < len(lines)
            else len(text)
        )
        block_text = text[start_off:end_off]
        if not block_text.strip():
            continue
        name = node.name
        priority = _score_block(name, block_text)
        blocks.append({
            "name": name,
            "text": block_text,
            "priority": priority,
            "start": start_off,
            "end": end_off,
        })
    return blocks


# Regex for JS/TS function and class declarations (including async/export variants).
_JS_BLOCK_RE = re.compile(
    r"(?:^|\n)"
    r"((?:export\s+default\s+|export\s+)?(?:async\s+)?function\s+(\w+)"
    r"|(?:export\s+)?class\s+(\w+)"
    r"|(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\()",
    re.MULTILINE,
)


def _extract_js_blocks(text: str) -> list[dict]:
    """Regex + brace-depth extraction for JS/TS functions and classes.

    Returns list of dicts: {name, text, priority, start, end}.
    """
    matches = list(_JS_BLOCK_RE.finditer(text))
    if not matches:
        return []

    blocks: list[dict] = []
    for m in matches:
        start = m.start() if text[m.start()] != "\n" else m.start() + 1
        # Extract name from whichever capture group matched
        name = m.group(2) or m.group(3) or m.group(4) or "__unknown__"

        # Find end by counting brace depth from the match start
        brace_depth = 0
        end = len(text)
        found_open = False
        for j in range(start, len(text)):
            if text[j] == "{":
                brace_depth += 1
                found_open = True
            elif text[j] == "}":
                brace_depth -= 1
                if found_open and brace_depth == 0:
                    end = j + 1
                    break

        block_text = text[start:end].strip()
        if not block_text:
            continue

        priority = _score_block(name, block_text)
        blocks.append({
            "name": name,
            "text": block_text,
            "priority": priority,
            "start": start,
            "end": end,
        })
    return blocks


def smart_split(content: str, filename: str, max_chars: int = 3000) -> dict:
    """Extract and prioritize code blocks within a max_chars budget.

    Dispatches to Python (AST) or JS/TS (regex+brace) extractors by extension.
    Unknown extensions or extractor failures fall back to a truncated block.

    Args:
        content:   File content to split.
        filename:  Used to determine the language extractor.
        max_chars: Maximum total characters in the selected blocks.

    Returns:
        {
            "blocks":          list[dict] — all extracted blocks (name, text, priority, start, end),
            "selected":        list[dict] — blocks chosen within max_chars (sorted by position),
            "skipped_summary": str        — one-line summary of omitted block names,
        }
    """
    if not content.strip():
        return {"blocks": [], "selected": [], "skipped_summary": ""}

    ext = _Path(filename).suffix.lower()

    if ext == ".py":
        blocks = _extract_python_blocks(content)
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        blocks = _extract_js_blocks(content)
    else:
        blocks = []

    # Fall back to truncated content when no blocks extracted (unknown ext or parse failure)
    if not blocks:
        fallback_text = content[:max_chars]
        fallback = {"name": "__fallback__", "text": fallback_text, "priority": 0, "start": 0, "end": len(fallback_text)}
        return {"blocks": [fallback], "selected": [fallback], "skipped_summary": ""}

    # Sort by priority DESC, then by position ASC (stable for equal priority)
    sorted_blocks = sorted(blocks, key=lambda b: (-b["priority"], b["start"]))

    # Greedy selection within budget
    selected: list[dict] = []
    budget = max_chars
    for block in sorted_blocks:
        block_len = len(block["text"])
        if block_len <= budget:
            selected.append(block)
            budget -= block_len

    # Re-sort selected by position for readable, top-to-bottom output
    selected.sort(key=lambda b: b["start"])

    # Build skipped summary
    selected_set = {id(b) for b in selected}
    skipped = [b for b in blocks if id(b) not in selected_set]
    if skipped:
        skipped_names = ", ".join(b["name"] for b in sorted(skipped, key=lambda b: b["start"]))
        skipped_summary = f"Skipped ({len(skipped)}): {skipped_names}"
    else:
        skipped_summary = ""

    return {"blocks": blocks, "selected": selected, "skipped_summary": skipped_summary}
