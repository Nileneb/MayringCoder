"""Parser für `# WHY(<refs>): <text>`-Marker → rationale-edges.

Extrahiert WHY-Comments aus einem Python-File und matched jeden
Marker an das nächste Code-Symbol (Assign / FunctionDef / ClassDef).
Comments vor non-trivialen Targets (for/if/try/while) werden mit
WARN-Log geskipped.

Edge-Schema (returned als list[dict]):
  source       — file path relativ zum Repo-Root (z.B. 'src/cli.py')
  target       — qualified name des Symbols (z.B. 'cli._SLUG_RE')
  type         — immer 'rationale'
  weight       — immer 1.0
  context      — die ref-list aus WHY(...) (z.B. '#185' oder '#182,perf')
  rationale    — der freie Text nach dem Doppelpunkt
  repo_slug    — vom Caller mitgegeben
  workspace_id — vom Caller mitgegeben
"""
from __future__ import annotations

import ast
import logging
import re
import tokenize
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Regex erfasst nur die ERSTE Zeile eines Markers.
# Multi-Line wird in Task 3 ergänzt.
_MARKER_RE = re.compile(r"^\s*WHY\(([^)]+)\):\s*(.+?)\s*$")


def extract_rationale_edges(
    file_path: Path,
    *,
    repo_slug: str,
    workspace_id: str,
) -> list[dict[str, Any]]:
    """Parse WHY-marker aus file_path. Returns [] bei Parse-Errors."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError:
        return []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        _log.warning("rationale_parser: skip %s (syntax error)", file_path)
        return []

    module_name = file_path.stem  # without .py

    # Tokenize, behalte nur Comment-Tokens mit WHY-pattern
    markers: list[tuple[int, str, str]] = []  # (line_no, refs, rationale)
    try:
        with file_path.open("rb") as f:
            tokens = list(tokenize.tokenize(f.readline))
    except (OSError, tokenize.TokenizeError):
        return []

    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        # tok.string ist '# WHY(#185): ...'  oder '# something else'
        text = tok.string.lstrip("#").strip()
        m = _MARKER_RE.match(text)
        if not m:
            continue
        markers.append((tok.start[0], m.group(1).strip(), m.group(2).strip()))

    if not markers:
        return []

    # Build a map: line_no → top-level ast.Node (Assign/FunctionDef/ClassDef)
    line_to_node: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            line_to_node[node.lineno] = node

    edges: list[dict[str, Any]] = []
    for marker_line, refs, rationale in markers:
        target_node: ast.AST | None = None
        for delta in range(1, 6):  # max 5 lines lookahead
            cand = line_to_node.get(marker_line + delta)
            if cand is not None:
                target_node = cand
                break
        if target_node is None:
            _log.warning(
                "rationale-skipped: file=%s line=%s reason=non-trivial-target",
                file_path, marker_line,
            )
            continue

        target_name = _node_target_name(target_node, module_name)
        if not target_name:
            continue
        edges.append({
            "source": file_path.name,
            "target": target_name,
            "type": "rationale",
            "weight": 1.0,
            "context": refs,
            "rationale": rationale,
            "repo_slug": repo_slug,
            "workspace_id": workspace_id,
        })
    return edges


def _node_target_name(node: ast.AST, module_name: str) -> str:
    """Module-qualified target-name. Task 4 erweitert das auf Class.method."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return f"{module_name}.{node.name}"
    if isinstance(node, ast.Assign):
        # Nimm den ersten Target-Namen (selten mehrere)
        if node.targets and isinstance(node.targets[0], ast.Name):
            return f"{module_name}.{node.targets[0].id}"
    if isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            return f"{module_name}.{node.target.id}"
    return ""
