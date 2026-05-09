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

_MARKER_RE = re.compile(r"^\s*WHY\(([^)]+)\):\s*(.+?)\s*$")


def extract_rationale_edges(
    file_path: Path,
    *,
    repo_slug: str,
    workspace_id: str,
    repo_root: Path | None = None,
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

    markers: list[tuple[int, str, str]] = []  # (line_no, refs, rationale)
    try:
        with file_path.open("rb") as f:
            tokens = list(tokenize.tokenize(f.readline))
    except (OSError, tokenize.TokenizeError):
        return []

    # Tokenize, sammle WHY-marker MIT Folgezeilen (`# ...` ohne WHY-keyword)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type != tokenize.COMMENT:
            i += 1
            continue
        text = tok.string.lstrip("#").strip()
        m = _MARKER_RE.match(text)
        if not m:
            i += 1
            continue
        marker_line = tok.start[0]
        refs = m.group(1).strip()
        rationale_lines = [m.group(2).strip()]
        # Sammle Folgezeilen die direkt anschließen UND `# ` sind UND NICHT
        # selbst ein WHY-marker (nächster Marker beginnt also einen neuen Block).
        j = i + 1
        prev_line = marker_line
        while j < len(tokens):
            t2 = tokens[j]
            if t2.type == tokenize.NL:           # blank-Zeile-token
                j += 1
                continue
            if t2.type != tokenize.COMMENT:
                break
            if t2.start[0] != prev_line + 1:     # Lücke → Block-Ende
                break
            t2text = t2.string.lstrip("#").strip()
            if _MARKER_RE.match(t2text):         # neuer WHY-block
                break
            rationale_lines.append(t2text)
            prev_line = t2.start[0]
            j += 1
        # `last_comment_line` ist der lookahead-Anker fürs target-symbol —
        # bei multi-line WHY-Blöcken steht der Code mehr als 5 Zeilen nach
        # dem ersten WHY-Token, der lookahead darf nicht ab marker_line
        # zählen, sonst skip mit reason='non-trivial-target' obwohl direkt
        # nach dem Block ein klares Assign/FunctionDef folgt.
        markers.append((prev_line, refs, "\n".join(rationale_lines)))
        i = j

    if not markers:
        return []

    # Build a map: line_no → (target_node, parent_class_name | "")
    line_to_node: dict[int, tuple[ast.AST, str]] = {}

    def _walk_with_parent(node: ast.AST, parent_class: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Assign, ast.AnnAssign, ast.FunctionDef,
                                  ast.AsyncFunctionDef, ast.ClassDef)):
                line_to_node[child.lineno] = (child, parent_class)
            if isinstance(child, ast.ClassDef):
                _walk_with_parent(child, child.name)
            else:
                _walk_with_parent(child, parent_class)

    _walk_with_parent(tree)

    edges: list[dict[str, Any]] = []
    for marker_line, refs, rationale in markers:
        target_info: tuple[ast.AST, str] | None = None
        for delta in range(1, 6):
            cand = line_to_node.get(marker_line + delta)
            if cand is not None:
                target_info = cand
                break
        if target_info is None:
            _log.warning(
                "rationale-skipped: file=%s line=%s reason=non-trivial-target",
                file_path, marker_line,
            )
            continue

        target_node, parent_class = target_info
        target_name = _node_target_name(target_node, module_name, parent_class)
        if not target_name:
            continue
        # WHY(production): source MUSS der full repo-relative Pfad sein
        # (z.B. 'src/cli.py'), nicht der basename 'cli.py'. Retrieval-JOIN
        # in retrieval.py:_rerank matcht `source = chunk.source.path` —
        # sources.path persistiert immer den vollen Pfad. Code-review fand
        # diesen production-blocker: parser schrieb basename → 0 matches in
        # production trotz grüner unit-tests.
        if repo_root is not None:
            try:
                source_str = str(file_path.relative_to(repo_root))
            except ValueError:
                # file_path nicht unter repo_root → fallback basename
                source_str = file_path.name
        else:
            source_str = file_path.name
        edges.append({
            "source": source_str,
            "target": target_name,
            "type": "rationale",
            "weight": 1.0,
            "context": refs,
            "rationale": rationale,
            "repo_slug": repo_slug,
            "workspace_id": workspace_id,
        })
    return edges


def extract_rationale_edges_for_repo(
    repo_root: Path,
    conn,
    *,
    repo_slug: str,
    workspace_id: str,
) -> int:
    """Walk repo_root for *.py files, extract rationale-edges, UPSERT into
    wiki_edges. Returns count of edges persisted."""
    persisted = 0
    # WHY(performance, #182): exclude vendor + build-artifacts. rglob('*.py')
    # auf cwd zieht sonst venv/.venv/site-packages mit (tausende Dateien
    # mit eigenen Comments die unsere Regex matchen könnten). Plus #182:
    # batch-commit alle 50 rows damit SQLite-busy_timeout nicht reißt.
    SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__",
                 "build", "dist", "site-packages", ".tox", "cache",
                 ".pytest_cache", ".mypy_cache"}
    BATCH = 50
    pending = 0
    for py_file in repo_root.rglob("*.py"):
        # Skip wenn IRGENDEIN parent-dir-name in SKIP_DIRS ist
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue
        edges = extract_rationale_edges(
            py_file, repo_slug=repo_slug, workspace_id=workspace_id,
            repo_root=repo_root,
        )
        for e in edges:
            conn.execute(
                "INSERT INTO wiki_edges(source, target, repo_slug, "
                "workspace_id, type, weight, context, rationale) "
                "VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(source,target,type,workspace_id) DO UPDATE SET "
                "rationale=excluded.rationale, context=excluded.context, "
                "weight=excluded.weight",
                (e["source"], e["target"], e["repo_slug"], e["workspace_id"],
                 e["type"], e["weight"], e["context"], e["rationale"]),
            )
            persisted += 1
            pending += 1
            if pending >= BATCH:
                conn.commit()
                pending = 0
    conn.commit()
    return persisted


def _node_target_name(node: ast.AST, module_name: str, parent_class: str = "") -> str:
    """Module/Class-qualified target name."""
    prefix = f"{module_name}."
    if parent_class:
        prefix += f"{parent_class}."
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return f"{prefix}{node.name}"
    if isinstance(node, ast.Assign):
        if node.targets and isinstance(node.targets[0], ast.Name):
            return f"{prefix}{node.targets[0].id}"
    if isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            return f"{prefix}{node.target.id}"
    return ""
