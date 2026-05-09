# Rationale-Edges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** WHY-marker im Code (`# WHY(#issue): text`) werden beim repo-analyze als wiki_edges mit `type='rationale'` persistiert und im /memory/search results als `rationale_edges`-Feld co-injected.

**Architecture:** Comment-Token-Parser (`rationale_parser.py`) + AST-target-Resolution für saubere qualified-name-targets. ATTACH DATABASE in init_memory_db verknüpft `cache/memory.db` mit `cache/wiki_v2.db` für JOINs in `_rerank()`. compress_for_prompt rendert rationale-edges als `**Rationale:**` Block.

**Tech Stack:** Python `tokenize` + `ast`-Module, sqlite3 ATTACH, existing wiki_v2.store + memory.retrieval, pytest TDD.

**Spec:** `docs/superpowers/specs/2026-05-09-rationale-edges-design.md`

**File Structure:**
- Create: `src/wiki_v2/rationale_parser.py` — Marker-Parser + AST-Target-Resolution
- Create: `tests/test_rationale_parser.py` — Parser-Tests
- Modify: `src/wiki_v2/store.py` — Idempotent `ALTER TABLE wiki_edges ADD rationale`
- Modify: `src/memory/store.py::init_memory_db` — ATTACH DATABASE wiki_v2.db
- Modify: `src/memory/schema.py::RetrievalRecord` — neues Field `rationale_edges`
- Modify: `src/memory/retrieval.py::_rerank` — Cross-DB-Lookup + Field-fill
- Modify: `src/memory/retrieval.py::compress_for_prompt` — Rendering der rationale-edges
- Modify: `src/wiki_v2/edge_extractor.py` — Aufruf-Stelle integrieren
- Modify: `src/cli.py` + `src/cli_args.py` — `--extract-rationale` Flag
- Modify: `tests/test_memory_retrieval.py` — Integration-Tests
- Modify: `docs/smoke_coverage_map.md` — Coverage-Eintrag

---

### Task 1: Schema-Migration in wiki_v2.store (idempotent ALTER TABLE)

**Files:**
- Modify: `src/wiki_v2/store.py` (init-Funktion, vermutlich `init_db` oder ähnlich)
- Test: `tests/test_wiki_v2_store.py` (Erweiterung)

- [ ] **Step 1: Locate init function in store.py**

```bash
grep -n "def init\|CREATE TABLE wiki_edges" /home/nileneb/Desktop/MayringCoder/src/wiki_v2/store.py
```

Die zentrale `init_db()`-Funktion (oder wie immer sie heißt) ist die Stelle wo die ALTER-TABLE-Idempotenz reinkommt. Wenn sie nicht existiert: in der Datei eine new Funktion `_ensure_rationale_column(conn)` erzeugen + sie nach den `CREATE TABLE`-Statements aufrufen.

- [ ] **Step 2: Write failing test for column-existence**

Add to `tests/test_wiki_v2_store.py` (oder neu erstellen wenn nicht da):

```python
def test_init_db_adds_rationale_column(tmp_path):
    """Issue #185/#182 follow-up: rationale-column muss nach init_db
    existieren. Idempotent: zweimal init darf nicht failen."""
    import sqlite3
    from src.wiki_v2.store import init_db  # adjust import to actual name

    db_path = tmp_path / "wiki.db"
    conn = sqlite3.connect(str(db_path))
    init_db(conn)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(wiki_edges)")}
    assert "rationale" in cols

    # Idempotency
    init_db(conn)
    cols2 = {row[1] for row in conn.execute("PRAGMA table_info(wiki_edges)")}
    assert "rationale" in cols2
```

- [ ] **Step 3: Run test — verify RED**

```bash
cd /home/nileneb/Desktop/MayringCoder
python3 -m pytest tests/test_wiki_v2_store.py::test_init_db_adds_rationale_column -v
```

Expected: FAIL with `AssertionError: 'rationale' not in cols` ODER `AttributeError`/`ImportError` falls init_db den Namen nicht hat. Im letzteren Fall import-pfad anpassen, dann re-run um genauen Fehler zu sehen.

- [ ] **Step 4: Add migration code to store.py**

Nach den existing `CREATE TABLE wiki_edges`-Statement, füge hinzu:

```python
def _ensure_rationale_column(conn) -> None:
    """Idempotent ALTER TABLE for the rationale column.

    Issue #185/#182 follow-up: rationale-edges document WHY-knowledge
    that's risky to lose during refactoring. Idempotent so repeated
    init_db calls (test fixtures, container restart) don't crash.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(wiki_edges)")}
    if "rationale" not in cols:
        conn.execute("ALTER TABLE wiki_edges ADD COLUMN rationale TEXT DEFAULT ''")
        conn.commit()
```

In `init_db` nach dem `CREATE TABLE wiki_edges (…)`-Block aufrufen:

```python
_ensure_rationale_column(conn)
```

- [ ] **Step 5: Run test — verify GREEN**

```bash
python3 -m pytest tests/test_wiki_v2_store.py::test_init_db_adds_rationale_column -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/nileneb/Desktop/MayringCoder
git add src/wiki_v2/store.py tests/test_wiki_v2_store.py
git commit -m "feat(wiki_v2): rationale TEXT column auf wiki_edges (idempotent migration)

Schema-Vorbereitung für Rationale-Edges (spec
docs/superpowers/specs/2026-05-09-rationale-edges-design.md). Column
defaultet auf '', existing edges sind unverändert."
```

---

### Task 2: Rationale-Parser Core (Single-line Marker vor Assign)

**Files:**
- Create: `src/wiki_v2/rationale_parser.py`
- Test: `tests/test_rationale_parser.py`

- [ ] **Step 1: Write failing test for the simplest case**

Create `tests/test_rationale_parser.py`:

```python
"""Tests for rationale-marker parser (Issue #185/#182 follow-up)."""
from __future__ import annotations
from pathlib import Path

from src.wiki_v2.rationale_parser import extract_rationale_edges


def test_extract_simple_marker_before_assign(tmp_path: Path) -> None:
    """Single-line WHY-marker direkt vor einer module-level Assignment."""
    src_file = tmp_path / "module.py"
    src_file.write_text(
        "import re\n"
        "\n"
        "# WHY(#185): path-traversal defence\n"
        "_SLUG_RE = re.compile(r'^[a-z]+$')\n"
    )
    edges = extract_rationale_edges(
        src_file, repo_slug="demo", workspace_id="bene",
    )
    assert len(edges) == 1
    e = edges[0]
    assert e["source"] == "module.py"
    assert e["target"] == "module._SLUG_RE"
    assert e["type"] == "rationale"
    assert e["context"] == "#185"
    assert e["rationale"] == "path-traversal defence"
    assert e["repo_slug"] == "demo"
    assert e["workspace_id"] == "bene"
```

- [ ] **Step 2: Run test — verify RED**

```bash
cd /home/nileneb/Desktop/MayringCoder
python3 -m pytest tests/test_rationale_parser.py::test_extract_simple_marker_before_assign -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.wiki_v2.rationale_parser'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/wiki_v2/rationale_parser.py`:

```python
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
    # Step 4 (Task 4) refines this for nested-class qualified names.
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
```

- [ ] **Step 4: Run test — verify GREEN**

```bash
python3 -m pytest tests/test_rationale_parser.py::test_extract_simple_marker_before_assign -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wiki_v2/rationale_parser.py tests/test_rationale_parser.py
git commit -m "feat(wiki_v2): rationale_parser Core — single-line WHY-marker

Erste TDD-iteration: Marker direkt vor module-level Assignment werden
zu rationale-edges mit qualified target ('module.var'). Parser lebt
unter src/wiki_v2/rationale_parser.py, Tests in
tests/test_rationale_parser.py. Multi-line, qualified-class-names,
async und non-trivial-skips folgen in Task 3-5."
```

---

### Task 3: Multi-line Rationale (Folgezeilen mit `# `)

**Files:**
- Modify: `src/wiki_v2/rationale_parser.py`
- Modify: `tests/test_rationale_parser.py`

- [ ] **Step 1: Add failing test for multi-line concatenation**

Append to `tests/test_rationale_parser.py`:

```python
def test_multi_line_rationale_concatenated(tmp_path: Path) -> None:
    """Folgezeilen mit '# ' (kein WHY-keyword) gehören zur rationale."""
    src_file = tmp_path / "module.py"
    src_file.write_text(
        "# WHY(#182, performance): SQLite busy_timeout=5s.\n"
        "# Single-Tx > 50 rows blockt smoke-test concurrent writes.\n"
        "# CHANGE WITH CARE.\n"
        "def commit_chunked(rows):\n"
        "    pass\n"
    )
    edges = extract_rationale_edges(
        src_file, repo_slug="demo", workspace_id="bene",
    )
    assert len(edges) == 1
    e = edges[0]
    assert e["target"] == "module.commit_chunked"
    assert e["context"] == "#182, performance"
    assert "SQLite busy_timeout=5s" in e["rationale"]
    assert "Single-Tx" in e["rationale"]
    assert "CHANGE WITH CARE" in e["rationale"]
    # Newlines preserved as join-char
    assert "\n" in e["rationale"]
```

- [ ] **Step 2: Run test — verify RED**

```bash
python3 -m pytest tests/test_rationale_parser.py::test_multi_line_rationale_concatenated -v
```

Expected: FAIL — current parser nimmt nur die erste Zeile.

- [ ] **Step 3: Refactor parser für multi-line**

Replace the `for tok in tokens` loop in `rationale_parser.py` with:

```python
    # Tokenize, sammle WHY-marker MIT Folgezeilen (`# ...` ohne WHY-keyword)
    markers: list[tuple[int, str, str]] = []  # (line_no, refs, rationale)
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
        markers.append((marker_line, refs, "\n".join(rationale_lines)))
        i = j
```

- [ ] **Step 4: Run test — verify GREEN + nothing else broken**

```bash
python3 -m pytest tests/test_rationale_parser.py -v
```

Expected: 2/2 PASS (Task 2's Test + new multi-line Test).

- [ ] **Step 5: Commit**

```bash
git add src/wiki_v2/rationale_parser.py tests/test_rationale_parser.py
git commit -m "feat(wiki_v2): multi-line WHY-blocks im rationale_parser

Folgezeilen mit '# ' (ohne WHY-keyword) werden mit \\n als join-char
an die rationale gehängt. Block endet bei Leer-Zeile, neuem WHY-marker,
oder code. Test mit 3-Zeilen-Begründung verifiziert."
```

---

### Task 4: Qualified Names mit Class-Hierarchy + Async

**Files:**
- Modify: `src/wiki_v2/rationale_parser.py::_node_target_name`
- Modify: `tests/test_rationale_parser.py`

- [ ] **Step 1: Write failing tests for class-method + async**

Append to `tests/test_rationale_parser.py`:

```python
def test_qualified_name_includes_class(tmp_path: Path) -> None:
    """Marker im class body produziert 'module.Class.method'."""
    src_file = tmp_path / "module.py"
    src_file.write_text(
        "class JobRunner:\n"
        "    # WHY(#100): retry-loop avoids transient deploy 502s\n"
        "    def run(self):\n"
        "        pass\n"
    )
    edges = extract_rationale_edges(
        src_file, repo_slug="demo", workspace_id="bene",
    )
    assert len(edges) == 1
    assert edges[0]["target"] == "module.JobRunner.run"


def test_handles_async_function(tmp_path: Path) -> None:
    """async def wird auch erkannt (AsyncFunctionDef-Node)."""
    src_file = tmp_path / "module.py"
    src_file.write_text(
        "# WHY(#88): async because the LLM-call blocks 30s\n"
        "async def fetch():\n"
        "    pass\n"
    )
    edges = extract_rationale_edges(
        src_file, repo_slug="demo", workspace_id="bene",
    )
    assert len(edges) == 1
    assert edges[0]["target"] == "module.fetch"
```

- [ ] **Step 2: Run tests — verify RED**

```bash
python3 -m pytest tests/test_rationale_parser.py::test_qualified_name_includes_class tests/test_rationale_parser.py::test_handles_async_function -v
```

Expected:
- `test_handles_async_function`: PASS already (AsyncFunctionDef ist im isinstance-Check)
- `test_qualified_name_includes_class`: FAIL — target ist `'module.run'` statt `'module.JobRunner.run'`

- [ ] **Step 3: Refactor for class-aware qualified-names**

Replace the `line_to_node` build in `extract_rationale_edges` with a parent-aware walk:

```python
    # Build a map: line_no → (target_node, parent_class_name | None)
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
```

Update the marker-loop:

```python
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
        edges.append({...})  # bleibt gleich
```

Update `_node_target_name`:

```python
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
```

- [ ] **Step 4: Run tests — verify GREEN**

```bash
python3 -m pytest tests/test_rationale_parser.py -v
```

Expected: 4/4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wiki_v2/rationale_parser.py tests/test_rationale_parser.py
git commit -m "feat(wiki_v2): qualified target-names inkl. class hierarchy

_node_target_name kennt jetzt Class-context: 'module.JobRunner.run'
statt nur 'module.run'. AsyncFunctionDef ebenfalls covered (war
schon im isinstance-Check, nur Test fehlte)."
```

---

### Task 5: Skip non-trivial Targets + Warn-Log

**Files:**
- Modify: `tests/test_rationale_parser.py`

- [ ] **Step 1: Write failing test for skip + warn**

Append to `tests/test_rationale_parser.py`:

```python
def test_skips_marker_before_for_loop(tmp_path: Path, caplog) -> None:
    """Marker vor non-trivialem Target (for/if/try/while) wird geskipped
    UND ein WARN-Log geschrieben (kein silent drop)."""
    import logging
    src_file = tmp_path / "module.py"
    src_file.write_text(
        "def main():\n"
        "    # WHY(#X): unklar wo das hin soll\n"
        "    for r in rows:\n"
        "        process(r)\n"
    )
    with caplog.at_level(logging.WARNING):
        edges = extract_rationale_edges(
            src_file, repo_slug="demo", workspace_id="bene",
        )
    assert edges == []
    assert any(
        "rationale-skipped" in rec.message and "non-trivial-target" in rec.message
        for rec in caplog.records
    )
```

- [ ] **Step 2: Run test — verify RED or already-GREEN**

```bash
python3 -m pytest tests/test_rationale_parser.py::test_skips_marker_before_for_loop -v
```

Expected: This may already PASS if Task 4's lookup misses the for-loop. Verify the LOG-assertion. Wenn die log-message fehlt: scope-fix in Task 4 hat den log-call gewahrt — sonst nochmal anpassen.

- [ ] **Step 3: Adjust if needed**

Wenn der Test wegen FEHLENDEM log-message fail't: prüfe in `extract_rationale_edges` dass das `_log.warning(...)` IMMER bei `target_info is None` aufgerufen wird (nicht in irgendeinem early-return weggekürzt).

- [ ] **Step 4: Run test — verify GREEN**

```bash
python3 -m pytest tests/test_rationale_parser.py -v
```

Expected: 5/5 PASS.

- [ ] **Step 5: Commit (auch wenn nur test added, sonst skip)**

```bash
git add tests/test_rationale_parser.py
git commit -m "test(rationale_parser): skip vor non-trivial target loggt warn

Regression-guard: Marker vor for/if/try wird mit
'rationale-skipped: file=X line=N reason=non-trivial-target' geloggt
und produziert keine edge — kein silent drop, User merkt dass sein
Marker nicht greift."
```

---

### Task 6: ATTACH wiki_v2.db in init_memory_db + RetrievalRecord field

**Files:**
- Modify: `src/memory/store.py::init_memory_db`
- Modify: `src/memory/schema.py::RetrievalRecord`
- Test: `tests/test_memory_store.py` (Erweiterung) + `tests/test_memory_retrieval.py`

- [ ] **Step 1: Write failing test for ATTACH**

Append to `tests/test_memory_store.py`:

```python
def test_init_memory_db_attaches_wiki_v2(tmp_path):
    """init_memory_db attached cache/wiki_v2.db wenn vorhanden, sodass
    Cross-DB-JOINs auf wikidb.wiki_edges direkt funktionieren."""
    import sqlite3
    from src.memory.store import init_memory_db
    from src.wiki_v2 import store as wstore

    cache = tmp_path / "cache"
    cache.mkdir()
    wiki_db = cache / "wiki_v2.db"
    wraw = sqlite3.connect(str(wiki_db))
    wstore.init_db(wraw)  # adjust if init differently named
    wraw.close()

    # Patch CACHE_DIR so init_memory_db looks at our tmp_path/cache
    import src.config as _cfg
    orig = _cfg.CACHE_DIR
    _cfg.CACHE_DIR = cache
    try:
        conn = init_memory_db(cache / "memory.db")
        assert getattr(conn, "_wiki_attached", False) is True
        # Cross-DB query should not crash
        n = conn.execute("SELECT COUNT(*) FROM wikidb.wiki_edges").fetchone()[0]
        assert n == 0  # empty after init
    finally:
        _cfg.CACHE_DIR = orig


def test_init_memory_db_skips_attach_when_no_wiki_db(tmp_path):
    """Cold-start ohne wiki_v2.db: Flag = False, keine Crash."""
    from src.memory.store import init_memory_db
    cache = tmp_path / "cache"
    cache.mkdir()
    import src.config as _cfg
    orig = _cfg.CACHE_DIR
    _cfg.CACHE_DIR = cache
    try:
        conn = init_memory_db(cache / "memory.db")
        assert getattr(conn, "_wiki_attached", True) is False
    finally:
        _cfg.CACHE_DIR = orig
```

- [ ] **Step 2: Run tests — verify RED**

```bash
python3 -m pytest tests/test_memory_store.py::test_init_memory_db_attaches_wiki_v2 tests/test_memory_store.py::test_init_memory_db_skips_attach_when_no_wiki_db -v
```

Expected: FAIL — `_wiki_attached` field gibt's noch nicht.

- [ ] **Step 3: Implement ATTACH in init_memory_db**

Open `src/memory/store.py`, find `init_memory_db()`, after the `init_db`/`CREATE TABLE`-Block, before `return conn` insert:

```python
    # ATTACH wiki_v2.db als 'wikidb' für Cross-DB-Joins (rationale-edges,
    # cluster-lookups). Spec: docs/superpowers/specs/2026-05-09-rationale-edges-design.md.
    # Cold-start ohne wiki_v2.db: Flag = False, alle JOINs silent skip.
    try:
        from src.config import CACHE_DIR
        wiki_path = (CACHE_DIR / "wiki_v2.db") if db_path is None \
            else db_path.parent / "wiki_v2.db"
    except Exception:
        wiki_path = None

    conn._wiki_attached = False  # type: ignore[attr-defined]
    if wiki_path is not None and wiki_path.exists():
        try:
            conn.execute("ATTACH DATABASE ? AS wikidb", (str(wiki_path),))
            conn._wiki_attached = True  # type: ignore[attr-defined]
        except sqlite3.OperationalError as e:
            import logging
            logging.getLogger(__name__).warning(
                "wiki_v2.db ATTACH failed (silent skip): %s", e,
            )
```

- [ ] **Step 4: Add RetrievalRecord field**

Open `src/memory/schema.py`. After `score_predicted_topic` field add:

```python
    # Issue #185/#182 follow-up: rationale-edges aus wiki_v2 für diesen chunk.
    # Liste von dicts {target, context, why} — wird im compress_for_prompt
    # als '**Rationale:**' Block gerendert. Empty wenn kein wiki-Match.
    rationale_edges: list[dict] = field(default_factory=list)
```

Plus in `to_dict()`:
```python
            "rationale_edges": self.rationale_edges,
```

- [ ] **Step 5: Run tests — verify GREEN**

```bash
python3 -m pytest tests/test_memory_store.py -v
```

Expected: PASS für die 2 neuen Tests + alle existierenden.

```bash
python3 -m pytest tests/test_memory_retrieval.py -v 2>&1 | tail -3
```

Expected: alle weiterhin grün.

- [ ] **Step 6: Commit**

```bash
git add src/memory/store.py src/memory/schema.py tests/test_memory_store.py
git commit -m "feat(memory): ATTACH wiki_v2.db + RetrievalRecord.rationale_edges

init_memory_db attached cache/wiki_v2.db als 'wikidb' wenn vorhanden,
sonst _wiki_attached=False (silent skip in retrieval). RetrievalRecord
bekommt rationale_edges-Feld als Trägermedium für die WHY-Strings die
gleich in compress_for_prompt eingebracht werden."
```

---

### Task 7: rationale-edges in /memory/search results + compress_for_prompt rendering

**Files:**
- Modify: `src/memory/retrieval.py::_rerank` (Cross-DB-Lookup)
- Modify: `src/memory/retrieval.py::compress_for_prompt`
- Test: `tests/test_memory_retrieval.py`

- [ ] **Step 1: Write failing test for end-to-end attach**

Append to `tests/test_memory_retrieval.py`:

```python
def test_search_attaches_rationale_edges(tmp_path):
    """End-to-end: chunk im wiki-rationale-edge → rationale_edges befüllt."""
    import sqlite3
    from src.memory.retrieval import search
    from src.memory.store import init_memory_db, upsert_source, insert_chunk
    from src.memory.schema import Source, Chunk
    from src.wiki_v2 import store as wstore

    # Setup wiki_v2.db with one rationale-edge
    cache = tmp_path / "cache"
    cache.mkdir()
    wraw = sqlite3.connect(str(cache / "wiki_v2.db"))
    wstore.init_db(wraw)
    wraw.execute(
        "INSERT INTO wiki_edges(source, target, repo_slug, workspace_id, "
        "type, weight, context, rationale) "
        "VALUES('src/cli.py','cli._SLUG_RE','demo','bene','rationale',1.0,"
        "'#185','path-traversal defence')"
    )
    wraw.commit()
    wraw.close()

    # Setup memory.db
    import src.config as _cfg
    orig_cache = _cfg.CACHE_DIR
    _cfg.CACHE_DIR = cache
    try:
        conn = init_memory_db(cache / "memory.db")
        assert conn._wiki_attached is True

        # 1 source mit path='src/cli.py' (matcht den wiki_node)
        src = Source(
            source_id="src::cli", source_type="repo_file",
            repo="demo", path="src/cli.py", content_hash="sha256:c",
        )
        upsert_source(conn, src)
        ch = Chunk(
            chunk_id=Chunk.make_id("src::cli", 0, "function"),
            source_id="src::cli", chunk_level="function", ordinal=0,
            text="_SLUG_RE = re.compile(r'^[a-z]+$')",
            text_hash="sha256:t", category_labels=["api"],
            created_at="2026-04-08T10:00:00+00:00",
        )
        insert_chunk(conn, ch)

        results = search(
            "slug regex", conn, None, "http://fake-ollama",
            opts={"top_k": 5, "include_text": False, "llm_prefilter": False,
                  "repo": "demo"},
        )
        assert len(results) >= 1
        match = next(r for r in results if r.chunk_id == ch.chunk_id)
        assert len(match.rationale_edges) == 1
        e = match.rationale_edges[0]
        assert e["target"] == "cli._SLUG_RE"
        assert e["context"] == "#185"
        assert e["why"] == "path-traversal defence"
    finally:
        _cfg.CACHE_DIR = orig_cache


def test_search_no_rationale_when_wiki_db_missing(tmp_path):
    """Cold-start ohne wiki_v2.db: rationale_edges == [], kein crash."""
    from src.memory.retrieval import search
    from src.memory.store import init_memory_db, upsert_source, insert_chunk
    from src.memory.schema import Source, Chunk
    cache = tmp_path / "cache"
    cache.mkdir()
    import src.config as _cfg
    orig_cache = _cfg.CACHE_DIR
    _cfg.CACHE_DIR = cache
    try:
        conn = init_memory_db(cache / "memory.db")
        assert conn._wiki_attached is False  # no wiki_v2.db

        src = Source(
            source_id="src::cli", source_type="repo_file",
            repo="demo", path="src/cli.py", content_hash="sha256:c",
        )
        upsert_source(conn, src)
        ch = Chunk(
            chunk_id=Chunk.make_id("src::cli", 0, "function"),
            source_id="src::cli", chunk_level="function", ordinal=0,
            text="x = 1", text_hash="sha256:t", category_labels=["api"],
            created_at="2026-04-08T10:00:00+00:00",
        )
        insert_chunk(conn, ch)

        results = search(
            "slug regex", conn, None, "http://fake-ollama",
            opts={"top_k": 5, "include_text": False, "llm_prefilter": False},
        )
        assert len(results) >= 1
        for r in results:
            assert r.rationale_edges == []
    finally:
        _cfg.CACHE_DIR = orig_cache
```

- [ ] **Step 2: Run tests — verify RED**

```bash
python3 -m pytest tests/test_memory_retrieval.py::test_search_attaches_rationale_edges tests/test_memory_retrieval.py::test_search_no_rationale_when_wiki_db_missing -v
```

Expected: FAIL — rationale_edges bleibt `[]` weil noch kein lookup in `_rerank`.

- [ ] **Step 3: Add rationale-lookup to _rerank**

In `src/memory/retrieval.py::_rerank`, NACH dem records-build-loop, vor `return ranked`:

```python
    # Issue #185/#182 follow-up: enrich each record with rationale-edges
    # from wiki_v2.db (if attached). Joinkey: chunk's source.path → wiki_edges.source.
    if getattr(conn, "_wiki_attached", False) and ranked:
        chunk_by_id = {c.chunk_id: c for c in candidates}
        for record in ranked:
            chunk = chunk_by_id.get(record.chunk_id)
            if chunk is None:
                continue
            # Lookup source.path via memory.db
            src_row = conn.execute(
                "SELECT path FROM sources WHERE source_id = ?",
                (chunk.source_id,),
            ).fetchone()
            if not src_row or not src_row[0]:
                continue
            try:
                rows = conn.execute(
                    "SELECT rationale, target, context FROM wikidb.wiki_edges "
                    "WHERE source = ? AND type = 'rationale' "
                    "AND rationale != ''",
                    (src_row[0],),
                ).fetchall()
            except Exception:
                continue
            record.rationale_edges = [
                {"target": t, "context": c, "why": r}
                for r, t, c in rows
            ]
```

- [ ] **Step 4: Run tests — verify GREEN**

```bash
python3 -m pytest tests/test_memory_retrieval.py::test_search_attaches_rationale_edges tests/test_memory_retrieval.py::test_search_no_rationale_when_wiki_db_missing -v
```

Expected: PASS.

- [ ] **Step 5: Add compress_for_prompt rendering test**

Append to `tests/test_memory_retrieval.py`:

```python
def test_compress_for_prompt_renders_rationale_block():
    """Wenn ein Record rationale_edges hat, taucht ein '**Rationale:**'
    Block im output auf — Claude sieht das WHY beim Inject."""
    from src.memory.retrieval import compress_for_prompt
    from src.memory.schema import RetrievalRecord
    rec = RetrievalRecord(
        chunk_id="c1", score_final=0.5,
        source_id="src::cli", text="_SLUG_RE = re.compile(...)",
        category_labels=["api"],
        rationale_edges=[{
            "target": "cli._SLUG_RE",
            "context": "#185",
            "why": "path-traversal defence",
        }],
    )
    out = compress_for_prompt([rec], char_budget=2000)
    assert "**Rationale:**" in out
    assert "cli._SLUG_RE" in out
    assert "path-traversal defence" in out
    assert "#185" in out
```

- [ ] **Step 6: Run test — verify RED**

```bash
python3 -m pytest tests/test_memory_retrieval.py::test_compress_for_prompt_renders_rationale_block -v
```

Expected: FAIL — `**Rationale:**` nicht im output.

- [ ] **Step 7: Add rationale-rendering to compress_for_prompt**

Find `compress_for_prompt` in `src/memory/retrieval.py`. After the chunk-text-rendering, BEFORE the per-record-block ends, insert:

```python
        if record.rationale_edges:
            block.append("\n**Rationale:**")
            for re_dict in record.rationale_edges:
                tgt = re_dict.get("target", "")
                why = re_dict.get("why", "")
                ctx = re_dict.get("context", "")
                ctx_suffix = f" ({ctx})" if ctx else ""
                block.append(f"- `{tgt}` — {why}{ctx_suffix}")
```

(The exact integration depends on existing block-build pattern in compress_for_prompt — adapt to match local style. Verify `block` variable is the per-record list.)

- [ ] **Step 8: Run tests — verify GREEN**

```bash
python3 -m pytest tests/test_memory_retrieval.py -v 2>&1 | tail -10
```

Expected: alle existing + 3 neue grün.

- [ ] **Step 9: Commit**

```bash
git add src/memory/retrieval.py tests/test_memory_retrieval.py
git commit -m "feat(retrieval): rationale_edges in /memory/search + compress_for_prompt

_rerank holt für jeden top-K chunk dessen wiki_edges-Rationale aus
wikidb (attached in init_memory_db). compress_for_prompt rendert
**Rationale:** block mit target + why + #issue-context. Memory-Inject
zeigt Claude die defensive WHY-Knowledge BEVOR er Code anfasst."
```

---

### Task 8: edge_extractor-Integration + CLI-Backfill

**Files:**
- Modify: `src/wiki_v2/edge_extractor.py` (oder wie immer der Hauptaufruf heißt)
- Modify: `src/cli_args.py` — neuen `--extract-rationale` Flag
- Modify: `src/cli.py` — Dispatcher
- Test: `tests/test_rationale_parser.py` (Integration mit edge_extractor)

- [ ] **Step 1: Locate edge_extractor entry point**

```bash
grep -rn "def extract_edges\|register.*edge\|concept_link\|label_cooccurrence" /home/nileneb/Desktop/MayringCoder/src/wiki_v2/ | head -10
```

Identifiziere die zentrale `extract_edges_for_repo()`-Funktion oder vergleichbar. Wenn `paper_rules.py` der Pattern ist (`detect_shared_concepts`-Style), dann gibt es eine Aufruf-Liste.

- [ ] **Step 2: Add failing integration test**

Append to `tests/test_rationale_parser.py`:

```python
def test_edge_extractor_persists_rationale_edges(tmp_path: Path) -> None:
    """edge_extractor-Aufruf für ein Repo mit WHY-marker → wiki_edges DB
    enthält rationale-rows."""
    import sqlite3
    from src.wiki_v2 import store as wstore
    from src.wiki_v2.rationale_parser import extract_rationale_edges_for_repo

    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "module.py").write_text(
        "# WHY(#185): path-traversal defence\n"
        "_SLUG_RE = 1\n"
    )

    db_path = tmp_path / "wiki.db"
    conn = sqlite3.connect(str(db_path))
    wstore.init_db(conn)

    n = extract_rationale_edges_for_repo(
        repo_root, conn, repo_slug="demo", workspace_id="bene",
    )
    assert n == 1

    rows = conn.execute(
        "SELECT source, target, type, rationale, context FROM wiki_edges "
        "WHERE type='rationale'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][3] == "path-traversal defence"
```

- [ ] **Step 3: Run test — verify RED**

```bash
python3 -m pytest tests/test_rationale_parser.py::test_edge_extractor_persists_rationale_edges -v
```

Expected: FAIL — `extract_rationale_edges_for_repo` existiert nicht.

- [ ] **Step 4: Implement repo-level wrapper**

Append to `src/wiki_v2/rationale_parser.py`:

```python
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
    for py_file in repo_root.rglob("*.py"):
        # Skip vendor/test_artifact paths
        if "/.git/" in str(py_file) or "/node_modules/" in str(py_file):
            continue
        edges = extract_rationale_edges(
            py_file, repo_slug=repo_slug, workspace_id=workspace_id,
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
    conn.commit()
    return persisted
```

- [ ] **Step 5: Run test — verify GREEN**

```bash
python3 -m pytest tests/test_rationale_parser.py::test_edge_extractor_persists_rationale_edges -v
```

Expected: PASS.

- [ ] **Step 6: Add CLI flag --extract-rationale**

Open `src/cli_args.py`. Find a sibling-flag (e.g. `--rebuild-transitions`). Add:

```python
    p.add_argument("--extract-rationale", action="store_true",
                   help="Extract WHY(...)-marker comments into wiki_edges (rationale-edges)")
```

Open `src/cli.py`. Add dispatcher:

```python
def _cmd_extract_rationale(args, repo_url: str) -> None:
    from src.config import CACHE_DIR
    from src.wiki_v2.rationale_parser import extract_rationale_edges_for_repo
    from src.wiki_v2 import store as wstore
    from src.identity.cli import resolve_cli_workspace
    import sqlite3

    if not repo_url:
        print("Fehler: --extract-rationale braucht --repo")
        return
    repo_root = Path(_repo_slug(repo_url) or "")
    # Caller (Cron / v2-chain) muss bereits ge'gitingest haben — repo_root ist
    # entweder ein lokaler Pfad oder ein cache-pfad. Wir nehmen CWD wenn nichts
    # explizit gesetzt.
    if not repo_root.is_absolute():
        repo_root = Path.cwd()
    conn = sqlite3.connect(str(CACHE_DIR / "wiki_v2.db"))
    wstore.init_db(conn)
    workspace = resolve_cli_workspace(args, conn=None, auto_create=False)
    n = extract_rationale_edges_for_repo(
        repo_root, conn, repo_slug=_repo_slug(repo_url),
        workspace_id=workspace,
    )
    conn.close()
    print(f"[rationale] persisted={n} edges")
```

In `main()`, near other dispatch lines (line ~450):

```python
    if getattr(args, "extract_rationale", False):    _cmd_extract_rationale(args, repo_url);              sys.exit(0)
```

Plus: Add `--extract-rationale` to v2-chain in `src/api/routes/jobs.py::_run_with_v2_postingest`:

```python
    rat_id = _make_job(workspace_id)
    v2_jobs["rationale"] = rat_id
    asyncio.create_task(_run_checker_job(
        rat_id,
        ["--repo", repo, "--extract-rationale", "--workspace-id", workspace_id],
        workspace_id,
    ))
```

- [ ] **Step 7: Run all tests + smoke check**

```bash
python3 -m pytest tests/ -x --ignore=tests/test_dashboard_e2e.py 2>&1 | tail -3
```

Expected: alle grün.

- [ ] **Step 8: Commit**

```bash
git add src/wiki_v2/rationale_parser.py src/cli.py src/cli_args.py src/api/routes/jobs.py tests/test_rationale_parser.py
git commit -m "feat(cli): --extract-rationale Flag + v2-chain Hook

extract_rationale_edges_for_repo läuft repo-walk, persistiert via
ON CONFLICT UPSERT in wiki_edges. CLI-Flag --extract-rationale +
v2-chain-Slot 'rationale' parallel zu ambient/predictive/images.
Beim nächsten /analyze auf einem Repo werden alle WHY-marker
automatisch geparsed."
```

---

### Task 9: Coverage-map + Issue-Updates + Smoke

**Files:**
- Modify: `docs/smoke_coverage_map.md`

- [ ] **Step 1: Add coverage entry**

Edit `docs/smoke_coverage_map.md`. In der "Aktive Smoke-Checks"-Tabelle hinzufügen:

```markdown
| 184a | Rationale-Edges in /memory/search | Pytest `tests/test_rationale_parser.py` (8 tests) + `tests/test_memory_retrieval.py::test_search_attaches_rationale_edges` (Cross-DB-JOIN) |
```

- [ ] **Step 2: Push**

```bash
git add docs/smoke_coverage_map.md
git commit -m "docs(coverage-map): rationale-edges feature"
git push origin master
```

- [ ] **Step 3: Verify deploy + smoke green**

```bash
DRID=$(gh run list -R Nileneb/MayringCoder --workflow="Build & Push" --limit 1 --json databaseId -q '.[0].databaseId')
until [ "$(gh run view $DRID -R Nileneb/MayringCoder --json status -q '.status' 2>/dev/null)" = "completed" ]; do sleep 18; done
echo "deploy: $(gh run view $DRID -R Nileneb/MayringCoder --json conclusion -q '.conclusion')"
gh run list -R Nileneb/MayringCoder --workflow="Post-deploy smoke (production)" --limit 1 --json conclusion 2>&1 | head
```

Expected: deploy success, smoke success (45/45 oder besser).

- [ ] **Step 4: Live-test on prod**

```bash
TOK=$(ssh nileneb@u-server 'grep ^MCP_SERVICE_TOKEN= ~/app.linn.games/.env | cut -d= -f2-')
ssh nileneb@u-server "curl -sS -X POST 'https://mcp.linn.games/analyze' -H 'Authorization: Bearer $TOK' -H 'Content-Type: application/json' -d '{\"repo\":\"https://github.com/nileneb/mayringcoder\",\"full\":false,\"adversarial\":false,\"no_pi\":true}'" 2>&1 | head
```

Expected: job_id zurück. Nach run: `wiki_edges WHERE type='rationale'` count > 0.

- [ ] **Step 5: Final state report**

Run:
```bash
ssh nileneb@u-server 'docker exec mayring-mayring-api-1 sh -c "sqlite3 /app/cache/wiki_v2.db \"SELECT COUNT(*) FROM wiki_edges WHERE type = \x27rationale\x27\""'
```

Expected: > 0 (depending on how many WHY-marker im aktuellen master sind — initial möglicherweise 0 weil noch keine markers existieren; backfill kommt in einem separaten PR der WHY-Marker auf z.B. _SLUG_RE und IGIO-batch-loop platziert).

---

## Self-Review Checklist

- [ ] **Spec coverage:** Alle 5 Architektur-Komponenten (Marker / Parser / Schema / search-Integration / ATTACH) haben Tasks
- [ ] **Placeholder scan:** Keine "TBD"/"TODO"/Skip-this-step Stellen
- [ ] **Type consistency:**
  - `extract_rationale_edges` (single-file) vs `extract_rationale_edges_for_repo` (repo-walk) — beide Namen konsistent
  - Edge-dict-keys: `source`, `target`, `type`, `weight`, `context`, `rationale`, `repo_slug`, `workspace_id` — gleich in Task 2 und Task 8
  - RetrievalRecord-field: `rationale_edges` — gleich in Task 6 und Task 7
  - compress_for_prompt-key: `why` vs Edge-key `rationale` — bewusst unterschiedlich (search-result-Output umbenannt zu `why` für UX), siehe Task 7 Step 7
- [ ] Spec-Anforderung "Backfill-Befehl" → Task 8 Step 6 (--extract-rationale flag) ✓
- [ ] Spec-Anforderung "Markdown-Block-Output mit Bullet" → Task 7 Step 7 ✓
- [ ] Spec-Anforderung "skip + WARN log bei non-trivial" → Task 5 ✓
