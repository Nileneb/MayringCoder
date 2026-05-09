# Rationale-Edges für wiki_v2 — Design

**Status:** Brainstorming abgeschlossen, awaiting user approval.
**Use-Case:** WHY-knowledge zu Code/Rules verknüpfen, im Memory-Inject co-rendern.
**Issue-Bezug:** Folge von #185 (path-traversal defence) und #182 (SQLite-lock workaround) — beide Workarounds würden ohne dokumentierte Rationale beim nächsten Refactoring "wegoptimiert".

---

## Ziel

Wenn ein Defensive-Pattern oder eine Performance-Decision im Code steht, soll
beim nächsten `/memory/search` der WHY-String automatisch mitkommen — sodass
Claude (oder ein menschlicher Maintainer) den Code nicht ohne Verständnis
ändert.

Beispiel-Trigger heute: Issue #185 fix `_SLUG_RE = re.compile(r"^[a-z0-9]...")`
ohne dokumentierte rationale → in 3 Monaten weiß niemand mehr warum so
strikt → Refactor lockert die regex → Path-Traversal kommt zurück.

---

## Architektur — 3 Komponenten

### 1. Comment-Marker im Code (Source-of-Truth)

Marker-Format:

```python
# WHY(#185): path-traversal defence — Slug-Input darf nicht ../etc/passwd ergeben
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}\Z")

# WHY(#182, performance): SQLite busy_timeout=5s. Single-Tx über >50 rows
# blockt smoke-test concurrent writes (10+ min stall). CHANGE WITH CARE.
def _commit_chunked(rows: list[Row]) -> None:
    ...
```

**Regex (Python-Multiline):**
```
^\s*#\s*WHY\(([^)]+)\):\s*(.+?)(?=\n\s*(?!#)|\Z)
```

Capture-Gruppen:
- `(1)` = ref-list (`#185` oder `#182, performance` — comma-separierte issue-ids + freie tags)
- `(2)` = freier rationale-Text, multi-line wenn die Folgezeile mit `# ` weitergeht

### 2. Parser & Target-Resolution

`src/wiki_v2/rationale_parser.py`:

```python
def extract_rationale_edges(
    file_path: Path,
    repo_slug: str,
    workspace_id: str,
) -> list[Edge]:
    """Findet alle WHY(...)-Marker und matcht jeden an das nächste Code-Symbol.

    Edge-Schema:
      source     = file_path relativ zum Repo (z.B. 'src/cli.py')
      target     = qualified name des nächsten Symbols (z.B. 'cli._SLUG_RE',
                   'cli._cmd_classify_igio', 'cli.JobRunner.run')
      type       = 'rationale'
      weight     = 1.0
      context    = ref-list aus capture(1) (z.B. '#185' oder '#182,performance')
      rationale  = capture(2) — multi-line ok, joined mit \\n

    Skipped wenn der Marker NICHT direkt vor:
      - Assign/AnnAssign (top-level oder in class body)
      - FunctionDef / AsyncFunctionDef
      - ClassDef
    steht — z.B. wenn der Marker vor einer for/if/try/while-Loop sitzt. Dann
    loggen wir 'rationale-skipped: file=X line=N reason=non-trivial-target'
    und persistieren KEINE edge. Klare Semantik > magical fallbacks.
    """
```

**Implementation-Skizze:**

1. Tokenize-Pass: `tokenize.tokenize(file)` → liste aller Tokens, Comment-Tokens behalten
2. Filter Comment-Tokens auf `# WHY(...)` Pattern
3. Für jeden Match: scanne folgende Tokens bis erstes `NAME` oder `NEWLINE+def/class`
4. AST-Lookup: `ast.parse(file)` und walk → finde Node mit gleicher line/col
5. Wenn Node-Type ∉ {Assign, AnnAssign, FunctionDef, AsyncFunctionDef, ClassDef}: skip + warn
6. Qualified-Name-Bauer: walk parents → join mit `.`

**Aufruf-Stelle:** in `src/wiki_v2/edge_extractor.py::extract_edges_for_repo`,
parallel zu den existierenden `concept_link`/`call`/`shared_type`-Extraktoren.
Läuft beim repo-analyze (also v2-chain → wiki-rebuild → edge-extraction).

### 3. DB-Schema

```sql
ALTER TABLE wiki_edges ADD COLUMN rationale TEXT DEFAULT '';
```

Idempotent in `init_wiki_v2_db()`:

```python
cur = conn.execute("PRAGMA table_info(wiki_edges)")
cols = {row[1] for row in cur.fetchall()}
if "rationale" not in cols:
    conn.execute("ALTER TABLE wiki_edges ADD COLUMN rationale TEXT DEFAULT ''")
```

`UNIQUE`-Constraint auf `(source, target, type, workspace_id)` bleibt — wenn
zwei WHY-Marker auf das gleiche Symbol zeigen (sehr selten), gewinnt der
zuletzt-extrahierte (UPSERT-on-conflict mit rationale-update).

### 4. /memory/search-Integration

In `src/memory/retrieval.py::_rerank` nach dem Score-Loop, vor `return ranked`:

```python
# Cross-DB-JOIN nur wenn wiki_v2.db attached ist (siehe init_memory_db)
if hasattr(conn, "_wiki_attached") and conn._wiki_attached:
    for record in ranked:
        # source.path lookup, kann fehlen wenn source_type != 'repo_file'
        chunk = next((c for c in candidates if c.chunk_id == record.chunk_id), None)
        if chunk is None:
            continue
        src_path = _source_path_for(conn, chunk.source_id)
        if not src_path:
            continue
        rows = conn.execute(
            "SELECT rationale, target, context FROM wikidb.wiki_edges "
            "WHERE source = ? AND type = 'rationale' "
            "AND rationale != '' AND repo_slug = ?",
            (src_path, repo_slug),
        ).fetchall()
        record.rationale_edges = [
            {"target": t, "context": c, "why": r} for r, t, c in rows
        ]
```

**Schema-Erweiterung in `RetrievalRecord`:**

```python
rationale_edges: list[dict] = field(default_factory=list)
```

**`compress_for_prompt`-Rendering** — wenn `record.rationale_edges`:

```
### <source_id>
<chunk text>

**Rationale:**
- `_SLUG_RE` — path-traversal defence (#185)
- `_cmd_classify_igio` — avoid SQLite lock 5s (#182, performance)
```

So sieht Claude die WHY-Knowledge BEVOR er den Code anfasst.

### 5. ATTACH DATABASE

`init_memory_db()` öffnet zusätzlich `cache/wiki_v2.db` als attached schema
`wikidb`. Falls die Datei nicht existiert (cold-start, nie wiki-rebuild
gelaufen): `conn._wiki_attached = False`, alle Lookups silently skip → keine
Crashes, keine Boost-Edges, nur Mainstream-Score.

```python
def init_memory_db(db_path: Path | None = None) -> DBAdapter:
    ...
    wiki_path = (db_path or DEFAULT_DB_PATH).parent / "wiki_v2.db"
    if wiki_path.exists():
        try:
            conn.execute("ATTACH DATABASE ? AS wikidb", (str(wiki_path),))
            conn._wiki_attached = True
        except sqlite3.OperationalError:
            conn._wiki_attached = False
    else:
        conn._wiki_attached = False
    return conn
```

---

## Tests (TDD red-green)

### `tests/test_rationale_parser.py` (neu)

| Test | Setup | Assertion |
|---|---|---|
| `test_extract_simple_marker_before_assign` | File mit `# WHY(#185): x` vor `_SLUG_RE = re.compile(...)` | edge `target='module._SLUG_RE'`, `rationale='x'`, `context='#185'` |
| `test_multi_line_rationale_concatenated` | `# WHY(#182): line1\\n# line2` vor function | rationale = `'line1\\nline2'` |
| `test_skips_marker_before_for_loop` | `# WHY(#X): ...` vor `for r in rows:` | 0 edges + 1 warning log mit reason='non-trivial-target' |
| `test_qualified_name_includes_class` | Marker im class body | target = `'module.MyClass.my_method'` |
| `test_handles_async_function` | vor `async def foo` | edge created with target |
| `test_multiple_markers_in_file` | 3 separate WHY-blocks | 3 edges |

### `tests/test_memory_retrieval.py` (Erweiterung)

| Test | Setup | Assertion |
|---|---|---|
| `test_search_attaches_rationale_edges_when_present` | wikidb.wiki_edges hat rationale-edge für source.path eines top-K chunks | `record.rationale_edges == [{"target": "...", "context": "#185", "why": "..."}]` |
| `test_search_empty_rationale_when_wiki_db_missing` | wiki_v2.db nicht vorhanden | `rationale_edges == []`, kein crash |
| `test_compress_for_prompt_renders_rationale_block` | record mit rationale_edges | Output enthält `**Rationale:**` + bullet |

---

## Migration & Rollout

1. **Schema-Migration**: `init_wiki_v2_db()` checkt + fügt column hinzu (idempotent, kein SQL-script-Pfad nötig)
2. **Backfill**: ein einmaliger CLI-Befehl `python -m src.cli --extract-rationale --repo X` läuft den Parser über alle existing wiki_nodes (= file-paths) und persistiert die rationale-edges. Plus auto-trigger als Step in v2-chain bei jedem `/analyze`-run.
3. **Doku**: Kurz-Eintrag in `docs/wiki_v2_extension_points.md` (oder vergleichbares) mit Marker-Spec für Contributor.
4. **Coverage-Map**: Eintrag bei "Aktive Smoke-Checks" für den neuen `tests/test_rationale_parser.py`.

---

## Out of Scope (für separates Issue)

- **HTTP-API** für nachträgliches Edit der rationale-strings (User wählte "Code-Marker only" in Brainstorming).
- **PR-Bot-Integration** (warnt bei Code-Änderungen die rationale-edge berühren).
- **Issue-Linking-Validation** — Marker `WHY(#9999)` zu nicht-existentem Issue: aktuell erlaubt, später optional GitHub-API-Check.
- **LLM-Auto-Generation** (User hat das gegen manuelles bevorzugt).

---

## Ergebnis-Erwartung

Nach Rollout + repo-analyze auf MayringCoder:
- `wiki_edges`-Anzahl steigt um ~10-30 (anhängig wieviel WHY-Marker im Code stehen)
- `/memory/search` results für `cli.py`/`predictive.py` enthalten `rationale_edges`-Block
- Memory-Inject zeigt Claude die Rationale BEFORE er Code lockert/wegoptimiert
