# Mayring Hybrid Categorization + UI Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vollständige induktiv/deduktiv/hybrid Mayring-Kategorisierung via Prompt-Templates, LLM-Modellauswahl in der UI, Conversation-Summary Ingestion, Compaction-aware Retrieval, CLAUDE.md Compact Instructions, und Issue #27 abschließen.

**Architecture:** Prompt-Templates in `prompts/` für jeden Mayring-Modus; `mayring_categorize()` lädt Template + Codebook dynamisch; `search()` bekommt `session_compacted`-Flag für Boost von Conversation-Summary-Chunks; `web_ui.py` bekommt Modell-Dropdown + Mayring-Modus-Felder + Conversation-Tab.

**Tech Stack:** Python, Gradio ≥4.0, SQLite, ChromaDB, Ollama (httpx), FastMCP, pytest, PyYAML

---

## File Map

| Datei | Aktion | Verantwortung |
|---|---|---|
| `prompts/mayring_deduktiv.md` | Create | System-Prompt: geschlossene Kategorienmenge |
| `prompts/mayring_induktiv.md` | Create | System-Prompt: freie Kategorienableitung |
| `prompts/mayring_hybrid.md` | Create | System-Prompt: Anker + neue Kategorien mit [neu] |
| `src/memory_ingest.py` | Modify | `mayring_categorize()` + Helpers + `ingest_conversation_summary()` |
| `src/memory_retrieval.py` | Modify | `search()` + `_rerank()` mit `session_compacted` |
| `src/mcp_server.py` | Modify | `search_memory()` Tool: `compacted` Parameter |
| `src/web_ui.py` | Modify | Modell-Dropdown, Modus/Codebook-Felder, Conversation-Tab |
| `.claude/CLAUDE.md` | Modify | Compact Instructions Abschnitt |
| `tests/test_memory_ingest.py` | Modify | Tests für neue `mayring_categorize()` + `ingest_conversation_summary()` |
| `tests/test_memory_retrieval.py` | Modify | Tests für `session_compacted` |
| `tests/test_web_ui.py` | Modify | Tests für Modell-Selektor + Conversation-Tab |

---

## Task 1: Prompt-Templates erstellen

**Files:**
- Create: `prompts/mayring_deduktiv.md`
- Create: `prompts/mayring_induktiv.md`
- Create: `prompts/mayring_hybrid.md`

- [ ] **Step 1: `prompts/mayring_deduktiv.md` erstellen**

```markdown
You are a qualitative content analyst applying Mayring's deductive content analysis method.

Use ONLY the following predefined categories: {{categories}}

Instructions:
- Assign 1 to 3 categories from the list above to the text chunk below.
- Choose only categories that clearly apply.
- Respond with ONLY a comma-separated list of category names from the provided list.
- Do NOT invent new categories. Do NOT explain your choice.

Example response: api, error_handling
```

- [ ] **Step 2: `prompts/mayring_induktiv.md` erstellen**

```markdown
You are a qualitative content analyst applying Mayring's inductive content analysis method.

Instructions:
- Derive 2 to 5 short category labels directly from the content of this text chunk.
- Labels must be lowercase, concise (1 to 3 words), hyphen-separated if multi-word.
- Labels should precisely describe the main themes present in the chunk.
- Do NOT use any predefined list. Derive labels from the text itself.
- Respond with ONLY a comma-separated list of new category labels.
- Do NOT explain your choice.

Example response: session-handling, token-validation, expiry-check
```

- [ ] **Step 3: `prompts/mayring_hybrid.md` erstellen**

```markdown
You are a qualitative content analyst applying Mayring's hybrid content analysis method.

Anchor categories (use if applicable): {{categories}}

Instructions:
- You MAY use any anchor category from the list above if it fits.
- You MAY also add new categories for themes NOT covered by the anchor list.
- Mark every new (non-anchor) category with the prefix [neu].
- Assign 3 to 5 labels total.
- Respond with ONLY a comma-separated list.
- Do NOT explain your choice.

Example response: api, error_handling, [neu]session-affinity
```

- [ ] **Step 4: Commit**

```bash
git add prompts/mayring_deduktiv.md prompts/mayring_induktiv.md prompts/mayring_hybrid.md
git commit -m "feat: add Mayring prompt templates (deduktiv/induktiv/hybrid)"
```

---

## Task 2: `mayring_categorize()` aktualisieren

**Files:**
- Modify: `src/memory_ingest.py`
- Modify: `tests/test_memory_ingest.py`

### Schritt A: Tests zuerst

- [ ] **Step 1: Tests in `tests/test_memory_ingest.py` schreiben**

Am Ende der Datei einfügen (nach den bestehenden Klassen):

```python
# ---------------------------------------------------------------------------
# Tests für mayring_categorize() — neue Signatur
# ---------------------------------------------------------------------------

from unittest.mock import patch as _patch


class TestMayringCategorize:
    """Tests für mayring_categorize() mit mode + codebook + source_type."""

    def _make_chunks(self, n: int = 2) -> list:
        from src.memory_ingest import _make_file_chunk
        return [_make_file_chunk(f"def func_{i}(): pass", f"repo:test:f{i}.py", i) for i in range(n)]

    def test_empty_model_returns_chunks_unchanged(self) -> None:
        from src.memory_ingest import mayring_categorize
        chunks = self._make_chunks(1)
        result = mayring_categorize(chunks, "http://localhost:11434", model="")
        assert result == chunks
        assert result[0].category_labels == []

    def test_deductive_mode_system_prompt_contains_codebook_categories(self) -> None:
        from src.memory_ingest import mayring_categorize
        chunks = self._make_chunks(1)
        captured: list[str] = []

        def fake_generate(prompt, ollama_url, model, label, *, system_prompt=None):
            captured.append(system_prompt or "")
            return "api, error_handling"

        with _patch("src.analyzer._ollama_generate", side_effect=fake_generate):
            mayring_categorize(
                chunks, "http://localhost:11434", model="test",
                mode="deductive", codebook="code", source_type="repo_file",
            )

        assert len(captured) == 1
        # Code codebook hat "api" als Kategorie — muss im System-Prompt erscheinen
        assert "api" in captured[0]
        # Deduktiv: kein [neu]-Hinweis im Prompt
        assert "[neu]" not in captured[0]

    def test_inductive_mode_prompt_has_no_category_placeholder(self) -> None:
        from src.memory_ingest import mayring_categorize
        chunks = self._make_chunks(1)
        captured: list[str] = []

        def fake_generate(prompt, ollama_url, model, label, *, system_prompt=None):
            captured.append(system_prompt or "")
            return "session-handling, token-check"

        with _patch("src.analyzer._ollama_generate", side_effect=fake_generate):
            mayring_categorize(
                chunks, "http://localhost:11434", model="test",
                mode="inductive", codebook="code", source_type="repo_file",
            )

        # Inductive template hat keine Kategorienliste — kein "api" etc.
        assert "{{categories}}" not in captured[0]
        # Labels werden trotzdem gesetzt
        assert chunks[0].category_labels == ["session-handling", "token-check"]

    def test_hybrid_mode_preserves_neu_prefix(self) -> None:
        from src.memory_ingest import mayring_categorize
        chunks = self._make_chunks(1)

        def fake_generate(prompt, ollama_url, model, label, *, system_prompt=None):
            return "api, [neu]custom-label"

        with _patch("src.analyzer._ollama_generate", side_effect=fake_generate):
            mayring_categorize(
                chunks, "http://localhost:11434", model="test",
                mode="hybrid", codebook="code", source_type="repo_file",
            )

        # [neu]-Prefix muss in category_labels erhalten bleiben
        assert "[neu]custom-label" in chunks[0].category_labels
        assert "api" in chunks[0].category_labels

    def test_auto_codebook_conversation_summary_uses_social(self) -> None:
        from src.memory_ingest import mayring_categorize, _resolve_codebook
        cats = _resolve_codebook("auto", "conversation_summary")
        # codebook_sozialforschung.yaml hat "argumentation" als Kategorie
        assert "argumentation" in cats

    def test_auto_codebook_repo_file_uses_code(self) -> None:
        from src.memory_ingest import _resolve_codebook
        cats = _resolve_codebook("auto", "repo_file")
        # codebook.yaml hat "api" als Kategorie
        assert "api" in cats

    def test_original_codebook_returns_mayring_basiskategorien(self) -> None:
        from src.memory_ingest import _resolve_codebook
        cats = _resolve_codebook("original", "repo_file")
        assert "Zusammenfassung" in cats
        assert "Explikation" in cats

    def test_ollama_exception_leaves_chunk_unchanged(self) -> None:
        from src.memory_ingest import mayring_categorize
        chunks = self._make_chunks(1)

        def boom(*args, **kwargs):
            raise RuntimeError("ollama down")

        with _patch("src.analyzer._ollama_generate", side_effect=boom):
            result = mayring_categorize(
                chunks, "http://localhost:11434", model="test",
                mode="hybrid", codebook="code",
            )

        assert result[0].category_labels == []
```

- [ ] **Step 2: Tests ausführen, Fehler bestätigen**

```bash
.venv/bin/python -m pytest tests/test_memory_ingest.py::TestMayringCategorize -v 2>&1 | tail -20
```

Erwartet: `ImportError` oder `TypeError` wegen fehlender `_resolve_codebook` und fehlender `mode`-Parameter.

### Schritt B: Implementierung

- [ ] **Step 3: Helpers + aktualisierte `mayring_categorize()` in `src/memory_ingest.py` einfügen**

Nach dem bestehenden `_CATEGORIZE_SYSTEM_PROMPT` Block (ca. Zeile 308) einfügen. Den alten `_CATEGORIZE_SYSTEM_PROMPT` und die alte `mayring_categorize()` **ersetzen** durch:

```python
# ---------------------------------------------------------------------------
# Mayring categorization — Prompt-Template Ansatz
# ---------------------------------------------------------------------------

_PROMPTS_DIR: Path = Path(__file__).parent.parent / "prompts"

_ORIGINAL_MAYRING_CATEGORIES: list[str] = [
    "Zusammenfassung",
    "Explikation",
    "Strukturierung",
    "Paraphrase",
    "Reduktion",
    "Kategoriensystem",
    "Ankerbeispiel",
]

_SOURCE_TYPE_TO_CODEBOOK: dict[str, str] = {
    "repo_file": "code",
    "note": "code",
    "conversation": "social",
    "conversation_summary": "social",
}

_MODE_TO_TEMPLATE: dict[str, str] = {
    "deductive": "mayring_deduktiv",
    "inductive": "mayring_induktiv",
    "hybrid": "mayring_hybrid",
}

_CODEBOOK_PATHS: dict[str, Path] = {
    "code": Path(__file__).parent.parent / "codebook.yaml",
    "social": Path(__file__).parent.parent / "codebook_sozialforschung.yaml",
}


def _resolve_codebook(codebook: str, source_type: str) -> list[str]:
    """Return list of category names for the given codebook/source_type.

    codebook: "auto" | "code" | "social" | "original"
    source_type: used only when codebook="auto"
    """
    if codebook == "auto":
        codebook = _SOURCE_TYPE_TO_CODEBOOK.get(source_type, "original")

    if codebook == "original" or codebook not in _CODEBOOK_PATHS:
        return list(_ORIGINAL_MAYRING_CATEGORIES)

    yaml_path = _CODEBOOK_PATHS[codebook]
    if not yaml_path.exists():
        return list(_ORIGINAL_MAYRING_CATEGORIES)

    try:
        if _HAS_YAML:
            import yaml as _yaml_local
            with yaml_path.open(encoding="utf-8") as f:
                data = _yaml_local.safe_load(f)
            return [cat["name"] for cat in data.get("categories", []) if "name" in cat]
    except Exception:
        pass

    return list(_ORIGINAL_MAYRING_CATEGORIES)


def _load_mayring_template(mode: str) -> str:
    """Load prompt template for the given mode. Falls back to inline default."""
    filename = _MODE_TO_TEMPLATE.get(mode, "mayring_hybrid") + ".md"
    template_path = _PROMPTS_DIR / filename
    try:
        return template_path.read_text(encoding="utf-8")
    except OSError:
        return (
            "Categorize this text chunk using these categories if applicable: {{categories}}. "
            "Respond with ONLY a comma-separated list of labels."
        )


def mayring_categorize(
    chunks: list[Chunk],
    ollama_url: str,
    model: str,
    mode: str = "hybrid",
    codebook: str = "auto",
    source_type: str = "repo_file",
) -> list[Chunk]:
    """Assign Mayring category labels to each chunk via LLM (optional, best-effort).

    Args:
        mode: "deductive" (closed category set), "inductive" (free derivation),
              "hybrid" (anchors + new categories marked with [neu])
        codebook: "auto" (detect from source_type), "code", "social", "original"
        source_type: used for auto-detection of codebook
    """
    if not model or not ollama_url:
        return chunks

    try:
        from src.analyzer import _ollama_generate
    except ImportError:
        return chunks

    categories = _resolve_codebook(codebook, source_type)
    template = _load_mayring_template(mode)
    system_prompt = template.replace("{{categories}}", ", ".join(categories))

    for chunk in chunks:
        try:
            prompt = f"Text chunk (first 400 chars):\n\n{chunk.text[:400]}"
            response = _ollama_generate(
                prompt=prompt,
                ollama_url=ollama_url,
                model=model,
                label=f"mayring:{chunk.chunk_id[:8]}",
                system_prompt=system_prompt,
            )
            labels = [lbl.strip() for lbl in response.split(",") if lbl.strip()]
            if labels:
                chunk.category_labels = labels[:5]
        except Exception:
            pass  # Categorization is strictly optional

    return chunks
```

- [ ] **Step 4: `ingest()` opts erweitern** — in `src/memory_ingest.py`, Funktion `ingest()`, nach `do_log: bool = bool(opts.get("log", False))` hinzufügen:

```python
    mode: str = opts.get("mode", "hybrid")
    codebook_choice: str = opts.get("codebook", "auto")
```

Und den bestehenden `mayring_categorize`-Aufruf ersetzen:
```python
    # Alt:
    # if do_categorize and model:
    #     chunks = mayring_categorize(chunks, ollama_url, model)

    # Neu:
    if do_categorize and model:
        chunks = mayring_categorize(
            chunks, ollama_url, model,
            mode=mode, codebook=codebook_choice,
            source_type=source.source_type,
        )
```

- [ ] **Step 5: Tests ausführen, Grün bestätigen**

```bash
.venv/bin/python -m pytest tests/test_memory_ingest.py::TestMayringCategorize -v 2>&1 | tail -20
```

Erwartet: alle 8 Tests PASS.

- [ ] **Step 6: Alle bisherigen Tests noch grün?**

```bash
.venv/bin/python -m pytest tests/test_memory_ingest.py -v 2>&1 | tail -10
```

Erwartet: alle Tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/memory_ingest.py tests/test_memory_ingest.py
git commit -m "feat: Mayring hybrid categorization via prompt templates (deduktiv/induktiv/hybrid)"
```

---

## Task 3: `ingest_conversation_summary()` hinzufügen

**Files:**
- Modify: `src/memory_ingest.py`
- Modify: `tests/test_memory_ingest.py`

### Schritt A: Tests zuerst

- [ ] **Step 1: Tests in `tests/test_memory_ingest.py` hinzufügen**

```python
class TestIngestConversationSummary:
    """Tests für ingest_conversation_summary()."""

    def test_creates_conversation_summary_source_type(self, tmp_path: Path) -> None:
        from src.memory_ingest import ingest_conversation_summary
        from src.memory_store import init_memory_db, get_source

        conn = init_memory_db(tmp_path / "mem.db")
        summary = "## Session Summary\n\nWir haben die MCP-Architektur implementiert.\n\n## Offene Punkte\n\nTests fehlen noch."

        with _patch("src.context._embed_texts", return_value=[[0.1, 0.2, 0.3]]):
            result = ingest_conversation_summary(
                summary_text=summary,
                conn=conn,
                chroma_collection=None,
                ollama_url="http://localhost:11434",
                model="",
                session_id="sess-123",
                run_id="run-456",
            )

        assert result["source_id"] is not None
        source = get_source(conn, result["source_id"])
        assert source is not None
        assert source.source_type == "conversation_summary"

    def test_session_id_stored_in_branch(self, tmp_path: Path) -> None:
        from src.memory_ingest import ingest_conversation_summary
        from src.memory_store import init_memory_db, get_source

        conn = init_memory_db(tmp_path / "mem2.db")

        with _patch("src.context._embed_texts", return_value=[[0.1, 0.2]]):
            result = ingest_conversation_summary(
                summary_text="## Summary\n\nKurzfassung.",
                conn=conn,
                chroma_collection=None,
                ollama_url="http://localhost:11434",
                model="",
                session_id="my-session",
                run_id="my-run",
            )

        source = get_source(conn, result["source_id"])
        assert source.branch == "my-session"
        assert source.commit == "my-run"

    def test_chunks_are_produced(self, tmp_path: Path) -> None:
        from src.memory_ingest import ingest_conversation_summary
        from src.memory_store import init_memory_db

        conn = init_memory_db(tmp_path / "mem3.db")
        summary = "## Teil 1\n\nErster Abschnitt.\n\n## Teil 2\n\nZweiter Abschnitt."

        with _patch("src.context._embed_texts", return_value=[[0.1]]):
            result = ingest_conversation_summary(
                summary_text=summary,
                conn=conn,
                chroma_collection=None,
                ollama_url="http://localhost:11434",
                model="",
            )

        # 2 Markdown-Sections → 2 Chunks
        assert len(result["chunk_ids"]) == 2
```

- [ ] **Step 2: Tests ausführen, Fehler bestätigen**

```bash
.venv/bin/python -m pytest tests/test_memory_ingest.py::TestIngestConversationSummary -v 2>&1 | tail -10
```

Erwartet: `ImportError: cannot import name 'ingest_conversation_summary'`

### Schritt B: Implementierung

- [ ] **Step 3: `ingest_conversation_summary()` in `src/memory_ingest.py` hinzufügen**

Nach der Funktion `ingest()` (am Ende der Datei) einfügen:

```python
# ---------------------------------------------------------------------------
# Task X — Conversation-Summary Ingestion
# ---------------------------------------------------------------------------

def ingest_conversation_summary(
    summary_text: str,
    conn: Any,
    chroma_collection: Any,
    ollama_url: str,
    model: str,
    session_id: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Ingest a Claude /compact summary as a conversation_summary source.

    Args:
        summary_text: Raw Markdown text of the compaction summary.
        conn: SQLite connection (from init_memory_db()).
        chroma_collection: ChromaDB collection or None.
        ollama_url: Ollama base URL.
        model: Ollama model name (empty string = no embedding / categorization).
        session_id: Optional session identifier, stored in Source.branch.
        run_id: Optional run identifier, stored in Source.commit.

    Returns:
        {source_id, chunk_ids, indexed, deduped, superseded}
    """
    import hashlib

    content_hash = "sha256:" + hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
    path = f"summary/{session_id or 'unknown'}"
    source_id = Source.make_id("conversation", path)

    source = Source(
        source_id=source_id,
        source_type="conversation_summary",
        repo="conversation",
        path=path,
        branch=session_id or "",
        commit=run_id or "",
        content_hash=content_hash,
        captured_at=_now_iso(),
    )

    return ingest(
        source=source,
        content=summary_text,
        conn=conn,
        chroma_collection=chroma_collection,
        ollama_url=ollama_url,
        model=model,
        opts={
            "categorize": bool(model),
            "codebook": "social",
            "mode": "hybrid",
            "log": True,
        },
    )
```

Außerdem sicherstellen, dass `get_source` in `src/memory_store.py` existiert (für den Test). Prüfen:

```bash
grep -n "def get_source" /home/nileneb/Desktop/MayringCoder/src/memory_store.py
```

Falls nicht vorhanden, in `src/memory_store.py` hinzufügen:

```python
def get_source(conn: sqlite3.Connection, source_id: str) -> Source | None:
    """Retrieve a Source by source_id. Returns None if not found."""
    row = conn.execute(
        "SELECT source_id, source_type, repo, path, branch, commit, content_hash, captured_at "
        "FROM sources WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        return None
    return Source(
        source_id=row[0], source_type=row[1], repo=row[2], path=row[3],
        branch=row[4], commit=row[5], content_hash=row[6], captured_at=row[7],
    )
```

- [ ] **Step 4: Tests ausführen, Grün bestätigen**

```bash
.venv/bin/python -m pytest tests/test_memory_ingest.py::TestIngestConversationSummary -v 2>&1 | tail -10
```

Erwartet: 3 Tests PASS.

- [ ] **Step 5: Alle ingest-Tests grün**

```bash
.venv/bin/python -m pytest tests/test_memory_ingest.py -v 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add src/memory_ingest.py src/memory_store.py tests/test_memory_ingest.py
git commit -m "feat: add ingest_conversation_summary() for /compact output (Task X)"
```

---

## Task 4: `session_compacted` in `search()` + `_rerank()`

**Files:**
- Modify: `src/memory_retrieval.py`
- Modify: `tests/test_memory_retrieval.py`

### Schritt A: Tests zuerst

- [ ] **Step 1: Tests in `tests/test_memory_retrieval.py` hinzufügen**

```python
class TestSessionCompacted:
    """Tests für session_compacted-Flag in search()."""

    def _make_conv_source(self, tmp_path) -> tuple:
        from src.memory_store import init_memory_db, upsert_source, insert_chunk
        conn = init_memory_db(tmp_path / "mc.db")
        src = Source(
            source_id="repo:conversation:summary/sess-1",
            source_type="conversation_summary",
            repo="conversation",
            path="summary/sess-1",
            branch="sess-1",
            commit="",
            content_hash="sha256:abc",
            captured_at="2026-04-08T10:00:00+00:00",
        )
        upsert_source(conn, src)
        # Create a section-level chunk (conversation summary → Markdown section)
        text_hash = Chunk.compute_text_hash("## Architektur\n\nWir haben MCP implementiert.")
        chunk = Chunk(
            chunk_id=Chunk.make_id(src.source_id, 0, "section"),
            source_id=src.source_id,
            chunk_level="section",
            ordinal=0,
            text="## Architektur\n\nWir haben MCP implementiert.",
            text_hash=text_hash,
            category_labels=["architektur"],
            created_at="2026-04-08T10:00:00+00:00",
        )
        insert_chunk(conn, chunk)
        return conn, chunk

    def test_compacted_boosts_section_chunks_from_conversation(self, tmp_path) -> None:
        from src.memory_retrieval import search

        conn, chunk = self._make_conv_source(tmp_path)

        results_normal = search(
            query="MCP Architektur",
            conn=conn,
            chroma_collection=None,
            ollama_url="http://localhost:11434",
            opts={"top_k": 5},
            session_compacted=False,
        )
        results_compacted = search(
            query="MCP Architektur",
            conn=conn,
            chroma_collection=None,
            ollama_url="http://localhost:11434",
            opts={"top_k": 5},
            session_compacted=True,
        )

        assert len(results_normal) == 1
        assert len(results_compacted) == 1
        # Compacted search should score higher for this section chunk
        assert results_compacted[0].score_final > results_normal[0].score_final

    def test_compacted_false_no_score_boost(self, tmp_path) -> None:
        from src.memory_retrieval import search

        conn, chunk = self._make_conv_source(tmp_path)

        r1 = search("MCP", conn, None, "http://localhost:11434", opts={"top_k": 5}, session_compacted=False)
        r2 = search("MCP", conn, None, "http://localhost:11434", opts={"top_k": 5}, session_compacted=False)

        assert len(r1) == 1 and len(r2) == 1
        # Same flag → same score
        assert r1[0].score_final == r2[0].score_final
```

- [ ] **Step 2: Tests ausführen, Fehler bestätigen**

```bash
.venv/bin/python -m pytest tests/test_memory_retrieval.py::TestSessionCompacted -v 2>&1 | tail -10
```

Erwartet: `TypeError: search() got an unexpected keyword argument 'session_compacted'`

### Schritt B: Implementierung

- [ ] **Step 3: `_rerank()` in `src/memory_retrieval.py` erweitern**

Signatur von `_rerank()` (Zeile ~158) ändern:

```python
def _rerank(
    candidates: list[Chunk],
    vector_scores: dict[str, float],
    symbolic_scores: dict[str, float],
    top_k: int,
    affinity_source_id: str | None = None,
    source_type_map: dict[str, str] | None = None,
    session_compacted: bool = False,
) -> list[RetrievalRecord]:
```

Im Loop (nach `score_final` Berechnung) hinzufügen:

```python
        # Compaction boost: prefer conversation_summary section chunks
        if session_compacted and source_type_map:
            chunk_source_type = source_type_map.get(chunk.source_id, "")
            if chunk_source_type == "conversation_summary" and chunk.chunk_level == "section":
                score_final = min(1.0, score_final + 0.10)
```

- [ ] **Step 4: `search()` in `src/memory_retrieval.py` erweitern**

Signatur von `search()` (Zeile ~215) ändern:

```python
def search(
    query: str,
    conn: sqlite3.Connection,
    chroma_collection: Any,
    ollama_url: str,
    opts: dict | None = None,
    session_compacted: bool = False,
) -> list[RetrievalRecord]:
```

Vor dem `_rerank()`-Aufruf hinzufügen:

```python
    # Build source_type lookup for session_compacted boost
    source_type_map: dict[str, str] = {}
    if session_compacted and candidates:
        source_ids = list({c.source_id for c in candidates})
        placeholders = ",".join(["?"] * len(source_ids))
        rows = conn.execute(
            f"SELECT source_id, source_type FROM sources WHERE source_id IN ({placeholders})",
            source_ids,
        ).fetchall()
        source_type_map = {r[0]: r[1] for r in rows}
```

`_rerank()`-Aufruf erweitern:

```python
    return _rerank(
        candidates, vector_scores, symbolic_scores, top_k, affinity_source_id,
        source_type_map=source_type_map,
        session_compacted=session_compacted,
    )
```

- [ ] **Step 5: Tests ausführen, Grün bestätigen**

```bash
.venv/bin/python -m pytest tests/test_memory_retrieval.py::TestSessionCompacted -v 2>&1 | tail -10
```

Erwartet: 2 Tests PASS.

- [ ] **Step 6: Alle retrieval-Tests grün**

```bash
.venv/bin/python -m pytest tests/test_memory_retrieval.py -v 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add src/memory_retrieval.py tests/test_memory_retrieval.py
git commit -m "feat: add session_compacted scoring boost to search() (Task Y)"
```

---

## Task 5: MCP `search_memory` Tool erweitern

**Files:**
- Modify: `src/mcp_server.py`

- [ ] **Step 1: `search_memory()` in `src/mcp_server.py` um `compacted`-Parameter erweitern**

Signatur des `search_memory`-Tools (Zeile ~164) ändern:

```python
@mcp.tool()
def search_memory(
    query: str,
    repo: str | None = None,
    categories: list[str] | None = None,
    source_type: str | None = None,
    top_k: int = 8,
    include_text: bool = True,
    source_affinity: str | None = None,
    char_budget: int = 6000,
    compacted: bool = False,
) -> dict:
    """Hybrid 4-stage memory search (scope filter → symbolic → vector → rerank).

    Args:
        query: Natural language search query
        repo: Filter by repository (e.g. "owner/name")
        categories: Filter by any of these Mayring category labels
        source_type: Filter by source type (e.g. "repo_file")
        top_k: Maximum number of results (default 8)
        include_text: Include chunk text in results (default True)
        source_affinity: source_id to boost in affinity scoring
        char_budget: Max chars for prompt_context output
        compacted: Set True after /compact to boost conversation_summary chunks

    Returns:
        {results: list[RetrievalRecord], prompt_context: str}
    """
    try:
        opts = {
            "repo": repo,
            "categories": categories,
            "source_type": source_type,
            "top_k": top_k,
            "include_text": include_text,
            "source_affinity": source_affinity,
        }
        results = search(
            query=query,
            conn=_get_conn(),
            chroma_collection=_get_chroma(),
            ollama_url=_OLLAMA_URL,
            opts=opts,
            session_compacted=compacted,
        )
        prompt_context = compress_for_prompt(results, char_budget)
        return {
            "results": [r.to_dict() for r in results],
            "prompt_context": prompt_context,
        }
    except Exception as exc:
        return {"error": str(exc), "results": [], "prompt_context": ""}
```

- [ ] **Step 2: Smoke-Test (kein Pytest, nur Import-Check)**

```bash
.venv/bin/python -c "from src.mcp_server import search_memory; print('OK')"
```

Erwartet: `OK`

- [ ] **Step 3: Alle bisherigen Tests noch grün**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add src/mcp_server.py
git commit -m "feat: add compacted flag to MCP search_memory tool"
```

---

## Task 6: `web_ui.py` — Modell-Dropdown + Mayring-Felder + Conversation-Tab

**Files:**
- Modify: `src/web_ui.py`
- Modify: `tests/test_web_ui.py`

### Schritt A: Tests zuerst

- [ ] **Step 1: Tests in `tests/test_web_ui.py` hinzufügen**

```python
class TestModelSelector:
    """Tests für Modell-Auswahl in der UI."""

    def test_do_ingest_uses_provided_model(self) -> None:
        """_do_ingest() übergibt model an ingest()."""
        from src.web_ui import _do_ingest
        captured: list[str] = []

        def fake_ingest(source, content, conn, chroma_collection, ollama_url, model, opts=None):
            captured.append(model)
            return {"source_id": "x", "chunk_ids": [], "indexed": False, "deduped": 0, "superseded": 0}

        with patch("src.web_ui.ingest", side_effect=fake_ingest), \
             patch("src.web_ui._get_conn", return_value=MagicMock()), \
             patch("src.web_ui._get_chroma", return_value=None), \
             patch("src.web_ui._MEMORY_READY", True):
            _do_ingest("hello world", None, "test.txt", "owner/repo",
                       categorize=False, mode="hybrid", codebook="auto",
                       model="mistral:7b", ollama_available=True)

        assert captured == ["mistral:7b"]

    def test_do_ingest_passes_mode_and_codebook_in_opts(self) -> None:
        """_do_ingest() schreibt mode + codebook in opts."""
        from src.web_ui import _do_ingest
        captured_opts: list[dict] = []

        def fake_ingest(source, content, conn, chroma_collection, ollama_url, model, opts=None):
            captured_opts.append(opts or {})
            return {"source_id": "x", "chunk_ids": [], "indexed": False, "deduped": 0, "superseded": 0}

        with patch("src.web_ui.ingest", side_effect=fake_ingest), \
             patch("src.web_ui._get_conn", return_value=MagicMock()), \
             patch("src.web_ui._get_chroma", return_value=None), \
             patch("src.web_ui._MEMORY_READY", True):
            _do_ingest("hello world", None, "test.txt", "owner/repo",
                       categorize=True, mode="deductive", codebook="social",
                       model="llama3", ollama_available=True)

        assert captured_opts[0]["mode"] == "deductive"
        assert captured_opts[0]["codebook"] == "social"
        assert captured_opts[0]["categorize"] is True


class TestConversationTab:
    """Tests für Conversation-Summary Ingestion via UI."""

    def test_do_ingest_conversation_calls_ingest_conversation_summary(self) -> None:
        from src.web_ui import _do_ingest_conversation
        captured: list[dict] = []

        def fake_ingest_conv(summary_text, conn, chroma_collection, ollama_url, model,
                             session_id=None, run_id=None):
            captured.append({"session_id": session_id, "run_id": run_id, "text": summary_text})
            return {"source_id": "conv:x", "chunk_ids": ["c1"], "indexed": False, "deduped": 0, "superseded": 0}

        with patch("src.web_ui.ingest_conversation_summary", side_effect=fake_ingest_conv), \
             patch("src.web_ui._get_conn", return_value=MagicMock()), \
             patch("src.web_ui._get_chroma", return_value=None), \
             patch("src.web_ui._MEMORY_READY", True):
            result = _do_ingest_conversation(
                summary_text="## Summary\n\nWas wir gemacht haben.",
                session_id="sess-abc",
                run_id="run-xyz",
                model="mistral:7b",
                ollama_available=True,
            )

        assert captured[0]["session_id"] == "sess-abc"
        assert captured[0]["run_id"] == "run-xyz"
        assert "conv:x" in result

    def test_do_ingest_conversation_empty_text_returns_error(self) -> None:
        from src.web_ui import _do_ingest_conversation
        with patch("src.web_ui._MEMORY_READY", True):
            result = _do_ingest_conversation("", "sess-1", "", "model", True)
        assert "error" in result.lower() or "Kein" in result
```

- [ ] **Step 2: Tests ausführen, Fehler bestätigen**

```bash
.venv/bin/python -m pytest tests/test_web_ui.py::TestModelSelector tests/test_web_ui.py::TestConversationTab -v 2>&1 | tail -15
```

Erwartet: Fehler wegen fehlender Signatur-Änderungen in `_do_ingest()` und fehlender `_do_ingest_conversation()`.

### Schritt B: Implementierung

- [ ] **Step 3: Imports in `src/web_ui.py` erweitern**

In den `try`-Block (Zeile ~41), `from src.memory_ingest import ingest, ...` erweitern:

```python
    from src.memory_ingest import ingest, get_or_create_chroma_collection, ingest_conversation_summary
```

- [ ] **Step 4: `_do_ingest()` Signatur und Inhalt aktualisieren**

Alte Signatur:
```python
def _do_ingest(text_input, file_upload, source_path, repo, categorize, ollama_available):
```

Neue Signatur:
```python
def _do_ingest(
    text_input: str,
    file_upload,
    source_path: str,
    repo: str,
    categorize: bool,
    mode: str,
    codebook: str,
    model: str,
    ollama_available: bool,
) -> str:
```

Den `ingest()`-Aufruf (Zeile ~229) ersetzen:

```python
    try:
        result = ingest(
            source=source,
            content=content,
            conn=conn,
            chroma_collection=chroma,
            ollama_url=_ollama_url,
            model=model,
            opts={
                "categorize": categorize and ollama_available,
                "mode": mode,
                "codebook": codebook,
                "log": True,
            },
        )
    except Exception as exc:
        return json.dumps({"error": f"Ingest fehlgeschlagen: {exc}"}, indent=2)
```

- [ ] **Step 5: `_do_ingest_conversation()` hinzufügen** (nach `_do_ingest()`, vor dem Browser-Block):

```python
def _do_ingest_conversation(
    summary_text: str,
    session_id: str,
    run_id: str,
    model: str,
    ollama_available: bool,
) -> str:
    """Ingest a /compact summary as conversation_summary source."""
    if not _MEMORY_READY:
        return json.dumps({"error": f"Memory-Module nicht geladen: {_IMPORT_ERROR}"}, indent=2)
    if not summary_text.strip():
        return json.dumps({"error": "Kein Inhalt angegeben."}, indent=2)

    conn = _get_conn()
    if conn is None:
        return json.dumps({"error": "SQLite-Datenbank nicht verfügbar."}, indent=2)

    chroma = _get_chroma() if ollama_available else None

    try:
        result = ingest_conversation_summary(
            summary_text=summary_text.strip(),
            conn=conn,
            chroma_collection=chroma,
            ollama_url=_ollama_url,
            model=model if ollama_available else "",
            session_id=session_id.strip() or None,
            run_id=run_id.strip() or None,
        )
    except Exception as exc:
        return json.dumps({"error": f"Conversation-Ingest fehlgeschlagen: {exc}"}, indent=2)

    return json.dumps(result, indent=2, ensure_ascii=False)
```

- [ ] **Step 6: `build_app()` in `src/web_ui.py` aktualisieren**

**6a**: Modell-Dropdown oben hinzufügen (nach dem `gr.HTML(...)` Status-Badge, ca. Zeile 359):

```python
        # --- Modell-Selektor ---
        with gr.Row():
            gr.HTML(_status_html(ollama_available, ollama_models))
            model_selector = gr.Dropdown(
                choices=ollama_models,
                value=ollama_models[0] if ollama_models else None,
                label="Modell",
                interactive=ollama_available,
                scale=1,
                min_width=200,
            )
```

(Den bestehenden alleinstehenden `gr.HTML(...)` Aufruf entfernen und durch die `gr.Row()` oben ersetzen.)

**6b**: Im Ingest-Tab nach der `ingest_categorize`-Checkbox hinzufügen:

```python
            with gr.Column(visible=False) as mayring_opts_col:
                ingest_mode = gr.Radio(
                    choices=["hybrid", "deductive", "inductive"],
                    value="hybrid",
                    label="Mayring-Modus",
                    info="hybrid = Anker + neue Kategorien | deductive = nur Codebook | inductive = frei",
                )
                ingest_codebook = gr.Dropdown(
                    choices=["auto", "code", "social", "original"],
                    value="auto",
                    label="Codebook",
                    info="auto = anhand source_type | code = Code-Analyse | social = Sozialforschung | original = Mayring-Basiskategorien",
                )

            ingest_categorize.change(
                fn=lambda v: gr.Column(visible=v),
                inputs=[ingest_categorize],
                outputs=[mayring_opts_col],
            )
```

**6c**: `_ingest_handler` im Ingest-Tab aktualisieren:

```python
            def _ingest_handler(text, file, path, repo, cat, mode, codebook, model):
                return _do_ingest(text, file, path, repo, cat, mode, codebook, model, ollama_available)

            ingest_btn.click(
                fn=_ingest_handler,
                inputs=[ingest_text, ingest_file, ingest_path, ingest_repo,
                        ingest_categorize, ingest_mode, ingest_codebook, model_selector],
                outputs=[ingest_output],
            )
```

**6d**: Neuen Tab "Conversation" hinzufügen (nach dem Feedback-Tab):

```python
        # -----------------------------------------------------------------------
        # Tab 5: Conversation (Task X)
        # -----------------------------------------------------------------------
        with gr.Tab("Conversation"):
            gr.Markdown(
                "### /compact-Output als Memory speichern\n\n"
                "Füge hier den Output von Claude's `/compact`-Befehl ein. "
                "Er wird als `conversation_summary`-Quelle gespeichert und ist via Suche abrufbar."
            )

            if not ollama_available:
                gr.Markdown(
                    "> **Ollama nicht erreichbar.** Embedding wird übersprungen. "
                    "Nur symbolische Suche nach dem Ingest."
                )

            conv_text = gr.Textbox(
                label="Zusammenfassung (/compact Output)",
                placeholder="## Session Summary\n\n...",
                lines=12,
            )
            with gr.Row():
                conv_session_id = gr.Textbox(
                    label="session_id (optional)",
                    placeholder="z.B. sess-2026-04-08",
                    scale=1,
                )
                conv_run_id = gr.Textbox(
                    label="run_id (optional)",
                    placeholder="z.B. run-001",
                    scale=1,
                )
            conv_btn = gr.Button("Als Memory speichern", variant="primary")
            conv_output = gr.Code(language="json", label="Ergebnis")

            def _conv_handler(text, session_id, run_id, model):
                return _do_ingest_conversation(text, session_id, run_id, model, ollama_available)

            conv_btn.click(
                fn=_conv_handler,
                inputs=[conv_text, conv_session_id, conv_run_id, model_selector],
                outputs=[conv_output],
            )
```

- [ ] **Step 7: Tests ausführen, Grün bestätigen**

```bash
.venv/bin/python -m pytest tests/test_web_ui.py::TestModelSelector tests/test_web_ui.py::TestConversationTab -v 2>&1 | tail -15
```

Erwartet: alle 4 Tests PASS.

- [ ] **Step 8: Alle web_ui-Tests grün**

```bash
.venv/bin/python -m pytest tests/test_web_ui.py -v 2>&1 | tail -10
```

- [ ] **Step 9: UI manuell starten (Smoke-Test)**

```bash
.venv/bin/python -m src.web_ui --help
```

Erwartet: Hilfe-Text ohne Fehler.

- [ ] **Step 10: Commit**

```bash
git add src/web_ui.py tests/test_web_ui.py
git commit -m "feat: add model selector, Mayring mode/codebook fields, Conversation tab to WebUI"
```

---

## Task 7: CLAUDE.md Compact Instructions

**Files:**
- Modify: `.claude/CLAUDE.md`

- [ ] **Step 1: Compact-Instructions-Abschnitt am Ende von `.claude/CLAUDE.md` einfügen**

```markdown
## Compact Instructions

Bei `/compact` folgende Informationen erhalten (werden für Memory-Retrieval nach Kompaktierung benötigt):

- **Architektur:** Zielzustand aus `Target-Architecture.md` + aktive Entscheidungen aus `CLAUDE.md`
- **Offene Tasks:** Akzeptanzkriterien aus Issue #27 (Nileneb/MayringCoder#27)
- **Editierte Module (diese Session):** Dateinamen und geänderte Funktionen
- **Aktive MCP-Tool-Verträge:** `memory.put`, `memory.get`, `memory.search_memory`,
  `memory.invalidate`, `memory.list_by_source`, `memory.explain`, `memory.reindex`, `memory.feedback`
- **Chunking-Invariante:** Strukturelles Chunking zuerst, Mayring-Kategorisierung als semantische Schicht danach
- **Codebook-Auto-Erkennung:** `repo_file`/`note` → `codebook.yaml`; `conversation`/`conversation_summary` → `codebook_sozialforschung.yaml`

Compact-Zusammenfassungen können via `mcp__memory__put` (source_type: conversation_summary) ins Memory gespeichert werden, damit sie bei zukünftigen Sessions per `mcp__memory__search_memory` abrufbar sind.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/CLAUDE.md
git commit -m "docs: add Compact Instructions section to .claude/CLAUDE.md (Task Z)"
```

---

## Task 8: Volltest + GitHub Issue kommentieren + schließen

**Files:** keine

- [ ] **Step 1: Alle Tests ausführen**

```bash
.venv/bin/python -m pytest tests/ -v 2>&1 | tail -20
```

Erwartet: alle Tests PASS (vorher 427, jetzt ~440+).

- [ ] **Step 2: GitHub Issue-Kommentar posten**

```bash
gh issue comment 27 --repo Nileneb/MayringCoder --body "$(cat <<'EOF'
## Abschluss-Kommentar — 2026-04-08

Alle Tasks aus Issue #27 und den Folge-Kommentaren sind implementiert und getestet.

### Neu implementiert (diese Session)

| Modul | Änderung |
|---|---|
| `prompts/mayring_deduktiv.md` | System-Prompt für geschlossene Kategorienmenge |
| `prompts/mayring_induktiv.md` | System-Prompt für freie Kategorienableitung |
| `prompts/mayring_hybrid.md` | System-Prompt für Anker + neue [neu]-Kategorien |
| `src/memory_ingest.py` | `mayring_categorize()` mit `mode`/`codebook`/`source_type`; `_resolve_codebook()`; `_load_mayring_template()`; `ingest_conversation_summary()` |
| `src/memory_retrieval.py` | `search()` + `_rerank()` mit `session_compacted=True` Boost |
| `src/mcp_server.py` | `search_memory` Tool: `compacted`-Parameter |
| `src/web_ui.py` | Modell-Dropdown, Mayring-Modus/Codebook-Felder (an Checkbox gebunden), Conversation-Tab |
| `.claude/CLAUDE.md` | Compact Instructions Abschnitt |
| `src/memory_store.py` | `get_source()` Helper |

### Akzeptanzkriterien

- [x] Claude kann über MCP Chunks schreiben und lesen
- [x] Geänderte Quellen werden inkrementell reingestiert
- [x] Retrieval kombiniert Kategorie, Quelle und Semantik
- [x] Jeder Treffer ist auf konkrete Quelle und Version zurückführbar
- [x] Dubletten und veraltete Versionen werden kontrolliert behandelt
- [x] System läuft vollständig lokal
- [x] Opt-in Logging aktiv
- [x] Mayring induktiv/deduktiv/hybrid via Prompt-Templates
- [x] LLM-Modellauswahl in der UI
- [x] Conversation-Summary Ingestion (Task X)
- [x] Compaction-aware Retrieval (Task Y)
- [x] CLAUDE.md Compact Instructions (Task Z)

### Tests

Alle Tests grün. Neue Tests abdecken: `TestMayringCategorize`, `TestIngestConversationSummary`, `TestSessionCompacted`, `TestModelSelector`, `TestConversationTab`.
EOF
)"
```

- [ ] **Step 3: Issue schließen**

```bash
gh issue close 27 --repo Nileneb/MayringCoder --comment "Alle Akzeptanzkriterien erfüllt. Issue wird geschlossen."
```

- [ ] **Step 4: Abschluss-Commit**

```bash
git add -A
git status
git commit -m "chore: issue #27 complete — all acceptance criteria met"
```
