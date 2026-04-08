# Design: Mayring-Integration, LLM-Auswahl & Issue #27 Completion

**Datum:** 2026-04-08  
**Status:** Genehmigt  
**Issue:** Nileneb/MayringCoder#27

---

## Überblick

Erweiterung der bestehenden Gradio WebUI und der Memory-Ingestion-Pipeline um:

1. Vollständige induktiv/deduktiv/hybrid Mayring-Kategorisierung via Prompt-Templates
2. LLM-Modellauswahl in der UI
3. Conversation-Summary Ingestion (Task X)
4. Compaction-aware Retrieval (Task Y)
5. CLAUDE.md Compact Instructions (Task Z)

---

## Abschnitt 1 — Mayring Prompt-Templates + Kategorisierung

### Neue Dateien

| Datei | Zweck |
|---|---|
| `prompts/mayring_deduktiv.md` | Geschlossene Kategorienmenge aus Codebook, LLM weist zu |
| `prompts/mayring_induktiv.md` | Kein Anker, LLM leitet frei aus Text ab |
| `prompts/mayring_hybrid.md` | Codebook-Kategorien als Anker + neue Kategorien erlaubt |

### Template-Variablen

Alle drei Templates nutzen `{{categories}}` als Platzhalter für die Codebook-Kategorien (wird leer gelassen bei `mayring_induktiv.md`). Neue, LLM-generierte Kategorien werden mit `[neu]`-Prefix markiert.

### Codebook-Auto-Erkennung

| `source_type` | Codebook |
|---|---|
| `repo_file`, `note` | `codebook.yaml` (Code-Analyse) |
| `conversation`, `conversation_summary` | `codebook_sozialforschung.yaml` |
| alle anderen | Original Mayring (hardcodiert: Zusammenfassung, Explikation, Strukturierung, Paraphrase, Reduktion, Kategoriensystem, Ankerbeispiel) |

### Funktionssignatur

```python
# src/memory_ingest.py
def mayring_categorize(
    chunks: list[Chunk],
    ollama_url: str,
    model: str,
    mode: str = "hybrid",          # "deductive" | "inductive" | "hybrid"
    codebook: str = "auto",        # "auto" | "code" | "social" | "original"
    source_type: str = "repo_file",
) -> list[Chunk]:
```

Interner Ablauf:
1. Codebook auflösen: `"auto"` → anhand `source_type`; sonst direkt
2. Kategorien aus YAML laden
3. Template laden (`prompts/mayring_{mode}.md` → `prompts/mayring_hybrid.md` etc.)
4. `{{categories}}` ersetzen
5. LLM-Aufruf pro Chunk via `_ollama_generate()`
6. Labels parsen; `[neu]`-Prefix für induktiv generierte Kategorien

### Invarianten

- Wenn Modell nicht erreichbar: Chunks bleiben ohne Labels, kein Fehler
- Codebook-Kategorien ersetzen nie Chunk-Grenzen aus strukturellem Chunking
- `ingest()` bekommt `mode` und `codebook` als neue `opts`-Schlüssel

---

## Abschnitt 2 — LLM-Auswahl in der UI

### Globaler Modell-Selektor

Position: oben in `build_app()`, direkt neben dem Ollama-Status-Badge.

```
[● Ollama online (3 Modelle)]   Modell: [mistral:7b-instruct ▾]
```

- `gr.Dropdown(choices=ollama_models, value=ollama_models[0] if ollama_models else "")` 
- Leer wenn Ollama offline
- Gespeichert in `gr.State` und an alle Handler weitergereicht

### Ingest-Tab Erweiterungen

Zwei neue Felder, sichtbar wenn "Mayring-Kategorisierung" aktiviert:

| Feld | Typ | Werte | Default |
|---|---|---|---|
| Mayring-Modus | `gr.Radio` | `hybrid`, `deductive`, `inductive` | `hybrid` |
| Codebook | `gr.Dropdown` | `auto`, `code`, `social`, `original` | `auto` |

`codebook="auto"` zeigt Hinweis: *"Wird anhand source_type erkannt"*.  
Modus/Codebook-Block ist per `gr.Column(visible=...)` an die Kategorisierungs-Checkbox gebunden.

### Änderungen an `_do_ingest()`

```python
def _do_ingest(text, file, path, repo, categorize, mode, codebook, model, ollama_available):
    ...
    result = ingest(source, content, conn, chroma,
        ollama_url=_ollama_url, model=model,
        opts={"categorize": categorize and ollama_available,
              "mode": mode, "codebook": codebook, "log": True})
```

---

## Abschnitt 3 — Tasks X / Y / Z

### Task X — Conversation-Summary Ingestion

**Neue Funktion** in `src/memory_ingest.py`:

```python
def ingest_conversation_summary(
    summary_text: str,
    conn,
    chroma_collection,
    ollama_url: str,
    model: str,
    session_id: str | None = None,
    run_id: str | None = None,
) -> dict:
```

- Erzeugt `Source(source_type="conversation_summary", ...)`
- `session_id` → `Source.branch`, `run_id` → `Source.commit` (Pragmatismus: kein Schema-Bruch; diese Felder sind bei conversation_summary semantisch leer)
- Chunking: Markdown-Heading-Sektionen (`_chunk_markdown`)
- Auto-Codebook: `codebook_sozialforschung.yaml`
- Ruft intern `ingest()` auf

**Neuer Tab "Conversation"** in `web_ui.py`:
- Textarea für `/compact`-Output
- Felder: `session_id` (optional), `run_id` (optional)
- Button "Als Memory speichern"
- Output: JSON-Ergebnis

### Task Y — Compaction-aware Retrieval

Erweiterung in `src/memory_retrieval.py`:

```python
def search(query, conn, chroma_collection, ollama_url,
           opts=None, session_compacted: bool = False) -> list[RetrievalRecord]:
```

Wenn `session_compacted=True`:
- Score-Multiplikator `+0.10` für Chunks mit `chunk_level="section"` und `source_type="conversation_summary"`
- Bevorzugt detaillierte Erinnerungen wenn Session-Kontext komprimiert wurde

Kein neues UI-Feature — internes Retrieval-Signal. Der MCP-Server (`mcp_server.py`) setzt `session_compacted=True` wenn der `memory.search`-Aufruf einen `compacted: true`-Parameter enthält; der Aufrufer (Claude Code nach `/compact`) ist verantwortlich diesen zu setzen.

### Task Z — CLAUDE.md Compact Instructions

Neuer Abschnitt in `.claude/CLAUDE.md`:

```markdown
## Compact Instructions

Bei /compact folgende Informationen erhalten:
- Aktive Architekturentscheidungen (Target-Architecture.md, CLAUDE.md)
- Offene Tasks / Akzeptanzkriterien aus Issue #27
- Editierte Module dieser Session (Dateinamen + geänderte Funktionen)
- Aktive MCP-Tool-Verträge: memory.put, memory.get, memory.search_memory,
  memory.invalidate, memory.list_by_source, memory.explain, memory.reindex, memory.feedback
- Chunking-Invariante: strukturell zuerst, Mayring als semantische Schicht
```

---

## Abhängigkeiten

```
Prompt-Templates (A)     → unabhängig
mayring_categorize() (B) → benötigt A
web_ui Modell-Selektor   → unabhängig
web_ui Ingest-Tab (C)    → benötigt B
ingest_conversation (X)  → unabhängig (neues source_type)
Retrieval session_compacted (Y) → unabhängig
CLAUDE.md (Z)            → unabhängig
Tests                    → benötigt B + C + X
GitHub Issue Kommentar   → nach allem anderen
```

---

## Akzeptanzkriterien

- [ ] `prompts/mayring_hybrid.md` / `_deduktiv.md` / `_induktiv.md` existieren mit `{{categories}}`-Platzhalter
- [ ] `mayring_categorize()` akzeptiert `mode` + `codebook` + `source_type`
- [ ] Neue Kategorien werden mit `[neu]` markiert
- [ ] UI zeigt Modell-Dropdown (aus `check_ollama()`)
- [ ] Ingest-Tab: Modus + Codebook-Felder, an Kategorisierungs-Checkbox gebunden
- [ ] `ingest_conversation_summary()` existiert und schreibt `source_type="conversation_summary"`
- [ ] `search()` akzeptiert `session_compacted`-Parameter
- [ ] `.claude/CLAUDE.md` hat Compact-Instructions-Abschnitt
- [ ] Alle bestehenden Tests grün + neue Tests für neue Funktionen
- [ ] GitHub Issue #27 bekommt abschließenden Kommentar + wird geschlossen
