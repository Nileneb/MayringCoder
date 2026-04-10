# Design: Issues #31, #32, #33, #34, #35, #37, #38

## Scope

Alle 7 offenen GitHub-Issues in einem koordinierten Batch abarbeiten. Drei Phasen: Fundament, Features, Doku & Tests.

---

## Phase 1 — Fundament

### #35 — Modellempfehlungen

Neue Sektion in README.md mit Tabelle:

| Modell | VRAM | Code-Review | Sozialforschung | Turbulenz | Vision |
|--------|------|-------------|-----------------|-----------|--------|
| mistral:7b-instruct | 4 GB | gut | gut | empfohlen | - |
| qwen2.5-coder:7b | 5 GB | sehr gut | - | gut | - |
| qwen3.5:9b | 7 GB | exzellent | exzellent | gut | - |
| qwen2.5vl:3b | 3 GB | - | - | - | empfohlen |

Plus VRAM-Hinweise und Ollama-Pull-Kommandos.

### #32 — Intelligenteres Chunking

**Ziel:** Statt nach fester Zeichenzahl abzuschneiden, logische Blöcke extrahieren und nach Relevanz priorisieren.

**Änderungen:**

1. `src/splitter.py` — neue Funktion `smart_split(content, language, max_chars)`:
   - Python: `ast.parse()` → Funktionen/Klassen als Blöcke
   - JavaScript/TypeScript: Regex-basiert (function/class/export Boundaries)
   - Markdown: Heading-basiert (bereits in `memory_ingest.py` vorhanden)
   - Fallback: Bisheriges Zeichenlimit-Truncation

2. Prioritäts-Scoring pro Block:
   - +3: security-relevante Namen (`auth`, `delete`, `admin`, `password`, `secret`, `token`)
   - +2: public API (export, `__init__`, Klassen-Methoden)
   - +1: Error-Handling (`except`, `catch`, `error`)
   - +0: Rest

3. Auswahl: Blöcke nach Score sortiert, bis `max_chars` erreicht. Übersprungene Blöcke als einzeilige Zusammenfassung anhängen.

4. `--max-chars` bleibt als CLI-Flag erhalten.

**Basis:** `structural_chunk()` aus `memory_ingest.py` wiederverwenden, nicht duplizieren.

---

## Phase 2 — Features

### #31 — Visuelle RAG mit Qwen2.5-VL

**Stufe 1 — Metadaten + SVG (sofort):**
- `codebook.yaml`: Bild-Extensions (png, jpg, jpeg, gif, svg) aus exclude_patterns entfernen
- SVG-Dateien: XML-Text als Chunk indexieren (source_type: `repo_file`, Kategorie: `diagram`)
- Andere Bilder: Metadaten-Chunk (Pfad, Größe, Typ) indexieren

**Stufe 2 — Qwen2.5-VL Captions:**
- Neues Modul `src/vision_captioner.py`:
  - `caption_image(image_path: Path, model: str = "qwen2.5vl:3b") -> str`
  - Ollama `/api/generate` mit base64-encodiertem Bild
  - Prompt: "Describe this image in detail. Focus on architecture, data flow, and technical content."
  - Rückgabe: Caption-Text
  - `caption_images_batch(paths: list[Path]) -> list[dict]` für Batch-Verarbeitung

- Integration in `memory_ingest.py`:
  - Neuer source_type: `image`
  - Caption wird als Text-Chunk gespeichert, Embedding über bestehende nomic-embed-text Pipeline
  - Metadaten: `original_path`, `image_type`, `dimensions` (via Pillow)

- Integration in `checker.py`:
  - `--caption-images` Flag (opt-in)
  - Während Overview-Phase: Bilder captionen und in Memory indexieren

**Stufe 3 — CLIP multimodal: bewusst aufgeschoben.** Erst wenn Stufe 2 Mehrwert zeigt.

**Abhängigkeit:** Pillow in requirements.txt.

### #38 — MCP Auth: JWT + Workspace-Isolation

**JWT-Authentifizierung:**
- Neue Dependency: `PyJWT` in requirements.txt
- Env-Vars: `MCP_AUTH_ENABLED` (default: false), `MCP_AUTH_SECRET` (HMAC HS256 Key)
- JWT Claims: `workspace_id` (str), `scope` (str, z.B. "repo:owner/repo"), `exp` (Unix timestamp)

**ASGI Middleware (ersetzt statische X-Auth-Token Middleware):**
```
Request → Header extrahieren (X-Auth-Token ODER Authorization: Bearer)
        → JWT dekodieren + validieren (Signatur, Expiry)
        → workspace_id in ASGI scope injizieren
        → Weiterleitung an FastMCP
```

- `MCP_AUTH_ENABLED=false`: Middleware durchlässig, workspace_id = "default"
- Ungültiger/fehlender Token bei enabled: 401

**Chroma-Isolation:**
- Collection-Name: `memory_chunks_{workspace_id}` statt `memory_chunks`
- `memory_store.py`: workspace_id Column in `sources` und `chunks` Tabellen
- Alle Queries in `memory_store.py` und `memory_retrieval.py` filtern nach workspace_id
- Migration: bestehende Daten bekommen workspace_id = "default"

**SQLite Schema-Migration:**
- `ALTER TABLE sources ADD COLUMN workspace_id TEXT DEFAULT 'default'`
- `ALTER TABLE chunks ADD COLUMN workspace_id TEXT DEFAULT 'default'`
- Automatisch beim Start (wie bestehende Schema-Erstellung)

**Token-Generator:**
- `tools/generate_mcp_token.py`: CLI-Tool
- `python tools/generate_mcp_token.py --workspace myapp --scope "repo:Nileneb/app.linn.games" --secret $MCP_AUTH_SECRET --expiry 30d`
- Gibt JWT-String aus

**Rate-Limiting (optional, niedrige Priorität):**
- In-Memory Counter pro workspace_id
- 429 bei Überschreitung
- Konfigurierbar via `MCP_RATE_LIMIT` (requests/min, default: 0 = unlimited)

### #37 — Docker-Compose Full Pipeline

**Neue Datei: `docker-compose.full.yml`** (bestehende `docker-compose.yml` bleibt unverändert)

Services:

1. **ollama** — `ollama/ollama:latest`
   - Volume: `ollama_models:/root/.ollama`
   - Healthcheck: `ollama list`
   - Entrypoint-Script: pullt konfiguriertes Modell beim ersten Start
   - GPU-Support: nvidia runtime (optional, via deploy.resources)

2. **analyzer** — MayringCoder-Image (erweitertes Dockerfile)
   - Depends_on: ollama (healthy)
   - Volumes: `./cache`, `./reports`, `./prompts`, `./codebook.yaml`
   - Environment: `OLLAMA_URL=http://ollama:11434`
   - Command: idle (für `docker compose run analyzer python checker.py ...`)

3. **web-ui** — MayringCoder-Image mit Gradio
   - Depends_on: ollama
   - Port: 7860
   - Command: `python -m src.web_ui`

4. **mcp-memory** — Bestehender MCP-Service (HTTP-Profil)
   - Port: 8000

Shared Volumes: `cache_data`, `reports_data`, `ollama_models`.

**Dockerfile-Erweiterung:** Zweites Stage oder erweiterte COPY-Befehle für checker.py, prompts/, codebook.yaml, turbulence_run.py.

---

## Phase 3 — Doku & Tests

### #34 — Memory-Roadmap

Neue Datei: `docs/memory_roadmap.md`

Inhalt:
- Phase 1 (erledigt): MCP Tools, SQLite+ChromaDB, Hybrid-Search, Multi-View Chunking
- Phase 2 (offen): Induktive Kategorisierung (Qwen), Re-Ranker (Cross-Encoder), Codebook-Auto-Erkennung
- Phase 3 (offen): Feedback-Loop, Training-Data-Export, Active Learning
- Phase 4 (offen): Retention Policies, Governance, Audit-Log, Workspace-Isolation (→ #38)
- Referenz auf Target-Architecture.md

### #33 — E2E-Tests Web-UI

Erweitert `tests/test_web_ui.py`:

1. Gradio `TestClient` für:
   - URL eingeben → Analyse starten → Report wird angezeigt
   - Overview-Modus → JSON-Cache wird erstellt
   - Leere/ungültige URL → Fehlermeldung

2. Mock-Setup:
   - Ollama-Responses gemockt (deterministische Ergebnisse)
   - Kein Netzwerk-Zugriff in Tests

3. Mindestens 5 neue Tests.

---

## Neue Dependencies

| Package | Issue | Zweck |
|---------|-------|-------|
| Pillow | #31 | Bild-Metadaten (Dimensionen) |
| PyJWT | #38 | JWT Token Validation |

Beide in `requirements.txt` aufnehmen.

---

## Reihenfolge der Implementierung

1. #35 (Doku, kein Code-Impact)
2. #32 (Chunking-Fundament)
3. #31 (Vision-RAG, nutzt Chunking)
4. #38 (JWT + Workspace, unabhängig von 31/32)
5. #37 (Docker-Full, profitiert von allen Features)
6. #34 (Roadmap-Doku)
7. #33 (E2E-Tests)

Parallelisierbar: #35 mit #32, #34 mit #33, #38 mit #31.
