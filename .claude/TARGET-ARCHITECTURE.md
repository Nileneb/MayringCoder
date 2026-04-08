# Target Architecture

## Zweck

MCP-basierte Architektur für Claude Code mit externem, persistentem Memory. Ziel ist nicht ein größeres Kontextfenster, sondern selektives Retrieval aus lokal gespeicherten, versionierten Wissenseinheiten.

Die Architektur dockt an MayringCoder an und nutzt vorhandene Bausteine:

- lokale LLM-Ausführung via Ollama
- Overview-Analyse als struktureller Einstiegspunkt
- SQLite-Snapshot-Cache für Änderungsdetektion
- ChromaDB für Embeddings und Similarity Search
- optionale finding-reactive RAG-Anreicherung
- opt-in JSONL-Logging für spätere Modellverbesserung

## Zielbild

Claude Code arbeitet als Orchestrator. Persistentes Wissen liegt nicht im Agenten, sondern in einer externen Memory-Schicht. Claude ruft diese Schicht über MCP-Tools auf.

Die Architektur trennt fünf Ebenen:

1. **Client-Ebene**: Claude Code
2. **Tooling-Ebene**: MCP-Server für Memory, Retrieval, Ingestion, Training, Governance
3. **Verarbeitungs-Ebene**: Chunking, Kategorisierung, Embedding, Ranking, Kompression
4. **Speicher-Ebene**: Key-Value/Cache, SQLite-Metadaten, ChromaDB-Vektorindex, Artefakt-Storage
5. **Steuerungs-Ebene**: Policies für Ingestion, Retrieval, Versionierung, TTL, Reindexing, Training

## Architekturprinzipien

- **Local first**: keine Pflicht zur Cloud
- **MCP als Schnittstelle**: Claude koppelt nur an stabile Tools, nicht direkt an interne Speicher
- **Chunking vor Embedding**: semantische Einheiten werden vor Indexierung erzeugt
- **Kategorie ist Metadatum, nicht Ersatz für Segmentierung**
- **Hybrid Retrieval**: strukturierte Filter + Vektor-Ranking + Cache-Hits
- **Versionierbarkeit**: jedes Memory ist an Quelle, Hash und Zeitstand gebunden
- **Deduplication**: identische oder fast identische Chunks werden nicht redundant gespeichert
- **Explainability**: jeder Retrieval-Treffer ist auf Quelle und Kategorie zurückführbar
- **Trainability**: Opt-in Logs erzeugen Trainingsdaten für spätere Modelle

## Systemkontext

### Primäre Inputs

- Repository-Dateien
- Projekt-Dokumentation
- Konversationsartefakte
- manuell gespeicherte Notizen
- später: WebUI-Events, Labels, Nutzerfeedback

### Primäre Outputs

- kontextrelevante Chunks für Claude
- strukturierte Memory-Treffer mit Herkunft
- Trainings- und Bewertungsdaten
- Reindex- und Invalidierungsereignisse

## Kernkomponenten

### 1. Claude Code

Verantwortung:

- Tool-Aufrufe über MCP
- Anfrageklassifikation: schreiben, suchen, aktualisieren, verdichten
- Zusammensetzen des Arbeitskontexts aus aktuellem Prompt + Retrieval-Ergebnissen
- keine direkte Persistenzlogik

### 2. MCP Gateway

Ein MCP-Server bündelt die fachlichen Tools. Minimaler Tool-Satz:

- `memory.put`
- `memory.get`
- `memory.search`
- `memory.update`
- `memory.invalidate`
- `memory.list_by_source`
- `memory.explain`
- `memory.reindex`
- `memory.feedback`

Optional getrennt nach Domänen:

- `ingest.*`
- `retrieve.*`
- `train.*`
- `admin.*`

### 3. Ingestion Pipeline

Verarbeitet neue oder geänderte Inhalte.

Schritte:

1. Source erfassen
2. Normalisieren
3. Basischunking
4. optionale semantische Verfeinerung durch induktive Mayring-Analyse
5. Metadaten anreichern
6. deduplizieren
7. in Cache und Vektorindex schreiben
8. Ingestion-Event protokollieren

### 4. Chunker

Der Chunker besteht aus zwei Stufen.

#### Stufe A: strukturelles Chunking

Primärsegmentierung entlang robuster Grenzen:

- Datei
- Abschnitt
- Klasse
- Funktion
- Markdown-Heading
- JSON/YAML-Objekt
- Gesprächseintrag

Ziel: stabile, reproduzierbare Basiseinheiten.

#### Stufe B: semantische Verfeinerung

Optionales kleines lokales Modell, z. B. Qwen via Ollama, erkennt innerhalb eines Basis-Chunks semantische Teilsegmente.

Regel:

- Wenn keine klare innere Trennung erkennbar ist: Basis-Chunk bleibt erhalten.
- Wenn wiederkehrende semantische Kategorien erkennbar sind: Split in Sub-Chunks.
- Wenn der Chunk bereits klein und homogen ist: kein weiterer Split.

**Schlussfolgerung:** Mayring-Induktion sollte **nicht** das einzige Chunking-Verfahren sein. Sie ist eine **semantische Verfeinerungs- und Indexierungsschicht** auf einem strukturellen Grundchunking.

### 5. Kategorisierer nach Mayring

Aufgabe:

- induktive Kategorien aus Chunk-Inhalten ableiten
- Kategorien normalisieren
- Kategorien versionieren
- Kategorien als Retrieval-Metadaten speichern

Kategorien dienen für:

- Key-Bildung
- Filter-Retrieval
- Clusterbildung
- Verdichtung
- Trainingslabels

Kategorien ersetzen nicht:

- Quellreferenz
- Positionsinformation
- Embeddings
- textuelle Repräsentation

### 6. Memory Store

Hybridmodell aus mehreren Speichern.

#### 6.1 Key-Value/Cache Store

Für schnelle direkte Zugriffe.

Geeignet für:

- exakte Keys
- zuletzt verwendete Memories
- Session-nahe Arbeitskontexte
- TTL-basierte Einträge

Beispieltechnologien:

- Redis
- SQLite-Key-Value-Tabelle
- DiskCache

#### 6.2 SQLite Metadata Store

Für robuste lokale Persistenz und relationale Abfragen.

Speichert:

- Chunk-Metadaten
- Versionen
- Hashes
- Source-Zuordnung
- Gültigkeitsstatus
- Ingestion-Logs
- Feedback

#### 6.3 ChromaDB Vector Store

Für semantisches Retrieval über Embeddings.

Speichert:

- Chunk-Text oder komprimierte Repräsentation
- Embedding
- Metadaten für Filterung

#### 6.4 Artifact Store

Für Originaltexte, Snapshots, Exporte, Trainingsdateien.

## Datenmodell

### Source

```json
{
  "source_id": "repo:owner/name:path/to/file.py",
  "source_type": "repo_file",
  "repo": "owner/name",
  "path": "path/to/file.py",
  "branch": "main",
  "commit": "abc123",
  "content_hash": "sha256:...",
  "captured_at": "2026-04-08T12:00:00Z"
}
```

### Chunk

```json
{
  "chunk_id": "chk_01...",
  "source_id": "repo:owner/name:path/to/file.py",
  "parent_chunk_id": null,
  "chunk_level": "function",
  "ordinal": 3,
  "start_offset": 1820,
  "end_offset": 2640,
  "text": "...",
  "text_hash": "sha256:...",
  "summary": "...",
  "category_labels": ["auth", "state-transition"],
  "category_version": "mayring-inductive-v1",
  "embedding_model": "nomic-embed-text",
  "embedding_id": "emb_01...",
  "quality_score": 0.91,
  "dedup_key": "sha256:...",
  "created_at": "2026-04-08T12:00:00Z",
  "superseded_by": null,
  "is_active": true
}
```

### Memory Key

Direkte Speicherung als `KEY(Kategorie):Chunk` ist als alleinige Struktur zu grob. Ziel ist ein zusammengesetzter Schlüssel:

```text
memory:{scope}:{category}:{source_fingerprint}:{chunk_hash_prefix}
```

Beispiel:

```text
memory:repo:auth:owner-name-src-user_service.py:9f3a1b2c
```

Zusätzliche Sekundärindizes:

- `source_id -> [chunk_id]`
- `category -> [chunk_id]`
- `dedup_key -> canonical_chunk_id`
- `entity/tag -> [chunk_id]`

### Retrieval Record

```json
{
  "chunk_id": "chk_01...",
  "score_vector": 0.82,
  "score_symbolic": 0.67,
  "score_recency": 0.30,
  "score_final": 0.76,
  "reasons": ["category_match", "same_repo", "embedding_similarity"],
  "source_id": "repo:owner/name:path/to/file.py"
}
```

## Speicherstrategie

### Was in welchen Store gehört

| Inhalt | KV/Cache | SQLite | ChromaDB | Artifact Store |
|---|---:|---:|---:|---:|
| Hot memory / recent context | x |  |  |  |
| Chunk-Metadaten |  | x | metadata |  |
| Volltext-Chunk | optional | optional | x | x |
| Embedding |  |  | x |  |
| Source-Snapshot |  | x |  | x |
| Training-Log |  | x |  | x |
| Feedback / Labels |  | x |  |  |

## Retrieval-Policy

Retrieval ist mehrstufig.

### Stufe 1: Scope Filter

Vorfilter nach harten Kriterien:

- Repository
- Branch oder Commit-Nähe
- Source-Type
- Kategorie
- Sprache/Dateityp
- Aktivstatus

### Stufe 2: Symbolisches Matching

Abgleich über:

- Dateipfad
- Funktionsname
- Entitäten
- Kategorie
- manuelle Tags

### Stufe 3: Vektor-Retrieval

Similarity Search in ChromaDB über:

- Query aus Nutzeranfrage
- Query aus Dateiinhalt
- Query aus Findings
- Query aus zusammengefasster Taskbeschreibung

### Stufe 4: Re-Ranking

Empfohlene Gewichtung:

$$score = 0.45 \cdot vector + 0.25 \cdot symbolic + 0.15 \cdot recency + 0.15 \cdot sourceAffinity$$

Optional ergänzt um Penalties:

- veraltete Version
- Dublette
- niedrige Chunk-Qualität
- zu großer Chunk

### Stufe 5: Kontextkompression

Vor Übergabe an Claude:

- Dubletten entfernen
- gleiche Quelle zusammenfassen
- lange Chunks kürzen
- optional summary-first, full-text-on-demand

## Kopplung an MayringCoder

### Bereits vorhandene Bausteine

Die Zielarchitektur nutzt vorhandene Strukturen von MayringCoder direkt:

- Overview-Resultate als erste Inventarschicht
- SQLite-Diff als Trigger für Reingestion geänderter Quellen
- bestehende Embedding-Funktionen für ChromaDB
- `query_similar_context` als Ausgangspunkt für semantisches Retrieval
- finding-reactive RAG als Muster für retrieval-spezifische Folgequeries
- opt-in JSONL-Logging im LLM-Call-Loop als Trainingsdatenquelle

### Geplante Erweiterung

MayringCoder wird um eine Memory-spezifische Ingestion erweitert.

Zusätzliche Pipeline:

```text
Source change
-> snapshot diff
-> structural chunking
-> optional Mayring inductive categorization
-> dedup/version resolution
-> kv write
-> sqlite metadata write
-> embedding write
-> chroma upsert
-> retrieval-ready
```

### Rolle der Overview-Stufe

Die Overview-Stufe bleibt wichtig, aber nicht als endgültiges Memory.

Funktion der Overview-Stufe:

- globale Projektkarte
- Kandidatenselektion für Detailingestion
- Quelle für Datei-, Funktions- und Verantwortungs-Metadaten
- Fallback-Kontext, wenn kein granularer Memory-Treffer existiert

## Chunking-Entscheidung

### Empfehlung

**Nicht**: ausschließlich `Kategorie -> Chunk`.

**Stattdessen**:

1. strukturellen Chunk erzeugen
2. optional semantisch unterteilen
3. Kategorien als Metadaten und Sekundärschlüssel speichern
4. Embeddings pro finalem Chunk erzeugen

### Begründung

Rein kategoriales Chunking erzeugt Probleme:

- instabile Chunk-Grenzen
- schwer reproduzierbare Reindexierung
- starke Modellabhängigkeit
- schwierige Diffs zwischen Versionen
- unklare Positionsreferenzen zur Quelle

Hybrides Chunking ist robuster:

- strukturale Grenzen bleiben nachvollziehbar
- induktive Kategorien erhöhen Abrufqualität
- Versionierung bleibt beherrschbar
- Training erhält saubere Labels pro Chunk

## Versionierung und Deduplication

### Versionierungsregeln

Jeder Chunk trägt:

- `source_id`
- `content_hash`
- `text_hash`
- `category_version`
- `embedding_model`
- `created_at`
- `superseded_by`

Regeln:

- gleicher `text_hash` und gleiche aktive Version -> kein neuer Chunk
- gleicher struktureller Ursprung, aber geänderter Inhalt -> neue Chunk-Version
- geänderte Kategorieschemata -> Reindexing, aber nicht zwingend neuer Source-Snapshot
- Embedding-Modellwechsel -> Re-Embedding ohne inhaltliche Neu-Ingestion

### Dedup-Strategie

Dedup in drei Ebenen:

1. **Exact dedup** über `text_hash`
2. **Near dedup** über Embedding-Similarity oberhalb eines Schwellwerts
3. **Canonicalization**: eine kanonische Chunk-ID, mehrere Referenzen

## MCP-Tool-Verträge

### `memory.put`

Input:

```json
{
  "source": {...},
  "content": "...",
  "scope": "repo",
  "tags": ["auth"],
  "ttl_seconds": null,
  "ingest_mode": "structural+semantic"
}
```

Output:

```json
{
  "source_id": "...",
  "chunk_ids": ["..."],
  "indexed": true,
  "deduped": 2,
  "superseded": 1
}
```

### `memory.search`

Input:

```json
{
  "query": "Where is auth state mutated?",
  "scope": "repo",
  "filters": {
    "repo": "owner/name",
    "categories": ["auth", "state-transition"]
  },
  "top_k": 8,
  "include_text": true
}
```

Output:

```json
{
  "results": [
    {
      "chunk_id": "...",
      "score": 0.81,
      "category_labels": ["auth"],
      "source_id": "...",
      "text": "...",
      "summary": "..."
    }
  ]
}
```

### `memory.explain`

Liefert für einen Treffer:

- warum er gefunden wurde
- welche Filter gegriffen haben
- welche Scores angesetzt wurden
- welche Quelle zugrunde liegt

## Trainingsarchitektur

### Ziel

Die Architektur soll später modellverbessernd genutzt werden, ohne die Kernfunktion davon abhängig zu machen.

### Zu loggende Ereignisse

- Ingestion-Entscheidungen
- Chunk-Splits
- Kategorien
- Retrieval-Queries
- Trefferlisten
- Nutzerfeedback
- akzeptierte vs. verworfene Treffer
- finale in Prompt übernommene Chunks

### Datennutzung

Später möglich für:

- Chunking-Optimierung
- Kategoriemodell-Finetuning
- Retrieval-Re-Ranker
- Query-Rewriting
- Halluzinationsreduktion durch bessere Kontextselektion

## Security und Governance

- lokaler Betrieb standardmäßig
- keine implizite Exfiltration
- Quellrechte und Scope pro Memory-Eintrag
- Invalidierung bei Source-Löschung oder Scope-Entzug
- TTL nur für flüchtige Memories, nicht für Reposnapshots
- Audit-Trail für `put`, `update`, `invalidate`, `reindex`

## Betriebsmodi

### Modus A: Offline Local Dev

- Ollama lokal
- SQLite lokal
- ChromaDB lokal
- optional Redis lokal
- keine externen Dienste

### Modus B: Team-Setup

- zentraler MCP-Server
- geteilte ChromaDB/Redis-Instanz
- getrennte Namespaces pro Projekt und Nutzer
- Freigaben über Scope und ACL

## WebUI-Perspektive

Spätere WebUI steuert:

- Ingestion starten/stoppen
- Reindexing auslösen
- Kategorien ansehen und korrigieren
- Retrieval-Treffer inspizieren
- Trainingslogs prüfen
- Policies konfigurieren
- Hot memories und tote Chunks anzeigen

Die WebUI ist Verwaltungsoberfläche, nicht Kernarchitektur.

## Referenzfluss

### Ingestion

```text
Repository/File/Doc
-> Snapshot
-> Structural Chunker
-> Optional Inductive Mayring Categorizer
-> Metadata + Dedup Resolver
-> SQLite
-> KV Cache
-> Embeddings
-> ChromaDB
```

### Retrieval

```text
Claude task
-> MCP memory.search
-> scope filter
-> symbolic retrieval
-> vector retrieval
-> re-rank
-> compress
-> prompt context
-> Claude response
```

### Feedback Loop

```text
Claude usage
-> selected chunks
-> user feedback / success signal
-> log store
-> evaluation dataset
-> future training or reranking
```

## Implementierungsphasen

### Phase 1: Minimal funktionsfähig

- MCP-Server mit `memory.put`, `memory.search`, `memory.invalidate`
- strukturelles Chunking
- SQLite + ChromaDB
- Source-/Chunk-Metadaten
- Retrieval für Claude

### Phase 2: Semantische Anreicherung

- Qwen-basiertes induktives Kategorisieren
- Kategorienormalisierung
- Hybrid Retrieval mit Filtern + Vektorsuche
- Explainability pro Treffer

### Phase 3: Qualitäts- und Governance-Schicht

- Dedup-Resolver
- Versiongraph
- Reindexing-Policies
- Feedbacklogging
- Retrieval-Metriken

### Phase 4: Trainings- und UI-Schicht

- Trainingsdatensammlung
- Re-Ranker
- Query-Rewriter
- WebUI

## Klare Architekturentscheidung

### Soll `Kategorie -> Chunk` direkt in den Cache?

**Ja, aber nur als Zugriffsmuster, nicht als Primärmodell.**

Konkret:

- Kategorie gehört in Key und Metadaten.
- Primäridentität eines Chunks kommt aus Quelle, Position, Hash und Version.
- Cache-Key darf Kategorie enthalten, aber Kategorie allein definiert keinen Chunk.

### Soll Mayring-Induktion das Chunking steuern?

**Teilweise.**

- strukturelle Grenzen definieren den Start
- induktive Analyse darf verfeinern und labeln
- finale Retrieval-Einheit ist ein stabiler, referenzierbarer Chunk mit semantischen Labels

### Ist MayringCoder eine gute Basis?

**Ja, als Ausgangsbasis für lokale Ingestion, Overview, Caching, Embedding und Retrieval.**

Nicht ausreichend sind aktuell:

- dediziertes externes Memory-Schema
- MCP-Server-Verträge
- Chunk-Versiongraph
- dedizierte Memory-Governance
- UI für Steuerung und Evaluation

## Akzeptanzkriterien

Die Zielarchitektur ist erreicht, wenn:

1. Claude über MCP persistent gespeicherte Chunks schreiben und lesen kann.
2. geänderte Quellen inkrementell reingestiert werden.
3. Retrieval nach Kategorie, Quelle und Semantik kombiniert funktioniert.
4. jeder Treffer auf eine konkrete Quelle und Version zurückführbar ist.
5. Dubletten und veraltete Versionen kontrolliert behandelt werden.
6. Trainings- und Feedbackdaten optional erfasst werden.
7. das System vollständig lokal betrieben werden kann.

## Kurzfazit

MayringCoder ist eine tragfähige Basis. Die Zielarchitektur sollte auf **hybridem Chunking**, **kategoriengestützter Indexierung**, **MCP-gekapseltem Memory-Zugriff** und **versioniertem Hybrid Storage** aufbauen. Induktive Mayring-Inhaltsanalyse ist dabei ein starker semantischer Layer, aber nicht die alleinige Definition der Chunk-Grenzen.
